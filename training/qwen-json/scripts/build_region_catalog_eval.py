#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_gmr_fk_directions_sft import clipped_sample_range, get_body_pos, joint_anchor, make_profile, set_joint_qpos  # noqa: E402
from resolve_high_level_motion import HUMANOID_ROOT, body_frame_origin_scale, load_manifest_scene  # noqa: E402


DEFAULT_ROBOTS = ["tienkung", "booster_k1", "fourier_n1", "berkeley_humanoid_lite", "openloong"]

DEFAULT_COMMANDS = [
    "подними правую руку вертикально вверх",
    "подними правую руку вверх под углом в сторону",
    "сделай Т-позу",
    "вытяни обе руки вперед и вверх",
    "скажи привет правой рукой",
]


def body_coords(profile, pos: np.ndarray, origin: np.ndarray, scale: float) -> dict[str, float]:
    v = pos - origin
    return {
        "x": round(float(np.dot(v, profile.front_axis) / scale), 3),
        "y": round(float(np.dot(v, profile.side_axis) / scale), 3),
        "z": round(float(np.dot(v, profile.up_axis) / scale), 3),
    }


def find_joint(joints: list[str], key: str) -> str | None:
    key = key.lower()
    for joint in joints:
        if key in joint.lower():
            return joint
    return None


def sample_reachable(profile, side: str, origin: np.ndarray, scale: float, n: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    model = profile.model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joints = profile.left_joints if side == "left" else profile.right_joints
    body = profile.left_body if side == "left" else profile.right_body
    active = joints[: max(2, min(len(joints), 7))]
    bounds = np.array([clipped_sample_range(model, j) for j in active], dtype=float)
    pts = [body_coords(profile, get_body_pos(model, data, body), origin, scale)]
    for _ in range(n):
        local = mujoco.MjData(model)
        mujoco.mj_forward(model, local)
        degs = rng.uniform(bounds[:, 0], bounds[:, 1])
        for joint, deg in zip(active, degs):
            set_joint_qpos(model, local, joint, math.radians(float(deg)))
        mujoco.mj_forward(model, local)
        pts.append(body_coords(profile, get_body_pos(model, local, body), origin, scale))
    arr = np.array([[p["x"], p["y"], p["z"]] for p in pts], dtype=float)
    center = arr.mean(axis=0)
    return {
        "x_min": round(float(arr[:, 0].min()), 3),
        "x_max": round(float(arr[:, 0].max()), 3),
        "y_min": round(float(arr[:, 1].min()), 3),
        "y_max": round(float(arr[:, 1].max()), 3),
        "z_min": round(float(arr[:, 2].min()), 3),
        "z_max": round(float(arr[:, 2].max()), 3),
        "sample_center": {"x": round(float(center[0]), 3), "y": round(float(center[1]), 3), "z": round(float(center[2]), 3)},
    }


def axis_interval(a: float, b: float) -> list[float]:
    return [round(min(a, b), 3), round(max(a, b), 3)]


def directed_interval(start: float, end: float) -> list[float]:
    return [round(start, 3), round(end, 3)]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def hand_catalog(profile, side: str, origin: np.ndarray, scale: float, seed: int) -> dict[str, Any]:
    model = profile.model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joints = profile.left_joints if side == "left" else profile.right_joints
    wrist_body = profile.left_body if side == "left" else profile.right_body
    shoulder_joint = joints[0]
    elbow_joint = find_joint(joints, "elbow")
    shoulder = body_coords(profile, joint_anchor(model, data, shoulder_joint), origin, scale)
    elbow = body_coords(profile, joint_anchor(model, data, elbow_joint), origin, scale) if elbow_joint else shoulder
    wrist = body_coords(profile, get_body_pos(model, data, wrist_body), origin, scale)
    box = sample_reachable(profile, side, origin, scale, n=480, seed=seed)
    shoulder_z = float(shoulder["z"])
    shoulder_x = float(shoulder["x"])
    chest_z = round((float(shoulder["z"]) + float(wrist["z"])) * 0.5, 3)
    shoulder_band = axis_interval(
        clamp(shoulder_z - 0.04, float(box["z_min"]), float(box["z_max"])),
        clamp(shoulder_z + 0.04, float(box["z_min"]), float(box["z_max"])),
    )
    above_shoulder = axis_interval(clamp(shoulder_z + 0.08, float(box["z_min"]), float(box["z_max"])), float(box["z_max"]))
    over_head = axis_interval(clamp(shoulder_z + 0.18, float(box["z_min"]), float(box["z_max"])), float(box["z_max"]))
    same_side_y = (
        directed_interval(float(wrist["y"]), float(box["y_max"]))
        if side == "left"
        else directed_interval(float(wrist["y"]), float(box["y_min"]))
    )
    center_y = (
        directed_interval(float(wrist["y"]), clamp(0.0, float(box["y_min"]), float(box["y_max"])))
    )
    cross_y = (
        directed_interval(float(wrist["y"]), float(box["y_min"]))
        if side == "left"
        else directed_interval(float(wrist["y"]), float(box["y_max"]))
    )
    shoulder_y = float(shoulder["y"])
    shoulder_lateral = axis_interval(
        clamp(shoulder_y - 0.04, float(box["y_min"]), float(box["y_max"])),
        clamp(shoulder_y + 0.04, float(box["y_min"]), float(box["y_max"])),
    )
    return {
        "current_wrist": wrist,
        "same_side_shoulder": shoulder,
        "elbow": elbow,
        "reachable_wrist_box": box,
        "regions": {
            "depth": {
                "keep": {"meaning": "keep the wrist at its current front/back depth", "x": [wrist["x"], wrist["x"]]},
                "body_side_plane": {
                    "meaning": "place the wrist near the shoulder/torso side plane in depth, neither forward nor backward",
                    "x": axis_interval(
                        clamp(shoulder_x - 0.04, float(box["x_min"]), float(box["x_max"])),
                        clamp(shoulder_x + 0.04, float(box["x_min"]), float(box["x_max"])),
                    ),
                },
                "forward": {"meaning": "move the wrist farther forward in front of the torso", "x": directed_interval(float(wrist["x"]), float(box["x_max"]))},
                "backward": {"meaning": "move the wrist farther backward behind the torso", "x": directed_interval(float(wrist["x"]), float(box["x_min"]))},
            },
            "lateral": {
                "keep": {"meaning": "keep the wrist at its current left/right side", "y": [wrist["y"], wrist["y"]]},
                "shoulder_lateral": {
                    "meaning": "near the same-side shoulder y-line; this is close to the upper arm root, not far outward",
                    "y": shoulder_lateral,
                },
                "same_side_outward": {
                    "meaning": "farther away from the torso centerline on this hand's own side; use for broad outward side placement",
                    "y": same_side_y,
                },
                "toward_body_center": {"meaning": "move inward toward the torso centerline without crossing to the other side", "y": center_y},
                "cross_body": {"meaning": "move across the torso toward the opposite side", "y": cross_y},
            },
            "height": {
                "keep": {"meaning": "keep the wrist at its current height", "z": [wrist["z"], wrist["z"]]},
                "low": {"meaning": "lower part of this wrist's reachable workspace", "z": directed_interval(float(wrist["z"]), float(box["z_min"]))},
                "chest_height": {"meaning": "around mid torso or chest height", "z": axis_interval(chest_z - 0.04, chest_z + 0.04)},
                "shoulder_height": {"meaning": "approximately level with the same-side shoulder", "z": shoulder_band},
                "above_shoulder": {"meaning": "above shoulder level but not necessarily maximum height", "z": above_shoulder},
                "over_head": {"meaning": "highest upward part of this wrist's reachable workspace", "z": over_head},
            },
        },
    }


def build_catalog(robot: str, scene: Path, seed: int) -> dict[str, Any]:
    profile = make_profile(robot, scene)
    if profile is None:
        raise RuntimeError(f"cannot build profile for {robot}: {scene}")
    origin, scale = body_frame_origin_scale(profile)
    return {
        "robot": robot,
        "scene": str(scene),
        "coordinate_frame": {
            "origin": "pelvis/torso body frame",
            "x": "larger is forward, smaller is backward",
            "y": "larger is robot-left, smaller is robot-right",
            "z": "larger is upward, smaller is downward",
            "scale_m": round(float(scale), 3),
        },
        "hands": {
            "left_hand": hand_catalog(profile, "left", origin, scale, seed + 1),
            "right_hand": hand_catalog(profile, "right", origin, scale, seed + 2),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--robots", nargs="*", default=DEFAULT_ROBOTS)
    parser.add_argument("--commands", nargs="*", default=DEFAULT_COMMANDS)
    parser.add_argument("--seed", type=int, default=941)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    idx = 1
    with args.out.open("w", encoding="utf-8") as f:
        for r_i, robot in enumerate(args.robots):
            scene = load_manifest_scene(robot, None)
            catalog = build_catalog(robot, scene, args.seed + r_i * 1000)
            for command in args.commands:
                row = {"idx": idx, "robot": robot, "scene": str(scene), "command": command, "region_catalog": catalog}
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                idx += 1
            print(f"[catalog] {robot}: {len(args.commands)} commands")
    print(f"[catalog] wrote {args.out}")


if __name__ == "__main__":
    main()
