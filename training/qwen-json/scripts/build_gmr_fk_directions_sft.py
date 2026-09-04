#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.optimize import least_squares


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUMANOID_ROOT = PROJECT_ROOT.parent


SYSTEM = """Ты робототехнический ассистент для motion JSON.
Тебе дают INITIAL_STATE, URDF/MJCF_CONTEXT, JOINT_CANDIDATES, LIMITS_DEG и FK_DIRECTIONS.
FK_DIRECTIONS показывает, куда двигается ладонь/end-effector при изменении каждого сустава; это подсказка для рассуждения, а не готовая поза.
Для /motion_json верни только валидный compact JSON-массив без markdown и без пояснений.
Compact format: [{"frame":{"exact_joint_name":angle_deg},"duration":seconds},{"frame":{},"duration":3.0}].
Используй только exact joint names из JOINT_CANDIDATES. Не используй имена body/link.
Если сустав не указан в новом frame, он сохраняет прошлое значение.
Учитывай INITIAL_STATE: не меняй суставы без FK-необходимости; no-op допустим только когда движение не улучшает FK-цель.
Все углы в compact JSON в градусах и должны быть внутри LIMITS_DEG."""


TASKS = {
    "t_pose": {
        "train": ["Разведи обе руки строго в стороны", "Поставь руки горизонтально в стороны", "Сделай руки в стороны как буква Т"],
        "val": ["Сделай Т-позу руками"],
        "hint": "Цель: левая ладонь уходит влево от робота, правая ладонь вправо от робота; обе примерно на высоте плеч.",
    },
    "arms_up": {
        "train": ["Подними обе руки вверх", "Руки вверх над головой", "Покажи обеими руками наверх"],
        "val": ["Подними руки к небу"],
        "hint": "Цель: обе ладони/end-effector идут вверх по +Z, руки подняты выше плеч.",
    },
    "greeting": {
        "train": ["Поздоровайся правой рукой", "Покажи приветствие правой рукой", "Подними правую руку для приветствия"],
        "val": ["Помаши привет правой рукой"],
        "hint": "Цель: правая ладонь поднята около головы/плеча, локоть согнут; затем маленькое движение кистью/предплечьем как приветствие. Левая рука спокойная.",
    },
}


@dataclass
class RobotProfile:
    robot: str
    scene: Path
    model: mujoco.MjModel
    left_joints: list[str]
    right_joints: list[str]
    left_body: str
    right_body: str
    fk_text: str
    limits_text: str
    mjcf_text: str
    initial_text: str
    body_center: np.ndarray
    target_z: float
    span: float
    side_axis: np.ndarray
    front_axis: np.ndarray
    up_axis: np.ndarray


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def safe_float(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(x)


def joint_names(model: mujoco.MjModel) -> list[str]:
    out = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name:
            out.append(name)
    return out


def side_of_name(name: str) -> str | None:
    s = name.lower()
    if re.search(r"(^|[_\W])(left|l)([_\W]|$)", s) or "arm_l" in s or "zarm_l" in s or s.endswith("_l") or s.endswith("_left"):
        return "left"
    if re.search(r"(^|[_\W])(right|r)([_\W]|$)", s) or "arm_r" in s or "zarm_r" in s or s.endswith("_r") or s.endswith("_right"):
        return "right"
    if "_l" in s and any(k in s for k in ("shoulder", "elbow", "wrist", "arm", "sho")):
        return "left"
    if "_r" in s and any(k in s for k in ("shoulder", "elbow", "wrist", "arm", "sho")):
        return "right"
    return None


def is_arm_joint(name: str) -> bool:
    s = name.lower()
    if any(k in s for k in ("hip", "knee", "ankle", "leg", "head", "waist", "torso", "wheel", "steer", "base", "root", "float")):
        return False
    return any(k in s for k in ("shoulder", "sho", "elbow", "wrist", "hand", "arm", "zarm", "upper"))


def split_arm_joints(model: mujoco.MjModel) -> tuple[list[str], list[str]]:
    left: list[str] = []
    right: list[str] = []
    for name in joint_names(model):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        if not is_arm_joint(name):
            continue
        side = side_of_name(name)
        if side == "left":
            left.append(name)
        elif side == "right":
            right.append(name)
    return left, right


def body_names(model: mujoco.MjModel) -> list[str]:
    out = []
    for bid in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name:
            out.append(name)
    return out


def joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return float(data.qpos[int(model.jnt_qposadr[jid])])


def set_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str, value_rad: float) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[int(model.jnt_qposadr[jid])] = value_rad


def get_body_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[bid].copy()


def affected_body_score(model: mujoco.MjModel, joints: list[str], body: str) -> float:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base = get_body_pos(model, data, body)
    total = 0.0
    for j in joints:
        old = joint_qpos(model, data, j)
        set_joint_qpos(model, data, j, old + math.radians(20))
        mujoco.mj_forward(model, data)
        total += float(np.linalg.norm(get_body_pos(model, data, body) - base))
        set_joint_qpos(model, data, j, old)
    return total


def select_end_body(model: mujoco.MjModel, side: str, joints: list[str]) -> str:
    candidates = []
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name in body_names(model):
        s = name.lower()
        if any(k in s for k in ("toe", "foot", "hip", "knee", "ankle", "leg", "waist", "head", "torso")):
            continue
        side_ok = side_of_name(name) == side
        if side == "left":
            side_ok = side_ok or bool(re.search(r"(^|[_\W])(left|l)([_\W]|$)", s)) or "arm_l" in s or "zarm_l" in s
        else:
            side_ok = side_ok or bool(re.search(r"(^|[_\W])(right|r)([_\W]|$)", s)) or "arm_r" in s or "zarm_r" in s
        if not side_ok:
            continue
        keyword_score = 0
        for kw, pts in [
            ("hand", 8),
            ("palm", 8),
            ("wrist", 6),
            ("gripper", 7),
            ("finger", 5),
            ("end", 5),
            ("link7", 4),
            ("_7", 3),
            ("forearm", 2),
        ]:
            if kw in s:
                keyword_score += pts
        if keyword_score <= 0:
            continue
        try:
            move_score = affected_body_score(model, joints, name)
        except Exception:
            move_score = 0.0
        candidates.append((keyword_score + move_score * 20.0, name))
    if candidates:
        return max(candidates)[1]

    # Some MJCFs use generic body names like "hand" / "hand_2" without left/right
    # tokens. In that case use neutral lateral position to assign the side.
    generic = []
    desired_y_sign = 1.0 if side == "left" else -1.0
    for name in body_names(model):
        s = name.lower()
        if not any(k in s for k in ("hand", "palm", "wrist", "gripper", "finger", "end")):
            continue
        y = float(get_body_pos(model, data, name)[1])
        lateral_score = desired_y_sign * y
        if lateral_score < -1e-4:
            continue
        try:
            move_score = affected_body_score(model, joints, name)
        except Exception:
            move_score = 0.0
        generic.append((10.0 + lateral_score * 10.0 + move_score * 20.0, name))
    if generic:
        return max(generic)[1]

    best = ("", -1.0)
    for name in body_names(model):
        s = name.lower()
        if any(k in s for k in ("toe", "foot", "hip", "knee", "ankle", "leg", "waist", "head", "torso")):
            continue
        score = affected_body_score(model, joints, name)
        if score > best[1]:
            best = (name, score)
    return best[0]


def limit_deg(model: mujoco.MjModel, name: str) -> tuple[float, float]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    lo, hi = model.jnt_range[jid]
    if not model.jnt_limited[jid] or (abs(lo) < 1e-9 and abs(hi) < 1e-9):
        lo, hi = -math.pi, math.pi
    return math.degrees(float(lo)), math.degrees(float(hi))


def clipped_sample_range(model: mujoco.MjModel, name: str) -> tuple[float, float]:
    lo, hi = limit_deg(model, name)
    return max(lo, -145.0), min(hi, 145.0)


def fk_direction_lines(model: mujoco.MjModel, side: str, joints: list[str], body: str) -> list[str]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base = get_body_pos(model, data, body)
    lines = [f"{side}_end_effector_body={body}", f"{side}_neutral_xyz=[{base[0]:.3f},{base[1]:.3f},{base[2]:.3f}]"]
    for j in joints:
        old = joint_qpos(model, data, j)
        parts = []
        for sign in (1, -1):
            set_joint_qpos(model, data, j, old + sign * math.radians(25))
            mujoco.mj_forward(model, data)
            delta = get_body_pos(model, data, body) - base
            label = "+25deg" if sign > 0 else "-25deg"
            parts.append(f"{label}: dXYZ=[{delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f}]")
            set_joint_qpos(model, data, j, old)
        lines.append(f"- {j}: " + "; ".join(parts))
    return lines


def xml_joint_snippets(scene: Path, wanted: set[str], max_chars: int = 8000) -> str:
    try:
        root = ET.parse(scene).getroot()
    except Exception:
        return "MJCF joint snippets unavailable: XML parse failed."
    snippets = []
    for elem in root.iter("joint"):
        name = elem.get("name")
        if name in wanted:
            snippets.append(ET.tostring(elem, encoding="unicode", short_empty_elements=True))
    text = "\n".join(snippets)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...truncated..."
    return text or "No inline joint snippets found; use JOINT_CANDIDATES and FK_DIRECTIONS."


def robot_geometry(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, float, float]:
    pts = data.xpos[1:].copy() if model.nbody > 1 else np.zeros((1, 3))
    center = np.mean(pts, axis=0)
    zmin = float(np.min(pts[:, 2]))
    zmax = float(np.max(pts[:, 2]))
    span = max(float(np.linalg.norm(np.max(pts, axis=0) - np.min(pts, axis=0))), 0.5)
    # A conservative shoulder/upper-body height proxy. It works better than
    # "neutral hand z + delta" when the default pose is already arms-out.
    target_z = zmin + 0.64 * (zmax - zmin)
    return center, target_z, span


def initial_state_text(model: mujoco.MjModel, left: list[str], right: list[str], left_body: str, right_body: str) -> tuple[str, np.ndarray, float, float]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    center, target_z, span = robot_geometry(model, data)
    left_pos = get_body_pos(model, data, left_body)
    right_pos = get_body_pos(model, data, right_body)
    lines = [
        "Initial pose is the state before executing the JSON.",
        f"body_center_xyz=[{center[0]:.3f},{center[1]:.3f},{center[2]:.3f}]",
        f"upper_body_target_z_for_T_pose≈{target_z:.3f}",
        f"left_end_effector_initial_xyz=[{left_pos[0]:.3f},{left_pos[1]:.3f},{left_pos[2]:.3f}]",
        f"right_end_effector_initial_xyz=[{right_pos[0]:.3f},{right_pos[1]:.3f},{right_pos[2]:.3f}]",
        "initial_joint_angles_deg:",
    ]
    for j in left + right:
        lines.append(f"- {j}: {math.degrees(joint_qpos(model, data, j)):.1f}")
    return "\n".join(lines), center, target_z, span


def unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return fallback.astype(float)
    return v / n


def infer_axes(model: mujoco.MjModel, left: list[str], right: list[str], left_body: str, right_body: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    left_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, left[0])
    right_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, right[0])
    left_anchor = data.xanchor[left_jid].copy()
    right_anchor = data.xanchor[right_jid].copy()
    left_hand = get_body_pos(model, data, left_body)
    right_hand = get_body_pos(model, data, right_body)

    up_axis = np.array([0.0, 0.0, 1.0])
    raw_side = (left_anchor - right_anchor) + 0.25 * (left_hand - right_hand)
    raw_side[2] = 0.0
    side_axis = unit(raw_side, np.array([0.0, 1.0, 0.0]))
    front_axis = unit(np.cross(side_axis, up_axis), np.array([1.0, 0.0, 0.0]))
    if front_axis[0] < 0:
        front_axis = -front_axis
    return side_axis, front_axis, up_axis


def make_profile(robot: str, scene: Path) -> RobotProfile | None:
    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
    except Exception as exc:
        print(f"[skip] {robot}: cannot load {scene}: {exc}")
        return None
    left, right = split_arm_joints(model)
    if not left or not right:
        print(f"[skip] {robot}: no bilateral arm joints found")
        return None
    left_body = select_end_body(model, "left", left)
    right_body = select_end_body(model, "right", right)
    if not left_body or not right_body:
        print(f"[skip] {robot}: no end effector body found")
        return None
    limits = []
    for j in left + right:
        lo, hi = limit_deg(model, j)
        limits.append(f"- {j}: [{lo:.1f},{hi:.1f}]")
    fk_lines = ["LEFT_ARM:"] + fk_direction_lines(model, "left", left, left_body) + ["RIGHT_ARM:"] + fk_direction_lines(model, "right", right, right_body)
    init_text, center, target_z, span = initial_state_text(model, left, right, left_body, right_body)
    side_axis, front_axis, up_axis = infer_axes(model, left, right, left_body, right_body)
    wanted = set(left + right)
    return RobotProfile(
        robot=robot,
        scene=scene,
        model=model,
        left_joints=left,
        right_joints=right,
        left_body=left_body,
        right_body=right_body,
        fk_text="\n".join(fk_lines),
        limits_text="\n".join(limits),
        mjcf_text=xml_joint_snippets(scene, wanted),
        initial_text=init_text,
        body_center=center,
        target_z=target_z,
        span=span,
        side_axis=side_axis,
        front_axis=front_axis,
        up_axis=up_axis,
    )


def current_joint_frame(model: mujoco.MjModel, data: mujoco.MjData, joints: list[str]) -> dict[str, float]:
    return {j: math.degrees(joint_qpos(model, data, j)) for j in joints}


def apply_frame(model: mujoco.MjModel, data: mujoco.MjData, frame: dict[str, float]) -> None:
    for j, deg in frame.items():
        set_joint_qpos(model, data, j, math.radians(deg))


def joint_anchor(model: mujoco.MjModel, data: mujoco.MjData, joint: str) -> np.ndarray:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    return data.xanchor[jid].copy()


def arm_reach(model: mujoco.MjModel, data: mujoco.MjData, joints: list[str], body: str) -> float:
    pts = [joint_anchor(model, data, j) for j in joints]
    pts.append(get_body_pos(model, data, body))
    reach = 0.0
    for a, b in zip(pts, pts[1:]):
        reach += float(np.linalg.norm(b - a))
    return max(reach, float(np.linalg.norm(pts[-1] - pts[0])), 0.25)


def target_points_for_side(profile: RobotProfile, side: str, task: str) -> list[tuple[str, str, np.ndarray, float]]:
    model = profile.model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joints = profile.left_joints if side == "left" else profile.right_joints
    body = profile.left_body if side == "left" else profile.right_body
    shoulder = joint_anchor(model, data, joints[0])
    mid_joint = joints[max(0, min(len(joints) - 1, len(joints) // 2))]
    reach = arm_reach(model, data, joints, body)
    side_sign = 1.0 if side == "left" else -1.0
    side_vec = side_sign * profile.side_axis

    if task == "t_pose":
        mid_target = shoulder + side_vec * (0.45 * reach)
        hand_target = shoulder + side_vec * (0.86 * reach)
        return [("joint", mid_joint, mid_target, 0.65), ("body", body, hand_target, 1.0)]
    if task == "arms_up":
        mid_target = shoulder + profile.up_axis * (0.46 * reach) + side_vec * (0.08 * reach)
        hand_target = shoulder + profile.up_axis * (0.86 * reach) + side_vec * (0.12 * reach)
        return [("joint", mid_joint, mid_target, 0.65), ("body", body, hand_target, 1.0)]
    if task == "greeting":
        mid_target = shoulder + profile.up_axis * (0.26 * reach) + side_vec * (0.18 * reach) + profile.front_axis * (0.10 * reach)
        hand_target = shoulder + profile.up_axis * (0.50 * reach) + side_vec * (0.24 * reach) + profile.front_axis * (0.18 * reach)
        return [("joint", mid_joint, mid_target, 0.45), ("body", body, hand_target, 1.0)]
    raise ValueError(task)


def solve_ik_side(profile: RobotProfile, side: str, task: str) -> tuple[dict[str, float], float, float]:
    model = profile.model
    joints = profile.left_joints if side == "left" else profile.right_joints
    body = profile.left_body if side == "left" else profile.right_body
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if task in {"t_pose", "arms_up"}:
        active_joints = joints[: max(2, min(len(joints), 4))]
    else:
        active_joints = joints[: max(2, min(len(joints), 5))]
    initial_deg = np.array([math.degrees(joint_qpos(model, data, j)) for j in active_joints], dtype=float)
    bounds = np.array([clipped_sample_range(model, j) for j in active_joints], dtype=float)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    x0 = np.clip(initial_deg, lo, hi)
    targets = target_points_for_side(profile, side, task)

    def fk_positions(x_deg: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
        local = mujoco.MjData(model)
        mujoco.mj_forward(model, local)
        for j, deg in zip(active_joints, x_deg):
            set_joint_qpos(model, local, j, math.radians(float(deg)))
        mujoco.mj_forward(model, local)
        out: dict[tuple[str, str], np.ndarray] = {}
        for kind, name, _, _ in targets:
            if kind == "body":
                out[(kind, name)] = get_body_pos(model, local, name)
            else:
                out[(kind, name)] = joint_anchor(model, local, name)
        return out

    def target_error(x_deg: np.ndarray) -> float:
        positions = fk_positions(x_deg)
        weighted = []
        for kind, name, target, weight in targets:
            weighted.append(weight * float(np.linalg.norm(positions[(kind, name)] - target)))
        return float(sum(weighted) / max(sum(w for *_, w in targets), 1e-6))

    initial_err = target_error(x0)
    reg_scale = max(profile.span, 0.5)

    def residual(x_deg: np.ndarray) -> np.ndarray:
        positions = fk_positions(x_deg)
        pos_res = []
        for kind, name, target, weight in targets:
            pos_res.extend((weight * (positions[(kind, name)] - target) / reg_scale).tolist())
        # Mild regularization makes "already correct" starts stay unchanged,
        # but does not block a real IK improvement.
        reg = 0.015 * (x_deg - x0) / 90.0
        return np.concatenate([np.array(pos_res), reg])

    starts = [x0]
    for bias in (-90, -45, 0, 45, 90):
        starts.append(np.clip(np.full_like(x0, bias, dtype=float), lo, hi))
    best_x = x0
    best_cost = float(np.linalg.norm(residual(x0)))
    for start in starts:
        res = least_squares(residual, start, bounds=(lo, hi), max_nfev=220, xtol=1e-5, ftol=1e-5, gtol=1e-5)
        cost = float(np.linalg.norm(res.fun))
        if cost < best_cost:
            best_cost = cost
            best_x = res.x

    solved_err = target_error(best_x)
    # If IK does not improve physical end-effector target meaningfully, keep the
    # initial state. This is not a semantic shortcut; it is the FK objective.
    if solved_err > initial_err - 0.035 * max(profile.span, 0.5):
        best_x = x0
        solved_err = initial_err

    out: dict[str, float] = {}
    for j, new, old in zip(active_joints, best_x, initial_deg):
        if abs(float(new - old)) >= 1.5:
            out[j] = round(float(new), 1)
    return out, initial_err, solved_err


def wave_joint(joints: list[str]) -> str | None:
    preferred = [j for j in joints if any(k in j.lower() for k in ("wrist", "yaw", "elbow"))]
    return (preferred or joints)[-1] if joints else None


def make_answer(profile: RobotProfile, task: str, rng: random.Random) -> list[dict[str, Any]]:
    if task == "t_pose":
        frame = {}
        left, _, _ = solve_ik_side(profile, "left", task)
        right, _, _ = solve_ik_side(profile, "right", task)
        frame.update(left)
        frame.update(right)
        if not frame:
            return [{"frame": {}, "duration": 3.0}]
        return [{"frame": frame, "duration": 2.2}, {"frame": {}, "duration": 3.0}]
    if task == "arms_up":
        frame = {}
        left, _, _ = solve_ik_side(profile, "left", task)
        right, _, _ = solve_ik_side(profile, "right", task)
        frame.update(left)
        frame.update(right)
        if not frame:
            return [{"frame": {}, "duration": 3.0}]
        return [{"frame": frame, "duration": 2.4}, {"frame": {}, "duration": 3.0}]
    if task == "greeting":
        frame, _, _ = solve_ik_side(profile, "right", task)
        wj = wave_joint(profile.right_joints)
        if not wj:
            return [{"frame": frame, "duration": 1.8}, {"frame": {}, "duration": 3.0}]
        lo, hi = clipped_sample_range(profile.model, wj)
        base = frame.get(wj, 0.0)
        a = max(lo, min(hi, base - 18.0))
        b = max(lo, min(hi, base + 18.0))
        return [
            {"frame": frame, "duration": 1.6},
            {"repeat": [{"frame": {wj: round(a, 1)}, "duration": 0.55}, {"frame": {wj: round(b, 1)}, "duration": 0.55}], "times": 2},
            {"frame": frame, "duration": 0.8},
            {"frame": {}, "duration": 3.0},
        ]
    raise ValueError(task)


def prompt_for(profile: RobotProfile, command: str, hint: str) -> str:
    return f"""/motion_json
ROBOT={profile.robot}
SCENE={profile.scene}

TASK_SEMANTICS:
{hint}

INITIAL_STATE:
{profile.initial_text}

JOINT_CANDIDATES:
left_arm={json.dumps(profile.left_joints, ensure_ascii=False)}
right_arm={json.dumps(profile.right_joints, ensure_ascii=False)}

LIMITS_DEG:
{profile.limits_text}

URDF/MJCF_CONTEXT:
{profile.mjcf_text}

FK_DIRECTIONS:
{profile.fk_text}

Command:
{command}"""


def strongest_fk_joint(profile: RobotProfile, side: str, axis: int, sign: float) -> str:
    model = profile.model
    joints = profile.left_joints if side == "left" else profile.right_joints
    body = profile.left_body if side == "left" else profile.right_body
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base = get_body_pos(model, data, body)
    best = (None, -1e9)
    for j in joints:
        old = joint_qpos(model, data, j)
        for delta in (25, -25):
            set_joint_qpos(model, data, j, old + math.radians(delta))
            mujoco.mj_forward(model, data)
            d = get_body_pos(model, data, body) - base
            score = sign * float(d[axis])
            set_joint_qpos(model, data, j, old)
            if score > best[1]:
                best = (f"{j} ({delta:+d}deg gives dXYZ[{axis}]={d[axis]:+.3f})", score)
    return best[0] or joints[0]


def make_qa_rows(profile: RobotProfile, sample_id: int) -> tuple[list[dict[str, Any]], int]:
    qa_specs = [
        (
            "robot_qa",
            f"/robot_qa\nROBOT={profile.robot}\nINITIAL_STATE:\n{profile.initial_text}\n\nJOINT_CANDIDATES:\nleft_arm={json.dumps(profile.left_joints, ensure_ascii=False)}\nright_arm={json.dumps(profile.right_joints, ensure_ascii=False)}\n\nFK_DIRECTIONS:\n{profile.fk_text}\n\nВопрос: Как использовать INITIAL_STATE при генерации движения?",
            "INITIAL_STATE - это стартовая FK-поза. Менять надо только те суставы, которые приближают end-effector к FK-цели команды. Если любое движение не улучшает FK-ошибку, надо вернуть no-op motion: [{\"frame\":{},\"duration\":3.0}].",
            "fk_initial_state_policy",
        ),
        (
            "robot_qa",
            f"/robot_qa\nROBOT={profile.robot}\nJOINT_CANDIDATES:\nleft_arm={json.dumps(profile.left_joints, ensure_ascii=False)}\nright_arm={json.dumps(profile.right_joints, ensure_ascii=False)}\n\nFK_DIRECTIONS:\n{profile.fk_text}\n\nВопрос: Какой сустав/знак сильнее всего помогает поднять правую ладонь вверх по +Z?",
            f"По FK_DIRECTIONS сильная подсказка для подъёма правой ладони вверх: {strongest_fk_joint(profile, 'right', 2, +1.0)}. Использовать надо exact joint name, не body/link name.",
            "fk_direction_up",
        ),
        (
            "robot_qa",
            f"/robot_qa\nROBOT={profile.robot}\nJOINT_CANDIDATES:\nleft_arm={json.dumps(profile.left_joints, ensure_ascii=False)}\nright_arm={json.dumps(profile.right_joints, ensure_ascii=False)}\n\nFK_DIRECTIONS:\n{profile.fk_text}\n\nВопрос: Какие имена можно писать в compact JSON?",
            "В compact JSON можно писать только exact joint names из JOINT_CANDIDATES. Имена body/link/end-effector из URDF/MJCF_CONTEXT писать нельзя.",
            "joint_name_policy",
        ),
    ]
    rows = []
    for task, user, answer, category in qa_specs:
        rows.append(
            {
                "sample_id": sample_id,
                "task": task,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": answer},
                ],
                "meta": {"robot": profile.robot, "scene": str(profile.scene), "name": f"{profile.robot}_{category}", "category": category},
            }
        )
        sample_id += 1
    return rows, sample_id


def make_rows(profiles: list[RobotProfile], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    sample_id = 9_000_000
    for profile in profiles:
        answers = {task: make_answer(profile, task, rng) for task in TASKS}
        qa_rows, sample_id = make_qa_rows(profile, sample_id)
        train.extend(qa_rows[:2])
        val.extend(qa_rows[2:])
        for task, spec in TASKS.items():
            answer_text = json.dumps(answers[task], ensure_ascii=False, separators=(",", ":"))
            for command in spec["train"]:
                train.append(
                    {
                        "sample_id": sample_id,
                        "task": "motion_json",
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt_for(profile, command, spec["hint"])},
                            {"role": "assistant", "content": answer_text},
                        ],
                        "meta": {"robot": profile.robot, "scene": str(profile.scene), "name": f"{profile.robot}_{task}_train"},
                    }
                )
                sample_id += 1
            for command in spec["val"]:
                val.append(
                    {
                        "sample_id": sample_id,
                        "task": "motion_json",
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt_for(profile, command, spec["hint"])},
                            {"role": "assistant", "content": answer_text},
                        ],
                        "meta": {"robot": profile.robot, "scene": str(profile.scene), "name": f"{profile.robot}_{task}_val"},
                    }
                )
                sample_id += 1
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=HUMANOID_ROOT / "resources" / "gmr_robots" / "manifest.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "dataset_gmr_fk_ik_initial_multitask_sft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-unitree", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    profiles: list[RobotProfile] = []
    for item in manifest:
        robot = item["robot"]
        if not args.include_unitree and robot.startswith("unitree"):
            continue
        if robot == "agibot_a2":
            continue
        if not item.get("available") or not item.get("scenes"):
            continue
        profile = make_profile(robot, Path(item["scenes"][0]))
        if profile is not None:
            profiles.append(profile)
            print(f"[profile] {robot}: left={len(profile.left_joints)} right={len(profile.right_joints)} bodies=({profile.left_body},{profile.right_body})")

    train, val = make_rows(profiles, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in [("train", train), ("val", val)]:
        write_jsonl(args.out_dir / f"{split}.jsonl", rows)
        write_jsonl(args.out_dir / f"{split}_compact_4096.jsonl", rows)
        write_jsonl(args.out_dir / f"{split}_compact_8192.jsonl", rows)

    manifest_out = {
        "dataset_kind": "gmr_fk_ik_initial_multitask_sft",
        "robots": [p.robot for p in profiles],
        "tasks": list(TASKS),
        "train_rows": len(train),
        "val_rows": len(val),
        "format": "INITIAL_STATE + URDF/MJCF + FK_DIRECTIONS + least-squares FK/IK target -> compact JSON, plus FK QA",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
