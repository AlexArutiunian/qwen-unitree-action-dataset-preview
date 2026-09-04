from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


MIN_ACTIVE_DURATION = 0.75
FORBIDDEN_JOINTS = {"floating_base_joint"}


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def normalize_motion_obj(obj: Any, min_active_duration: float = MIN_ACTIVE_DURATION) -> tuple[Any, list[str]]:
    issues: list[str] = []

    def rec(x: Any) -> Any:
        if isinstance(x, list):
            return [rec(v) for v in x]
        if not isinstance(x, dict):
            return x
        y = dict(x)
        if "duration" in y:
            try:
                y["duration"] = float(y["duration"])
            except Exception:
                issues.append(f"bad_duration:{y.get('duration')}")
        if "frame" in y and isinstance(y["frame"], list):
            frame = []
            for joint in y["frame"]:
                jj = dict(joint) if isinstance(joint, dict) else joint
                if isinstance(jj, dict) and "angle" in jj:
                    try:
                        jj["angle"] = float(jj["angle"])
                    except Exception:
                        issues.append(f"bad_angle:{jj.get('angle')}")
                frame.append(jj)
            y["frame"] = frame
        if "repeat" in y:
            y["repeat"] = rec(y["repeat"])
        return y

    return rec(obj), issues


def iter_steps(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, list):
        for step in obj:
            yield from iter_steps(step)
    elif isinstance(obj, dict):
        yield obj
        if "repeat" in obj:
            yield from iter_steps(obj["repeat"])


def validate_motion_obj(
    obj: Any,
    joint_ranges: dict[str, tuple[float, float]],
    min_active_duration: float = MIN_ACTIVE_DURATION,
) -> list[str]:
    errors: list[str] = []
    allowed_joints = set(joint_ranges) - FORBIDDEN_JOINTS

    if not isinstance(obj, list):
        return ["top_level_not_array"]

    def validate_step(step: Any, path: str) -> None:
        if not isinstance(step, dict):
            errors.append(f"{path}:step_not_object")
            return
        extra = set(step) - {"frame", "duration", "repeat", "times"}
        if extra:
            errors.append(f"{path}:extra_keys:{sorted(extra)}")
        has_frame = "frame" in step
        has_repeat = "repeat" in step
        if has_frame == has_repeat:
            errors.append(f"{path}:must_have_exactly_one_of_frame_or_repeat")
            return
        if has_frame:
            frame = step.get("frame")
            if not isinstance(frame, list):
                errors.append(f"{path}:bad_frame")
            else:
                for i, joint in enumerate(frame):
                    if not isinstance(joint, dict):
                        errors.append(f"{path}.frame[{i}]:joint_not_object")
                        continue
                    name = joint.get("name")
                    angle = joint.get("angle")
                    if name in FORBIDDEN_JOINTS:
                        errors.append(f"{path}.frame[{i}]:forbidden_joint:{name}")
                    elif name not in allowed_joints:
                        errors.append(f"{path}.frame[{i}]:unknown_joint:{name}")
                    if not isinstance(angle, (int, float)):
                        errors.append(f"{path}.frame[{i}]:bad_angle:{angle}")
                    elif name in joint_ranges:
                        lo, hi = joint_ranges[name]
                        if not (lo <= float(angle) <= hi):
                            errors.append(f"{path}.frame[{i}]:angle_out_of_range:{name}:{angle}")
            active = isinstance(frame, list) and len(frame) > 0
            if "duration" not in step:
                errors.append(f"{path}:missing_duration")
            else:
                dur = step.get("duration")
                if not isinstance(dur, (int, float)):
                    errors.append(f"{path}:bad_duration:{dur}")
                elif active and float(dur) < min_active_duration:
                    errors.append(f"{path}:duration_below_min:{dur}")
        if has_repeat:
            rep = step.get("repeat")
            times = step.get("times")
            if not isinstance(rep, list) or len(rep) == 0:
                errors.append(f"{path}:bad_repeat")
            else:
                for i, sub in enumerate(rep):
                    validate_step(sub, f"{path}.repeat[{i}]")
            if not isinstance(times, int) or times < 1:
                errors.append(f"{path}:bad_times:{times}")

    for i, step in enumerate(obj):
        validate_step(step, f"$[{i}]")
    return errors


def validate_json_text(text: str, joint_ranges: dict[str, tuple[float, float]]) -> tuple[bool, bool, list[str], Any | None]:
    try:
        obj = json.loads(text)
    except Exception as exc:
        return False, False, [f"json_parse_error:{exc}"], None
    errors = validate_motion_obj(obj, joint_ranges)
    return True, len(errors) == 0, errors, obj


def count_error_kind(errors: list[str], needle: str) -> int:
    return sum(1 for e in errors if needle in e)
