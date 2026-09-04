#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_id_ranges(s: str) -> set[int]:
    out: set[int] = set()
    if not s.strip():
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def parse_joint_ranges(spec_text: str) -> dict[str, tuple[float, float]]:
    pattern = re.compile(
        r"^\s*([A-Za-z0-9_]+_joint):\s*([-+]?\d+(?:\.\d+)?)\s*(?:to|\.\.)\s*([-+]?\d+(?:\.\d+)?)",
        re.MULTILINE,
    )
    return {name: (float(lo), float(hi)) for name, lo, hi in pattern.findall(spec_text) if name != "floating_base_joint"}


def normalize_motion_obj(obj: Any, min_duration: float) -> tuple[Any, list[str]]:
    issues: list[str] = []

    def rec(x: Any) -> Any:
        if isinstance(x, list):
            return [rec(v) for v in x]
        if not isinstance(x, dict):
            return x
        y = dict(x)
        if "duration" in y:
            try:
                dur = float(y["duration"])
                if dur < min_duration:
                    issues.append(f"duration_clamped:{dur}->{min_duration}")
                    dur = min_duration
                y["duration"] = dur
            except Exception:
                issues.append(f"bad_duration:{y.get('duration')}")
        if "frame" in y and isinstance(y["frame"], list):
            frame = []
            for joint in y["frame"]:
                jj = dict(joint)
                if "angle" in jj:
                    try:
                        jj["angle"] = float(jj["angle"])
                    except Exception:
                        issues.append(f"bad_angle:{jj}")
                frame.append(jj)
            y["frame"] = frame
        if "repeat" in y:
            y["repeat"] = rec(y["repeat"])
        return y

    return rec(obj), issues


def validate_motion_obj(obj: Any, joint_ranges: dict[str, tuple[float, float]], min_duration: float) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, list):
        return ["top_level_not_list"]

    def validate_step(step: Any, path: str) -> None:
        if not isinstance(step, dict):
            errors.append(f"{path}:step_not_object")
            return
        has_frame = "frame" in step
        has_repeat = "repeat" in step
        if has_frame == has_repeat:
            errors.append(f"{path}:must_have_exactly_one_of_frame_or_repeat")
        extra = set(step) - {"frame", "duration", "repeat", "times"}
        if extra:
            errors.append(f"{path}:extra_keys:{sorted(extra)}")
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
                    if name not in joint_ranges:
                        errors.append(f"{path}.frame[{i}]:unknown_joint:{name}")
                    if not isinstance(angle, (int, float)):
                        errors.append(f"{path}.frame[{i}]:bad_angle:{angle}")
                    elif name in joint_ranges:
                        lo, hi = joint_ranges[name]
                        if not lo <= float(angle) <= hi:
                            errors.append(f"{path}.frame[{i}]:angle_out_of_range:{name}:{angle}")
            dur = step.get("duration")
            if dur is not None and (not isinstance(dur, (int, float)) or float(dur) < min_duration):
                errors.append(f"{path}:bad_duration:{dur}")
        if has_repeat:
            rep = step.get("repeat")
            times = step.get("times")
            if not isinstance(rep, list) or not rep:
                errors.append(f"{path}:bad_repeat")
            else:
                for i, sub in enumerate(rep):
                    validate_step(sub, f"{path}.repeat[{i}]")
            if not isinstance(times, int) or times < 1:
                errors.append(f"{path}:bad_times:{times}")

    for i, step in enumerate(obj):
        validate_step(step, f"$[{i}]")
    return errors


def compress_redundant_frame_joints(obj: Any) -> tuple[Any, int]:
    """Remove repeated joint targets from later frames while preserving hold durations."""
    removed = 0

    def norm_angle(value: Any) -> float | None:
        try:
            return round(float(value), 6)
        except Exception:
            return None

    def compress_steps(steps: Any, state: dict[str, float]) -> Any:
        nonlocal removed
        if not isinstance(steps, list):
            return steps
        out = []
        for step in steps:
            if not isinstance(step, dict):
                out.append(step)
                continue
            new_step = dict(step)
            if isinstance(new_step.get("frame"), list):
                new_frame = []
                for item in new_step["frame"]:
                    if not isinstance(item, dict):
                        new_frame.append(item)
                        continue
                    name = item.get("name")
                    angle = norm_angle(item.get("angle"))
                    if isinstance(name, str) and angle is not None and state.get(name) == angle:
                        removed += 1
                        continue
                    if isinstance(name, str) and angle is not None:
                        state[name] = angle
                    new_frame.append(item)
                new_step["frame"] = new_frame
            elif isinstance(new_step.get("repeat"), list):
                # Keep repeat blocks conservative: compress only within the first written body.
                new_step["repeat"] = compress_steps(new_step["repeat"], dict(state))
            out.append(new_step)
        return out

    return compress_steps(obj, {}), removed


def build_system_prompt(spec_text: str) -> str:
    joint_ranges = parse_joint_ranges(spec_text)
    joint_lines = "\n".join(f"{name}: {lo:g} to {hi:g}" for name, (lo, hi) in sorted(joint_ranges.items()))
    return (
        "You convert Russian natural-language robot motion commands into Unitree G1 joint-control JSON.\n"
        "Return only a valid JSON array, with no markdown, comments, or explanations.\n"
        "Allowed step forms:\n"
        '{"frame":[{"name":"<joint>","angle":<degrees>}],"duration":<seconds>}\n'
        '{"repeat":[<steps>],"times":<integer>}\n'
        "Rules:\n"
        "Use only allowed joints. floating_base_joint is forbidden.\n"
        "Angles are degrees and must stay within joint limits.\n"
        "frame: [] is valid as a pause or hold step.\n"
        "Use duration >= 0.75 seconds for active movement steps.\n"
        "Prefer one frame for simultaneous joint movement.\n"
        "Critical joint semantics:\n"
        "- shoulder_pitch: negative angle moves the arm forward/up; positive angle moves it backward.\n"
        "- left_shoulder_roll: positive moves the left arm outward; negative moves it inward.\n"
        "- right_shoulder_roll: negative moves the right arm outward; positive moves it inward.\n"
        "- elbow joints: 0 degrees means about a 90-degree bend; +90 degrees means a straight elbow.\n"
        "- For straight arms, use elbow angles near +90, not 0.\n"
        "- When both arms act together, put left and right joints in the same frame.\n"
        "- Do not command finger joints unless the user asks for fingers, fist, open hand, grasping, or palm/finger gestures.\n"
        "Allowed joints and limits:\n"
        f"{joint_lines}"
    ).strip()


def make_messages(command: str, target_json: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Command:\n{command.strip()}"},
        {"role": "assistant", "content": target_json.strip()},
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]], system_prompt: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            obj = {
                "sample_id": int(row["sample_id"]),
                "source_id": int(row["source_id"]),
                "class": row.get("class", ""),
                "augmentation_type": "original",
                "split": row["split"],
                "messages": make_messages(row["text"], row["target_json"], system_prompt),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("/home/alpc/humanoid/dataset_panto/dataset_approved.csv"))
    parser.add_argument("--json-dir", type=Path, default=Path("/home/alpc/humanoid/dataset_panto/dataset_approved"))
    parser.add_argument("--spec", type=Path, default=Path("/home/alpc/humanoid/lora-robot/outputs/dataset_viewer/robot_spec_joints-only.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/alpc/humanoid/lora-robot/outputs/dataset_approved_clean_sft"))
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--reserved-ids", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--prompt-mode", default="compact")
    parser.add_argument("--min-duration", type=float, default=0.75)
    parser.add_argument("--no-compress-redundant-joints", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec_text = args.spec.read_text(encoding="utf-8")
    joint_ranges = parse_joint_ranges(spec_text)
    system_prompt = build_system_prompt(spec_text)
    (args.out_dir / "robot_spec_joints-only.txt").write_text(spec_text, encoding="utf-8")
    (args.out_dir / f"system_prompt_{args.prompt_mode}.txt").write_text(system_prompt + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    with args.csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            decision = str(row.get("review_decision", "")).strip().lower()
            if decision and decision != "approve":
                continue
            json_name = str(row.get("json_name") or "").strip()
            if not json_name:
                missing.append({"id": row.get("ID"), "reason": "empty_json_name"})
                continue
            target_path = args.json_dir / json_name
            if not target_path.exists():
                missing.append({"id": row.get("ID"), "json_name": json_name})
                continue
            try:
                obj = json.loads(target_path.read_text(encoding="utf-8"))
                obj, issues = normalize_motion_obj(obj, args.min_duration)
                removed_redundant = 0
                if not args.no_compress_redundant_joints:
                    obj, removed_redundant = compress_redundant_frame_joints(obj)
                    if removed_redundant:
                        issues.append(f"redundant_joint_targets_removed:{removed_redundant}")
                errors = validate_motion_obj(obj, joint_ranges, args.min_duration)
                target_json = stable_json_dumps(obj)
            except Exception as exc:
                invalid.append({"id": row.get("ID"), "json_name": json_name, "error": repr(exc)})
                continue
            if errors:
                invalid.append({"id": row.get("ID"), "json_name": json_name, "error": ";".join(errors[:30])})
                continue
            sid = int(row["ID"])
            rows.append(
                {
                    "sample_id": sid,
                    "source_id": sid,
                    "text": str(row.get("Text", "")).strip(),
                    "target_json": target_json,
                    "class": str(row.get("class", "")).strip(),
                    "json_name": json_name,
                    "source_json": str(row.get("source_json", json_name)).strip(),
                    "target_sha256": sha256_text(target_json),
                    "target_len_chars": len(target_json),
                    "sanitize_issues": ";".join(issues),
                }
            )

    reserved = parse_id_ranges(args.reserved_ids)
    train_val_ids = [int(r["source_id"]) for r in rows if int(r["source_id"]) not in reserved]
    rng = random.Random(args.seed)
    rng.shuffle(train_val_ids)
    val_count = max(1, round(len(train_val_ids) * args.val_ratio)) if len(train_val_ids) > 1 else 0
    val_ids = set(train_val_ids[:val_count])

    for row in rows:
        sid = int(row["source_id"])
        row["split"] = "reserved" if sid in reserved else ("val" if sid in val_ids else "train")

    prefix = f"{args.prompt_mode}_{args.max_seq_length}"
    for split in ["train", "val", "reserved"]:
        part = [r for r in rows if r["split"] == split]
        write_jsonl(args.out_dir / f"{split}.jsonl", part, system_prompt)
        write_jsonl(args.out_dir / f"{split}_{prefix}.jsonl", part, system_prompt)

    with (args.out_dir / "robot_sft.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "sample_id",
            "source_id",
            "split",
            "text",
            "target_json",
            "class",
            "augmentation_type",
            "json_name",
            "source_json",
            "target_sha256",
            "target_len_chars",
            "sanitize_issues",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["augmentation_type"] = "original"
            writer.writerow({k: out.get(k, "") for k in fieldnames})

    with (args.out_dir / "missing_targets.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "json_name", "reason"])
        writer.writeheader()
        writer.writerows(missing)
    with (args.out_dir / "invalid_targets.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "json_name", "error"])
        writer.writeheader()
        writer.writerows(invalid)

    split_counts = {split: sum(1 for r in rows if r["split"] == split) for split in ["train", "val", "reserved"]}
    manifest = {
        "dataset_kind": "approved_clean_no_augmentation",
        "csv": str(args.csv),
        "json_dir": str(args.json_dir),
        "spec": str(args.spec),
        "prompt_mode": args.prompt_mode,
        "max_seq_length": args.max_seq_length,
        "total_rows": len(rows),
        "split_counts": split_counts,
        "reserved_ids": sorted(reserved),
        "val_ratio": args.val_ratio,
        "missing_rows": len(missing),
        "invalid_rows": len(invalid),
        "joint_count": len(joint_ranges),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Saved: {args.out_dir}")


if __name__ == "__main__":
    main()
