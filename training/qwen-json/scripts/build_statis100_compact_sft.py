#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUMANOID_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.joint_aliases import alias_to_canonical, canonical_to_alias


DEFAULT_PROMPT = PROJECT_ROOT / "outputs" / "dataset_approved_clean_sft_alias_delta_json" / "system_prompt_compact_json.txt"


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def make_system_prompt(base: str) -> str:
    prompt = base.strip()
    prompt = prompt.replace(
        "Use duration >= 0.75 seconds for active movement steps.",
        "Use duration >= 0.75 seconds for active movement steps on the real robot; prefer 1.0-3.5 seconds for smooth safe motions.",
    )
    if "End every motion with a hold step" not in prompt:
        prompt += (
            "\n- End every motion with a hold step: "
            '{"frame":{},"duration":3.0 to 5.0}.'
        )
    return prompt


def validate_motion(obj: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, list):
        return ["top_level_not_list"]
    for i, step in enumerate(obj):
        if not isinstance(step, dict):
            errors.append(f"{i}:step_not_object")
            continue
        if "frame" in step:
            frame = step["frame"]
            if not isinstance(frame, dict):
                errors.append(f"{i}:frame_not_compact_object")
            else:
                for name, angle in frame.items():
                    try:
                        alias_to_canonical(str(name))
                        float(angle)
                    except Exception:
                        errors.append(f"{i}:bad_joint_or_angle:{name}={angle}")
            try:
                if float(step.get("duration", 0)) < 0:
                    errors.append(f"{i}:negative_duration")
            except Exception:
                errors.append(f"{i}:bad_duration")
        elif "repeat" in step:
            if not isinstance(step.get("repeat"), list) or not isinstance(step.get("times"), int):
                errors.append(f"{i}:bad_repeat")
        else:
            errors.append(f"{i}:missing_frame_or_repeat")
    if obj:
        last = obj[-1]
        if not isinstance(last, dict) or last.get("frame") != {}:
            errors.append("missing_final_compact_hold")
        else:
            duration = float(last.get("duration", 0))
            if not 3.0 <= duration <= 5.0:
                errors.append(f"bad_final_hold_duration:{duration}")
    return errors


def make_messages(system_prompt: str, command: str, target_json: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Command:\n{command.strip()}"},
        {"role": "assistant", "content": target_json.strip()},
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]], system_prompt: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            item = {
                "sample_id": row["id"],
                "source_id": row["id"],
                "class": "",
                "augmentation_type": "original_statis100_compact_slow",
                "split": row["split"],
                "messages": make_messages(system_prompt, row["text"], row["target_json"]),
            }
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=HUMANOID_ROOT / "dataset_panto" / "100_statis.csv")
    parser.add_argument("--json-dir", type=Path, default=HUMANOID_ROOT / "dataset_panto" / "dataset_100_statis")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "dataset_100_statis_compact_slow_sft")
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = make_system_prompt(args.system_prompt.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    with args.csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sid = int(str(row["ID"]).strip())
            text = str(row["Text"]).strip()
            json_name = str(row.get("json_name") or "").strip() or f"{sid}.json"
            path = args.json_dir / json_name
            if not path.exists():
                path = args.json_dir / f"{sid}.json"
            if not path.exists():
                invalid.append({"id": str(sid), "json_name": json_name, "error": "missing_json"})
                continue
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
                errors = validate_motion(obj)
            except Exception as exc:
                invalid.append({"id": str(sid), "json_name": path.name, "error": repr(exc)})
                continue
            if errors:
                invalid.append({"id": str(sid), "json_name": path.name, "error": ";".join(errors)})
                continue
            rows.append({"id": sid, "text": text, "json_name": path.name, "target_json": compact_json(obj)})

    rng = random.Random(args.seed)
    ids = [r["id"] for r in rows]
    rng.shuffle(ids)
    val_n = max(1, round(len(ids) * args.val_ratio)) if len(ids) > 1 else 0
    val_ids = set(ids[:val_n])
    for row in rows:
        row["split"] = "val" if row["id"] in val_ids else "train"

    for split in ("train", "val"):
        part = [r for r in rows if r["split"] == split]
        write_jsonl(args.out_dir / f"{split}.jsonl", part, system_prompt)
        write_jsonl(args.out_dir / f"{split}_compact_4096.jsonl", part, system_prompt)
        write_jsonl(args.out_dir / f"{split}_compact_8192.jsonl", part, system_prompt)

    (args.out_dir / "system_prompt_compact_json.txt").write_text(system_prompt + "\n", encoding="utf-8")
    with (args.out_dir / "robot_sft.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["sample_id", "split", "text", "json_name", "target_json"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row["id"],
                    "split": row["split"],
                    "text": row["text"],
                    "json_name": row["json_name"],
                    "target_json": row["target_json"],
                }
            )
    with (args.out_dir / "invalid_targets.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["id", "json_name", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(invalid)

    manifest = {
        "dataset_kind": "statis100_compact_slow_no_augmentation",
        "csv": str(args.csv),
        "json_dir": str(args.json_dir),
        "system_prompt": str(args.system_prompt),
        "out_dir": str(args.out_dir),
        "format": "alias_delta_compact_json",
        "total_rows": len(rows),
        "split_counts": {
            "train": sum(1 for r in rows if r["split"] == "train"),
            "val": sum(1 for r in rows if r["split"] == "val"),
        },
        "invalid_rows": len(invalid),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
