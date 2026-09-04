#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


MOTION_SYSTEM = """You are a Unitree G1 FTP robot-control assistant.
The user message starts with /motion_json or /robot_qa.
For /motion_json return only a valid compact JSON array, no markdown and no explanations.
Compact motion format:
{"frame":{"<joint_alias>":<degrees>},"duration":<seconds>}
{"repeat":[<steps>],"times":<integer>}
Use short aliases: L_ for left, R_ for right, omit _joint. Every motion starts from neutral pose; omitted joints keep their previous value. Use duration >= 0.75 seconds, prefer 1.0-3.5 seconds. End every motion with {"frame":{},"duration":3.0 to 5.0}.
Important G1 FTP semantics: shoulder_pitch negative moves the arm forward/up; positive moves it backward/down. Elbow 0 degrees means about a 90-degree bend; +90 degrees means a straight elbow.
For /robot_qa answer in Russian, briefly and technically; do not output motion JSON unless the user explicitly asks for /motion_json."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def make_motion_rows(rows: list[dict[str, Any]], split: str, offset: int = 0) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        messages = row["messages"]
        command = messages[1]["content"]
        if not command.startswith("/motion_json"):
            command = "/motion_json\n" + command
        out.append(
            {
                "sample_id": int(row.get("sample_id", 0)) + offset,
                "source_id": row.get("source_id"),
                "task": "motion_json",
                "split": split,
                "messages": [
                    {"role": "system", "content": MOTION_SYSTEM},
                    {"role": "user", "content": command},
                    {"role": "assistant", "content": messages[2]["content"]},
                ],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-sft", type=Path, default=PROJECT_ROOT / "outputs" / "dataset_100_statis_compact_slow_sft")
    parser.add_argument("--qa-dir", type=Path, default=PROJECT_ROOT / "outputs" / "robot_qa_g1_ftp_v2")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "dataset_100_statis_multitask_robot_sft_v2")
    parser.add_argument("--motion-repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    motion_train_base = read_jsonl(args.motion_sft / "train_compact_4096.jsonl")
    motion_val_base = read_jsonl(args.motion_sft / "val_compact_4096.jsonl")
    qa_train = read_jsonl(args.qa_dir / "train_robot_qa.jsonl")
    qa_val = read_jsonl(args.qa_dir / "val_robot_qa.jsonl")

    train: list[dict[str, Any]] = []
    for rep in range(args.motion_repeat):
        train.extend(make_motion_rows(motion_train_base, "train", offset=rep * 100_000))
    train.extend(qa_train)
    val = make_motion_rows(motion_val_base, "val") + qa_val

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(val)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in [("train", train), ("val", val)]:
        write_jsonl(args.out_dir / f"{split}.jsonl", rows)
        write_jsonl(args.out_dir / f"{split}_compact_4096.jsonl", rows)
        write_jsonl(args.out_dir / f"{split}_compact_8192.jsonl", rows)

    manifest = {
        "dataset_kind": "statis100_motion_json_plus_robot_qa_g1_ftp_v2",
        "motion_sft": str(args.motion_sft),
        "qa_dir": str(args.qa_dir),
        "out_dir": str(args.out_dir),
        "motion_repeat": args.motion_repeat,
        "train_rows": len(train),
        "val_rows": len(val),
        "train_motion_rows": len(motion_train_base) * args.motion_repeat,
        "val_motion_rows": len(motion_val_base),
        "train_qa_rows": len(qa_train),
        "val_qa_rows": len(qa_val),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
