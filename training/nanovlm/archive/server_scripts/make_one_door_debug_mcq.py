#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
from pathlib import Path
import pandas as pd

OPEN_CHOICES = [
    "Yes, the door is open.",
    "No, the door is closed.",
    "Cannot tell from the image.",
    "The door is not visible.",
]

STATE_CHOICES = [
    "Open.",
    "Closed.",
    "Partially open.",
    "Cannot tell.",
]

PASS_CHOICES = [
    "Yes, the path is open.",
    "No, the path is blocked or closed.",
    "Cannot tell from the image.",
    "There is no doorway visible.",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--state", required=True, choices=["open", "closed"])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    src = Path(args.image)
    out = Path(args.out_dir)
    img_dir = out / "images"

    if out.exists():
        shutil.rmtree(out)
    img_dir.mkdir(parents=True, exist_ok=True)

    dst = img_dir / src.name
    shutil.copy2(src, dst)
    rel = f"images/{src.name}"

    state = args.state

    if state == "open":
        ans_open = "Yes, the door is open."
        ans_state = "Open."
        ans_pass = "Yes, the path is open."
    else:
        ans_open = "No, the door is closed."
        ans_state = "Closed."
        ans_pass = "No, the path is blocked or closed."

    rows = [
        {
            "id": "debug_q1_open_direct",
            "image": rel,
            "question": "Is the door open?",
            "choices": str(OPEN_CHOICES),
            "gt_idx": OPEN_CHOICES.index(ans_open),
            "answer": ans_open,
            "task": "door_open_direct",
            "door_state": state,
        },
        {
            "id": "debug_q2_state",
            "image": rel,
            "question": "What is the state of the door?",
            "choices": str(STATE_CHOICES),
            "gt_idx": STATE_CHOICES.index(ans_state),
            "answer": ans_state,
            "task": "door_state_mcq",
            "door_state": state,
        },
        {
            "id": "debug_q3_passability",
            "image": rel,
            "question": "Can the robot pass through this doorway?",
            "choices": str(PASS_CHOICES),
            "gt_idx": PASS_CHOICES.index(ans_pass),
            "answer": ans_pass,
            "task": "door_passability",
            "door_state": state,
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(out / "metadata.csv", index=False)
    with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("DONE:", out)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
