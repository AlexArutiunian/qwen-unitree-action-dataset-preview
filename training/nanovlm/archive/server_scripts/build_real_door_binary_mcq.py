#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import random
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


def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for x in candidates:
        if x.lower() in lower:
            return lower[x.lower()]
    return None


def detect_state_from_row(row, image_path):
    text = " ".join(str(x).lower() for x in list(row.values) + [str(image_path).lower()])

    if "door_open" in text or "elevator_open" in text or "/open" in text or "_open" in text:
        return "open"
    if "door_closed" in text or "elevator_closed" in text or "/closed" in text or "_closed" in text:
        return "closed"

    raise ValueError(f"Cannot detect state for image={image_path}")


def make_choices(base_choices, answer, seed):
    choices = list(base_choices)
    rng = random.Random(seed)
    rng.shuffle(choices)
    return choices, choices.index(answer)


def add_row(rows, img_rel, q, choices_base, ans, task, sample_id):
    choices, gt_idx = make_choices(choices_base, ans, sample_id + "_" + task)
    rows.append({
        "id": sample_id + "_" + task,
        "image": img_rel,
        "question": q,
        "choices": str(choices),
        "gt_idx": gt_idx,
        "answer": ans,
        "task": task,
        "door_state": "open" if "open" in ans.lower() and "closed" not in ans.lower() else "",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    img_out = out_dir / "images"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    img_out.mkdir(parents=True, exist_ok=True)

    manifest = Path(args.manifest) if args.manifest else raw_dir / "manifest_binary.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)

    df = pd.read_csv(manifest)
    print("manifest rows:", len(df))
    print("columns:", df.columns.tolist())

    image_col = find_col(df, ["image", "path", "filepath", "file", "filename", "rel_path", "relative_path"])
    state_col = find_col(df, ["label", "class", "state", "door_state", "folder", "category"])

    if image_col is None:
        # fallback: first column that looks like image path
        for c in df.columns:
            vals = df[c].astype(str).head(20).str.lower()
            if vals.str.contains(r"\.(jpg|jpeg|png|bmp)", regex=True).any():
                image_col = c
                break

    if image_col is None:
        raise RuntimeError("Could not find image/path column in manifest")

    print("image_col:", image_col)
    print("state_col:", state_col)

    rows = []
    copied = 0

    for i, r in df.iterrows():
        src_str = str(r[image_col]).strip()

        src = Path(src_str)
        if not src.is_absolute():
            src = raw_dir / src

        if not src.exists():
            # попробовать поиск по имени внутри raw_dir
            matches = list(raw_dir.rglob(Path(src_str).name))
            if matches:
                src = matches[0]

        if not src.exists():
            raise FileNotFoundError(f"Image not found for row {i}: {src_str}")

        if state_col:
            label_text = str(r[state_col]).lower()
            if "open" in label_text and "closed" not in label_text:
                state = "open"
            elif "closed" in label_text:
                state = "closed"
            else:
                state = detect_state_from_row(r, src)
        else:
            state = detect_state_from_row(r, src)

        sample_id = f"realdoor_binary_{i:04d}_{state}"
        ext = src.suffix.lower()
        img_rel = f"images/{sample_id}{ext}"
        dst = out_dir / img_rel
        shutil.copy2(src, dst)
        copied += 1

        # 1. Direct open question
        q = "Is the door open?"
        ans = "Yes, the door is open." if state == "open" else "No, the door is closed."
        choices, gt_idx = make_choices(OPEN_CHOICES, ans, sample_id + "_door_open_direct")
        rows.append({
            "id": sample_id + "_door_open_direct",
            "image": img_rel,
            "question": q,
            "choices": str(choices),
            "gt_idx": gt_idx,
            "answer": ans,
            "task": "door_open_direct",
            "door_state": state,
        })

        # 2. Door state question
        q = "What is the state of the door?"
        ans = "Open." if state == "open" else "Closed."
        choices, gt_idx = make_choices(STATE_CHOICES, ans, sample_id + "_door_state_mcq")
        rows.append({
            "id": sample_id + "_door_state_mcq",
            "image": img_rel,
            "question": q,
            "choices": str(choices),
            "gt_idx": gt_idx,
            "answer": ans,
            "task": "door_state_mcq",
            "door_state": state,
        })

        # 3. Passability question
        q = "Can the robot pass through this doorway?"
        ans = "Yes, the path is open." if state == "open" else "No, the path is blocked or closed."
        choices, gt_idx = make_choices(PASS_CHOICES, ans, sample_id + "_door_passability")
        rows.append({
            "id": sample_id + "_door_passability",
            "image": img_rel,
            "question": q,
            "choices": str(choices),
            "gt_idx": gt_idx,
            "answer": ans,
            "task": "door_passability",
            "door_state": state,
        })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    out_df.to_csv(out_dir / "metadata.csv", index=False)
    with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for rec in out_df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\nDONE")
    print("copied images:", copied)
    print("qa rows:", len(out_df))
    print("out_dir:", out_dir)
    print("\nstates:")
    print(out_df["door_state"].value_counts().to_string())
    print("\ntasks:")
    print(out_df["task"].value_counts().to_string())
    print("\nanswers:")
    print(out_df["answer"].value_counts().to_string())
    print("\npreview:")
    print(out_df.head(12)[["image", "question", "choices", "answer", "task", "door_state"]].to_string(index=False))


if __name__ == "__main__":
    main()
