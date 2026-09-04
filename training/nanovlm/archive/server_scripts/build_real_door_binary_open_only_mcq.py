#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import shutil
from pathlib import Path

import pandas as pd


CHOICES = [
    "Yes, the door is open.",
    "No, the door is closed.",
    "Cannot tell from the image.",
    "The door is not visible.",
]


def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for x in candidates:
        if x.lower() in lower:
            return lower[x.lower()]
    return None


def detect_state(row, image_path):
    text = " ".join(str(x).lower() for x in list(row.values) + [str(image_path).lower()])

    if "door_open" in text or "elevator_open" in text:
        return "open"
    if "door_closed" in text or "elevator_closed" in text:
        return "closed"
    if "/door_open/" in text or "/elevator_open/" in text:
        return "open"
    if "/door_closed/" in text or "/elevator_closed/" in text:
        return "closed"

    raise ValueError(f"Cannot detect state for image={image_path}")


def make_choices(answer, seed):
    choices = list(CHOICES)
    rng = random.Random(seed)
    rng.shuffle(choices)
    return choices, choices.index(answer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    manifest = Path(args.manifest) if args.manifest else raw_dir / "manifest_binary.csv"
    out_dir = Path(args.out_dir)
    img_out = out_dir / "images"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    img_out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest)
    print("manifest rows:", len(df))
    print("columns:", df.columns.tolist())

    image_col = find_col(df, ["image", "path", "filepath", "file", "filename", "rel_path", "relative_path"])
    label_col = find_col(df, ["label", "class", "state", "door_state", "folder", "category"])

    if image_col is None:
        for c in df.columns:
            if df[c].astype(str).str.contains(r"\.(jpg|jpeg|png|bmp)", case=False, regex=True).any():
                image_col = c
                break

    if image_col is None:
        raise RuntimeError("Could not find image/path column")

    print("image_col:", image_col)
    print("label_col:", label_col)

    rows = []
    copied = 0

    for i, r in df.iterrows():
        src_str = str(r[image_col]).strip()
        src = Path(src_str)

        if not src.is_absolute():
            src = raw_dir / src

        if not src.exists():
            matches = list(raw_dir.rglob(Path(src_str).name))
            if matches:
                src = matches[0]

        if not src.exists():
            raise FileNotFoundError(f"Image not found row={i}: {src_str}")

        if label_col:
            label = str(r[label_col]).lower()
            if "open" in label and "closed" not in label:
                state = "open"
            elif "closed" in label:
                state = "closed"
            else:
                state = detect_state(r, src)
        else:
            state = detect_state(r, src)

        sample_id = f"realdoor_openonly_{i:04d}_{state}"
        rel_img = f"images/{sample_id}{src.suffix.lower()}"
        dst = out_dir / rel_img
        shutil.copy2(src, dst)
        copied += 1

        answer = "Yes, the door is open." if state == "open" else "No, the door is closed."
        choices, gt_idx = make_choices(answer, sample_id)

        rows.append({
            "id": sample_id,
            "image": rel_img,
            "question": "Is the door open?",
            "choices": str(choices),
            "gt_idx": gt_idx,
            "answer": answer,
            "task": "door_open_binary",
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
    print("\nanswers:")
    print(out_df["answer"].value_counts().to_string())
    print("\npreview:")
    print(out_df.head(12)[["image", "question", "choices", "answer", "door_state"]].to_string(index=False))


if __name__ == "__main__":
    main()
