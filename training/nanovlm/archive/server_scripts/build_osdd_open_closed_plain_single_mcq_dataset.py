#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import shutil
from pathlib import Path
from collections import Counter

import pandas as pd


CLASS_MAP = {
    0: "closed",
    1: "open",
}

CHOICES = ["closed", "open", "cannot tell", "not visible"]

QUESTION_TEMPLATES = [
    "Is the main object in the image open or closed?",
    "What is the state of the main object in the image?",
    "Choose the correct state of the main object.",
]

IMG_EXTS = [".png", ".jpg", ".jpeg"]


def find_image_for_txt(txt_path: Path):
    for ext in IMG_EXTS:
        p = txt_path.with_suffix(ext)
        if p.exists():
            return p
    return None


def parse_all_yolo(txt_path: Path):
    objs = []
    for line in txt_path.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            box = tuple(map(float, parts[1:5]))
        except Exception:
            continue
        objs.append({"class_id": cid, "box": box})
    return objs


def make_choices(answer, seed):
    choices = list(CHOICES)
    rng = random.Random(seed)
    rng.shuffle(choices)
    return choices, choices.index(answer)


def build_split(split_root: Path, split_name: str, out_dir: Path):
    out_img_dir = out_dir / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = Counter()
    class_counts = Counter()

    txts = sorted(split_root.rglob("*.txt"))

    for txt_path in txts:
        img_path = find_image_for_txt(txt_path)
        if img_path is None:
            skipped["missing_image"] += 1
            continue

        objs_all = parse_all_yolo(txt_path)
        if len(objs_all) != 1:
            skipped["not_single_object_total"] += 1
            continue

        cid = objs_all[0]["class_id"]
        if cid not in CLASS_MAP:
            skipped["single_object_not_open_closed"] += 1
            continue

        state = CLASS_MAP[cid]

        for q_i, question in enumerate(QUESTION_TEMPLATES):
            ex_id = f"osdd_oc_plain_single_{split_name}_{txt_path.stem}_q{q_i}_c{cid}"
            rel_img = f"images/{ex_id}{img_path.suffix.lower()}"
            dst_img = out_dir / rel_img

            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            choices, gt_idx = make_choices(state, ex_id)

            rows.append({
                "id": ex_id,
                "image": rel_img,
                "question": question,
                "choices": str(choices),
                "gt_idx": gt_idx,
                "answer": state,
                "task": "open_closed_plain_single",
                "state": state,
                "class_id": cid,
                "source_dataset": "OSDD",
                "source_image": str(img_path),
                "source_txt": str(txt_path),
                "question_template_id": q_i,
                "split": split_name,
            })

        class_counts[cid] += 1

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    df.to_csv(out_dir / "metadata.csv", index=False)
    with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in df.to_dict(orient="records"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 100)
    print("split:", split_name)
    print("rows:", len(df))
    print("single open/closed objects:", len(df) // len(QUESTION_TEMPLATES) if len(df) else 0)

    print("\nclass object counts:")
    for cid, n in sorted(class_counts.items()):
        print(cid, CLASS_MAP[cid], n)

    print("\nanswer counts:")
    print(df["answer"].value_counts().to_string() if len(df) else "")

    print("\nskipped:")
    print(pd.Series(skipped).to_string() if skipped else "none")

    print("\npreview:")
    if len(df):
        print(df[["image", "question", "choices", "answer", "gt_idx"]].head(8).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osdd-root", default="osdd_data")
    ap.add_argument("--out-root", default="osdd_open_closed_plain_single_mcq")
    args = ap.parse_args()

    root = Path(args.osdd_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split, split_root in {
        "train": root / "train",
        "val": root / "val",
        "test": root / "test",
    }.items():
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        build_split(split_root, split, out_dir)

    print("\nDONE:", out_root)


if __name__ == "__main__":
    main()
