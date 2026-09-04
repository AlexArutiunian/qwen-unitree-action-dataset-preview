#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


CLASS_MAP = {
    0: "closed",
    1: "open",
    2: "empty",
    3: "occupied",
    4: "folded",
    5: "unfolded",
    6: "filled",
    7: "unconnected",
    8: "connected",
}

STATE_QUESTIONS = {
    "closed": "Is there a closed object in the image?",
    "open": "Is there an open object in the image?",
    "empty": "Is there an empty container or object in the image?",
    "occupied": "Is there an occupied container or object in the image?",
    "folded": "Is there a folded object in the image?",
    "unfolded": "Is there an unfolded object in the image?",
    "filled": "Is there a filled container or object in the image?",
    "unconnected": "Is there an unconnected object in the image?",
    "connected": "Is there a connected object in the image?",
}

CHOICES = ["yes", "no", "cannot tell", "not visible"]
IMG_EXTS = [".png", ".jpg", ".jpeg"]


def find_image_for_txt(txt_path: Path):
    for ext in IMG_EXTS:
        p = txt_path.with_suffix(ext)
        if p.exists():
            return p
    return None


def parse_txt(txt_path: Path):
    class_ids = []
    for line in txt_path.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
        except Exception:
            continue
        if cid in CLASS_MAP:
            class_ids.append(cid)
    return class_ids


def make_choices(answer, seed):
    rng = random.Random(seed)
    choices = list(CHOICES)
    rng.shuffle(choices)
    return choices, choices.index(answer)


def collect_split(split_root: Path, split_name: str, out_dir: Path, negatives_per_positive_state: int, max_images: int | None):
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    txts = sorted(split_root.rglob("*.txt"))
    if max_images:
        txts = txts[:max_images]

    image_records = []
    state_to_pos_images = defaultdict(list)
    all_images = []

    for txt_path in txts:
        img_path = find_image_for_txt(txt_path)
        if img_path is None:
            continue

        cids = parse_txt(txt_path)
        if not cids:
            continue

        states = sorted(set(CLASS_MAP[c] for c in cids))
        rec = {
            "img_path": img_path,
            "txt_path": txt_path,
            "states": states,
            "class_ids": sorted(set(cids)),
        }
        image_records.append(rec)
        all_images.append(rec)

        for st in states:
            state_to_pos_images[st].append(rec)

    rows = []
    copied = 0
    rng = random.Random(42)

    # Copy each full image once.
    src_to_rel = {}
    for i, rec in enumerate(image_records):
        src = rec["img_path"]
        rel = f"images/{split_name}_{src.stem}.jpg"
        dst = out_dir / rel

        if src not in src_to_rel:
            if not dst.exists():
                try:
                    # сохраняем как jpg через copy если уже jpg, иначе просто копируем с расширением jpg нельзя.
                    # Поэтому копируем оригинал с исходным suffix.
                    rel = f"images/{split_name}_{src.name}"
                    dst = out_dir / rel
                    shutil.copy2(src, dst)
                    copied += 1
                except Exception:
                    continue
            src_to_rel[src] = rel

    # Positive rows: for each image state present => yes.
    for rec in image_records:
        rel = src_to_rel.get(rec["img_path"])
        if not rel:
            continue

        for st in rec["states"]:
            ex_id = f"osdd_full_{split_name}_{rec['img_path'].stem}_{st}_yes"
            choices, gt_idx = make_choices("yes", ex_id)
            rows.append({
                "id": ex_id,
                "image": rel,
                "question": STATE_QUESTIONS[st],
                "choices": str(choices),
                "gt_idx": gt_idx,
                "answer": "yes",
                "task": f"state_presence_{st}",
                "state": st,
                "source_dataset": "OSDD",
                "source_image": str(rec["img_path"]),
                "source_txt": str(rec["txt_path"]),
                "present_states": str(rec["states"]),
                "class_ids": str(rec["class_ids"]),
                "split": split_name,
            })

    # Negative rows: for each state choose images where this state absent.
    states_all = list(STATE_QUESTIONS.keys())
    for st in states_all:
        pos_count = len(state_to_pos_images.get(st, []))
        target_neg = pos_count * negatives_per_positive_state

        neg_pool = [r for r in image_records if st not in r["states"]]
        rng.shuffle(neg_pool)
        neg_pool = neg_pool[:target_neg]

        for rec in neg_pool:
            rel = src_to_rel.get(rec["img_path"])
            if not rel:
                continue

            ex_id = f"osdd_full_{split_name}_{rec['img_path'].stem}_{st}_no"
            choices, gt_idx = make_choices("no", ex_id)
            rows.append({
                "id": ex_id,
                "image": rel,
                "question": STATE_QUESTIONS[st],
                "choices": str(choices),
                "gt_idx": gt_idx,
                "answer": "no",
                "task": f"state_presence_{st}",
                "state": st,
                "source_dataset": "OSDD",
                "source_image": str(rec["img_path"]),
                "source_txt": str(rec["txt_path"]),
                "present_states": str(rec["states"]),
                "class_ids": str(rec["class_ids"]),
                "split": split_name,
            })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=123).reset_index(drop=True)

    df.to_csv(out_dir / "metadata.csv", index=False)
    with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in df.to_dict(orient="records"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 100)
    print("split:", split_name)
    print("split_root:", split_root)
    print("images parsed:", len(image_records))
    print("images copied:", copied)
    print("rows:", len(df))
    print("\nanswer counts:")
    print(df["answer"].value_counts().to_string())
    print("\ntask counts:")
    print(df["task"].value_counts().to_string())
    print("\nstate counts:")
    print(df["state"].value_counts().to_string())
    print("\npreview:")
    print(df[["image", "question", "choices", "answer", "task", "present_states"]].head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--osdd-root", default="osdd_data")
    parser.add_argument("--out-root", default="osdd_fullimage_state_mcq")
    parser.add_argument("--negatives-per-positive-state", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    osdd_root = Path(args.osdd_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    split_roots = {
        "train": osdd_root / "train",
        "val": osdd_root / "val",
        "test": osdd_root / "test",
    }

    for split, root in split_roots.items():
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        collect_split(
            split_root=root,
            split_name=split,
            out_dir=out_dir,
            negatives_per_positive_state=args.negatives_per_positive_state,
            max_images=args.max_images,
        )

    print("\nDONE:", out_root)


if __name__ == "__main__":
    main()
