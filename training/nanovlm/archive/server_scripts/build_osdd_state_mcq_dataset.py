#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from PIL import Image


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

# Группы состояний. Каждая группа превращается в отдельный MCQ с 4 вариантами.
GROUPS = {
    "openable_state": {
        "class_ids": {0, 1},
        "choices": ["closed", "open", "cannot tell", "not visible"],
        "question_templates": [
            "What is the state of the object?",
            "Is the object open or closed?",
            "Choose the correct state of the object.",
        ],
    },
    "container_state": {
        "class_ids": {2, 3, 6},
        "choices": ["empty", "occupied", "filled", "cannot tell"],
        "question_templates": [
            "What is the state of the container?",
            "Is the container empty, occupied, or filled?",
            "Choose the correct state of the container.",
        ],
    },
    "fold_state": {
        "class_ids": {4, 5},
        "choices": ["folded", "unfolded", "cannot tell", "not visible"],
        "question_templates": [
            "What is the state of the object?",
            "Is the object folded or unfolded?",
            "Choose the correct state of the object.",
        ],
    },
    "connection_state": {
        "class_ids": {7, 8},
        "choices": ["unconnected", "connected", "cannot tell", "not visible"],
        "question_templates": [
            "What is the connection state of the object?",
            "Is the object connected or unconnected?",
            "Choose the correct connection state.",
        ],
    },
}


IMG_EXTS = [".png", ".jpg", ".jpeg"]


def find_image_for_txt(txt_path: Path):
    for ext in IMG_EXTS:
        p = txt_path.with_suffix(ext)
        if p.exists():
            return p
    return None


def yolo_to_xyxy(box, w, h, pad=0.20):
    x, y, bw, bh = box
    x1 = (x - bw / 2) * w
    y1 = (y - bh / 2) * h
    x2 = (x + bw / 2) * w
    y2 = (y + bh / 2) * h

    box_w = x2 - x1
    box_h = y2 - y1

    x1 -= box_w * pad
    y1 -= box_h * pad
    x2 += box_w * pad
    y2 += box_h * pad

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(w, int(round(x2)))
    y2 = min(h, int(round(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def group_for_class(cid):
    for group_name, spec in GROUPS.items():
        if cid in spec["class_ids"]:
            return group_name, spec
    return None, None


def deterministic_choices(base_choices, answer, seed_text):
    choices = list(base_choices)
    rng = random.Random(seed_text)
    rng.shuffle(choices)

    # safety: answer must exist
    if answer not in choices:
        raise ValueError(f"answer {answer} not in choices {choices}")

    gt_idx = choices.index(answer)
    return choices, gt_idx


def collect_split(split_root: Path, split_name: str, out_dir: Path, crop_padding: float, max_per_class: int | None):
    txts = sorted(split_root.rglob("*.txt"))

    rows = []
    per_class_saved = Counter()
    per_class_seen = Counter()
    skipped = Counter()

    img_out_dir = out_dir / "images"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    for txt_path in txts:
        img_path = find_image_for_txt(txt_path)
        if img_path is None:
            skipped["missing_image"] += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped["bad_image"] += 1
            continue

        w, h = img.size
        lines = txt_path.read_text(errors="ignore").splitlines()

        for obj_i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                skipped["bad_line"] += 1
                continue

            try:
                cid = int(float(parts[0]))
                box = tuple(map(float, parts[1:5]))
            except Exception:
                skipped["bad_parse"] += 1
                continue

            if cid not in CLASS_MAP:
                skipped["unknown_class"] += 1
                continue

            per_class_seen[cid] += 1

            if max_per_class is not None and per_class_saved[cid] >= max_per_class:
                skipped["max_per_class"] += 1
                continue

            group_name, spec = group_for_class(cid)
            if group_name is None:
                skipped["no_group"] += 1
                continue

            crop_box = yolo_to_xyxy(box, w, h, pad=crop_padding)
            if crop_box is None:
                skipped["bad_box"] += 1
                continue

            crop = img.crop(crop_box)

            state = CLASS_MAP[cid]
            q_templates = spec["question_templates"]

            # 3 вопроса на один crop: это качественная формулировочная аугментация,
            # но уже внутри правильной state-задачи, а не общего RoboVQA affordance.
            for q_i, question in enumerate(q_templates):
                ex_id = f"osdd_{split_name}_{txt_path.stem}_obj{obj_i:02d}_q{q_i}_c{cid}"
                rel_img = f"images/{ex_id}.jpg"
                crop.save(out_dir / rel_img, quality=95)

                choices, gt_idx = deterministic_choices(
                    spec["choices"],
                    state,
                    seed_text=ex_id,
                )

                rows.append({
                    "id": ex_id,
                    "image": rel_img,
                    "question": question,
                    "choices": str(choices),
                    "gt_idx": gt_idx,
                    "answer": state,
                    "task": group_name,
                    "source_dataset": "OSDD",
                    "source_image": str(img_path),
                    "source_txt": str(txt_path),
                    "class_id": cid,
                    "state": state,
                    "bbox_xywh_yolo": str(box),
                    "crop_xyxy": str(crop_box),
                    "question_template_id": q_i,
                    "split": split_name,
                })

            per_class_saved[cid] += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metadata.csv", index=False)

    with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in df.to_dict(orient="records"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 100)
    print("SPLIT:", split_name)
    print("split_root:", split_root)
    print("out_dir:", out_dir)
    print("rows:", len(df))
    print("unique crop objects approx:", len(df) // 3)
    print("\nseen class counts:")
    for k in sorted(per_class_seen):
        print(k, CLASS_MAP[k], per_class_seen[k])

    print("\nsaved object counts before question x3:")
    for k in sorted(per_class_saved):
        print(k, CLASS_MAP[k], per_class_saved[k])

    print("\nrow answer counts:")
    if len(df):
        print(df["answer"].value_counts().to_string())

    print("\ntask counts:")
    if len(df):
        print(df["task"].value_counts().to_string())

    print("\nskipped:")
    for k, v in skipped.items():
        print(k, v)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--osdd-root", default="osdd_data")
    parser.add_argument("--out-root", default="osdd_state_mcq")
    parser.add_argument("--crop-padding", type=float, default=0.25)
    parser.add_argument("--max-per-class", type=int, default=None)
    args = parser.parse_args()

    osdd_root = Path(args.osdd_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    split_roots = {
        "train": osdd_root / "train",
        "val": osdd_root / "val",
        "test": osdd_root / "test",
    }

    all_reports = []
    for split_name, split_root in split_roots.items():
        out_dir = out_root / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        df = collect_split(
            split_root=split_root,
            split_name=split_name,
            out_dir=out_dir,
            crop_padding=args.crop_padding,
            max_per_class=args.max_per_class,
        )
        all_reports.append((split_name, len(df)))

    print("\nDONE")
    print("out_root:", out_root)
    print("reports:", all_reports)


if __name__ == "__main__":
    main()
