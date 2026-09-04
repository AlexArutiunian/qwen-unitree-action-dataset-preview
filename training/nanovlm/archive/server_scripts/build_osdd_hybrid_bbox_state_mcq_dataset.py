#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import shutil
from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image, ImageDraw


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

GROUPS = {
    "openable_state": {
        "class_ids": {0, 1},
        "choices": ["closed", "open", "cannot tell", "not visible"],
        "q_bbox": "Look at the object highlighted with the red box. Is it open or closed?",
        "q_plain": "Is the main object in the image open or closed?",
    },
    "container_state": {
        "class_ids": {2, 3, 6},
        "choices": ["empty", "occupied", "filled", "cannot tell"],
        "q_bbox": "Look at the object highlighted with the red box. Is it empty, occupied, or filled?",
        "q_plain": "Is the main container or object in the image empty, occupied, or filled?",
    },
    "fold_state": {
        "class_ids": {4, 5},
        "choices": ["folded", "unfolded", "cannot tell", "not visible"],
        "q_bbox": "Look at the object highlighted with the red box. Is it folded or unfolded?",
        "q_plain": "Is the main object in the image folded or unfolded?",
    },
    "connection_state": {
        "class_ids": {7, 8},
        "choices": ["unconnected", "connected", "cannot tell", "not visible"],
        "q_bbox": "Look at the object highlighted with the red box. Is it connected or unconnected?",
        "q_plain": "Is the main object in the image connected or unconnected?",
    },
}

IMG_EXTS = [".png", ".jpg", ".jpeg"]


def find_image_for_txt(txt_path: Path):
    for ext in IMG_EXTS:
        p = txt_path.with_suffix(ext)
        if p.exists():
            return p
    return None


def group_for_class(cid: int):
    for g, spec in GROUPS.items():
        if cid in spec["class_ids"]:
            return g, spec
    return None, None


def parse_yolo(txt_path: Path):
    objs = []
    for line in txt_path.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            x, y, bw, bh = map(float, parts[1:5])
        except Exception:
            continue

        if cid not in CLASS_MAP:
            continue

        group, spec = group_for_class(cid)
        if group is None:
            continue

        objs.append({
            "class_id": cid,
            "state": CLASS_MAP[cid],
            "group": group,
            "box": (x, y, bw, bh),
        })
    return objs


def yolo_to_xyxy(box, w, h, pad=0.05):
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

    return (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(w, int(round(x2))),
        min(h, int(round(y2))),
    )


def draw_bbox_full_image(src_img_path: Path, box, out_path: Path, pad=0.05):
    img = Image.open(src_img_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = yolo_to_xyxy(box, w, h, pad=pad)

    draw = ImageDraw.Draw(img)
    line_w = max(3, int(min(w, h) * 0.008))
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=line_w)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def make_choices(choices_base, answer, seed):
    choices = list(choices_base)
    rng = random.Random(seed)
    rng.shuffle(choices)
    return choices, choices.index(answer)


def build_split(split_root: Path, split_name: str, out_dir: Path, bbox_prob_single: float, max_objects_per_class=None):
    out_img_dir = out_dir / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(123)
    rows = []
    saved_per_class = Counter()
    skipped = Counter()
    mode_counts = Counter()

    txts = sorted(split_root.rglob("*.txt"))

    for txt_path in txts:
        img_path = find_image_for_txt(txt_path)
        if img_path is None:
            skipped["missing_image"] += 1
            continue

        objs = parse_yolo(txt_path)
        if not objs:
            skipped["no_valid_objects"] += 1
            continue

        total_objs = len(objs)

        for obj_i, obj in enumerate(objs):
            cid = obj["class_id"]
            state = obj["state"]
            group = obj["group"]
            spec = GROUPS[group]

            if max_objects_per_class is not None and saved_per_class[cid] >= max_objects_per_class:
                skipped["max_per_class"] += 1
                continue

            # Главное правило:
            # 1 объект на фото => половина с bbox, половина без bbox
            # несколько объектов => обязательно bbox, иначе вопрос неоднозначен
            if total_objs == 1:
                use_bbox = rng.random() < bbox_prob_single
            else:
                use_bbox = True

            mode = "bbox" if use_bbox else "plain"
            question = spec["q_bbox"] if use_bbox else spec["q_plain"]

            ex_id = f"osdd_{split_name}_{txt_path.stem}_obj{obj_i:02d}_c{cid}_{mode}"
            rel_img = f"images/{ex_id}.jpg"
            out_img_path = out_dir / rel_img

            if use_bbox:
                draw_bbox_full_image(img_path, obj["box"], out_img_path)
            else:
                # Полное изображение без bbox.
                shutil.copy2(img_path, out_img_path)

            choices, gt_idx = make_choices(spec["choices"], state, ex_id)

            rows.append({
                "id": ex_id,
                "image": rel_img,
                "question": question,
                "choices": str(choices),
                "gt_idx": gt_idx,
                "answer": state,
                "task": group,
                "state": state,
                "class_id": cid,
                "mode": mode,
                "source_dataset": "OSDD",
                "source_image": str(img_path),
                "source_txt": str(txt_path),
                "bbox_xywh_yolo": str(obj["box"]),
                "objects_in_source_image": total_objs,
                "split": split_name,
            })

            saved_per_class[cid] += 1
            mode_counts[mode] += 1

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    df.to_csv(out_dir / "metadata.csv", index=False)
    with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in df.to_dict(orient="records"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 100)
    print("SPLIT:", split_name)
    print("split_root:", split_root)
    print("out_dir:", out_dir)
    print("rows:", len(df))
    print("mode counts:")
    print(pd.Series(mode_counts).to_string())
    print("\nanswer counts:")
    print(df["answer"].value_counts().to_string() if len(df) else "")
    print("\ntask counts:")
    print(df["task"].value_counts().to_string() if len(df) else "")
    print("\nclass counts:")
    for cid, n in sorted(saved_per_class.items()):
        print(cid, CLASS_MAP[cid], n)
    print("\nskipped:")
    print(pd.Series(skipped).to_string() if skipped else "none")
    print("\npreview:")
    if len(df):
        print(df[["image", "mode", "question", "choices", "answer", "task", "objects_in_source_image"]].head(10).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osdd-root", default="osdd_data")
    ap.add_argument("--out-root", default="osdd_hybrid_bbox_state_mcq")
    ap.add_argument("--bbox-prob-single", type=float, default=0.5)
    ap.add_argument("--max-objects-per-class", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.osdd_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    split_roots = {
        "train": root / "train",
        "val": root / "val",
        "test": root / "test",
    }

    for split, split_root in split_roots.items():
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        build_split(
            split_root=split_root,
            split_name=split,
            out_dir=out_dir,
            bbox_prob_single=args.bbox_prob_single,
            max_objects_per_class=args.max_objects_per_class,
        )

    print("\nDONE:", out_root)


if __name__ == "__main__":
    main()
