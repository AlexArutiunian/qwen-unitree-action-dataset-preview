#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm


def get_image(ex):
    img = ex.get("image", None)

    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if isinstance(img, str):
        return Image.open(img).convert("RGB")

    if isinstance(img, dict):
        if "path" in img and img["path"]:
            return Image.open(img["path"]).convert("RGB")

    raise ValueError(f"Cannot read image. Keys={list(ex.keys())}")


def get_question(ex):
    for k in ["question", "Question", "query"]:
        if k in ex:
            return str(ex[k])
    raise ValueError(f"No question field. Keys={list(ex.keys())}")


def get_choices(ex):
    for k in ["choices", "Choices", "answer_choices"]:
        if k in ex:
            choices = list(ex[k])
            if len(choices) >= 4:
                return [str(x) for x in choices[:4]]

    direct = []
    for k in ["A", "B", "C", "D"]:
        if k in ex:
            direct.append(str(ex[k]))
    if len(direct) == 4:
        return direct

    raise ValueError(f"No choices field. Keys={list(ex.keys())}")


def get_gt_idx(ex, choices):
    for k in ["correct_choice_idx", "gt_idx", "label", "answer_idx", "correct_idx"]:
        if k in ex:
            return int(ex[k])

    for k in ["answer", "correct_answer", "direct_answer"]:
        if k in ex:
            ans = str(ex[k]).strip().lower()
            for i, ch in enumerate(choices):
                if str(ch).strip().lower() == ans:
                    return i

    raise ValueError(f"No gt idx field. Keys={list(ex.keys())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="HuggingFaceM4/A-OKVQA")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out-dir", default="aokvqa_eval_300")
    parser.add_argument("--max-examples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Loading:", args.dataset_name, args.split)

    try:
        ds = load_dataset(args.dataset_name, split=args.split)
    except Exception as e:
        print("Failed split:", args.split, e)
        print("Trying validation...")
        ds = load_dataset(args.dataset_name, split="validation")

    indices = list(range(len(ds)))
    random.shuffle(indices)

    rows = []
    used = 0
    skipped = 0

    for idx in tqdm(indices, desc="Building A-OKVQA image-MCQ dataset"):
        if used >= args.max_examples:
            break

        try:
            ex = ds[idx]
            img = get_image(ex)
            question = get_question(ex)
            choices = get_choices(ex)
            gt_idx = get_gt_idx(ex, choices)

            if len(choices) != 4:
                skipped += 1
                continue
            if gt_idx < 0 or gt_idx >= 4:
                skipped += 1
                continue

            img_name = f"aokvqa_{used:06d}_idx{idx}.jpg"
            img.save(img_dir / img_name, quality=95)

            rows.append({
                "id": f"aokvqa_{used:06d}",
                "source_idx": idx,
                "image": f"images/{img_name}",
                "question": question,
                "choices": str(choices),
                "gt_idx": gt_idx,
                "answer": choices[gt_idx],
                "task": "aokvqa_mcq",
            })

            used += 1

        except Exception as e:
            skipped += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metadata.csv", index=False)

    with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nDONE")
    print("out_dir:", out_dir)
    print("rows:", len(df))
    print("skipped:", skipped)

    if len(df):
        print("\nGT letter counts:")
        print(df["gt_idx"].value_counts().sort_index())
        print("\nPreview:")
        print(df[["image", "question", "choices", "gt_idx", "answer"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
