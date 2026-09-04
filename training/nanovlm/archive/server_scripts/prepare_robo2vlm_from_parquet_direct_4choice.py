#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import io
import json
import random
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


def answer_to_idx(ans):
    s = str(ans).strip()
    if s in ["A", "B", "C", "D", "E", "F"]:
        return ord(s) - ord("A")
    try:
        return int(float(s))
    except Exception:
        return None


def save_image(obj, out_path):
    if isinstance(obj, Image.Image):
        im = obj.convert("RGB")
    elif isinstance(obj, bytes):
        im = Image.open(io.BytesIO(obj)).convert("RGB")
    elif isinstance(obj, dict):
        if obj.get("bytes") is not None:
            im = Image.open(io.BytesIO(obj["bytes"])).convert("RGB")
        elif obj.get("path"):
            im = Image.open(obj["path"]).convert("RGB")
        else:
            raise ValueError(f"bad image dict keys={obj.keys()}")
    else:
        try:
            im = obj.convert("RGB")
        except Exception as e:
            raise ValueError(f"unsupported image type {type(obj)}") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=95)


def make_4_choices(choices, correct_answer, seed):
    choices = [str(x) for x in list(choices)]
    gt_old = answer_to_idx(correct_answer)

    if gt_old is None or gt_old < 0 or gt_old >= len(choices):
        return None, None, None

    correct_text = choices[gt_old]
    distractors = [c for i, c in enumerate(choices) if i != gt_old]

    rng = random.Random(seed)
    rng.shuffle(distractors)

    new_choices = [correct_text] + distractors[:3]

    # если вариантов меньше 4 — добиваем заглушками
    pads = ["Cannot tell.", "Not applicable.", "Unknown.", "None of the above."]
    for p in pads:
        if len(new_choices) >= 4:
            break
        if p not in new_choices:
            new_choices.append(p)

    new_choices = new_choices[:4]
    rng.shuffle(new_choices)
    gt_new = new_choices.index(correct_text)

    return new_choices, gt_new, correct_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet-glob",
        default="/home/arutiunyan_ag/.cache/huggingface/hub/datasets--keplerccc--ManipulationVQA-60k/snapshots/27bac3049503042784397d9a7f0cc83064f96087/data/train-*.parquet",
    )
    ap.add_argument("--out-dir", default="robo2vlm_50k_mcq")
    ap.add_argument("--max-train", type=int, default=50000)
    ap.add_argument("--max-val", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(Path("/").glob(args.parquet_glob.lstrip("/")))
    print("parquet files:", len(files))
    for p in files[:5]:
        print(" ", p, f"{p.stat().st_size / 1024 / 1024:.1f} MB")

    if not files:
        raise RuntimeError("No parquet files found")

    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "train" / "images").mkdir(parents=True, exist_ok=True)
    (out / "val" / "images").mkdir(parents=True, exist_ok=True)

    rows_train = []
    rows_val = []
    bad = 0
    kept = 0
    total_need = args.max_train + args.max_val

    pbar = tqdm(total=total_need, desc="convert Robo2VLM parquet")

    for file in files:
        if kept >= total_need:
            break

        pf = pq.ParquetFile(file)

        for batch in pf.iter_batches(batch_size=args.batch_size):
            if kept >= total_need:
                break

            df = batch.to_pandas()

            for _, row in df.iterrows():
                if kept >= total_need:
                    break

                try:
                    choices, gt_idx, answer_text = make_4_choices(
                        row["choices"],
                        row["correct_answer"],
                        seed=f"{args.seed}_{row.get('id', kept)}",
                    )
                    if choices is None:
                        bad += 1
                        continue

                    split = "train" if kept < args.max_train else "val"
                    j = kept if split == "train" else kept - args.max_train

                    sample_id = f"robo2vlm_{split}_{j:06d}"
                    rel_img = f"images/{sample_id}.jpg"

                    save_image(row["image"], out / split / rel_img)

                    rec = {
                        "id": sample_id,
                        "image": rel_img,
                        "question": str(row["question"]),
                        "choices": str(choices),
                        "gt_idx": int(gt_idx),
                        "answer": str(answer_text),
                        "task": "robo2vlm",
                        "source_id": str(row.get("id", "")),
                        "source_file": str(file),
                        "orig_correct_answer": str(row["correct_answer"]),
                        "orig_num_choices": int(len(row["choices"])),
                    }

                    if split == "train":
                        rows_train.append(rec)
                    else:
                        rows_val.append(rec)

                    kept += 1
                    pbar.update(1)

                except Exception:
                    bad += 1
                    continue

    pbar.close()

    for split, rows in [("train", rows_train), ("val", rows_val)]:
        df = pd.DataFrame(rows)
        df.to_csv(out / split / "metadata.csv", index=False)

        with open(out / split / "metadata.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print("\n" + "=" * 100)
        print(split)
        print("rows:", len(df))
        if len(df):
            print("unique images:", df["image"].nunique())
            print("\norig_num_choices:")
            print(df["orig_num_choices"].value_counts().to_string())
            print("\ngt_idx:")
            print(df["gt_idx"].value_counts().sort_index().to_string())
            print("\nanswer counts:")
            print(df["answer"].value_counts().head(20).to_string())
            print("\npreview:")
            print(df[["image", "question", "choices", "answer", "gt_idx"]].head(10).to_string(index=False, max_colwidth=180))

    print("\nDONE:", out.resolve())
    print("bad/skipped:", bad)


if __name__ == "__main__":
    main()
