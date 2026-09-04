#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset


def pick_col(cols, names):
    low = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    for c in cols:
        cl = c.lower()
        for n in names:
            if n.lower() in cl:
                return c
    return None


def normalize_choices(x):
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, tuple):
        return [str(v) for v in x]
    if isinstance(x, dict):
        vals = []
        for k in ["A", "B", "C", "D", "0", "1", "2", "3"]:
            if k in x:
                vals.append(str(x[k]))
        if len(vals) >= 2:
            return vals[:4]
        return [str(v) for v in x.values()]
    if isinstance(x, str):
        s = x.strip()
        try:
            y = ast.literal_eval(s)
            return normalize_choices(y)
        except Exception:
            pass
        for sep in ["|||", "\n", ";"]:
            if sep in s:
                parts = [p.strip(" -\t") for p in s.split(sep) if p.strip()]
                if len(parts) >= 2:
                    return parts[:4]
    return None


def save_image(obj, out_path):
    if isinstance(obj, Image.Image):
        im = obj.convert("RGB")
    elif isinstance(obj, dict) and "bytes" in obj:
        import io
        im = Image.open(io.BytesIO(obj["bytes"])).convert("RGB")
    elif isinstance(obj, str):
        im = Image.open(obj).convert("RGB")
    else:
        im = obj.convert("RGB")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=95)


def infer_answer(row, choices, answer_col, idx_col):
    if idx_col and row.get(idx_col) is not None:
        try:
            idx = int(row[idx_col])
            if 0 <= idx < len(choices):
                return idx, choices[idx]
        except Exception:
            pass

    if answer_col and row.get(answer_col) is not None:
        ans = row.get(answer_col)

        if isinstance(ans, int):
            if 0 <= ans < len(choices):
                return ans, choices[ans]

        s = str(ans).strip()

        if s in ["A", "B", "C", "D"]:
            idx = ord(s) - ord("A")
            if 0 <= idx < len(choices):
                return idx, choices[idx]

        try:
            idx = int(float(s))
            if 0 <= idx < len(choices):
                return idx, choices[idx]
        except Exception:
            pass

        for i, c in enumerate(choices):
            if s.lower() == str(c).strip().lower():
                return i, choices[i]

    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="keplerccc/ManipulationVQA-60k")
    ap.add_argument("--out-dir", default="robo2vlm_50k_mcq")
    ap.add_argument("--max-train", type=int, default=50000)
    ap.add_argument("--max-val", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)

    (out / "train" / "images").mkdir(parents=True, exist_ok=True)
    (out / "val" / "images").mkdir(parents=True, exist_ok=True)

    print("Loading streaming:", args.dataset)
    ds0 = load_dataset(args.dataset, split="train", streaming=True)
    first = next(iter(ds0))
    cols = list(first.keys())

    print("columns:", cols)

    image_col = pick_col(cols, ["image", "rgb", "obs_image", "camera_image", "image_0"])
    question_col = pick_col(cols, ["question", "query", "prompt"])
    choices_col = pick_col(cols, ["choices", "options", "answer_choices", "candidates"])
    answer_col = pick_col(cols, ["answer", "correct_answer", "label", "gt_answer"])
    idx_col = pick_col(cols, ["answer_idx", "correct_idx", "gt_idx", "label_idx", "correct_choice"])

    print("detected:")
    print(" image_col:", image_col)
    print(" question_col:", question_col)
    print(" choices_col:", choices_col)
    print(" answer_col:", answer_col)
    print(" idx_col:", idx_col)

    if not image_col or not question_col or not choices_col or (not answer_col and not idx_col):
        print("\nFirst sample:")
        for k, v in first.items():
            print(k, type(v), str(v)[:1000])
        raise SystemExit("Could not detect schema")

    total_need = args.max_train + args.max_val
    rows_train, rows_val = [], []
    bad = 0
    kept = 0

    ds = load_dataset(args.dataset, split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10000)

    for row in tqdm(ds, total=total_need, desc="convert streaming"):
        if kept >= total_need:
            break

        choices = normalize_choices(row[choices_col])
        if not choices or len(choices) < 2:
            bad += 1
            continue

        choices = choices[:4]
        while len(choices) < 4:
            for pad in ["Cannot tell.", "Not applicable.", "Unknown.", "None of the above."]:
                if len(choices) >= 4:
                    break
                if pad not in choices:
                    choices.append(pad)

        gt_idx, answer = infer_answer(row, choices, answer_col, idx_col)
        if gt_idx is None:
            bad += 1
            continue

        split = "train" if kept < args.max_train else "val"
        j = kept if split == "train" else kept - args.max_train
        sample_id = f"robo2vlm_{split}_{j:06d}"
        rel_img = f"images/{sample_id}.jpg"

        try:
            save_image(row[image_col], out / split / rel_img)
        except Exception:
            bad += 1
            continue

        rec = {
            "id": sample_id,
            "image": rel_img,
            "question": str(row[question_col]),
            "choices": str(choices),
            "gt_idx": int(gt_idx),
            "answer": str(answer),
            "task": "robo2vlm",
            "source": args.dataset,
        }

        if split == "train":
            rows_train.append(rec)
        else:
            rows_val.append(rec)

        kept += 1

    for split, rows in [("train", rows_train), ("val", rows_val)]:
        df = pd.DataFrame(rows)
        df.to_csv(out / split / "metadata.csv", index=False)

        with open(out / split / "metadata.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print("\n", split)
        print("rows:", len(df))
        if len(df):
            print("answer counts:")
            print(df["answer"].value_counts().head(20).to_string())
            print(df.head(3).to_string(index=False))

    print("\nDONE:", out.resolve())
    print("bad/skipped:", bad)


if __name__ == "__main__":
    main()
