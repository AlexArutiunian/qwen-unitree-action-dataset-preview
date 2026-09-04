#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import pandas as pd


def copy_subset(df, src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "images").mkdir(parents=True, exist_ok=True)

    df.to_csv(dst_dir / "metadata.csv", index=False)

    copied = 0
    for p in df["image"].dropna().unique():
        src = src_dir / p
        dst = dst_dir / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src_dir = Path(args.source_dir)
    df = pd.read_csv(src_dir / "metadata.csv")

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    n = len(df)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    out_prefix = Path(args.out_prefix)

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for name, part in splits.items():
        out_dir = Path(f"{out_prefix}_{name}")
        copied = copy_subset(part, src_dir, out_dir)
        print(name, "rows:", len(part), "copied_images:", copied, "dir:", out_dir)

        if "answer" in part.columns:
            print(part["answer"].value_counts())
            print()

    print("DONE")


if __name__ == "__main__":
    main()
