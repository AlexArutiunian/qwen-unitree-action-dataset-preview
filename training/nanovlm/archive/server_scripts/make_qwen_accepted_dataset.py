#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source-dataset-dir", required=True)
    parser.add_argument("--accepted-csv", required=True)
    parser.add_argument("--out-dir", required=True)

    args = parser.parse_args()

    source_dir = Path(args.source_dataset_dir)
    accepted_csv = Path(args.accepted_csv)
    out_dir = Path(args.out_dir)

    out_img_dir = out_dir / "images"
    out_vid_dir = out_dir / "videos"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_vid_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(accepted_csv)

    # Важно: для train оставляем исходную разметку, потому что она совпала с Qwen.
    keep_cols = [
        "id", "uid", "video", "image", "question", "choices", "gt_idx",
        "answer", "task", "instruction", "task_header",
        "raw_video_name", "hf_video_path", "frame_ratio",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df2 = df[keep_cols].copy()

    copied_images = 0
    copied_videos = 0

    for p in df2["image"].dropna().unique():
        src = source_dir / p
        dst = out_dir / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied_images += 1

    # видео для train не обязательно, но оставим опционально только ссылки в CSV.
    # Физически видео не копируем, чтобы не раздувать папку.

    df2.to_csv(out_dir / "metadata.csv", index=False)

    print("DONE")
    print("accepted rows:", len(df2))
    print("copied images:", copied_images)
    print("out_dir:", out_dir)
    print("answer counts:")
    print(df2["answer"].value_counts())
    print("task counts:")
    print(df2["task"].value_counts())


if __name__ == "__main__":
    main()
