#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import pandas as pd

SRC_ROOT = Path("deepdoors2_door_state_mcq")
OUT_ROOT = Path("deepdoors2_door_state_binary_mcq")

for split in ["train", "val", "test"]:
    src_dir = SRC_ROOT / split
    out_dir = OUT_ROOT / split
    out_img_dir = out_dir / "images"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src_dir / "metadata.csv")

    # убираем partially_open
    df = df[df["door_state"].isin(["open", "closed"])].copy()

    # оставляем все 3 типа вопросов для open/closed
    # door_state_mcq включает 2 формулировки на картинку, это ок
    used_images = sorted(df["image"].unique())

    for rel in used_images:
        src_img = src_dir / rel
        dst_img = out_dir / rel
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if src_img.exists():
            shutil.copy2(src_img, dst_img)

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(out_dir / "metadata.csv", index=False)

    with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in df.to_dict(orient="records"):
            import json
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "="*80)
    print(split)
    print("rows:", len(df))
    print("images:", len(used_images))
    print("states:")
    print(df["door_state"].value_counts().to_string())
    print("tasks:")
    print(df["task"].value_counts().to_string())
    print("answers:")
    print(df["answer"].value_counts().to_string())

print("\nDONE:", OUT_ROOT)
