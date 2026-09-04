#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil, json, random
import pandas as pd

RAW = Path("door_closed_glass")
OUT = Path("real_glass_closed_door_mcq")

CHOICES = [
    "Yes, there is a closed glass door.",
    "No, the passage is open.",
    "Cannot tell from the image.",
    "There is no passage visible.",
]

QUESTION = "Is there a closed glass door in front of the passage?"
ANSWER = "Yes, there is a closed glass door."

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "images").mkdir(parents=True, exist_ok=True)

imgs = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]:
    imgs.extend(RAW.rglob(ext))

imgs = sorted(set(imgs))
print("images found:", len(imgs))

rows = []
for i, src in enumerate(imgs):
    sample_id = f"glass_closed_{i:04d}"
    rel = f"images/{sample_id}{src.suffix.lower()}"
    shutil.copy2(src, OUT / rel)

    choices = list(CHOICES)
    random.Random(sample_id).shuffle(choices)

    rows.append({
        "id": sample_id,
        "image": rel,
        "question": QUESTION,
        "choices": str(choices),
        "gt_idx": choices.index(ANSWER),
        "answer": ANSWER,
        "task": "closed_glass_door_detection",
        "door_state": "closed",
        "visual_case": "closed_glass",
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "metadata.csv", index=False)

with open(OUT / "metadata.jsonl", "w", encoding="utf-8") as f:
    for r in df.to_dict(orient="records"):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("DONE:", OUT)
print("rows:", len(df))
print(df.head(20).to_string(index=False))
