#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil, json, random
import pandas as pd

RAW = Path("door_close_noreflect")
OUT = Path("real_door_close_noreflect_mcq")

QUESTION = "Is the door open?"
ANSWER = "No, the door is closed."

CHOICES = [
    "Yes, the door is open.",
    "No, the door is closed.",
    "Cannot tell from the image.",
    "The door is not visible.",
]

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "images").mkdir(parents=True, exist_ok=True)

imgs = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]:
    imgs.extend(RAW.rglob(ext))
imgs = sorted(set(imgs))

print("images found:", len(imgs))
if not imgs:
    raise RuntimeError(f"No images found in {RAW.resolve()}")

rows = []
for i, src in enumerate(imgs):
    sample_id = f"door_close_noreflect_{i:04d}"
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
        "task": "door_open_binary",
        "door_state": "closed",
        "visual_case": "closed_noreflect",
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "metadata.csv", index=False)

with open(OUT / "metadata.jsonl", "w", encoding="utf-8") as f:
    for r in df.to_dict(orient="records"):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("DONE:", OUT)
print("rows:", len(df))
print("answers:")
print(df["answer"].value_counts().to_string())
print(df.head(20).to_string(index=False))
