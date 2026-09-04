#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil, json, random
import pandas as pd

RAW = Path("door_elevator_rgb_binary_open_closed")
MAN = RAW / "manifest_binary.csv"
OUT = Path("real_robot_door_only_open_mcq")

CHOICES = [
    "Yes, the door is open.",
    "No, the door is closed.",
    "Cannot tell from the image.",
    "The door is not visible.",
]

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "images").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MAN)
print("columns:", df.columns.tolist())
print("rows:", len(df))

# Ищем строки, где есть именно door_open, но не elevator_open
mask = df.astype(str).apply(lambda col: col.str.contains("door_open", case=False, regex=False)).any(axis=1)
mask &= ~df.astype(str).apply(lambda col: col.str.contains("elevator_open", case=False, regex=False)).any(axis=1)

sub = df[mask].copy().reset_index(drop=True)
print("door_open rows:", len(sub))

# колонка с путем
path_col = None
for c in sub.columns:
    if sub[c].astype(str).str.contains(r"\.(jpg|jpeg|png|bmp)", case=False, regex=True).any():
        path_col = c
        break

if path_col is None:
    raise RuntimeError("No image path column found")

print("path_col:", path_col)

rows = []
for i, r in sub.iterrows():
    src_str = str(r[path_col]).strip()
    src = Path(src_str)
    if not src.is_absolute():
        src = RAW / src

    if not src.exists():
        matches = list(RAW.rglob(Path(src_str).name))
        if matches:
            src = matches[0]

    if not src.exists():
        raise FileNotFoundError(src_str)

    sample_id = f"realdoor_door_open_{i:04d}"
    rel = f"images/{sample_id}{src.suffix.lower()}"
    shutil.copy2(src, OUT / rel)

    answer = "Yes, the door is open."
    choices = list(CHOICES)
    random.Random(sample_id).shuffle(choices)

    rows.append({
        "id": sample_id,
        "image": rel,
        "question": "Is the door open?",
        "choices": str(choices),
        "gt_idx": choices.index(answer),
        "answer": answer,
        "task": "door_open_binary",
        "door_state": "open",
        "source_class": "door_open",
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT / "metadata.csv", index=False)

with open(OUT / "metadata.jsonl", "w", encoding="utf-8") as f:
    for rec in out_df.to_dict(orient="records"):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("\nDONE:", OUT)
print("rows:", len(out_df))
print("unique images:", out_df["image"].nunique())
print(out_df.head(20).to_string(index=False))
