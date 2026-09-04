#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd
import json
import shutil

OUT = Path("debug_nano_raw_general_mcq")
if OUT.exists():
    shutil.rmtree(OUT)

(OUT / "images").mkdir(parents=True, exist_ok=True)

img = Image.new("RGB", (512, 512), "white")
draw = ImageDraw.Draw(img)
draw.text((30, 30), "General MCQ test", fill=(0, 0, 0))
img.save(OUT / "images" / "blank.jpg", quality=95)

rows = [
    {
        "id": "rainbow_colors",
        "image": "images/blank.jpg",
        "question": "How many colors are in a rainbow?",
        "choices": str(["Seven.", "Three.", "Twelve.", "One."]),
        "gt_idx": 0,
        "answer": "Seven.",
        "task": "general",
    },
    {
        "id": "two_plus_two",
        "image": "images/blank.jpg",
        "question": "What is 2 plus 2?",
        "choices": str(["Four.", "Five.", "Two.", "Twenty two."]),
        "gt_idx": 0,
        "answer": "Four.",
        "task": "math",
    },
    {
        "id": "capital_france",
        "image": "images/blank.jpg",
        "question": "What is the capital of France?",
        "choices": str(["Paris.", "London.", "Berlin.", "Rome."]),
        "gt_idx": 0,
        "answer": "Paris.",
        "task": "general",
    },
    {
        "id": "hello_reply",
        "image": "images/blank.jpg",
        "question": "If someone says hello, what is a normal reply?",
        "choices": str(["Hello.", "Goodbye forever.", "Seven.", "Closed."]),
        "gt_idx": 0,
        "answer": "Hello.",
        "task": "conversation",
    },
    {
        "id": "door_open_text_only",
        "image": "images/blank.jpg",
        "question": "If a door is fully open, can a person usually pass through it?",
        "choices": str(["Yes.", "No.", "Cannot tell.", "The door is not visible."]),
        "gt_idx": 0,
        "answer": "Yes.",
        "task": "commonsense",
    },
]

df = pd.DataFrame(rows)
df.to_csv(OUT / "metadata.csv", index=False)

with open(OUT / "metadata.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("DONE:", OUT)
print(df.to_string(index=False))
