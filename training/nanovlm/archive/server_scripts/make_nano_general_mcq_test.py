#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd
import json

OUT = Path("debug_nano_general_mcq")
if OUT.exists():
    import shutil
    shutil.rmtree(OUT)

(OUT / "images").mkdir(parents=True, exist_ok=True)

# простая белая картинка, чтобы модель получила image input
img = Image.new("RGB", (512, 512), "white")
draw = ImageDraw.Draw(img)
draw.text((30, 30), "General knowledge test", fill=(0, 0, 0))
img.save(OUT / "images" / "blank.jpg", quality=95)

rows = [
    {
        "id": "rainbow_colors",
        "image": "images/blank.jpg",
        "question": "How many colors are in a rainbow?",
        "choices": str(["Seven.", "Three.", "Twelve.", "One."]),
        "gt_idx": 0,
        "answer": "Seven.",
        "task": "general_knowledge",
    },
    {
        "id": "two_plus_two",
        "image": "images/blank.jpg",
        "question": "What is 2 plus 2?",
        "choices": str(["Four.", "Five.", "Two.", "Twenty two."]),
        "gt_idx": 0,
        "answer": "Four.",
        "task": "math_simple",
    },
    {
        "id": "capital_france",
        "image": "images/blank.jpg",
        "question": "What is the capital of France?",
        "choices": str(["Paris.", "London.", "Berlin.", "Rome."]),
        "gt_idx": 0,
        "answer": "Paris.",
        "task": "general_knowledge",
    },
    {
        "id": "sky_color",
        "image": "images/blank.jpg",
        "question": "What color is the clear daytime sky usually?",
        "choices": str(["Blue.", "Green.", "Black.", "Orange."]),
        "gt_idx": 0,
        "answer": "Blue.",
        "task": "general_knowledge",
    },
    {
        "id": "hello_reply",
        "image": "images/blank.jpg",
        "question": "If someone says hello, what is a normal reply?",
        "choices": str(["Hello.", "Goodbye forever.", "Seven.", "Closed."]),
        "gt_idx": 0,
        "answer": "Hello.",
        "task": "conversation_simple",
    },
]

df = pd.DataFrame(rows)
df.to_csv(OUT / "metadata.csv", index=False)

with open(OUT / "metadata.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("DONE:", OUT)
print(df.to_string(index=False))
