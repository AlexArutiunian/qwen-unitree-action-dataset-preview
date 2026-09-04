from pathlib import Path
import pandas as pd
import shutil
import random

SRC = Path("robotpct")
OUT = Path("robot_tpose_mcq_shuffled")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

QUESTION = "Which option best describes the image?"
CORRECT = "The robot is standing with its arms spread out to the sides."
ALL_CHOICES = [
    CORRECT,
    "The robot is running.",
    "The robot is lying down.",
    "The robot is leaning backward.",
]

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "images").mkdir(parents=True, exist_ok=True)

imgs = [p for p in sorted(SRC.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]
rows = []

for img_i, src in enumerate(imgs):
    ext = src.suffix.lower()
    dst_rel = f"images/robot_tpose_{img_i:05d}{ext}"
    shutil.copy2(src, OUT / dst_rel)

    for rep in range(24):
        rng = random.Random(1000 + rep)
        choices = ALL_CHOICES[:]
        rng.shuffle(choices)

        rows.append({
            "id": f"robot_tpose_{img_i:05d}_rep{rep:02d}",
            "image": dst_rel,
            "question": QUESTION,
            "choices": str(choices),
            "gt_idx": choices.index(CORRECT),
            "answer": CORRECT,
            "task": "robot_pose_mcq_shuffled",
            "source_path": str(src),
        })

df = pd.DataFrame(rows)
df.to_csv(OUT / "metadata.csv", index=False)

print("DONE:", OUT)
print("rows:", len(df))
print("gt_idx counts:")
print(df["gt_idx"].value_counts().sort_index().to_string())
print(df[["image", "choices", "answer", "gt_idx"]].head(10).to_string(index=False, max_colwidth=220))
