from pathlib import Path
import pandas as pd
import shutil

SRC = Path("robotpct")
OUT = Path("robot_tpose_mcq")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

QUESTION = "Which option best describes the image?"

CHOICES = [
    "The robot is standing with its arms spread out to the sides.",
    "The robot is running.",
    "The robot is lying down.",
    "The robot is leaning backward.",
]

ANSWER = CHOICES[0]
GT_IDX = 0

if OUT.exists():
    shutil.rmtree(OUT)

(OUT / "images").mkdir(parents=True, exist_ok=True)

imgs = [p for p in sorted(SRC.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]

rows = []
for i, src in enumerate(imgs):
    ext = src.suffix.lower()
    dst_rel = f"images/robot_tpose_{i:05d}{ext}"
    shutil.copy2(src, OUT / dst_rel)

    rows.append({
        "id": f"robot_tpose_{i:05d}",
        "image": dst_rel,
        "question": QUESTION,
        "choices": str(CHOICES),
        "gt_idx": GT_IDX,
        "answer": ANSWER,
        "task": "robot_pose_mcq",
        "source_path": str(src),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "metadata.csv", index=False)

print("DONE:", OUT)
print("rows:", len(df))
print(df[["image", "question", "choices", "answer", "gt_idx", "source_path"]].to_string(index=False, max_colwidth=200))
