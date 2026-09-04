from pathlib import Path
import pandas as pd
import shutil
import random

SRC = Path("dataset_doors_reflect")
OUT = Path("reflect_doors_all_yesno4_eval")

QUESTION = "Is the door closed?"
CHOICES_BASE = ["Yes", "No", "Not applicable", "Invalid image"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

if OUT.exists():
    shutil.rmtree(OUT)

(OUT / "images").mkdir(parents=True, exist_ok=True)

rng = random.Random(777)

rows = []
idx = 0

def add_folder(folder_name, door_state, answer):
    global idx

    src_dir = SRC / folder_name
    files = [
        p for p in sorted(src_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]

    print(folder_name, "files:", len(files))

    for src in files:
        ext = src.suffix.lower()
        dst_rel = f"images/reflect_all_{idx:05d}{ext}"
        shutil.copy2(src, OUT / dst_rel)

        choices = CHOICES_BASE[:]
        rng.shuffle(choices)

        rows.append({
            "id": f"reflect_all_{idx:05d}",
            "image": dst_rel,
            "question": QUESTION,
            "choices": str(choices),
            "gt_idx": choices.index(answer),
            "answer": answer,
            "task": "reflect_doors_all_yesno4_eval",
            "door_state": door_state,
            "manual_label": folder_name,
            "source_dataset": "dataset_doors_reflect_all",
            "source_path": str(src),
        })

        idx += 1

add_folder("close", "closed", "Yes")
add_folder("open", "open", "No")

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(OUT / "metadata.csv", index=False)

print("\nDONE:", OUT)
print("rows:", len(df))
print("\ndoor_state:")
print(df["door_state"].value_counts().to_string())
print("\nanswer:")
print(df["answer"].value_counts().to_string())
print("\ngt_idx:")
print(df["gt_idx"].value_counts().sort_index().to_string())
print("\nchoices examples:")
print(df["choices"].value_counts().head(10).to_string())
print("\npreview:")
print(df[["image", "door_state", "answer", "gt_idx", "source_path"]].head(20).to_string(index=False, max_colwidth=160))
