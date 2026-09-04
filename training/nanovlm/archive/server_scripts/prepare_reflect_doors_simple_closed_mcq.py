from pathlib import Path
import pandas as pd
import shutil
import random

SRC = Path("dataset_doors_reflect")
OUT = Path("reflect_doors_simple_closed_mcq_split")

QUESTION = "Is the door closed?"
CHOICES = ["Yes", "No", "There is no any door", "I do not see image"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

random.seed(42)

if OUT.exists():
    shutil.rmtree(OUT)

for split in ["train", "val", "test"]:
    (OUT / split / "images").mkdir(parents=True, exist_ok=True)

def collect(label_dir):
    p = SRC / label_dir
    files = [x for x in sorted(p.rglob("*")) if x.is_file() and x.suffix.lower() in IMG_EXTS]
    random.shuffle(files)
    return files

open_files = collect("open")
close_files = collect("close")

print("open:", len(open_files))
print("close:", len(close_files))

def split_files(files):
    n = len(files)
    n_test = max(5, int(n * 0.15))
    n_val = max(5, int(n * 0.15))
    test = files[:n_test]
    val = files[n_test:n_test+n_val]
    train = files[n_test+n_val:]
    return train, val, test

open_train, open_val, open_test = split_files(open_files)
close_train, close_val, close_test = split_files(close_files)

splits = {
    "train": [("open", open_train), ("close", close_train)],
    "val": [("open", open_val), ("close", close_val)],
    "test": [("open", open_test), ("close", close_test)],
}

for split, groups in splits.items():
    rows = []
    idx = 0

    for label, files in groups:
        for src in files:
            ext = src.suffix.lower()
            dst_rel = f"images/reflect_{split}_{idx:05d}{ext}"
            dst = OUT / split / dst_rel
            shutil.copy2(src, dst)

            if label == "close":
                gt_idx = 0
                answer = "Yes"
                door_state = "closed"
            else:
                gt_idx = 1
                answer = "No"
                door_state = "open"

            rows.append({
                "id": f"reflect_{split}_{idx:05d}",
                "image": dst_rel,
                "question": QUESTION,
                "choices": str(CHOICES),
                "gt_idx": gt_idx,
                "answer": answer,
                "task": "door_closed_simple",
                "door_state": door_state,
                "manual_label": label,
                "source_dataset": "dataset_doors_reflect",
                "source_path": str(src),
            })
            idx += 1

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUT / split / "metadata.csv", index=False)

    print("\n" + "="*80)
    print(split, "rows:", len(df))
    print("unique images:", df["image"].nunique())
    print(df["door_state"].value_counts().to_string())
    print(df[["image", "door_state", "answer", "gt_idx", "source_path"]].head(10).to_string(index=False, max_colwidth=140))

print("\nDONE:", OUT)
