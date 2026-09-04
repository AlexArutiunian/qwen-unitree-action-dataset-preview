from pathlib import Path
import pandas as pd
import shutil
import random

SRC = Path("dataset_doors_reflect")
OUT = Path("reflect_doors_2choice_closed_mcq_split")

QUESTION = "Is the door closed?"
BASE_CHOICES = ["Yes", "No"]

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
    rng = random.Random(1000 + len(split))

    for label, files in groups:
        for src in files:
            ext = src.suffix.lower()
            dst_rel = f"images/reflect_{split}_{idx:05d}{ext}"
            shutil.copy2(src, OUT / split / dst_rel)

            if label == "close":
                answer = "Yes"
                door_state = "closed"
            else:
                answer = "No"
                door_state = "open"

            choices = BASE_CHOICES[:]
            rng.shuffle(choices)

            rows.append({
                "id": f"reflect_{split}_{idx:05d}",
                "image": dst_rel,
                "question": QUESTION,
                "choices": str(choices),
                "gt_idx": choices.index(answer),
                "answer": answer,
                "task": "door_closed_2choice",
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
    print("door_state:")
    print(df["door_state"].value_counts().to_string())
    print("gt_idx:")
    print(df["gt_idx"].value_counts().sort_index().to_string())
    print("choices:")
    print(df["choices"].value_counts().to_string())

print("\nDONE:", OUT)
