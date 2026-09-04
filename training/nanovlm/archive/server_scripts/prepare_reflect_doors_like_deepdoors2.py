from pathlib import Path
import pandas as pd
import shutil
import random

SRC = Path("dataset_doors_reflect")
DEEP_META = Path("deepdoors2_door_state_mcq/train/metadata.csv")
OUT = Path("reflect_doors_like_deepdoors2_split")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

random.seed(42)

if OUT.exists():
    shutil.rmtree(OUT)

for split in ["train", "val", "test"]:
    (OUT / split / "images").mkdir(parents=True, exist_ok=True)

deep = pd.read_csv(DEEP_META)

# Берём только те состояния, которые есть в твоём датасете
deep = deep[deep["door_state"].isin(["open", "closed"])].copy()

# Эти задачи уже были в обучении DeepDoors2-модели
TASKS = ["door_state_mcq", "door_open_direct", "door_passability"]
deep = deep[deep["task"].isin(TASKS)].copy()

templates = {}
for state in ["open", "closed"]:
    for task in TASKS:
        t = deep[(deep["door_state"] == state) & (deep["task"] == task)].copy()
        if len(t) == 0:
            print("WARNING: no template", state, task)
        templates[(state, task)] = t.to_dict("records")
        print("templates", state, task, len(t))

def collect(label_dir):
    p = SRC / label_dir
    files = [x for x in sorted(p.rglob("*")) if x.is_file() and x.suffix.lower() in IMG_EXTS]
    random.shuffle(files)
    return files

open_files = collect("open")
close_files = collect("close")

print("\nopen files:", len(open_files))
print("close files:", len(close_files))

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
    "train": [("open", open_train), ("closed", close_train)],
    "val": [("open", open_val), ("closed", close_val)],
    "test": [("open", open_test), ("closed", close_test)],
}

for split, groups in splits.items():
    rows = []
    img_idx = 0
    row_idx = 0

    for state, files in groups:
        for src in files:
            ext = src.suffix.lower()
            dst_rel = f"images/reflect_{split}_{img_idx:05d}{ext}"
            dst = OUT / split / dst_rel
            shutil.copy2(src, dst)

            # На каждую картинку делаем 3 строки — как в DeepDoors2:
            # state question, direct open question, passability question.
            for task in TASKS:
                pool = templates[(state, task)]
                if not pool:
                    continue

                tmpl = random.choice(pool)

                rows.append({
                    "id": f"reflect_{split}_{row_idx:06d}",
                    "image": dst_rel,
                    "question": tmpl["question"],
                    "choices": tmpl["choices"],
                    "gt_idx": int(tmpl["gt_idx"]),
                    "answer": tmpl["answer"],
                    "task": task,
                    "door_state": state,
                    "manual_label": "open" if state == "open" else "close",
                    "source_dataset": "dataset_doors_reflect",
                    "source_path": str(src),
                })
                row_idx += 1

            img_idx += 1

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUT / split / "metadata.csv", index=False)

    print("\n" + "="*100)
    print(split)
    print("rows:", len(df))
    print("unique images:", df["image"].nunique())
    print("door_state:")
    print(df["door_state"].value_counts().to_string())
    print("task:")
    print(df["task"].value_counts().to_string())
    print("answer:")
    print(df["answer"].value_counts().to_string())
    print(df[["image", "door_state", "task", "question", "answer", "gt_idx"]].head(12).to_string(index=False, max_colwidth=160))

print("\nDONE:", OUT)
