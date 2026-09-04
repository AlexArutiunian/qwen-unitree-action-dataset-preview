from pathlib import Path
import pandas as pd
import shutil
import random

SRC = Path("reflect_doors_all_yesno4_eval")
OUT = Path("reflect_only_yesno4_split")

CHOICES_BASE = ["Yes", "No", "Not applicable", "Invalid image"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42

if OUT.exists():
    shutil.rmtree(OUT)

for split in ["train", "val", "test"]:
    (OUT / split / "images").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC / "metadata.csv")
rng = random.Random(SEED)

def split_group(g):
    idxs = list(g.index)
    rng.shuffle(idxs)

    n = len(idxs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = idxs[:n_train]
    val = idxs[n_train:n_train+n_val]
    test = idxs[n_train+n_val:]

    return train, val, test

splits = {"train": [], "val": [], "test": []}

for state, g in df.groupby("door_state"):
    tr, va, te = split_group(g)
    splits["train"].extend(tr)
    splits["val"].extend(va)
    splits["test"].extend(te)

def make_split(split, idxs):
    rows = []
    local_rng = random.Random(SEED + len(split))

    for new_i, old_i in enumerate(idxs):
        r = df.loc[old_i].copy()

        src_img = SRC / str(r["image"])
        ext = src_img.suffix or ".jpg"
        dst_rel = f"images/reflect_{split}_{new_i:05d}{ext}"
        shutil.copy2(src_img, OUT / split / dst_rel)

        answer = str(r["answer"]).strip()
        choices = CHOICES_BASE[:]
        local_rng.shuffle(choices)

        r["id"] = f"reflect_{split}_{new_i:05d}"
        r["image"] = dst_rel
        r["question"] = "Is the door closed?"
        r["choices"] = str(choices)
        r["gt_idx"] = choices.index(answer)
        r["answer"] = answer
        r["task"] = "reflect_only_yesno4"
        rows.append(r)

    out = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    out.to_csv(OUT / split / "metadata.csv", index=False)

    print("\n" + "="*80)
    print(split)
    print("rows:", len(out))
    print("door_state:")
    print(out["door_state"].value_counts().to_string())
    print("answer:")
    print(out["answer"].value_counts().to_string())
    print("gt_idx:")
    print(out["gt_idx"].value_counts().sort_index().to_string())

for split, idxs in splits.items():
    make_split(split, idxs)

print("\nDONE:", OUT)
