from pathlib import Path
import shutil
import random

SRC = Path("dataset_doors_reflect")
OUT = Path("door_cls_reflect")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SEED = 42
TRAIN = 0.70
VAL = 0.15

if OUT.exists():
    shutil.rmtree(OUT)

for split in ["train", "val", "test"]:
    for cls in ["closed", "open"]:
        (OUT / split / cls).mkdir(parents=True, exist_ok=True)

def collect(folder):
    files = [p for p in sorted((SRC / folder).rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    random.Random(SEED).shuffle(files)
    return files

groups = {
    "closed": collect("close"),
    "open": collect("open"),
}

for cls, files in groups.items():
    n = len(files)
    n_train = int(n * TRAIN)
    n_val = int(n * VAL)

    split_files = {
        "train": files[:n_train],
        "val": files[n_train:n_train+n_val],
        "test": files[n_train+n_val:],
    }

    for split, fs in split_files.items():
        for i, src in enumerate(fs):
            ext = src.suffix.lower()
            dst = OUT / split / cls / f"reflect_{split}_{cls}_{i:05d}{ext}"
            shutil.copy2(src, dst)

    print(cls, "total:", n, "train:", len(split_files["train"]), "val:", len(split_files["val"]), "test:", len(split_files["test"]))

print("\nDONE:", OUT)
