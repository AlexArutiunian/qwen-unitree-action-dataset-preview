from pathlib import Path
import pandas as pd
import shutil

SRC_ROOT = Path("deepdoors2_door_state_mcq")
OUT = Path("door_cls_deepdoors2")

if OUT.exists():
    shutil.rmtree(OUT)

for split in ["train", "val", "test"]:
    for cls in ["closed", "open"]:
        (OUT / split / cls).mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    src_dir = SRC_ROOT / split
    df = pd.read_csv(src_dir / "metadata.csv")

    # Только open/closed. partially_open пока выкидываем.
    df = df[df["door_state"].isin(["open", "closed"])].copy()

    # В DeepDoors2 metadata много строк на одну картинку из-за разных questions.
    # Для классификатора нужна одна копия на уникальную картинку.
    df = df.drop_duplicates(subset=["image", "door_state"]).reset_index(drop=True)

    counts = {"open": 0, "closed": 0}

    for i, r in df.iterrows():
        state = str(r["door_state"]).lower()
        src = src_dir / str(r["image"])
        if not src.exists():
            print("missing:", src)
            continue

        ext = src.suffix or ".jpg"
        dst = OUT / split / state / f"{split}_{state}_{counts[state]:06d}{ext}"
        shutil.copy2(src, dst)
        counts[state] += 1

    print("\n", split)
    print(counts)

print("\nDONE:", OUT)
