from pathlib import Path
import pandas as pd
import shutil

DEEP_TRAIN = Path("deepdoors2_simple_closed_mcq/train")
DEEP_VAL = Path("deepdoors2_simple_closed_mcq/val")

REF_TRAIN = Path("reflect_doors_simple_closed_mcq_split/train")
REF_VAL = Path("reflect_doors_simple_closed_mcq_split/val")

OUT_TRAIN = Path("deepdoors2_plus_reflect_simple_closed_train")
OUT_VAL = Path("deepdoors2_plus_reflect_simple_closed_val")

REFLECT_REPEAT_TRAIN = 8
REFLECT_REPEAT_VAL = 1

def reset_dir(p):
    if p.exists():
        shutil.rmtree(p)
    (p / "images").mkdir(parents=True, exist_ok=True)

def copy_dataset(src_dir, out_dir, prefix, repeat=1):
    df = pd.read_csv(src_dir / "metadata.csv")
    rows = []

    for rep in range(repeat):
        for i, r in df.iterrows():
            src_img = src_dir / str(r["image"])
            if not src_img.exists():
                print("SKIP missing:", src_img)
                continue

            ext = src_img.suffix or ".jpg"
            dst_rel = f"images/{prefix}_rep{rep}_{i:07d}{ext}"
            shutil.copy2(src_img, out_dir / dst_rel)

            rr = r.copy()
            rr["id"] = f"{prefix}_rep{rep}_{i:07d}"
            rr["image"] = dst_rel
            rr["source_dataset"] = prefix
            rows.append(rr)

    return pd.DataFrame(rows)

for out in [OUT_TRAIN, OUT_VAL]:
    reset_dir(out)

deep_train = copy_dataset(DEEP_TRAIN, OUT_TRAIN, "deepdoors2", repeat=1)
ref_train = copy_dataset(REF_TRAIN, OUT_TRAIN, "reflect_doors", repeat=REFLECT_REPEAT_TRAIN)
train = pd.concat([deep_train, ref_train], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
train.to_csv(OUT_TRAIN / "metadata.csv", index=False)

deep_val = copy_dataset(DEEP_VAL, OUT_VAL, "deepdoors2", repeat=1)
ref_val = copy_dataset(REF_VAL, OUT_VAL, "reflect_doors", repeat=REFLECT_REPEAT_VAL)
val = pd.concat([deep_val, ref_val], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
val.to_csv(OUT_VAL / "metadata.csv", index=False)

for name, df in [("train", train), ("val", val)]:
    print("\n" + "="*100)
    print(name)
    print("rows:", len(df))
    print("unique images:", df["image"].nunique())
    print("\nsource_dataset:")
    print(df["source_dataset"].value_counts().to_string())
    print("\ndoor_state:")
    print(df["door_state"].value_counts().to_string())
    print("\nanswer:")
    print(df["answer"].value_counts().to_string())

print("\nDONE")
print("train:", OUT_TRAIN)
print("val:", OUT_VAL)
