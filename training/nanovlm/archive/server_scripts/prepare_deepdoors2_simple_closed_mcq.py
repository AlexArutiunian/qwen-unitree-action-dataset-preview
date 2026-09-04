from pathlib import Path
import pandas as pd
import shutil

QUESTION = "Is the door closed?"
CHOICES = ["Yes", "No", "There is no any door", "I do not see image"]

SRC_ROOT = Path("deepdoors2_door_state_mcq")
OUT_ROOT = Path("deepdoors2_simple_closed_mcq")

if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)

for split in ["train", "val", "test"]:
    src_dir = SRC_ROOT / split
    out_dir = OUT_ROOT / split
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src_dir / "metadata.csv")
    df = df[df["door_state"].isin(["open", "closed"])].copy().reset_index(drop=True)

    rows = []
    for i, r in df.iterrows():
        src_img = src_dir / str(r["image"])
        if not src_img.exists():
            print("SKIP missing:", src_img)
            continue

        ext = src_img.suffix or ".png"
        dst_rel = f"images/deepdoors2_{split}_{i:06d}{ext}"
        shutil.copy2(src_img, out_dir / dst_rel)

        state = str(r["door_state"]).strip().lower()

        rr = r.copy()
        rr["id"] = f"deepdoors2_{split}_{i:06d}"
        rr["image"] = dst_rel
        rr["question"] = QUESTION
        rr["choices"] = str(CHOICES)

        if state == "closed":
            rr["gt_idx"] = 0
            rr["answer"] = "Yes"
            rr["manual_label"] = "closed"
        else:
            rr["gt_idx"] = 1
            rr["answer"] = "No"
            rr["manual_label"] = "open"

        rr["task"] = "door_closed_simple"
        rr["source_dataset"] = "deepdoors2"
        rows.append(rr)

    out = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    out.to_csv(out_dir / "metadata.csv", index=False)

    print("\n" + "="*80)
    print(split, "rows:", len(out))
    print("unique images:", out["image"].nunique())
    print(out["door_state"].value_counts().to_string())
    print(out[["image", "door_state", "answer", "gt_idx"]].head(10).to_string(index=False))

print("\nDONE:", OUT_ROOT)
