from pathlib import Path
import pandas as pd
import shutil
import random

QUESTION = "Is the door closed?"
BASE_CHOICES = ["Yes", "No"]

SRC_ROOT = Path("deepdoors2_door_state_mcq")
OUT_ROOT = Path("deepdoors2_2choice_closed_mcq")

if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)

for split in ["train", "val", "test"]:
    rng = random.Random(2000 + len(split))

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

        if state == "closed":
            answer = "Yes"
            manual_label = "closed"
        else:
            answer = "No"
            manual_label = "open"

        choices = BASE_CHOICES[:]
        rng.shuffle(choices)

        rr = r.copy()
        rr["id"] = f"deepdoors2_{split}_{i:06d}"
        rr["image"] = dst_rel
        rr["question"] = QUESTION
        rr["choices"] = str(choices)
        rr["gt_idx"] = choices.index(answer)
        rr["answer"] = answer
        rr["task"] = "door_closed_2choice"
        rr["manual_label"] = manual_label
        rr["source_dataset"] = "deepdoors2"
        rows.append(rr)

    out = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    out.to_csv(out_dir / "metadata.csv", index=False)

    print("\n" + "="*80)
    print(split, "rows:", len(out))
    print("door_state:")
    print(out["door_state"].value_counts().to_string())
    print("gt_idx:")
    print(out["gt_idx"].value_counts().sort_index().to_string())
    print("choices:")
    print(out["choices"].value_counts().to_string())

print("\nDONE:", OUT_ROOT)
