from pathlib import Path
import pandas as pd
import shutil
import random

QUESTION = "Is the door closed?"
BASE_CHOICES = ["Yes", "No"]

def make_from_existing(src_dir, out_dir, label_col="door_state", seed=42):
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(src_dir / "images", out_dir / "images")

    df = pd.read_csv(src_dir / "metadata.csv").copy()
    rng = random.Random(seed)

    rows = []
    for i, r in df.iterrows():
        rr = r.copy()

        if label_col in rr:
            label = str(rr[label_col]).strip().lower()
        else:
            label = str(rr.get("manual_label", "")).strip().lower()

        if label in ["closed", "close", "blocked"]:
            answer = "Yes"
            door_state = "closed"
        elif label in ["open", "clear"]:
            answer = "No"
            door_state = "open"
        else:
            continue

        choices = BASE_CHOICES[:]
        rng.shuffle(choices)

        rr["question"] = QUESTION
        rr["choices"] = str(choices)
        rr["gt_idx"] = choices.index(answer)
        rr["answer"] = answer
        rr["door_state"] = door_state
        rr["task"] = "door_closed_2choice_eval"
        rows.append(rr)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "metadata.csv", index=False)

    print("\nDONE:", out_dir)
    print("rows:", len(out))
    print("door_state:")
    print(out["door_state"].value_counts().to_string())
    print("gt_idx:")
    print(out["gt_idx"].value_counts().sort_index().to_string())
    print("choices:")
    print(out["choices"].value_counts().to_string())

make_from_existing(
    "reflect_doors_2choice_closed_mcq_split/test",
    "reflect_doors_2choice_closed_mcq_split_test_eval",
    label_col="door_state",
    seed=44,
)

make_from_existing(
    "deepdoors2_door_closed_simple_20_qwen",
    "deepdoors2_door_closed_simple_20_2choice_eval",
    label_col="door_state",
    seed=45,
)

make_from_existing(
    "real_glass_hard22_simple_closed_mcq",
    "real_glass_hard22_2choice_eval",
    label_col="door_state",
    seed=46,
)
