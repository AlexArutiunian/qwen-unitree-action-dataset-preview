from pathlib import Path
import pandas as pd
import shutil
import random
import argparse

CANONICAL = [
    "Yes",
    "No",
    "There is no any door",
    "I do not see image",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    if out.exists():
        shutil.rmtree(out)

    shutil.copytree(src / "images", out / "images")

    df = pd.read_csv(src / "metadata.csv")
    rows = []

    rng = random.Random(args.seed)

    for i, r in df.iterrows():
        answer = str(r["answer"]).strip()
        if answer not in CANONICAL:
            raise ValueError(f"Unknown answer at row {i}: {answer}")

        choices = CANONICAL[:]
        rng.shuffle(choices)

        rr = r.copy()
        rr["choices"] = str(choices)
        rr["gt_idx"] = choices.index(answer)
        rr["answer"] = answer
        rows.append(rr)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out / "metadata.csv", index=False)

    print("DONE:", out)
    print("rows:", len(out_df))
    print("\nchoices examples:")
    print(out_df["choices"].value_counts().head(10).to_string())
    print("\ngt_idx counts:")
    print(out_df["gt_idx"].value_counts().sort_index().to_string())
    print("\nanswer counts:")
    print(out_df["answer"].value_counts().to_string())

if __name__ == "__main__":
    main()
