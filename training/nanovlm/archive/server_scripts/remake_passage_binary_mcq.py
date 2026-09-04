from pathlib import Path
import argparse
import pandas as pd
import random
import shutil

QUESTION = "Can the robot pass through this passage without opening or moving anything?"
BASE_CHOICES = [
    "Yes, the passage is clear.",
    "No, the passage is blocked.",
    "Cannot tell from the image.",
    "There is no visible passage.",
]

def read_src_dataset(ds_dir: Path):
    meta = ds_dir / "metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(f"metadata.csv not found in {ds_dir}")
    df = pd.read_csv(meta)
    if "image" not in df.columns:
        raise RuntimeError(f"'image' column not found in {meta}")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-datasets", nargs="+", required=True)
    ap.add_argument("--blocked-datasets", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    rows = []
    idx = 0

    def process_dataset(ds_path_str, target_label):
        nonlocal idx
        ds_dir = Path(ds_path_str)
        df = read_src_dataset(ds_dir)

        for _, r in df.iterrows():
            rel_img = str(r["image"])
            src_img = ds_dir / rel_img
            if not src_img.exists():
                print(f"SKIP missing image: {src_img}")
                continue

            ext = src_img.suffix.lower() or ".jpg"
            dst_name = f"sample_{idx:05d}{ext}"
            dst_rel = f"images/{dst_name}"
            dst_img = img_out / dst_name
            shutil.copy2(src_img, dst_img)

            choices = BASE_CHOICES[:]
            random.shuffle(choices)

            if target_label == "open":
                gt_answer = "Yes, the passage is clear."
            else:
                gt_answer = "No, the passage is blocked."

            gt_idx = choices.index(gt_answer)

            rows.append({
                "id": f"passage_{idx:05d}",
                "image": dst_rel,
                "question": QUESTION,
                "choices": str(choices),
                "gt_idx": gt_idx,
                "answer": gt_answer,
                "task": "passage_binary",
                "source_label": target_label,
                "source_dataset": ds_dir.name,
            })
            idx += 1

    for ds in args.open_datasets:
        process_dataset(ds, "open")

    for ds in args.blocked_datasets:
        process_dataset(ds, "blocked")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "metadata.csv", index=False)

    print("DONE:", out_dir)
    print("rows:", len(out_df))
    if len(out_df):
        print("\nsource_label counts:")
        print(out_df["source_label"].value_counts().to_string())
        print("\nsource_dataset counts:")
        print(out_df["source_dataset"].value_counts().to_string())
        print("\npreview:")
        print(out_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
