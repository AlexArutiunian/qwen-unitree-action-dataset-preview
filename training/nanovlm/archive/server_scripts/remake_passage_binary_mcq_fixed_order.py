from pathlib import Path
import argparse
import pandas as pd
import shutil

QUESTION = "Can the robot pass through this passage without opening or moving anything?"
CHOICES = [
    "Yes, the passage is clear.",
    "No, the passage is blocked.",
    "Cannot tell from the image.",
    "There is no visible passage.",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-datasets", nargs="+", required=True)
    ap.add_argument("--blocked-datasets", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    rows = []
    idx = 0

    def process(ds_path_str, label):
        nonlocal idx
        ds_dir = Path(ds_path_str)
        df = pd.read_csv(ds_dir / "metadata.csv")

        for _, r in df.iterrows():
            src = ds_dir / str(r["image"])
            if not src.exists():
                print("SKIP missing:", src)
                continue

            ext = src.suffix or ".jpg"
            dst_rel = f"images/sample_{idx:05d}{ext}"
            shutil.copy2(src, out_dir / dst_rel)

            if label == "open":
                gt_idx = 0
                answer = "Yes, the passage is clear."
            else:
                gt_idx = 1
                answer = "No, the passage is blocked."

            rows.append({
                "id": f"passage_fixed_{idx:05d}",
                "image": dst_rel,
                "question": QUESTION,
                "choices": str(CHOICES),
                "gt_idx": gt_idx,
                "answer": answer,
                "task": "passage_binary_fixed_order",
                "source_label": label,
                "source_dataset": ds_dir.name,
            })
            idx += 1

    for ds in args.open_datasets:
        process(ds, "open")

    for ds in args.blocked_datasets:
        process(ds, "blocked")

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "metadata.csv", index=False)

    print("DONE:", out_dir)
    print("rows:", len(out))
    print(out["source_label"].value_counts().to_string())
    print(out["source_dataset"].value_counts().to_string())
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
