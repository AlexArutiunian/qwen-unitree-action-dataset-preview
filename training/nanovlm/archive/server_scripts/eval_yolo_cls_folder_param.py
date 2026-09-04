from pathlib import Path
import pandas as pd
from ultralytics import YOLO
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--imgsz", type=int, default=224)
args = ap.parse_args()

MODEL = Path(args.model)
DATA = Path(args.data)
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

model = YOLO(str(MODEL))

rows = []
for gt in ["closed", "open"]:
    d = DATA / gt
    for img in sorted(d.glob("*")):
        if img.suffix.lower() not in IMG_EXTS:
            continue

        res = model.predict(str(img), imgsz=args.imgsz, verbose=False)[0]

        top1_idx = int(res.probs.top1)
        pred = str(res.names[top1_idx])
        conf = float(res.probs.top1conf)

        probs = res.probs.data.detach().cpu().numpy().tolist()
        prob_by_name = {str(res.names[i]): float(p) for i, p in enumerate(probs)}

        rows.append({
            "image": str(img),
            "gt": gt,
            "pred": pred,
            "correct": int(pred == gt),
            "conf": conf,
            "prob_closed": prob_by_name.get("closed", None),
            "prob_open": prob_by_name.get("open", None),
        })

df = pd.DataFrame(rows)
df.to_csv(OUT / "results.csv", index=False)

print("MODEL:", MODEL)
print("DATA:", DATA)
print("rows:", len(df))
print("accuracy:", df["correct"].mean() * 100, int(df["correct"].sum()), "/", len(df))

print("\nBy class:")
print(df.groupby("gt")["correct"].agg(["count", "sum", "mean"]).to_string())

print("\nPred distribution:")
print(df["pred"].value_counts().to_string())

print("\nErrors:")
err = df[df["correct"] == 0]
print(err.to_string(index=False, max_colwidth=180))

print("\nSaved:", OUT / "results.csv")
