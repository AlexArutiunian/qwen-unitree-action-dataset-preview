from pathlib import Path
import pandas as pd
import shutil

SRC = Path("deepdoors2_door_state_mcq/test")
OUT = Path("deepdoors2_door_closed_simple_20_qwen")

QUESTION = "Is the door closed?"
CHOICES = ["Yes", "No", "There is no any door", "I see black image"]

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "images").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC / "metadata.csv")
df = df[df["door_state"].isin(["open", "closed"])].copy()

closed = df[df["door_state"] == "closed"].sample(n=10, random_state=42)
open_ = df[df["door_state"] == "open"].sample(n=10, random_state=42)

df20 = pd.concat([closed, open_]).sample(frac=1, random_state=42).reset_index(drop=True)

rows = []
for i, r in df20.iterrows():
    src_img = SRC / str(r["image"])
    if not src_img.exists():
        print("SKIP missing:", src_img)
        continue

    ext = src_img.suffix or ".jpg"
    dst_rel = f"images/deepdoors2_{i:05d}{ext}"
    shutil.copy2(src_img, OUT / dst_rel)

    state = str(r["door_state"]).strip().lower()

    rr = r.copy()
    rr["id"] = f"deepdoors2_20_{i:05d}"
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

    rr["task"] = "deepdoors2_door_closed_simple_20"
    rows.append(rr)

out = pd.DataFrame(rows)
out.to_csv(OUT / "metadata.csv", index=False)

print("DONE:", OUT)
print("rows:", len(out))
print("\ndoor_state counts:")
print(out["door_state"].value_counts().to_string())
print("\npreview:")
print(out[["image", "door_state", "answer", "gt_idx"]].to_string(index=True))
