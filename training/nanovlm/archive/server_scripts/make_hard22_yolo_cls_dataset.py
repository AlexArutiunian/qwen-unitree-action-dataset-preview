from pathlib import Path
import pandas as pd
import shutil

SRC = Path("real_glass_hard22_yesno4_eval")
OUT = Path("door_cls_hard22")

if OUT.exists():
    shutil.rmtree(OUT)

for cls in ["closed", "open"]:
    (OUT / "test" / cls).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC / "metadata.csv")

counts = {"open": 0, "closed": 0}

for _, r in df.iterrows():
    state = str(r["door_state"]).lower()
    if state not in ["open", "closed"]:
        continue

    src = SRC / str(r["image"])
    ext = src.suffix or ".jpg"
    dst = OUT / "test" / state / f"hard22_{state}_{counts[state]:03d}{ext}"
    shutil.copy2(src, dst)
    counts[state] += 1

print("DONE:", OUT)
print(counts)
