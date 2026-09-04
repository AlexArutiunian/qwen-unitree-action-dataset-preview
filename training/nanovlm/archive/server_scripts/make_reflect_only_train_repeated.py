from pathlib import Path
import pandas as pd
import shutil
import random

SRC = Path("reflect_only_yesno4_split/train")
OUT = Path("reflect_only_yesno4_train_repeat")

REPEAT = 10
CHOICES_BASE = ["Yes", "No", "Not applicable", "Invalid image"]

if OUT.exists():
    shutil.rmtree(OUT)

shutil.copytree(SRC / "images", OUT / "images")

df = pd.read_csv(SRC / "metadata.csv")
rows = []

for rep in range(REPEAT):
    rng = random.Random(1000 + rep)

    for i, r in df.iterrows():
        rr = r.copy()
        answer = str(rr["answer"]).strip()

        choices = CHOICES_BASE[:]
        rng.shuffle(choices)

        rr["id"] = f"{rr['id']}_rep{rep}"
        rr["choices"] = str(choices)
        rr["gt_idx"] = choices.index(answer)
        rr["task"] = "reflect_only_yesno4_repeat"
        rows.append(rr)

out = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
out.to_csv(OUT / "metadata.csv", index=False)

print("DONE:", OUT)
print("rows:", len(out))
print("door_state:")
print(out["door_state"].value_counts().to_string())
print("answer:")
print(out["answer"].value_counts().to_string())
print("gt_idx:")
print(out["gt_idx"].value_counts().sort_index().to_string())
