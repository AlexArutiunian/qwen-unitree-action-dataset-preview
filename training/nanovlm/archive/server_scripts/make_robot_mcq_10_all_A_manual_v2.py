from pathlib import Path
import pandas as pd
import shutil
import ast
import csv

SRC_DATASET = Path("robot_mcq_10_all_A_manual")
OUT = Path("robot_mcq_10_all_A_manual_v2")
SRC_IMG_DIR = Path("robotpct_final")

if OUT.exists():
    shutil.rmtree(OUT)

shutil.copytree(SRC_DATASET, OUT)

df = pd.read_csv(OUT / "metadata.csv")

# row numbers are 1-based for readability
fixes = {
    4: {
        "question": "What is the robot doing?",
        "choices": [
            "Opening the door.",
            "Waving hello.",
            "Reaching toward a wall without touching the door handle.",
            "Covering the camera with its hand.",
        ],
    },
    8: {
        "question": "What is the robot doing?",
        "choices": [
            "Stomping with one foot.",
            "Falling onto a person.",
            "Standing still with both feet flat on the floor.",
            "Waving its arms.",
        ],
    },
    10: {
        "question": "Which option best describes the image?",
        "choices": [
            "Two robots are visible, and one appears inactive while the other appears active.",
            "Only one robot is visible.",
            "The robots are hanging from a suspension system.",
            "A robot is opening a door.",
        ],
    },
}

for row_num, fix in fixes.items():
    idx = row_num - 1
    choices = fix["choices"]
    df.loc[idx, "question"] = fix["question"]
    df.loc[idx, "choices"] = str(choices)
    df.loc[idx, "answer"] = choices[0]
    df.loc[idx, "gt_idx"] = 0

df["task"] = "robot_mcq_10_all_A_manual_v2"
df.to_csv(OUT / "metadata.csv", index=False)

# also update English CSV qe.csv for consistency
items_en = []
for _, r in df.iterrows():
    choices = ast.literal_eval(r["choices"])
    idx = int(str(r["id"]).split("_")[-1])
    items_en.append([idx, r["question"], choices[0], choices[1], choices[2], choices[3]])

items_en = sorted(items_en, key=lambda x: x[0])

with open(SRC_IMG_DIR / "qe_v2.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["idx", "question", "A", "B", "C", "D"])
    w.writerows(items_en)

print("DONE:", OUT)
print(df[["id", "image", "question", "answer", "gt_idx", "source_path"]].to_string(index=False, max_colwidth=240))
print("\ngt_idx counts:")
print(df["gt_idx"].value_counts().sort_index().to_string())
print("\nsaved:", SRC_IMG_DIR / "qe_v2.csv")
