from pathlib import Path
import pandas as pd
import shutil

SRC_IMG = Path("robot_mcq_10_all_A_manual_v2/images/robot_09.png")
OUT = Path("robot_09_hand_mcq")

if OUT.exists():
    shutil.rmtree(OUT)

(OUT / "images").mkdir(parents=True, exist_ok=True)

dst_rel = "images/robot_09.png"
shutil.copy2(SRC_IMG, OUT / dst_rel)

choices = [
    "Right hand.",
    "Left hand.",
    "Both hands.",
    "No hand.",
]

df = pd.DataFrame([{
    "id": "robot_09_hand",
    "image": dst_rel,
    "question": "Which hand is the robot raising?",
    "choices": str(choices),
    "gt_idx": 0,
    "answer": choices[0],
    "task": "robot_raised_hand_mcq",
    "source_path": str(SRC_IMG),
}])

df.to_csv(OUT / "metadata.csv", index=False)

print("DONE:", OUT)
print(df.to_string(index=False, max_colwidth=220))
