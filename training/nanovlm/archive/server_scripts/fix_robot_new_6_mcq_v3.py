from pathlib import Path
import pandas as pd
import shutil
import ast

SRC = Path("robot_new_6_mcq_v2")
OUT = Path("robot_new_6_mcq_v3")

if OUT.exists():
    shutil.rmtree(OUT)

shutil.copytree(SRC, OUT)

df = pd.read_csv(OUT / "metadata.csv")

# Исправления по индексам 0-based:
# robot_new_02 -> вариант B: Leaning backward.
# robot_new_05 -> вариант B: No, it is fully closed...
# robot_new_06 -> вариант B: It has spread both arms...
fixes = {
    1: "Leaning backward.",
    4: "No, it is fully closed and the robot cannot pass.",
    5: "It has spread both arms out to the sides.",
}

for idx, new_answer in fixes.items():
    choices = ast.literal_eval(df.loc[idx, "choices"])
    if new_answer not in choices:
        raise RuntimeError(f"Answer not in choices for row {idx}: {new_answer} | {choices}")

    df.loc[idx, "answer"] = new_answer
    df.loc[idx, "gt_idx"] = choices.index(new_answer)

df["task"] = "robot_action_pose_mcq_v3_fixed_gt"
df.to_csv(OUT / "metadata.csv", index=False)

print("DONE:", OUT)
print(df[["id", "image", "question", "choices", "answer", "gt_idx", "source_path"]].to_string(index=False, max_colwidth=240))
