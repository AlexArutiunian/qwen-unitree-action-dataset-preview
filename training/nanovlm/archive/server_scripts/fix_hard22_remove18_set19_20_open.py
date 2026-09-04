from pathlib import Path
import pandas as pd
import shutil
import ast

SRC = Path("real_glass_hard22_yesno4_eval")
OUT = Path("real_glass_hard22_yesno4_eval_fix_remove18_open19_20")

if OUT.exists():
    shutil.rmtree(OUT)

shutil.copytree(SRC, OUT)

meta = OUT / "metadata.csv"
df = pd.read_csv(meta)

print("BEFORE rows:", len(df))
print(df[["image", "answer", "gt_idx"]].to_string(index=False, max_colwidth=160))

# 1) удалить sample_00018
remove_names = {"sample_00018.jpg"}
df = df[~df["image"].apply(lambda x: Path(str(x)).name in remove_names)].copy()

# можно удалить сам файл, чтобы не путался
for name in remove_names:
    p = OUT / "images" / name
    if p.exists():
        p.unlink()
        print("deleted image:", p)

# 2) sample_00019 и sample_00020 сделать OPEN
# В вопросе "Is the door closed?" open => answer = "No"
set_open_names = {"sample_00019.jpg", "sample_00020.jpg"}

for idx, row in df.iterrows():
    name = Path(str(row["image"])).name
    if name in set_open_names:
        choices = ast.literal_eval(row["choices"])
        if "No" not in choices:
            raise RuntimeError(f"No option not found in choices for {name}: {choices}")

        df.loc[idx, "answer"] = "No"
        df.loc[idx, "gt_idx"] = choices.index("No")

        if "door_state" in df.columns:
            df.loc[idx, "door_state"] = "open"
        if "manual_label" in df.columns:
            df.loc[idx, "manual_label"] = "open"
        if "dataset_gt_answer" in df.columns:
            df.loc[idx, "dataset_gt_answer"] = "No"

df.to_csv(meta, index=False)

print("\nAFTER rows:", len(df))
print(df[["image", "answer", "gt_idx"]].to_string(index=False, max_colwidth=160))

print("\nChanged/check:")
print(df[df["image"].apply(lambda x: Path(str(x)).name in set_open_names)][
    [c for c in ["image", "door_state", "manual_label", "answer", "gt_idx", "choices"] if c in df.columns]
].to_string(index=False, max_colwidth=240))

assert len(df) == 21, f"Expected 21 rows after deleting one, got {len(df)}"
for name in set_open_names:
    row = df[df["image"].apply(lambda x: Path(str(x)).name == name)]
    assert len(row) == 1, name
    assert row.iloc[0]["answer"] == "No", row.iloc[0].to_dict()

print("\nOK:", OUT)
