from pathlib import Path
import pandas as pd
import shutil

OPEN_DATASETS = ["real_robot_door_only_open_mcq"]
BLOCKED_DATASETS = ["real_door_close_noreflect_mcq", "real_glass_closed_door_mcq"]

CHOICES = [
    "Yes, the passage is clear.",
    "No, the passage is blocked.",
    "Cannot tell from the image.",
    "There is no visible passage.",
]

PROMPTS = {
    "v1_direct_robot": (
        "For robot navigation, decide whether the robot can move straight through the passage "
        "without opening or moving anything. Answer clear only if the physical path is unobstructed."
    ),

    "v2_glass_careful": (
        "For robot navigation, decide physical passability. A glass or reflective door may look like an open hallway. "
        "Do not rely only on seeing space behind it. Look for a physical barrier across the opening: a glass panel, "
        "door leaf, vertical edge, handle, frame line, or reflection. Can the robot move straight through without opening or moving anything?"
    ),

    "v3_not_overblocked": (
        "Decide if the robot can pass straight through. A visible room or corridor behind glass does not mean the path is open. "
        "However, do not mark blocked just because there is a doorway or reflections. Mark blocked only when a closed door, "
        "glass panel, mirror-like surface, or obstacle physically crosses the robot's path."
    ),

    "v4_binary_rule": (
        "Classify physical passability for a robot. Choose Yes only when there is an actual open gap large enough to pass through. "
        "Choose No when a closed opaque door, closed glass door, reflective panel, or obstacle blocks the opening. "
        "If the image does not show enough evidence, choose Cannot tell."
    ),

    "v5_gap_based": (
        "Look at the passage opening. Is there a continuous open gap that a robot could pass through now, without touching a door or obstacle? "
        "Glass, transparent doors, mirror-like panels, and closed door leaves count as blocked even if the space behind them is visible."
    ),
}

def copy_rows(src_dir, label, out_dir, rows, idx):
    df = pd.read_csv(Path(src_dir) / "metadata.csv")
    for _, r in df.iterrows():
        src = Path(src_dir) / str(r["image"])
        if not src.exists():
            print("SKIP", src)
            continue

        ext = src.suffix or ".jpg"
        rel = f"images/sample_{idx:05d}{ext}"
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        if label == "open":
            gt_idx = 0
            answer = CHOICES[0]
        else:
            gt_idx = 1
            answer = CHOICES[1]

        rows.append({
            "id": f"sample_{idx:05d}",
            "image": rel,
            "gt_idx": gt_idx,
            "answer": answer,
            "task": "passage_prompt_variant",
            "source_label": label,
            "source_dataset": Path(src_dir).name,
        })
        idx += 1
    return idx

for key, question in PROMPTS.items():
    out_dir = Path(f"real_passage_{key}_mcq")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    rows = []
    idx = 0

    for ds in OPEN_DATASETS:
        idx = copy_rows(ds, "open", out_dir, rows, idx)

    for ds in BLOCKED_DATASETS:
        idx = copy_rows(ds, "blocked", out_dir, rows, idx)

    df = pd.DataFrame(rows)
    df["question"] = question
    df["choices"] = str(CHOICES)

    cols = ["id", "image", "question", "choices", "gt_idx", "answer", "task", "source_label", "source_dataset"]
    df = df[cols]
    df.to_csv(out_dir / "metadata.csv", index=False)

    print("\nDONE", out_dir, "rows", len(df))
    print(question)
