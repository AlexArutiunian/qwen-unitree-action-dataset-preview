from pathlib import Path
import pandas as pd
import shutil

SRC = Path("robotpct_new")
OUT = Path("robot_new_6_mcq_v5")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

items = [
    {
        "question": "Which option best describes what the robot is doing?",
        "choices": [
            "The robot is leaning backward and looking at the board.",
            "The robot is crouching and touching the floor with its hands.",
            "The robot is standing straight in front of the wall.",
            "The robot is showing a surprised reaction.",
        ],
        "answer": "The robot is leaning backward and looking at the board.",
    },
    {
        "question": "What is the robot doing?",
        "choices": [
            "The robot is standing straight.",
            "The robot is leaning backward.",
            "The robot is preparing to press a button.",
            "The robot is showing a surprised reaction.",
        ],
        "answer": "The robot is standing straight.",
    },
    {
        "question": "What is the robot doing?",
        "choices": [
            "The robot is waving hello with its raised hand.",
            "The robot is standing with both arms spread out to the sides.",
            "The robot is leaning backward.",
            "The robot is deliberately covering the camera lens with its hand.",
        ],
        "answer": "The robot is waving hello with its raised hand.",
    },
    {
        "question": "What is the robot doing?",
        "choices": [
            "The robot is reaching toward the door handle to open the door.",
            "The robot is waving hello.",
            "The robot is simply extending its right arm sideways without interacting with the door.",
            "The robot is covering the camera with its hand.",
        ],
        "answer": "The robot is reaching toward the door handle to open the door.",
    },
    {
        "question": "Is the door open for the robot?",
        "choices": [
            "Yes, the doorway is open and the robot can go through.",
            "No, the door is fully closed and blocks the robot.",
            "The robot has already passed through the door.",
            "The robot is only starting to open the door.",
        ],
        "answer": "Yes, the doorway is open and the robot can go through.",
    },
    {
        "question": "What pose is the robot in?",
        "choices": [
            "The robot is waving with its left hand.",
            "The robot is standing in a T-pose with both arms spread horizontally.",
            "The robot has raised only its left arm straight upward.",
            "The robot is opening the door.",
        ],
        "answer": "The robot is waving with its left hand.",
    },
]

if OUT.exists():
    shutil.rmtree(OUT)

(OUT / "images").mkdir(parents=True, exist_ok=True)

imgs = [p for p in sorted(SRC.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]

print("found images:", len(imgs))
for i, p in enumerate(imgs):
    print(i + 1, p)

if len(imgs) < len(items):
    raise RuntimeError(f"Need at least {len(items)} images, found {len(imgs)}")

rows = []
for i, item in enumerate(items):
    src = imgs[i]
    ext = src.suffix.lower()
    dst_rel = f"images/robot_new_{i+1:02d}{ext}"
    shutil.copy2(src, OUT / dst_rel)

    choices = item["choices"]
    answer = item["answer"]

    rows.append({
        "id": f"robot_new_{i+1:02d}",
        "image": dst_rel,
        "question": item["question"],
        "choices": str(choices),
        "gt_idx": choices.index(answer),
        "answer": answer,
        "task": "robot_action_pose_mcq_v5_clear_choices",
        "source_path": str(src),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "metadata.csv", index=False)

print("\nDONE:", OUT)
print(df[["id", "image", "question", "answer", "gt_idx", "source_path"]].to_string(index=False, max_colwidth=220))
