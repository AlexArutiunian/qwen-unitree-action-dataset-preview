#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import re
import shutil
from pathlib import Path
from collections import Counter

import pandas as pd


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

CLASS_ALIASES = {
    "closed": "closed",
    "close": "closed",
    "open": "open",
    "semi-open": "partially_open",
    "semi_open": "partially_open",
    "semiopen": "partially_open",
    "semi": "partially_open",
    "partially_open": "partially_open",
    "partial": "partially_open",
}

STATE_TO_ANSWER = {
    "closed": "Closed.",
    "open": "Open.",
    "partially_open": "Partially open.",
}

# Все 4 варианта точно совпадают с твоим door_eval third question
STATE_CHOICES = ["Open.", "Closed.", "Partially open.", "Cannot tell."]

# Для passability делаем простую разметку:
# open -> yes, closed -> no, partially_open -> cannot tell
PASS_CHOICES = [
    "Yes, the path is open.",
    "No, the path is blocked or closed.",
    "Cannot tell from the image.",
    "There is no doorway visible.",
]

# Для direct open:
OPEN_CHOICES = [
    "Yes, the door is open.",
    "No, the door is closed.",
    "Cannot tell from the image.",
    "The door is not visible.",
]


def norm_name(s: str) -> str:
    s = s.lower().strip()
    s = s.replace(" ", "_")
    s = s.replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def infer_state_from_path(p: Path):
    parts = [norm_name(x) for x in p.parts]
    for part in reversed(parts):
        for k, v in CLASS_ALIASES.items():
            if norm_name(k) == part:
                return v
    return None


def find_rgb_original_root(root: Path):
    # Ищем папку, внутри которой есть Train/Test/Val и классы
    candidates = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        dn = norm_name(d.name)
        path_s = norm_name(str(d))
        if "rgb" in dn or "/rgb" in path_s:
            # приоритет OriginalSize/RGB
            score = 0
            if "original" in path_s:
                score += 10
            if "cropped" in path_s:
                score -= 10
            if "door_classification" in path_s or "classification" in path_s:
                score += 5
            # есть ли картинки ниже
            n_imgs = 0
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    n_imgs += 1
                    if n_imgs > 20:
                        break
            if n_imgs:
                candidates.append((score, n_imgs, d))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    print("RGB root candidates:")
    for score, n, d in candidates[:10]:
        print(" ", "score", score, "sample_imgs", n, d)

    return candidates[0][2]


def split_from_path(p: Path):
    parts = [norm_name(x) for x in p.parts]
    for part in parts:
        if part in {"train", "training"}:
            return "train"
        if part in {"val", "valid", "validation"}:
            return "val"
        if part in {"test", "testing"}:
            return "test"
    return "unknown"


def make_choices(base, answer, seed):
    choices = list(base)
    rng = random.Random(seed)
    rng.shuffle(choices)
    return choices, choices.index(answer)


def build_rows_for_image(rel_img, state, img_id):
    rows = []

    # 1. Прямой вопрос: Is the door open?
    if state == "open":
        ans = "Yes, the door is open."
    elif state == "closed":
        ans = "No, the door is closed."
    else:
        ans = "Cannot tell from the image."

    q = "Is the door open?"
    choices, gt_idx = make_choices(OPEN_CHOICES, ans, img_id + "_open")
    rows.append({
        "id": img_id + "_q_open",
        "image": rel_img,
        "question": q,
        "choices": str(choices),
        "gt_idx": gt_idx,
        "answer": ans,
        "task": "door_open_direct",
        "door_state": state,
    })

    # 2. State question: What is the state of the door?
    ans = STATE_TO_ANSWER[state]
    q = "What is the state of the door?"
    choices, gt_idx = make_choices(STATE_CHOICES, ans, img_id + "_state")
    rows.append({
        "id": img_id + "_q_state",
        "image": rel_img,
        "question": q,
        "choices": str(choices),
        "gt_idx": gt_idx,
        "answer": ans,
        "task": "door_state_mcq",
        "door_state": state,
    })

    # 3. Passability question
    if state == "open":
        ans = "Yes, the path is open."
    elif state == "closed":
        ans = "No, the path is blocked or closed."
    else:
        ans = "Cannot tell from the image."

    q = "Can the robot pass through this doorway?"
    choices, gt_idx = make_choices(PASS_CHOICES, ans, img_id + "_pass")
    rows.append({
        "id": img_id + "_q_pass",
        "image": rel_img,
        "question": q,
        "choices": str(choices),
        "gt_idx": gt_idx,
        "answer": ans,
        "task": "door_passability",
        "door_state": state,
    })

    # 4. Доп. формулировка для open/closed
    q = "Is the doorway open or closed?"
    ans = STATE_TO_ANSWER[state]
    choices, gt_idx = make_choices(STATE_CHOICES, ans, img_id + "_doorway_state")
    rows.append({
        "id": img_id + "_q_doorway_state",
        "image": rel_img,
        "question": q,
        "choices": str(choices),
        "gt_idx": gt_idx,
        "answer": ans,
        "task": "door_state_mcq",
        "door_state": state,
    })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--out-root", default="deepdoors2_door_state_mcq")
    ap.add_argument("--rgb-root", default=None)
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.rgb_root:
        rgb_root = Path(args.rgb_root)
    else:
        rgb_root = find_rgb_original_root(raw_root)

    if rgb_root is None:
        raise RuntimeError("Could not find RGB root. Pass --rgb-root manually.")

    print("Using RGB root:", rgb_root)

    imgs = [p for p in rgb_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    print("images found:", len(imgs))

    by_split = {"train": [], "val": [], "test": []}
    skipped = Counter()

    for p in imgs:
        state = infer_state_from_path(p)
        split = split_from_path(p)

        if state is None:
            skipped["unknown_state"] += 1
            continue
        if split not in by_split:
            skipped["unknown_split"] += 1
            continue

        by_split[split].append((p, state))

    for split, items in by_split.items():
        out_dir = out_root / split
        img_dir = out_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        counts = Counter()

        for i, (src, state) in enumerate(sorted(items, key=lambda x: str(x[0]))):
            img_id = f"deepdoors2_{split}_{i:06d}_{state}"
            rel_img = f"images/{img_id}{src.suffix.lower()}"
            dst = out_dir / rel_img
            shutil.copy2(src, dst)

            rows.extend(build_rows_for_image(rel_img, state, img_id))
            counts[state] += 1

        df = pd.DataFrame(rows)
        if len(df):
            df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        df.to_csv(out_dir / "metadata.csv", index=False)
        with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
            for r in df.to_dict(orient="records"):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print("\n" + "=" * 100)
        print(split)
        print("images:", len(items))
        print("rows:", len(df))
        print("image state counts:")
        print(pd.Series(counts).to_string())
        if len(df):
            print("\ntask counts:")
            print(df["task"].value_counts().to_string())
            print("\nanswer counts:")
            print(df["answer"].value_counts().to_string())
            print("\npreview:")
            print(df[["image", "question", "choices", "answer", "task", "door_state"]].head(10).to_string(index=False))

    print("\nskipped:", dict(skipped))
    print("DONE:", out_root)


if __name__ == "__main__":
    main()
