#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import re
import shutil
from pathlib import Path

import cv2
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from tqdm import tqdm


REPO_ID = "Tianli/robovqa"


def parse_affordance_tasks(text):
    """
    Берём только задачи вида:
    <task:affordance:discriminative:discrete:True/False>
    ...
    <PRED:BINARY>yes/no</PRED:BINARY>
    """
    results = []

    pattern = re.compile(
        r"<task:([^>]+)>(.*?)(?=<task:|$)",
        flags=re.DOTALL,
    )

    for m in pattern.finditer(text):
        task_header = m.group(1).strip()
        body = m.group(2).strip()

        header_low = task_header.lower()

        if "affordance" not in header_low:
            continue
        if "discrete" not in header_low:
            continue

        ans_m = re.search(
            r"<PRED:BINARY>\s*(yes|no)\s*</PRED:BINARY>",
            body,
            flags=re.IGNORECASE,
        )
        if not ans_m:
            continue

        answer = ans_m.group(1).lower().strip()

        before_pred = body.split("<PRED>")[0].strip()

        if " Q:" in before_pred:
            instr, q = before_pred.split(" Q:", 1)
            instr = instr.strip()
            q = "Q:" + q.strip()
        else:
            instr = before_pred.strip()
            q = ""

        if not instr:
            continue

        question = f"Is this action possible right now: {instr}?"

        results.append({
            "task_header": task_header,
            "task_type": "affordance_possible",
            "instruction": instr,
            "question": question,
            "answer": answer,
        })

    return results


def make_mcq_choices(answer):
    choices = ["yes", "no", "cannot tell", "not visible"]
    random.shuffle(choices)
    gt_idx = choices.index(answer)
    return choices, gt_idx


def extract_frame_by_ratio(video_path, out_path, ratio=0.10):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if n_frames <= 0:
        frame_idx = 0
    else:
        frame_idx = int(max(0, min(n_frames - 1, n_frames * ratio)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame from: {video_path}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img.save(out_path, quality=95)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out-dir", type=str, default="robovqa_affordance_1000")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--target-examples", type=int, default=1000)
    parser.add_argument("--max-rows-scan", type=int, default=500000)
    parser.add_argument("--frame-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance-yes-no", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    video_dir = out_dir / "videos"
    image_dir = out_dir / "images"

    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print("Listing available RoboVQA video files on HF...")
    repo_files = list_repo_files(REPO_ID, repo_type="dataset")
    available_video_paths = {
        Path(f).name: f
        for f in repo_files
        if f.endswith(".mp4")
    }
    print("available videos:", len(available_video_paths))

    print("Loading RoboVQA streaming metadata...")
    ds = load_dataset(REPO_ID, split=args.split, streaming=True)

    rows = []
    scanned = 0
    used = 0

    skipped_missing_video = 0
    skipped_no_affordance = 0
    skipped_download = 0

    answer_counts = {"yes": 0, "no": 0}

    if args.balance_yes_no:
        target_per_answer = args.target_examples // 2
    else:
        target_per_answer = None

    pbar = tqdm(total=args.target_examples, desc="Building RoboVQA affordance dataset")

    for ex in ds:
        scanned += 1

        if scanned > args.max_rows_scan:
            break

        if used >= args.target_examples:
            break

        uid = str(ex.get("uid", ""))
        text = ex.get("text", "")
        video_name = ex.get("video", "")

        if not text or not video_name:
            continue

        if video_name not in available_video_paths:
            skipped_missing_video += 1
            continue

        tasks = parse_affordance_tasks(text)

        if not tasks:
            skipped_no_affordance += 1
            continue

        for task in tasks:
            if used >= args.target_examples:
                break

            answer = task["answer"]

            if answer not in {"yes", "no"}:
                continue

            if args.balance_yes_no:
                if answer_counts[answer] >= target_per_answer:
                    continue

            try:
                local_video = hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=available_video_paths[video_name],
                )

                dst_video = video_dir / video_name
                if not dst_video.exists():
                    shutil.copy2(local_video, dst_video)

                image_name = f"frame_{used:06d}_affordance_{Path(video_name).stem}.jpg"
                image_path = image_dir / image_name

                if not image_path.exists():
                    extract_frame_by_ratio(
                        dst_video,
                        image_path,
                        ratio=args.frame_ratio,
                    )

            except Exception as e:
                skipped_download += 1
                print(f"\nSkip video {video_name}: {type(e).__name__}: {e}")
                continue

            choices, gt_idx = make_mcq_choices(answer)

            rows.append({
                "id": f"robovqa_affordance_{used:06d}",
                "uid": uid,
                "video": f"videos/{video_name}",
                "image": f"images/{image_name}",
                "question": task["question"],
                "choices": str(choices),
                "gt_idx": gt_idx,
                "answer": answer,
                "task": "affordance_possible",
                "instruction": task["instruction"],
                "task_header": task["task_header"],
                "raw_video_name": video_name,
                "hf_video_path": available_video_paths[video_name],
                "frame_ratio": args.frame_ratio,
            })

            answer_counts[answer] += 1
            used += 1
            pbar.update(1)

    pbar.close()

    df = pd.DataFrame(rows)

    csv_path = out_dir / "metadata.csv"
    jsonl_path = out_dir / "metadata.jsonl"

    df.to_csv(csv_path, index=False)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nDONE")
    print("scanned rows:", scanned)
    print("saved examples:", len(df))
    print("skipped_missing_video:", skipped_missing_video)
    print("skipped_no_affordance:", skipped_no_affordance)
    print("skipped_download:", skipped_download)
    print("out_dir:", out_dir)
    print("csv:", csv_path)
    print("jsonl:", jsonl_path)

    if len(df):
        print("\nAnswer counts:")
        print(df["answer"].value_counts())

        print("\nPreview:")
        print(df[[
            "image", "video", "question", "choices",
            "gt_idx", "answer", "task"
        ]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
