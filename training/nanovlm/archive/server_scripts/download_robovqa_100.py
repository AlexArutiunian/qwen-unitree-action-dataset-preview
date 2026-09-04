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


def parse_discrete_tasks(text):
    """
    Extract discrete yes/no tasks from RoboVQA text.
    """
    results = []

    pattern = re.compile(
        r"<task:([^>]+)>(.*?)(?=<task:|$)",
        flags=re.DOTALL,
    )

    for m in pattern.finditer(text):
        task_header = m.group(1).strip()
        body = m.group(2).strip()

        if "discrete" not in task_header:
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
            instr = ""
            q = before_pred

        q_clean = q.replace("Q:", "").strip()

        if "possible right now" in q_clean.lower():
            question = f"Is this action possible right now: {instr}?"
            task_type = "affordance_possible"
        elif "satisfied" in q_clean.lower():
            question = f"Has this instruction been satisfied: {instr}?"
            task_type = "success_satisfied"
        else:
            question = q_clean
            task_type = "discrete_binary"

        if not instr and len(question) < 5:
            continue

        results.append(
            {
                "task_header": task_header,
                "task_type": task_type,
                "instruction": instr,
                "question": question,
                "answer": answer,
            }
        )

    return results


def make_mcq_choices(answer):
    choices = ["yes", "no", "cannot tell", "not visible"]
    random.shuffle(choices)
    gt_idx = choices.index(answer)
    return choices, gt_idx


def extract_frame_by_ratio(video_path, out_path, ratio=0.5):
    """
    Extract frame by video ratio:
      affordance -> early frame, e.g. 0.15
      success    -> late frame, e.g. 0.90
    """
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


def choose_frame_ratio(task_type):
    # For affordance, use early frame: action should be possible "right now".
    if task_type == "affordance_possible":
        return 0.10

    # For success, use almost-final frame: the action should already be completed.
    if task_type == "success_satisfied":
        return 0.98

    return 0.50


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out-dir", type=str, default="robovqa_100_preview")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--target-examples", type=int, default=100)
    parser.add_argument("--max-rows-scan", type=int, default=50000)
    parser.add_argument("--download-videos", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

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
    skipped_missing_video = 0
    skipped_no_task = 0
    skipped_download = 0
    used = 0

    pbar = tqdm(total=args.target_examples, desc="Collecting RoboVQA MCQ examples")

    for ex in ds:
        scanned += 1

        if scanned > args.max_rows_scan:
            break

        uid = str(ex.get("uid", ""))
        text = ex.get("text", "")
        video_name = ex.get("video", "")

        if not text or not video_name:
            continue

        # HF repo contains only a subset of videos.
        # Skip metadata rows whose mp4 is not actually present.
        if video_name not in available_video_paths:
            skipped_missing_video += 1
            continue

        tasks = parse_discrete_tasks(text)

        if not tasks:
            skipped_no_task += 1
            continue

        for task in tasks:
            answer = task["answer"]

            if answer not in {"yes", "no"}:
                continue

            choices, gt_idx = make_mcq_choices(answer)

            video_rel = f"videos/{video_name}"
            image_rel = ""

            if args.download_videos:
                try:
                    local_video = hf_hub_download(
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        filename=available_video_paths[video_name],
                    )

                    dst_video = video_dir / video_name
                    if not dst_video.exists():
                        shutil.copy2(local_video, dst_video)

                    ratio = choose_frame_ratio(task["task_type"])
                    image_name = f"frame_{used:06d}_{task['task_type']}_{Path(video_name).stem}.jpg"
                    image_path = image_dir / image_name

                    if not image_path.exists():
                        extract_frame_by_ratio(dst_video, image_path, ratio=ratio)

                    image_rel = f"images/{image_name}"

                except Exception as e:
                    skipped_download += 1
                    print(f"\nSkip video {video_name}: {type(e).__name__}: {e}")
                    continue

            rows.append(
                {
                    "id": f"robovqa_{used:06d}",
                    "uid": uid,
                    "video": video_rel,
                    "image": image_rel,
                    "question": task["question"],
                    "choices": str(choices),
                    "gt_idx": gt_idx,
                    "answer": answer,
                    "task": task["task_type"],
                    "instruction": task["instruction"],
                    "task_header": task["task_header"],
                    "raw_video_name": video_name,
                    "hf_video_path": available_video_paths[video_name],
                }
            )

            used += 1
            pbar.update(1)

            if used >= args.target_examples:
                break

        if used >= args.target_examples:
            break

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
    print("skipped_no_task:", skipped_no_task)
    print("skipped_download:", skipped_download)
    print("out_dir:", out_dir)
    print("csv:", csv_path)
    print("jsonl:", jsonl_path)

    if len(df):
        print("\nTask counts:")
        print(df["task"].value_counts())

        print("\nAnswer counts:")
        print(df["answer"].value_counts())

        print("\nPreview:")
        cols = ["image", "video", "question", "choices", "gt_idx", "answer", "task"]
        print(df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
