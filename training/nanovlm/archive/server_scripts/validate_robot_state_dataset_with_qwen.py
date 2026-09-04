import argparse
import ast
import json
import re
import time
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info


LETTERS = ["A", "B", "C", "D"]


def parse_choices(x):
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, str):
        x = x.strip()
        try:
            v = ast.literal_eval(x)
            return [str(t) for t in v]
        except Exception:
            pass
        if "|" in x:
            return [t.strip() for t in x.split("|")]
    raise ValueError(f"Cannot parse choices: {x}")


def extract_letter(text, choices=None):
    raw = str(text).strip()
    up = raw.upper()

    # best case: model outputs A/B/C/D
    m = re.search(r"\b([ABCD])\b", up)
    if m:
        return m.group(1)

    # fallback: model outputs answer text
    if choices is not None:
        low = raw.lower()
        pairs = sorted(list(enumerate(choices)), key=lambda x: len(str(x[1])), reverse=True)
        for i, ch in pairs:
            ch_low = str(ch).strip().lower()
            if ch_low and ch_low in low:
                return LETTERS[i]

    # yes/no fallback
    low = raw.lower()
    if re.search(r"\byes\b", low):
        if choices:
            for i, ch in enumerate(choices):
                if str(ch).lower().strip() == "yes":
                    return LETTERS[i]
    if re.search(r"\bno\b", low):
        if choices:
            for i, ch in enumerate(choices):
                if str(ch).lower().strip() == "no":
                    return LETTERS[i]

    return None


def build_prompt(question, choices):
    return f"""You are a careful visual robot-state judge.

You must answer the multiple-choice question using ONLY the image.

Question: {question}

A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Return only one uppercase letter: A, B, C, or D.
Answer:"""


@torch.no_grad()
def qwen_predict(model, processor, image_path, question, choices, device, max_new_tokens=16):
    prompt = build_prompt(question, choices)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(device)

    t0 = time.perf_counter()

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    dt = time.perf_counter() - t0

    generated_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]

    out = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    pred_letter = extract_letter(out, choices)
    pred_idx = LETTERS.index(pred_letter) if pred_letter in LETTERS else -1

    return pred_idx, pred_letter, out, dt


def safe_percent(x, total):
    return 100.0 * x / total if total else 0.0


def plot_confusion(cm, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Qwen prediction")
    ax.set_ylabel("Dataset label")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(LETTERS)
    ax.set_yticklabels(LETTERS)

    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_task_accuracy(task_df, out_path):
    if len(task_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(task_df["task"], task_df["agreement_percent"])
    ax.set_ylabel("Qwen-label agreement, %")
    ax.set_xlabel("Task")
    ax.set_title("Dataset correctness by task type")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_answer_distribution(gt_counts, pred_counts, out_path):
    x = np.arange(4)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, gt_counts, width, label="Dataset label")
    ax.bar(x + width / 2, pred_counts, width, label="Qwen prediction")
    ax.set_title("Answer distribution")
    ax.set_xlabel("Answer letter")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(LETTERS)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--metadata", default="metadata.csv")
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max-examples", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=16)

    parser.add_argument("--accept-threshold", type=float, default=80.0)

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    metadata_path = dataset_dir / args.metadata
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("dataset_dir:", dataset_dir)
    print("metadata:", metadata_path)

    df = pd.read_csv(metadata_path)

    # Нужно, чтобы image был заполнен
    df = df[df["image"].notna()]
    df = df[df["image"].astype(str).str.len() > 0].copy()

    if args.max_examples > 0:
        df = df.head(args.max_examples).copy()

    print("examples to validate:", len(df))

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    print("Loading processor:", args.model)
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    print("Loading model:", args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    device = next(model.parameters()).device
    print("model device:", device)

    rows = []

    cm = np.zeros((4, 4), dtype=int)
    gt_counts = np.zeros(4, dtype=int)
    pred_counts = np.zeros(4, dtype=int)

    total = 0
    agree = 0
    invalid = 0
    latencies = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Qwen validating dataset"):
        try:
            image_rel = str(row["image"])
            image_path = dataset_dir / image_rel

            if not image_path.exists():
                raise FileNotFoundError(str(image_path))

            question = str(row["question"])
            choices = parse_choices(row["choices"])
            gt_idx = int(row["gt_idx"])

            if len(choices) != 4:
                raise ValueError(f"choices len != 4: {choices}")

            pred_idx, pred_letter, raw_out, dt = qwen_predict(
                model=model,
                processor=processor,
                image_path=image_path,
                question=question,
                choices=choices,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )

            ok = int(pred_idx == gt_idx)

            total += 1
            agree += ok
            latencies.append(dt)

            if 0 <= gt_idx < 4:
                gt_counts[gt_idx] += 1

            if 0 <= pred_idx < 4:
                pred_counts[pred_idx] += 1
                cm[gt_idx, pred_idx] += 1
            else:
                invalid += 1

            rows.append({
                **row.to_dict(),
                "dataset_gt_idx": gt_idx,
                "dataset_gt_letter": LETTERS[gt_idx],
                "dataset_gt_answer": choices[gt_idx],
                "qwen_pred_idx": pred_idx,
                "qwen_pred_letter": pred_letter,
                "qwen_pred_answer": choices[pred_idx] if 0 <= pred_idx < 4 else "",
                "qwen_raw_output": raw_out,
                "qwen_agrees_with_dataset": ok,
                "latency_sec": dt,
                "error": "",
            })

        except Exception as e:
            invalid += 1
            rows.append({
                **row.to_dict(),
                "dataset_gt_idx": row.get("gt_idx", ""),
                "dataset_gt_letter": "",
                "dataset_gt_answer": "",
                "qwen_pred_idx": -1,
                "qwen_pred_letter": "",
                "qwen_pred_answer": "",
                "qwen_raw_output": "",
                "qwen_agrees_with_dataset": 0,
                "latency_sec": "",
                "error": f"{type(e).__name__}: {e}",
            })

    res = pd.DataFrame(rows)

    agreement = safe_percent(agree, total)

    res.to_csv(out_dir / "qwen_validation_results.csv", index=False)

    accepted = res[res["qwen_agrees_with_dataset"] == 1].copy()
    rejected = res[res["qwen_agrees_with_dataset"] == 0].copy()

    accepted.to_csv(out_dir / "accepted_by_qwen.csv", index=False)
    rejected.to_csv(out_dir / "rejected_by_qwen.csv", index=False)

    task_rows = []
    if "task" in res.columns:
        for task, g in res.groupby("task"):
            valid_g = g[g["error"].astype(str).str.len() == 0]
            n = len(valid_g)
            c = int(valid_g["qwen_agrees_with_dataset"].sum()) if n else 0
            task_rows.append({
                "task": task,
                "total": n,
                "agree": c,
                "agreement_percent": safe_percent(c, n),
            })

    task_df = pd.DataFrame(task_rows).sort_values("agreement_percent", ascending=False) if task_rows else pd.DataFrame()
    task_df.to_csv(out_dir / "task_accuracy.csv", index=False)

    summary = {
        "dataset_dir": str(dataset_dir),
        "metadata": str(metadata_path),
        "model": args.model,
        "total_validated": total,
        "qwen_agrees": agree,
        "invalid_or_errors": invalid,
        "agreement_percent": agreement,
        "accept_threshold_percent": args.accept_threshold,
        "dataset_passes_threshold": bool(agreement >= args.accept_threshold),
        "gt_counts": {LETTERS[i]: int(gt_counts[i]) for i in range(4)},
        "qwen_pred_counts": {LETTERS[i]: int(pred_counts[i]) for i in range(4)},
        "confusion_matrix_rows_dataset_cols_qwen": cm.tolist(),
        "latency_avg_sec": float(mean(latencies)) if latencies else None,
        "latency_p50_sec": float(np.percentile(latencies, 50)) if latencies else None,
        "latency_p95_sec": float(np.percentile(latencies, 95)) if latencies else None,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_confusion(
        cm,
        out_dir / "confusion_matrix_dataset_vs_qwen.png",
        f"Dataset vs Qwen | agreement={agreement:.2f}%",
    )

    plot_answer_distribution(
        gt_counts,
        pred_counts,
        out_dir / "answer_distribution_dataset_vs_qwen.png",
    )

    if len(task_df):
        plot_task_accuracy(
            task_df,
            out_dir / "task_accuracy.png",
        )

    print("\n" + "=" * 80)
    print("QWEN DATASET VALIDATION DONE")
    print("total_validated:", total)
    print("qwen_agrees:", agree)
    print("invalid_or_errors:", invalid)
    print(f"agreement_percent: {agreement:.2f}%")
    print("threshold:", args.accept_threshold)
    print("dataset_passes_threshold:", agreement >= args.accept_threshold)
    print("saved to:", out_dir)

    if len(task_df):
        print("\nTask accuracy:")
        print(task_df.to_string(index=False))


if __name__ == "__main__":
    main()
