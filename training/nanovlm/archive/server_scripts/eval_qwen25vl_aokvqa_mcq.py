import argparse
import json
import os
import random
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
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info


LETTERS = ["A", "B", "C", "D"]


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = int(round((len(xs) - 1) * p / 100))
    return xs[k]


def load_aokvqa_dataset(dataset_name, split):
    print(f"Loading dataset: {dataset_name} split={split}")
    try:
        return load_dataset(dataset_name, split=split)
    except Exception as e1:
        print(f"Failed with split={split}: {type(e1).__name__}: {e1}")

        for alt_split in ["validation", "val", "test"]:
            if alt_split == split:
                continue
            try:
                print(f"Trying split={alt_split}")
                return load_dataset(dataset_name, split=alt_split)
            except Exception as e2:
                print(f"Failed split={alt_split}: {type(e2).__name__}: {e2}")

        raise RuntimeError(
            "Could not load A-OKVQA dataset. "
            "Pass correct --dataset-name and --split."
        )


def get_image(ex):
    img = ex.get("image", None)

    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if isinstance(img, str):
        return Image.open(img).convert("RGB")

    if isinstance(img, dict):
        # HF image feature sometimes returns {"path": ..., "bytes": ...}
        if "path" in img and img["path"]:
            return Image.open(img["path"]).convert("RGB")

    raise ValueError(f"Cannot read image field. Keys: {list(ex.keys())}")


def get_question(ex):
    for k in ["question", "Question", "query"]:
        if k in ex:
            return str(ex[k])
    raise ValueError(f"No question field. Keys: {list(ex.keys())}")


def get_choices(ex):
    for k in ["choices", "Choices", "answer_choices", "multiple_choice_answer"]:
        if k in ex:
            choices = ex[k]
            if isinstance(choices, str):
                # Could be stringified list
                try:
                    import ast
                    choices = ast.literal_eval(choices)
                except Exception:
                    choices = [x.strip() for x in choices.split("|")]
            choices = list(choices)
            if len(choices) >= 4:
                return [str(x) for x in choices[:4]]

    # Some datasets use direct columns
    direct = []
    for k in ["A", "B", "C", "D"]:
        if k in ex:
            direct.append(str(ex[k]))
    if len(direct) == 4:
        return direct

    raise ValueError(f"No choices field. Keys: {list(ex.keys())}")


def get_gt_idx(ex, choices):
    for k in ["correct_choice_idx", "gt_idx", "label", "answer_idx", "correct_idx"]:
        if k in ex:
            return int(ex[k])

    # If answer text exists, map it to choices
    for k in ["answer", "correct_answer", "direct_answer"]:
        if k in ex:
            ans = str(ex[k]).strip().lower()
            for i, ch in enumerate(choices):
                if str(ch).strip().lower() == ans:
                    return i

    raise ValueError(f"No gt index field. Keys: {list(ex.keys())}")


def normalize_example(ex):
    image = get_image(ex)
    question = get_question(ex)
    choices = get_choices(ex)
    gt_idx = get_gt_idx(ex, choices)

    if gt_idx < 0 or gt_idx >= 4:
        raise ValueError(f"Bad gt_idx={gt_idx}")

    return image, question, choices, gt_idx


def build_prompt(question, choices):
    return f"""You are a visual question answering model.

Look at the image carefully and answer the multiple-choice question.

Question: {question}

A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Return only one uppercase letter: A, B, C, or D.
Answer:"""


def extract_letter(text, choices=None):
    raw = str(text).strip()
    upper = raw.upper()

    # Best case: just "A"
    m = re.search(r"\b([ABCD])\b", upper)
    if m:
        return m.group(1)

    # Fallback: sometimes model outputs choice text
    if choices is not None:
        low = raw.lower()
        # longer first to avoid partial collisions
        pairs = sorted(list(enumerate(choices)), key=lambda x: len(str(x[1])), reverse=True)
        for i, ch in pairs:
            ch_low = str(ch).strip().lower()
            if ch_low and ch_low in low:
                return LETTERS[i]

    return None


@torch.no_grad()
def qwen_predict(model, processor, image, question, choices, device, max_new_tokens=16):
    prompt = build_prompt(question, choices)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
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


def make_black_image(img):
    return Image.new("RGB", img.size, (0, 0, 0))


def plot_confusion(cm, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
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


def plot_distribution(gt_counts, pred_counts, out_path, title):
    x = np.arange(4)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, gt_counts, width, label="GT")
    ax.bar(x + width / 2, pred_counts, width, label="Pred")

    ax.set_title(title)
    ax.set_xlabel("Letter")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(LETTERS)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_latency(latencies, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(latencies, bins=30)
    ax.set_title(title)
    ax.set_xlabel("Latency, sec")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluate(
    model,
    processor,
    ds,
    indices,
    device,
    out_dir,
    run_name,
    max_new_tokens=16,
    mode="normal",
):
    rows = []
    cm = np.zeros((4, 4), dtype=int)
    gt_counts = np.zeros(4, dtype=int)
    pred_counts = np.zeros(4, dtype=int)
    latencies = []

    correct = 0
    total = 0
    invalid = 0

    # For wrong-image eval: fixed shifted index map
    wrong_map = {}
    if mode == "wrong":
        shuffled = indices[:]
        random.shuffle(shuffled)
        for a, b in zip(indices, shuffled):
            if a == b:
                b = indices[(indices.index(a) + 1) % len(indices)]
            wrong_map[a] = b

    for n, idx in enumerate(tqdm(indices, desc=f"{run_name}")):
        try:
            ex = ds[int(idx)]
            image, question, choices, gt_idx = normalize_example(ex)

            if mode == "black":
                image = make_black_image(image)

            elif mode == "wrong":
                wrong_ex = ds[int(wrong_map[idx])]
                image, _, _, _ = normalize_example(wrong_ex)

            elif mode == "shuffled":
                correct_choice = choices[gt_idx]
                new_choices = choices[:]
                random.shuffle(new_choices)
                choices = new_choices
                gt_idx = choices.index(correct_choice)

            pred_idx, pred_letter, raw_out, dt = qwen_predict(
                model=model,
                processor=processor,
                image=image,
                question=question,
                choices=choices,
                device=device,
                max_new_tokens=max_new_tokens,
            )

            ok = int(pred_idx == gt_idx)
            correct += ok
            total += 1

            gt_counts[gt_idx] += 1

            if 0 <= pred_idx < 4:
                pred_counts[pred_idx] += 1
                cm[gt_idx, pred_idx] += 1
            else:
                invalid += 1

            latencies.append(dt)

            rows.append(
                {
                    "n": n,
                    "idx": int(idx),
                    "question": question,
                    "A": choices[0],
                    "B": choices[1],
                    "C": choices[2],
                    "D": choices[3],
                    "gt_idx": gt_idx,
                    "gt_letter": LETTERS[gt_idx],
                    "gt_answer": choices[gt_idx],
                    "pred_idx": pred_idx,
                    "pred_letter": pred_letter,
                    "pred_answer": choices[pred_idx] if 0 <= pred_idx < 4 else "",
                    "correct": ok,
                    "raw_output": raw_out,
                    "latency_sec": dt,
                    "mode": mode,
                }
            )

        except Exception as e:
            invalid += 1
            rows.append(
                {
                    "n": n,
                    "idx": int(idx),
                    "error": f"{type(e).__name__}: {e}",
                    "mode": mode,
                }
            )

    acc = 100.0 * correct / total if total else 0.0

    result_dir = out_dir / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(result_dir / "results.csv", index=False)

    summary = {
        "run_name": run_name,
        "mode": mode,
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "accuracy_percent": acc,
        "gt_counts": {LETTERS[i]: int(gt_counts[i]) for i in range(4)},
        "pred_counts": {LETTERS[i]: int(pred_counts[i]) for i in range(4)},
        "confusion_matrix_rows_gt_cols_pred": cm.tolist(),
        "latency_avg_sec": mean(latencies) if latencies else None,
        "latency_p50_sec": percentile(latencies, 50),
        "latency_p95_sec": percentile(latencies, 95),
        "latency_min_sec": min(latencies) if latencies else None,
        "latency_max_sec": max(latencies) if latencies else None,
    }

    with open(result_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_confusion(
        cm,
        result_dir / "confusion_matrix.png",
        f"{run_name} confusion matrix | acc={acc:.2f}%",
    )

    plot_distribution(
        gt_counts,
        pred_counts,
        result_dir / "answer_distribution.png",
        f"{run_name} answer distribution",
    )

    if latencies:
        plot_latency(
            latencies,
            result_dir / "latency_hist.png",
            f"{run_name} latency histogram",
        )

    print("\n" + "=" * 80)
    print(f"{run_name}")
    print(f"mode: {mode}")
    print(f"total: {total}")
    print(f"correct: {correct}")
    print(f"invalid: {invalid}")
    print(f"ACC: {acc:.2f}%")
    print("GT counts:", summary["gt_counts"])
    print("Pred counts:", summary["pred_counts"])
    print("Latency avg:", summary["latency_avg_sec"])
    print("Latency p50:", summary["latency_p50_sec"])
    print("Latency p95:", summary["latency_p95_sec"])
    print("Saved to:", result_dir)

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--dataset-name", type=str, default="HuggingFaceM4/A-OKVQA")
    parser.add_argument("--split", type=str, default="validation")

    parser.add_argument("--out-dir", type=str, default="qwen_aokvqa_eval")
    parser.add_argument("--max-examples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=16)

    parser.add_argument("--control-evals", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_aokvqa_dataset(args.dataset_name, args.split)

    n = len(ds)
    indices = list(range(n))
    random.shuffle(indices)
    indices = indices[: min(args.max_examples, n)]

    print("Dataset size:", n)
    print("Eval examples:", len(indices))

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
    print("Model first device:", device)

    all_summaries = {}

    all_summaries["normal"] = evaluate(
        model=model,
        processor=processor,
        ds=ds,
        indices=indices,
        device=device,
        out_dir=out_dir,
        run_name="normal",
        max_new_tokens=args.max_new_tokens,
        mode="normal",
    )

    if args.control_evals:
        all_summaries["black"] = evaluate(
            model=model,
            processor=processor,
            ds=ds,
            indices=indices,
            device=device,
            out_dir=out_dir,
            run_name="black_image",
            max_new_tokens=args.max_new_tokens,
            mode="black",
        )

        all_summaries["wrong"] = evaluate(
            model=model,
            processor=processor,
            ds=ds,
            indices=indices,
            device=device,
            out_dir=out_dir,
            run_name="wrong_image",
            max_new_tokens=args.max_new_tokens,
            mode="wrong",
        )

        all_summaries["shuffled"] = evaluate(
            model=model,
            processor=processor,
            ds=ds,
            indices=indices,
            device=device,
            out_dir=out_dir,
            run_name="shuffled_choices",
            max_new_tokens=args.max_new_tokens,
            mode="shuffled",
        )

    with open(out_dir / "all_summaries.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print("\nDONE")
    print("All results saved to:", out_dir)


if __name__ == "__main__":
    main()
