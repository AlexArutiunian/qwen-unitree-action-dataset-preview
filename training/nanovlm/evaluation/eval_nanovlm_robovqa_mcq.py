#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tqdm import tqdm


LETTERS = ["A", "B", "C", "D"]


def run(cmd, cwd=None):
    print("$", " ".join(map(str, cmd)))
    subprocess.check_call(list(map(str, cmd)), cwd=cwd)


def ensure_repo(repo_dir: Path):
    if repo_dir.exists() and (repo_dir / "models").exists():
        print("Using existing nanoVLM repo:", repo_dir)
        return
    run(["git", "clone", "https://github.com/huggingface/nanoVLM.git", str(repo_dir)])


def download_kaggle_dataset(handle: str, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)

    if any(dst.rglob("model.safetensors")) or any(dst.rglob("*.bin")) or any(dst.rglob("*.pt")):
        print("Kaggle model already downloaded:", dst)
        return

    if shutil.which("kaggle") is None:
        run([sys.executable, "-m", "pip", "install", "-q", "kaggle"])

    print("Downloading Kaggle dataset:", handle)
    run(["kaggle", "datasets", "download", "-d", handle, "-p", str(dst), "--unzip"])


def looks_like_model_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    names = {x.name for x in p.iterdir() if x.is_file()}
    has_cfg = any(n in names for n in ["config.json", "model_config.json", "vlm_config.json"])
    has_weights = any(n.endswith((".safetensors", ".bin", ".pt", ".pth")) for n in names)
    return has_cfg and has_weights


def find_model_dir(root: Path) -> Path:
    root = Path(root)
    candidates = []

    if looks_like_model_dir(root):
        candidates.append(root)

    for p in root.rglob("*"):
        if looks_like_model_dir(p):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"No model dir found under {root}")

    candidates = sorted(
        candidates,
        key=lambda p: (
            len(p.parts),
            -sum(x.stat().st_size for x in p.glob("*") if x.is_file())
        )
    )

    print("Model dir candidates:")
    for c in candidates[:10]:
        print(" ", c)

    return candidates[0]


class MCQHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Linear(hidden_dim, 4)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.out(self.dropout(hidden[:, -1, :]))


def patch_nanovlm(VisionLanguageModel):
    def forward_mcq_logits(self, input_ids: torch.Tensor, images: torch.Tensor, attention_mask=None) -> torch.Tensor:
        token_embd = self.decoder.token_embedding(input_ids)

        images_tensor = self._process_images(images, input_ids.device)
        if images_tensor is not None:
            image_embd = self.vision_encoder(images_tensor)
            image_embd = self.MP(image_embd)
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        hidden, _ = self.decoder(
            token_embd,
            attention_mask=attention_mask,
            kv_cache=None,
            start_pos=0,
        )

        return self.mcq_head(hidden)

    if not getattr(VisionLanguageModel, "_mcq_patched", False):
        orig_init = VisionLanguageModel.__init__

        def patched_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            hidden_dim = self.decoder.token_embedding.embedding_dim
            if not hasattr(self, "mcq_head"):
                self.mcq_head = MCQHead(hidden_dim)

        VisionLanguageModel.__init__ = patched_init
        VisionLanguageModel.forward_mcq_logits = forward_mcq_logits
        VisionLanguageModel._mcq_patched = True


def coerce_cfg_ints(cfg):
    for name in ["max_img_size", "vit_img_size", "mp_image_token_length"]:
        if hasattr(cfg, name):
            try:
                setattr(cfg, name, int(getattr(cfg, name)))
            except Exception:
                pass


def normalize_image(image):
    return ImageOps.exif_transpose(image).convert("RGB")


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


def build_mcq_prompt(question: str, choices):
    return (
        f"Question: {question}\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n"
        "Answer:"
    )


def force_token_ids(ids, tokenizer):
    if isinstance(ids, str):
        ids = tokenizer.encode(ids, add_special_tokens=False)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    if torch.is_tensor(ids):
        ids = ids.detach().cpu().tolist()
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def percentile(xs, p):
    if not xs:
        return None
    return float(np.percentile(xs, p))


def make_black_image(img):
    return Image.new("RGB", img.size, (0, 0, 0))


def plot_confusion(cm, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("nanoVLM prediction")
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


def plot_answer_distribution(gt_counts, pred_counts, out_path):
    x = np.arange(4)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, gt_counts, width, label="Dataset label")
    ax.bar(x + width / 2, pred_counts, width, label="nanoVLM prediction")

    ax.set_title("Answer distribution")
    ax.set_xlabel("Answer letter")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(LETTERS)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_task_accuracy(task_df, out_path):
    if len(task_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(task_df["task"], task_df["accuracy_percent"])
    ax.set_ylabel("Accuracy, %")
    ax.set_xlabel("Task")
    ax.set_title("nanoVLM accuracy by RoboVQA task type")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_latency(latencies, out_path):
    if not latencies:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(latencies, bins=30)
    ax.set_title("nanoVLM latency")
    ax.set_xlabel("Latency, sec")
    ax.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def safe_percent(x, total):
    return 100.0 * x / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--metadata", default="metadata.csv")
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--kaggle-dataset", default="alexandrdixsept/nanovlm-mcq-upd-arch-exp4-5")
    parser.add_argument("--work-dir", default="nanovlm_eval_work")
    parser.add_argument("--repo-dir", default="nanoVLM")

    parser.add_argument("--max-examples", type=int, default=-1)
    parser.add_argument("--mode", default="normal", choices=["normal", "black", "wrong", "shuffled"])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_dir = Path(args.dataset_dir).resolve()
    metadata_path = dataset_dir / args.metadata
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("dataset_dir:", dataset_dir)
    print("metadata:", metadata_path)
    print("out_dir:", out_dir)
    print("mode:", args.mode)

    repo_dir = Path(args.repo_dir).resolve()
    ensure_repo(repo_dir)

    sys.path.insert(0, str(repo_dir))
    from models.vision_language_model import VisionLanguageModel
    from data.processors import get_tokenizer, get_image_processor

    patch_nanovlm(VisionLanguageModel)

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.model_dir:
        model_dir = Path(args.model_dir).resolve()
    else:
        kaggle_root = work_dir / "kaggle_model_dataset"
        download_kaggle_dataset(args.kaggle_dataset, kaggle_root)
        model_dir = find_model_dir(kaggle_root)

    print("Using model_dir:", model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

    model = VisionLanguageModel.from_pretrained(str(model_dir)).to(device)
    coerce_cfg_ints(model.cfg)
    model.eval()

    tokenizer = get_tokenizer(
        model.cfg.lm_tokenizer,
        model.cfg.vlm_extra_tokens,
        model.cfg.lm_chat_template,
    )
    imgproc = get_image_processor(
        int(model.cfg.max_img_size),
        int(model.cfg.vit_img_size),
    )

    df = pd.read_csv(metadata_path)
    df = df[df["image"].notna()]
    df = df[df["image"].astype(str).str.len() > 0].copy()

    if args.max_examples > 0:
        df = df.head(args.max_examples).copy()

    print("examples:", len(df))
    if "task" in df.columns:
        print("Task counts:")
        print(df["task"].value_counts())
    if "answer" in df.columns:
        print("Answer counts:")
        print(df["answer"].value_counts())

    rows = []

    cm = np.zeros((4, 4), dtype=int)
    gt_counts = np.zeros(4, dtype=int)
    pred_counts = np.zeros(4, dtype=int)

    total = 0
    correct = 0
    invalid = 0
    latencies = []

    wrong_images = None
    if args.mode == "wrong":
        image_list = list(df["image"])
        wrong_images = image_list[:]
        random.shuffle(wrong_images)

    @torch.no_grad()
    def predict_one(image_path, question, choices):
        img = Image.open(image_path).convert("RGB")
        img = normalize_image(img)

        if args.mode == "black":
            img = make_black_image(img)

        proc_img, _ = imgproc(img)
        n_imgs = int(proc_img.shape[0])

        image_tokens = tokenizer.image_token * int(model.cfg.mp_image_token_length) * n_imgs
        prompt = build_mcq_prompt(question, choices)
        messages = [{"role": "user", "content": image_tokens + prompt}]

        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        ids = force_token_ids(ids, tokenizer)

        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        images = proc_img.to(device)

        t0 = time.perf_counter()
        logits = model.forward_mcq_logits(input_ids, images).squeeze(0)
        dt = time.perf_counter() - t0

        probs = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))

        return pred_idx, probs, dt

    for n, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"nanoVLM eval {args.mode}")):
        try:
            image_rel = str(row["image"])
            image_path = dataset_dir / image_rel

            if args.mode == "wrong":
                wrong_rel = wrong_images[n]
                image_path = dataset_dir / wrong_rel

            if not image_path.exists():
                raise FileNotFoundError(str(image_path))

            question = str(row["question"])
            choices = parse_choices(row["choices"])
            gt_idx = int(row["gt_idx"])

            if len(choices) != 4:
                raise ValueError(f"choices len != 4: {choices}")

            if args.mode == "shuffled":
                correct_choice = choices[gt_idx]
                random.shuffle(choices)
                gt_idx = choices.index(correct_choice)

            pred_idx, probs, dt = predict_one(
                image_path=image_path,
                question=question,
                choices=choices,
            )

            ok = int(pred_idx == gt_idx)

            total += 1
            correct += ok
            latencies.append(dt)

            gt_counts[gt_idx] += 1
            pred_counts[pred_idx] += 1
            cm[gt_idx, pred_idx] += 1

            rows.append({
                **row.to_dict(),
                "eval_mode": args.mode,
                "dataset_gt_idx": gt_idx,
                "dataset_gt_letter": LETTERS[gt_idx],
                "dataset_gt_answer": choices[gt_idx],
                "nanovlm_pred_idx": pred_idx,
                "nanovlm_pred_letter": LETTERS[pred_idx],
                "nanovlm_pred_answer": choices[pred_idx],
                "nanovlm_correct": ok,
                "prob_A": float(probs[0]),
                "prob_B": float(probs[1]),
                "prob_C": float(probs[2]),
                "prob_D": float(probs[3]),
                "latency_sec": dt,
                "error": "",
            })

        except Exception as e:
            invalid += 1
            rows.append({
                **row.to_dict(),
                "eval_mode": args.mode,
                "dataset_gt_idx": row.get("gt_idx", ""),
                "dataset_gt_letter": "",
                "dataset_gt_answer": "",
                "nanovlm_pred_idx": -1,
                "nanovlm_pred_letter": "",
                "nanovlm_pred_answer": "",
                "nanovlm_correct": 0,
                "prob_A": "",
                "prob_B": "",
                "prob_C": "",
                "prob_D": "",
                "latency_sec": "",
                "error": f"{type(e).__name__}: {e}",
            })

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "nanovlm_results.csv", index=False)

    task_rows = []
    if "task" in res.columns:
        for task, g in res.groupby("task"):
            valid_g = g[g["error"].astype(str).str.len() == 0]
            n = len(valid_g)
            c = int(valid_g["nanovlm_correct"].sum()) if n else 0
            task_rows.append({
                "task": task,
                "total": n,
                "correct": c,
                "accuracy_percent": safe_percent(c, n),
            })

    task_df = pd.DataFrame(task_rows)
    if len(task_df):
        task_df = task_df.sort_values("accuracy_percent", ascending=False)
    task_df.to_csv(out_dir / "task_accuracy.csv", index=False)

    acc = safe_percent(correct, total)

    summary = {
        "dataset_dir": str(dataset_dir),
        "metadata": str(metadata_path),
        "model_dir": str(model_dir),
        "mode": args.mode,
        "total_validated": total,
        "correct": correct,
        "invalid_or_errors": invalid,
        "accuracy_percent": acc,
        "gt_counts": {LETTERS[i]: int(gt_counts[i]) for i in range(4)},
        "nanovlm_pred_counts": {LETTERS[i]: int(pred_counts[i]) for i in range(4)},
        "confusion_matrix_rows_dataset_cols_nanovlm": cm.tolist(),
        "latency_avg_sec": float(mean(latencies)) if latencies else None,
        "latency_p50_sec": percentile(latencies, 50),
        "latency_p95_sec": percentile(latencies, 95),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_confusion(
        cm,
        out_dir / "confusion_matrix_dataset_vs_nanovlm.png",
        f"Dataset vs nanoVLM | {args.mode} | acc={acc:.2f}%",
    )

    plot_answer_distribution(
        gt_counts,
        pred_counts,
        out_dir / "answer_distribution_dataset_vs_nanovlm.png",
    )

    if len(task_df):
        plot_task_accuracy(
            task_df,
            out_dir / "task_accuracy.png",
        )

    plot_latency(
        latencies,
        out_dir / "latency_hist.png",
    )

    print("\n" + "=" * 80)
    print("nanoVLM ROBOVQA EVAL DONE")
    print("mode:", args.mode)
    print("total_validated:", total)
    print("correct:", correct)
    print("invalid_or_errors:", invalid)
    print(f"accuracy_percent: {acc:.2f}%")
    print("saved to:", out_dir)

    if len(task_df):
        print("\nTask accuracy:")
        print(task_df.to_string(index=False))


if __name__ == "__main__":
    main()
