#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune already trained nanoVLM MCQ model on local AI2-THOR generated MCQ dataset.

Expected dataset:
  dataset_dir/images/*.jpg
  dataset_dir/metadata.jsonl

Example:
  python train_nanovlm_ai2thor_local.py \
    --dataset-dir ai2thor_openable_dataset \
    --kaggle-dataset alexandrdixsept/nanovlm-mcq-upd-arch-exp4-5 \
    --max-steps 8000 --eval-every 500
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=str, required=True)
    p.add_argument("--metadata", type=str, default=None)
    p.add_argument("--kaggle-dataset", type=str, default="alexandrdixsept/nanovlm-mcq-upd-arch-exp4-5")
    p.add_argument("--model-dir", type=str, default=None, help="Use local model dir instead of downloading Kaggle dataset")
    p.add_argument("--repo-dir", type=str, default="nanoVLM")
    p.add_argument("--work-dir", type=str, default="nanovlm_ai2thor_train_run")
    p.add_argument("--base-model-id", type=str, default="lusxvr/nanoVLM")
    p.add_argument("--max-steps", type=int, default=8000)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--val-max-examples", type=int, default=400)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unfreeze-last-decoder-layers", type=int, default=2)
    p.add_argument("--train-mp", action="store_true", default=True)
    p.add_argument("--no-train-mp", action="store_false", dest="train_mp")
    p.add_argument("--lr-head", type=float, default=8e-5)
    p.add_argument("--lr-mp", type=float, default=2e-5)
    p.add_argument("--lr-decoder", type=float, default=8e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.03)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp-dtype", type=str, default="bf16", choices=["fp16", "bf16", "none"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--limit-examples", type=int, default=None)
    return p.parse_args()


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
    # If already contains model-looking files, do not redownload.
    if any(dst.rglob("model.safetensors")) or any(dst.rglob("*.bin")) or any(dst.rglob("*.pt")):
        print("Kaggle dataset already downloaded:", dst)
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
    # Prefer the shallowest, then largest weights dir.
    candidates = sorted(candidates, key=lambda p: (len(p.parts), -sum(x.stat().st_size for x in p.glob('*') if x.is_file())))
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
        hidden, _ = self.decoder(token_embd, attention_mask=attention_mask, kv_cache=None, start_pos=0)
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


def build_mcq_prompt(question: str, choices: List[str]) -> str:
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


class LocalMCQDataset:
    def __init__(self, items, root: Path):
        self.items = items
        self.root = Path(root)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = dict(self.items[idx])
        img_path = Path(ex["image"])
        if not img_path.is_absolute():
            img_path = self.root / img_path
        ex["image"] = Image.open(img_path).convert("RGB")
        return ex


def get_decoder_layers(model):
    for attr in ["layers", "blocks", "h"]:
        if hasattr(model.decoder, attr):
            layers = getattr(model.decoder, attr)
            if isinstance(layers, (list, nn.ModuleList)):
                return layers
    return None


def fmt_time(sec):
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_dir = Path(args.dataset_dir).resolve()
    metadata_path = Path(args.metadata).resolve() if args.metadata else dataset_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(args.repo_dir).resolve()
    ensure_repo(repo_dir)

    sys.path.insert(0, str(repo_dir))
    from models.vision_language_model import VisionLanguageModel
    from data.processors import get_tokenizer, get_image_processor
    patch_nanovlm(VisionLanguageModel)

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
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer, model.cfg.vlm_extra_tokens, model.cfg.lm_chat_template)
    imgproc = get_image_processor(int(model.cfg.max_img_size), int(model.cfg.vit_img_size))

    # Load metadata
    items = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                if "choices" in ex and len(ex["choices"]) == 4 and "gt_idx" in ex:
                    items.append(ex)
    if args.limit_examples:
        items = items[:args.limit_examples]
    random.shuffle(items)
    split = int(len(items) * (1.0 - args.val_ratio))
    train_items = items[:split]
    val_items = items[split:]
    ds_train = LocalMCQDataset(train_items, dataset_dir)
    ds_val = LocalMCQDataset(val_items, dataset_dir)
    print("items:", len(items), "train:", len(ds_train), "val:", len(ds_val))

    def encode_mcq_example(ex):
        img = normalize_image(ex["image"])
        proc_img, _ = imgproc(img)
        n_imgs = int(proc_img.shape[0])
        image_tokens = tokenizer.image_token * int(model.cfg.mp_image_token_length) * n_imgs
        prompt = build_mcq_prompt(ex["question"], list(ex["choices"]))
        messages = [{"role": "user", "content": image_tokens + prompt}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        ids = force_token_ids(ids, tokenizer)
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        images = proc_img.to(device)
        label = torch.tensor([int(ex["gt_idx"])], dtype=torch.long, device=device)
        return input_ids, images, label

    @torch.no_grad()
    def eval_dataset(ds, max_examples=300, desc="eval"):
        model.eval()
        total = min(max_examples, len(ds))
        correct = 0
        for i in tqdm(range(total), desc=desc):
            input_ids, images, label = encode_mcq_example(ds[i])
            logits = model.forward_mcq_logits(input_ids, images).squeeze(0)
            pred = int(torch.argmax(logits).item())
            correct += int(pred == int(label.item()))
        acc = correct / total * 100 if total else 0.0
        print(f"{desc} ACC: {acc:.2f}% ({correct}/{total})")
        return acc

    # Freeze/unfreeze
    for p in model.parameters():
        p.requires_grad = False
    for p in model.mcq_head.parameters():
        p.requires_grad = True
    if args.train_mp and hasattr(model, "MP"):
        for p in model.MP.parameters():
            p.requires_grad = True
    layers = get_decoder_layers(model)
    if layers is not None and args.unfreeze_last_decoder_layers > 0:
        for block in layers[-args.unfreeze_last_decoder_layers:]:
            for p in block.parameters():
                p.requires_grad = True
        print("Unfrozen decoder layers:", args.unfreeze_last_decoder_layers)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.4f}%)")

    head_params, mp_params, decoder_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "mcq_head" in name:
            head_params.append(p)
        elif name.startswith("MP") or ".MP" in name:
            mp_params.append(p)
        else:
            decoder_params.append(p)
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr_head})
    if mp_params:
        param_groups.append({"params": mp_params, "lr": args.lr_mp})
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": args.lr_decoder})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    use_amp = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and args.amp_dtype == "fp16"))

    save_root = work_dir / "runs"
    best_dir = save_root / "best_model"
    last_dir = save_root / "last_model"
    final_dir = save_root / "final_model"
    log_path = save_root / "train_log.json"
    save_root.mkdir(parents=True, exist_ok=True)

    baseline_acc = eval_dataset(ds_val, max_examples=min(args.val_max_examples, len(ds_val)), desc="val before train")
    best_acc = baseline_acc
    model.eval()
    model.save_pretrained(str(best_dir))
    model.train()

    train_log = [{"step": 0, "val_acc": float(baseline_acc), "event": "baseline"}]
    log_path.write_text(json.dumps(train_log, indent=2), encoding="utf-8")

    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_count = 0
    step_times = []
    eval_times = []
    start_time = time.time()

    pbar = tqdm(range(1, args.max_steps + 1), desc="nanoVLM AI2-THOR fine-tuning", dynamic_ncols=True)
    for step in pbar:
        t0 = time.time()
        ex = ds_train[random.randrange(len(ds_train))]
        input_ids, images, label = encode_mcq_example(ex)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model.forward_mcq_logits(input_ids, images)
            loss = F.cross_entropy(logits, label, label_smoothing=args.label_smoothing)
            loss_back = loss / args.grad_accum

        if use_amp and args.amp_dtype == "fp16":
            scaler.scale(loss_back).backward()
        else:
            loss_back.backward()

        running_loss += float(loss.detach().cpu())
        running_count += 1

        if step % args.grad_accum == 0:
            if use_amp and args.amp_dtype == "fp16":
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            if use_amp and args.amp_dtype == "fp16":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        step_times.append(time.time() - t0)
        avg_step = sum(step_times[-200:]) / min(len(step_times), 200)
        eval_left = max(0, args.max_steps // args.eval_every - step // args.eval_every)
        avg_eval = sum(eval_times) / len(eval_times) if eval_times else 0
        eta = (args.max_steps - step) * avg_step + eval_left * avg_eval
        avg_loss = running_loss / max(1, running_count)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "best": f"{best_acc:.2f}%", "ETA": fmt_time(eta)})

        if step % 50 == 0:
            print(f"step={step}/{args.max_steps} loss={avg_loss:.4f} best={best_acc:.2f}% elapsed={fmt_time(time.time()-start_time)} ETA={fmt_time(eta)}")
            train_log.append({"step": step, "loss": float(avg_loss), "best_acc": float(best_acc), "event": "train"})
            running_loss = 0.0
            running_count = 0
            log_path.write_text(json.dumps(train_log, indent=2), encoding="utf-8")

        if step % args.eval_every == 0:
            print(f"\n=== Eval step {step} ===")
            te = time.time()
            val_acc = eval_dataset(ds_val, max_examples=min(args.val_max_examples, len(ds_val)), desc=f"val step {step}")
            eval_times.append(time.time() - te)
            model.eval()
            model.save_pretrained(str(last_dir))
            if val_acc > best_acc:
                best_acc = val_acc
                model.save_pretrained(str(best_dir))
                print("NEW BEST", best_acc, "saved to", best_dir)
            train_log.append({"step": step, "val_acc": float(val_acc), "best_acc": float(best_acc), "event": "eval"})
            log_path.write_text(json.dumps(train_log, indent=2), encoding="utf-8")
            model.train()

    model.eval()
    model.save_pretrained(str(final_dir))
    print("\nDONE")
    print("best_acc:", best_acc)
    print("best_model:", best_dir)
    print("last_model:", last_dir)
    print("final_model:", final_dir)


if __name__ == "__main__":
    main()
