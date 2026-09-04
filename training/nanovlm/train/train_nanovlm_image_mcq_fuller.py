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
import torch.nn.functional as F
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


def parse_choices(x):
    if isinstance(x, list):
        out = [str(v) for v in x]
    elif isinstance(x, str):
        x = x.strip()
        try:
            out = [str(v) for v in ast.literal_eval(x)]
        except Exception:
            if "|" in x:
                out = [t.strip() for t in x.split("|")]
            else:
                raise ValueError(f"Cannot parse choices: {x}")
    else:
        raise ValueError(f"Cannot parse choices type: {type(x)}")

    if len(out) != 4:
        raise ValueError(f"Need exactly 4 choices, got {len(out)}: {out}")
    return out


def read_metadata(dataset_dir: Path, metadata_name: str = None):
    if metadata_name is not None:
        p = dataset_dir / metadata_name
    else:
        if (dataset_dir / "metadata.csv").exists():
            p = dataset_dir / "metadata.csv"
        elif (dataset_dir / "metadata.jsonl").exists():
            p = dataset_dir / "metadata.jsonl"
        else:
            raise FileNotFoundError(f"No metadata.csv or metadata.jsonl in {dataset_dir}")

    print("Reading metadata:", p)

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() == ".jsonl":
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"Unsupported metadata format: {p}")

    required = ["image", "question", "choices", "gt_idx"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}. Columns: {list(df.columns)}")

    df = df[df["image"].notna()].copy()
    df = df[df["image"].astype(str).str.len() > 0].copy()

    clean_rows = []
    skipped = 0

    for _, r in df.iterrows():
        try:
            img_path = dataset_dir / str(r["image"])
            if not img_path.exists():
                skipped += 1
                continue

            choices = parse_choices(r["choices"])
            gt_idx = int(r["gt_idx"])

            if gt_idx < 0 or gt_idx >= 4:
                skipped += 1
                continue

            clean_rows.append({
                **r.to_dict(),
                "choices": choices,
                "gt_idx": gt_idx,
                "abs_image_path": str(img_path),
            })
        except Exception:
            skipped += 1

    out = pd.DataFrame(clean_rows)
    print(f"Loaded valid rows: {len(out)} | skipped: {skipped}")

    if len(out) and "answer" in out.columns:
        print("Answer counts:")
        print(out["answer"].value_counts())

    if len(out) and "task" in out.columns:
        print("Task counts:")
        print(out["task"].value_counts())

    return out


def normalize_image(image):
    return ImageOps.exif_transpose(image).convert("RGB")


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


class ImageMCQDataset:
    def __init__(self, df, tokenizer, imgproc, model_cfg, device):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.imgproc = imgproc
        self.model_cfg = model_cfg
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[int(idx)]

        img = Image.open(r["abs_image_path"]).convert("RGB")
        img = normalize_image(img)

        proc_img, _ = self.imgproc(img)
        n_imgs = int(proc_img.shape[0])

        choices = r["choices"]
        question = str(r["question"])
        gt_idx = int(r["gt_idx"])

        image_tokens = self.tokenizer.image_token * int(self.model_cfg.mp_image_token_length) * n_imgs
        prompt = build_mcq_prompt(question, choices)

        messages = [{"role": "user", "content": image_tokens + prompt}]

        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        ids = force_token_ids(ids, self.tokenizer)

        input_ids = torch.tensor(ids, dtype=torch.long)
        label = torch.tensor(gt_idx, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "images": proc_img,
            "label": label,
        }


def unfreeze_for_mcq(model, unfreeze_last_decoder_layers=2, unfreeze_last_vision_layers=0):
    for p in model.parameters():
        p.requires_grad = False

    # Always train MCQ head
    if hasattr(model, "mcq_head"):
        for p in model.mcq_head.parameters():
            p.requires_grad = True

    # Train multimodal projector
    if hasattr(model, "MP"):
        for p in model.MP.parameters():
            p.requires_grad = True

    # Unfreeze last decoder layers if we can find them
    decoder = getattr(model, "decoder", None)
    layers = None

    if decoder is not None:
        for attr in ["layers", "blocks", "h"]:
            if hasattr(decoder, attr):
                candidate = getattr(decoder, attr)
                try:
                    if len(candidate) > 0:
                        layers = candidate
                        print(f"Found decoder layers: decoder.{attr}, count={len(layers)}")
                        break
                except Exception:
                    pass

    if layers is not None and unfreeze_last_decoder_layers > 0:
        for layer in list(layers)[-unfreeze_last_decoder_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
    else:
        print("WARNING: decoder layers not found or unfreeze_last_decoder_layers=0")


    # Unfreeze last vision encoder layers if requested
    vision = getattr(model, "vision_encoder", None)
    vision_layers = None

    if vision is not None:
        for attr in ["blocks", "layers", "h", "encoder"]:
            if hasattr(vision, attr):
                candidate = getattr(vision, attr)
                # Some encoders store layers under encoder.layers
                if attr == "encoder" and hasattr(candidate, "layers"):
                    candidate = getattr(candidate, "layers")
                try:
                    if len(candidate) > 0:
                        vision_layers = candidate
                        print(f"Found vision layers: vision_encoder.{attr}, count={len(vision_layers)}")
                        break
                except Exception:
                    pass

    if vision_layers is not None and unfreeze_last_vision_layers > 0:
        for layer in list(vision_layers)[-unfreeze_last_vision_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
    elif unfreeze_last_vision_layers > 0:
        print("WARNING: vision layers not found, requested unfreeze_last_vision_layers > 0")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable / total * 100:.4f}%)")

    return trainable, total


def make_optimizer(model, lr_head, lr_mp, lr_decoder, lr_vision, weight_decay):
    head_params = []
    mp_params = []
    decoder_params = []
    vision_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if "mcq_head" in name:
            head_params.append(p)
        elif ".MP." in name or name.startswith("MP."):
            mp_params.append(p)
        elif name.startswith("decoder."):
            decoder_params.append(p)
        elif name.startswith("vision_encoder."):
            vision_params.append(p)
        else:
            other_params.append(p)

    groups = []

    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})
    if mp_params:
        groups.append({"params": mp_params, "lr": lr_mp, "weight_decay": weight_decay})
    if decoder_params:
        groups.append({"params": decoder_params, "lr": lr_decoder, "weight_decay": weight_decay})
    if vision_params:
        groups.append({"params": vision_params, "lr": lr_vision, "weight_decay": weight_decay})
    if other_params:
        groups.append({"params": other_params, "lr": lr_head, "weight_decay": weight_decay})

    print("Optimizer groups:")
    print("  head:", sum(p.numel() for p in head_params))
    print("  MP:", sum(p.numel() for p in mp_params))
    print("  decoder:", sum(p.numel() for p in decoder_params))
    print("  vision:", sum(p.numel() for p in vision_params))
    print("  other:", sum(p.numel() for p in other_params))

    return torch.optim.AdamW(groups)


@torch.no_grad()
def eval_dataset(model, dataset, device, max_examples=-1, desc="val"):
    model.eval()

    n = len(dataset)
    if max_examples > 0:
        n = min(n, max_examples)

    correct = 0
    total = 0
    cm = np.zeros((4, 4), dtype=int)

    for i in tqdm(range(n), desc=desc):
        ex = dataset[i]

        input_ids = ex["input_ids"].unsqueeze(0).to(device)
        images = ex["images"].to(device)
        label = int(ex["label"].item())

        logits = model.forward_mcq_logits(input_ids, images).squeeze(0)
        pred = int(torch.argmax(logits).item())

        correct += int(pred == label)
        total += 1

        if 0 <= label < 4 and 0 <= pred < 4:
            cm[label, pred] += 1

    acc = 100.0 * correct / total if total else 0.0
    print(f"{desc} ACC: {acc:.2f}% ({correct}/{total})")
    print("confusion rows=GT cols=PRED:")
    print(cm)

    model.train()
    return acc


def save_model(model, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(path))
    else:
        torch.save(model.state_dict(), path / "model.pt")
    print("Saved:", path)


def fmt_time(sec):
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dataset-dir", required=True)
    parser.add_argument("--val-dataset-dir", default=None)
    parser.add_argument("--metadata", default=None)

    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--repo-dir", default="nanoVLM")
    parser.add_argument("--work-dir", required=True)

    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--val-max-examples", type=int, default=150)

    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr-head", type=float, default=8e-5)
    parser.add_argument("--lr-mp", type=float, default=2e-5)
    parser.add_argument("--lr-decoder", type=float, default=8e-6)
    parser.add_argument("--lr-vision", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--unfreeze-last-decoder-layers", type=int, default=2)
    parser.add_argument("--unfreeze-last-vision-layers", type=int, default=0)

    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--internal-val-ratio", type=float, default=0.15)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dir = Path(args.train_dataset_dir).resolve()
    val_dir = Path(args.val_dataset_dir).resolve() if args.val_dataset_dir else None
    model_dir = Path(args.model_dir).resolve()
    repo_dir = Path(args.repo_dir).resolve()
    work_dir = Path(args.work_dir).resolve()

    runs_dir = work_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print("train_dir:", train_dir)
    print("val_dir:", val_dir)
    print("model_dir:", model_dir)
    print("work_dir:", work_dir)

    ensure_repo(repo_dir)

    sys.path.insert(0, str(repo_dir))
    from models.vision_language_model import VisionLanguageModel
    from data.processors import get_tokenizer, get_image_processor

    patch_nanovlm(VisionLanguageModel)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

    model = VisionLanguageModel.from_pretrained(str(model_dir)).to(device)
    coerce_cfg_ints(model.cfg)
    model.train()

    tokenizer = get_tokenizer(
        model.cfg.lm_tokenizer,
        model.cfg.vlm_extra_tokens,
        model.cfg.lm_chat_template,
    )
    imgproc = get_image_processor(
        int(model.cfg.max_img_size),
        int(model.cfg.vit_img_size),
    )

    train_df = read_metadata(train_dir, args.metadata)

    if val_dir is not None:
        val_df = read_metadata(val_dir, args.metadata)
    else:
        train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        n_val = max(1, int(len(train_df) * args.internal_val_ratio))
        val_df = train_df.iloc[:n_val].copy()
        train_df = train_df.iloc[n_val:].copy()

    print("Final split:")
    print("  train:", len(train_df))
    print("  val:", len(val_df))

    if len(train_df) == 0:
        raise RuntimeError("Train dataset is empty")
    if len(val_df) == 0:
        raise RuntimeError("Val dataset is empty")

    train_ds = ImageMCQDataset(train_df, tokenizer, imgproc, model.cfg, device)
    val_ds = ImageMCQDataset(val_df, tokenizer, imgproc, model.cfg, device)

    unfreeze_for_mcq(
        model,
        unfreeze_last_decoder_layers=args.unfreeze_last_decoder_layers,
        unfreeze_last_vision_layers=args.unfreeze_last_vision_layers,
    )

    optimizer = make_optimizer(
        model,
        lr_head=args.lr_head,
        lr_mp=args.lr_mp,
        lr_decoder=args.lr_decoder,
        lr_vision=args.lr_vision,
        weight_decay=args.weight_decay,
    )

    if args.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
        use_amp = True
        use_scaler = False
    elif args.amp_dtype == "fp16":
        amp_dtype = torch.float16
        use_amp = True
        use_scaler = True
    else:
        amp_dtype = torch.float32
        use_amp = False
        use_scaler = False

    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    print("amp_dtype:", args.amp_dtype)
    print("grad_accum:", args.grad_accum)

    best_acc = eval_dataset(
        model,
        val_ds,
        device,
        max_examples=args.val_max_examples,
        desc="val before train",
    )

    best_dir = runs_dir / "best_model"
    last_dir = runs_dir / "last_model"
    final_dir = runs_dir / "final_model"

    save_model(model, best_dir)

    optimizer.zero_grad(set_to_none=True)

    start = time.time()
    recent_losses = []

    pbar = tqdm(range(1, args.max_steps + 1), desc="nanoVLM image-MCQ fine-tuning")

    for step in pbar:
        idx = random.randrange(len(train_ds))
        ex = train_ds[idx]

        input_ids = ex["input_ids"].unsqueeze(0).to(device)
        images = ex["images"].to(device)
        label = ex["label"].unsqueeze(0).to(device)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model.forward_mcq_logits(input_ids, images)
                loss = F.cross_entropy(
                    logits.float(),
                    label,
                    label_smoothing=args.label_smoothing,
                )
                loss = loss / args.grad_accum
        else:
            logits = model.forward_mcq_logits(input_ids, images)
            loss = F.cross_entropy(
                logits.float(),
                label,
                label_smoothing=args.label_smoothing,
            )
            loss = loss / args.grad_accum

        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        recent_losses.append(float(loss.item() * args.grad_accum))

        if step % args.grad_accum == 0:
            if use_scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    1.0,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    1.0,
                )
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        elapsed = time.time() - start
        avg_step = elapsed / step
        eta = avg_step * (args.max_steps - step)
        loss_show = mean(recent_losses[-50:]) if recent_losses else 0.0

        pbar.set_postfix({
            "loss": f"{loss_show:.4f}",
            "best": f"{best_acc:.2f}%",
            "ETA": fmt_time(eta),
        })

        if step % 50 == 0:
            print(
                f"step={step}/{args.max_steps} "
                f"loss={loss_show:.4f} "
                f"best={best_acc:.2f}% "
                f"elapsed={fmt_time(elapsed)} "
                f"ETA={fmt_time(eta)}"
            )

        if step % args.eval_every == 0:
            print(f"\n=== Eval step {step} ===")
            acc = eval_dataset(
                model,
                val_ds,
                device,
                max_examples=args.val_max_examples,
                desc=f"val step {step}",
            )

            if acc > best_acc:
                best_acc = acc
                print("New best:", best_acc)
                save_model(model, best_dir)

            save_model(model, last_dir)

    save_model(model, final_dir)

    print("\nDONE")
    print("best_acc:", best_acc)
    print("best_model:", best_dir)
    print("last_model:", last_dir)
    print("final_model:", final_dir)


if __name__ == "__main__":
    main()
