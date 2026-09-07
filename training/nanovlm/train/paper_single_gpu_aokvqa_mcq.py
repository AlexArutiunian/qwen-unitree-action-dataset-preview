import argparse, math, random
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor

# -----------------------------
# Специализированная MCQA-голова
# -----------------------------

class MCQHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.out = nn.Linear(hidden_dim, 4)  # 4 варианта ответа

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [B, T, D]
        last = hidden[:, -1, :]   # [B, D]
        last = self.dropout(last)
        return self.out(last)     # [B, 4]


# -----------------------------
# Монкипатч: forward_mcq_logits
# -----------------------------

def forward_mcq_logits(self, input_ids: torch.Tensor, images: torch.Tensor, attention_mask=None) -> torch.Tensor:
    """
    Вариант forward, который:
      - считает только скрытые состояния декодера
      - не делает проекцию на словарь
      - отдаёт логиты 4-классовой головы MCQHead
    """
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
        start_pos=0
    )  # [B, T, D]

    logits = self.mcq_head(hidden)  # [B, 4]
    return logits

# подвешиваем метод к классу
VisionLanguageModel.forward_mcq_logits = forward_mcq_logits

# -----------------------------
# Остальной тренировочный код
# -----------------------------

LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}
IDX_TO_LETTER = "ABCD"

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def device_auto():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_mc_prompt(question: str, choices: List[str]) -> str:
    return (
        f"Question: {question}\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n"
        "Answer:"
    )

class AOKVQAMCQDataset(Dataset):
    def __init__(self, split="train"):
        self.ds = load_dataset("HuggingFaceM4/A-OKVQA", split=split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.ds[int(idx)]
        return {
            "image": ex["image"],
            "question": ex["question"],
            "choices": list(ex["choices"]),
            "gt_idx": int(ex["correct_choice_idx"]),  # 0..3
            "qid": str(ex["question_id"])
        }

# -----------------------------
# Настройка trainable модулей
# -----------------------------

def configure_trainable_modules(model: VisionLanguageModel, args):
    """
    Настраиваем, какие части модели обучаем:
      - vision_encoder: freeze по флагу
      - MP: freeze по флагу
      - decoder: либо полностью frozen, либо размораживаем последние N слоёв
    """
    # vision
    if args.freeze_vision and hasattr(model, "vision_encoder"):
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        print("[freeze] vision_encoder")
    else:
        print("[unfreeze] vision_encoder")

    # MP
    if args.freeze_proj and hasattr(model, "MP"):
        for p in model.MP.parameters():
            p.requires_grad = False
        print("[freeze] modality projector (MP)")
    else:
        print("[unfreeze] modality projector (MP)")

    # decoder
    dec = model.decoder

    # пытаемся найти список слоёв
    if hasattr(dec, "layers"):
        layers = dec.layers
    elif hasattr(dec, "model") and hasattr(dec.model, "layers"):
        layers = dec.model.layers
    else:
        print("[warn] cannot find decoder layers (no .layers or .model.layers). Decoder untouched.")
        return

    num_layers = len(layers)
    k = max(0, int(args.unfreeze_decoder_layers))

    if k == 0:
        # все слои декодера замораживаем
        for layer in layers:
            for p in layer.parameters():
                p.requires_grad = False
        print(f"[freeze] all {num_layers} decoder layers")
    else:
        cutoff = num_layers - k
        for i, layer in enumerate(layers):
            train_this = (i >= cutoff)
            for p in layer.parameters():
                p.requires_grad = train_this
        print(f"[freeze] first {cutoff} decoder layers, [unfreeze] last {k} decoder layers")

# -----------------------------
# TRAIN
# -----------------------------

def train(args):
    seed_everything(args.seed)
    device = device_auto()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Loading model: {args.model_id}")
    model = VisionLanguageModel.from_pretrained(args.model_id).to(device)
    model.train()

    # сначала настраиваем, что тренируем в vision/MP/decoder
    configure_trainable_modules(model, args)

    # навешиваем MCQ-голову (всегда trainable)
    hidden_dim = model.decoder.token_embedding.embedding_dim
    model.mcq_head = MCQHead(hidden_dim, dropout=args.head_dropout).to(device)

    tokenizer = get_tokenizer(model.cfg.lm_tokenizer, model.cfg.vlm_extra_tokens, model.cfg.lm_chat_template)
    imgproc = get_image_processor(model.cfg.max_img_size, model.cfg.vit_img_size)

    ds_train = AOKVQAMCQDataset(split="train")
    ds_val   = AOKVQAMCQDataset(split="validation")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    total_steps = math.ceil(len(ds_train) * args.epochs / max(1, args.grad_accum))
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.label_smoothing > 0.0:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(f"[loss] CrossEntropyLoss with label_smoothing={args.label_smoothing}")
    else:
        loss_fn = nn.CrossEntropyLoss()
        print("[loss] Plain CrossEntropyLoss (no label smoothing)")

    global_step, best_val = 0, -1.0

    def step_on_example(sample: Dict[str,Any]) -> torch.Tensor:
        img: Image.Image = sample["image"].convert("RGB")
        q, choices = sample["question"], sample["choices"]
        gt_idx = sample["gt_idx"]

        prompt = build_mc_prompt(q, choices)

        proc_img, _ = imgproc(img)
        n_imgs = proc_img.shape[0]
        num_img_tokens = model.cfg.mp_image_token_length * n_imgs
        image_tokens = tokenizer.image_token * num_img_tokens

        # только пользовательский промпт, без правильного ответа внутри контекста
        messages = [{"role": "user", "content": image_tokens + prompt}]
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True
        )
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        logits = model.forward_mcq_logits(input_ids, proc_img.to(device))  # [1, 4]
        labels = torch.tensor([gt_idx], dtype=torch.long, device=device)   # [1]
        loss = loss_fn(logits, labels)
        return loss

    @torch.no_grad()
    def evaluate_mcq(split_ds) -> float:
        model.eval()
        correct = 0
        total = len(split_ds)
        for ex in tqdm(split_ds, total=total, desc="Eval-MCQ"):
            img: Image.Image = ex["image"].convert("RGB")
            q, choices = ex["question"], list(ex["choices"])
            gt_idx = int(ex["gt_idx"])

            proc_img, _ = imgproc(img)
            n_imgs = proc_img.shape[0]
            num_img_tokens = model.cfg.mp_image_token_length * n_imgs
            image_tokens = tokenizer.image_token * num_img_tokens

            prompt = build_mc_prompt(q, choices)
            messages = [{"role": "user", "content": image_tokens + prompt}]
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True
            )
            input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

            logits = model.forward_mcq_logits(input_ids, proc_img.to(device))  # [1, 4]
            pred_idx = int(logits.argmax(dim=-1).item())
            correct += int(pred_idx == gt_idx)

        model.train()
        return 100.0 * correct / total

    print(f"Train size: {len(ds_train)} | Val size: {len(ds_val)}")
    accum = args.grad_accum
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(total_steps), desc="Training", dynamic_ncols=True)
    train_iter_idx = 0

    for step in pbar:
        loss_accum = 0.0
        for _ in range(accum):
            if train_iter_idx >= len(ds_train):
                train_iter_idx = 0
            sample = ds_train[train_iter_idx]
            train_iter_idx += 1

            loss = step_on_example(sample) / accum
            loss.backward()
            loss_accum += loss.item()

        clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        pbar.set_postfix(loss=f"{loss_accum:.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")

        if (global_step % args.eval_every) == 0:
            val_acc = evaluate_mcq(ds_val)
            print(f"\n[Eval] step={global_step} | val MCQ acc = {val_acc:.2f}%")
            if val_acc > best_val:
                best_val = val_acc
                ckpt_path = out_dir / f"best_mcq_{val_acc:.2f}.pt"
                print(f"[Checkpoint] Saving to {ckpt_path}")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "val_acc": val_acc,
                        "step": global_step,
                    },
                    ckpt_path,
                )

    final_acc = evaluate_mcq(ds_val)
    print(f"\n[Final] val MCQ acc = {final_acc:.2f}%  (best={best_val:.2f}%)")

    full_out = out_dir / "mcq_finetuned"
    model.save_pretrained(str(full_out))
    print(f"Saved HF checkpoint to: {full_out}")

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="lusxvr/nanoVLM", help="Pretrained HF model id or local folder")
    ap.add_argument("--output_dir", default="checkpoints_mcq")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.10)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=500)

    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--head_dropout", type=float, default=0.0)
    ap.add_argument("--unfreeze_decoder_layers", type=int, default=0)

    ap.add_argument("--freeze_vision", action="store_true", default=True)
    ap.add_argument("--freeze_proj",  action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    train(args)

if __name__ == "__main__":
    cli()
