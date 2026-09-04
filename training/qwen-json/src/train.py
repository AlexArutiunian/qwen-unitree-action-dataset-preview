from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .hf_utils import (
    PROJECT_ROOT,
    json_dump,
    load_config,
    load_tokenizer,
    now_utc_iso,
    resolve_path,
    set_runtime_env,
    set_seed,
)
from .prepare_dataset import prepare_from_config


class ChatSFTDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_seq_length: int, drop_overlength: bool = False):
        self.items: list[dict[str, torch.Tensor]] = []
        self.dropped_overlength: list[dict[str, Any]] = []
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                messages = obj["messages"]
                prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
                full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                encoded = tokenizer(full_text, add_special_tokens=False, truncation=False)
                input_ids = encoded["input_ids"]
                if len(input_ids) > max_seq_length:
                    if drop_overlength:
                        self.dropped_overlength.append(
                            {
                                "sample_id": obj.get("sample_id"),
                                "source_id": obj.get("source_id"),
                                "n_tokens": len(input_ids),
                                "max_seq_length": max_seq_length,
                            }
                        )
                        continue
                    raise ValueError(
                        f"Example sample_id={obj.get('sample_id')} has {len(input_ids)} tokens > {max_seq_length}. "
                        "Run prepare_dataset filtering first; target JSON is intentionally not truncated."
                    )
                labels = list(input_ids)
                labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
                if all(x == -100 for x in labels):
                    continue
                self.items.append(
                    {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }
                )
        if not self.items:
            raise RuntimeError(f"No usable examples loaded from {path}")
        if self.dropped_overlength:
            print(f"[dataset] dropped overlength from {path.name}: {len(self.dropped_overlength)}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


class DataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        return {
            "input_ids": pad_sequence([f["input_ids"] for f in features], batch_first=True, padding_value=self.pad_token_id),
            "attention_mask": pad_sequence([f["attention_mask"] for f in features], batch_first=True, padding_value=0),
            "labels": pad_sequence([f["labels"] for f in features], batch_first=True, padding_value=-100),
        }


class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row: dict[str, Any] = {
            "time": time.time(),
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
        }
        for key in ["loss", "eval_loss", "learning_rate", "grad_norm"]:
            if key in logs:
                row[key] = logs[key]
        if torch.cuda.is_available():
            row["gpu_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 4)
            row["gpu_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 4)
            row["gpu_max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _training_args_kwargs() -> dict[str, str]:
    import inspect

    params = inspect.signature(TrainingArguments.__init__).parameters
    return {
        "eval_key": "eval_strategy" if "eval_strategy" in params else "evaluation_strategy",
    }


def load_model(cfg: dict[str, Any]):
    compute_dtype_name = cfg.get("bnb_4bit_compute_dtype", "float16")
    compute_dtype = torch.bfloat16 if compute_dtype_name == "bfloat16" else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.get("model_name", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    lora = cfg.get("lora", {})
    lora_config = LoraConfig(
        r=int(lora.get("r", 8)),
        lora_alpha=int(lora.get("alpha", 16)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora.get("target_modules", ["q_proj", "v_proj"]),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train_from_config(cfg: dict[str, Any]) -> Path:
    set_runtime_env(str(cfg.get("cuda_visible_devices", "0")))
    set_seed(int(cfg.get("seed", 42)))
    root = resolve_path(cfg.get("project_root", PROJECT_ROOT)) or PROJECT_ROOT
    run_name = cfg.get("run_name") or f"{cfg.get('prompt_mode', 'compact')}_{cfg.get('max_seq_length', 4096)}"
    run_dir = resolve_path(cfg.get("output_dir", f"outputs/{run_name}"), root) or (root / "outputs" / run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] run_dir: {run_dir}")
    dataset_cfg = cfg.get("dataset", {})
    local_sft_dir = dataset_cfg.get("local_sft_dir")
    if local_sft_dir:
        dataset_dir = resolve_path(local_sft_dir, root)
        if dataset_dir is None or not dataset_dir.exists():
            raise FileNotFoundError(f"Local SFT dataset not found: {local_sft_dir}")
        manifest_path = dataset_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
            "dataset_kind": "local_sft_dir",
            "dataset_dir": str(dataset_dir),
        }
        print(f"[train] using local SFT dataset: {dataset_dir}")
    else:
        manifest = prepare_from_config(cfg)
        dataset_dir = resolve_path(dataset_cfg.get("work_dir", "outputs/dataset"), root) or (root / "outputs" / "dataset")
    prefix = f"{cfg.get('prompt_mode', 'compact')}_{int(cfg.get('max_seq_length', 4096))}"
    train_path = dataset_dir / f"train_{prefix}.jsonl"
    val_path = dataset_dir / f"val_{prefix}.jsonl"

    tokenizer = load_tokenizer(cfg.get("model_name", "Qwen/Qwen2.5-Coder-7B-Instruct"))
    drop_overlength = bool(dataset_cfg.get("drop_overlength_examples", False))
    train_dataset = ChatSFTDataset(train_path, tokenizer, int(cfg.get("max_seq_length", 4096)), drop_overlength=drop_overlength)
    eval_dataset = ChatSFTDataset(val_path, tokenizer, int(cfg.get("max_seq_length", 4096)), drop_overlength=drop_overlength)
    print(f"[train] train examples: {len(train_dataset)}")
    print(f"[train] val examples: {len(eval_dataset)}")
    dropped_overlength = train_dataset.dropped_overlength + eval_dataset.dropped_overlength
    if dropped_overlength:
        dropped_path = run_dir / "dropped_overlength_runtime.json"
        json_dump(dropped_overlength, dropped_path)
        print(f"[train] runtime dropped overlength examples: {len(dropped_overlength)} -> {dropped_path}")

    model = load_model(cfg)
    tcfg = cfg.get("training", {})
    sig = _training_args_kwargs()
    kwargs = {
        "output_dir": str(run_dir / "checkpoints"),
        "num_train_epochs": float(tcfg.get("num_train_epochs", 10)),
        "max_steps": int(tcfg.get("max_steps", 600)),
        "per_device_train_batch_size": int(tcfg.get("batch_size", 1)),
        "per_device_eval_batch_size": int(tcfg.get("eval_batch_size", 1)),
        "gradient_accumulation_steps": int(tcfg.get("grad_accum", 16)),
        "learning_rate": float(tcfg.get("learning_rate", 2e-4)),
        "lr_scheduler_type": tcfg.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": float(tcfg.get("warmup_ratio", 0.03)),
        "logging_strategy": "steps",
        "logging_steps": int(tcfg.get("logging_steps", 1)),
        sig["eval_key"]: "steps",
        "eval_steps": int(tcfg.get("eval_steps", 50)),
        "save_strategy": "steps",
        "save_steps": int(tcfg.get("save_steps", 50)),
        "save_total_limit": int(tcfg.get("save_total_limit", 3)),
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": cfg.get("bnb_4bit_compute_dtype", "float16") == "float16",
        "bf16": cfg.get("bnb_4bit_compute_dtype") == "bfloat16",
        "optim": tcfg.get("optim", "paged_adamw_8bit"),
        "report_to": tcfg.get("report_to", ["tensorboard"]),
        "logging_dir": str(run_dir / "runs"),
        "remove_unused_columns": False,
        "dataloader_num_workers": int(tcfg.get("dataloader_num_workers", 0)),
        "gradient_checkpointing": False,
    }
    args = TrainingArguments(**kwargs)
    live_metrics = run_dir / "training_metrics_live.jsonl"
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(tokenizer.pad_token_id),
        callbacks=[
            MetricsLoggerCallback(live_metrics),
            EarlyStoppingCallback(
                early_stopping_patience=int(tcfg.get("early_stopping_patience", 4)),
                early_stopping_threshold=float(tcfg.get("early_stopping_threshold", 0.001)),
            ),
        ],
    )

    trainer.train()
    best_ckpt = trainer.state.best_model_checkpoint
    best_metric = trainer.state.best_metric
    print(f"[train] best checkpoint: {best_ckpt}")
    print(f"[train] best metric: {best_metric}")

    best_adapter = run_dir / "best_adapter"
    trainer.save_model(str(best_adapter))
    tokenizer.save_pretrained(str(best_adapter))
    shared_best = root / "outputs" / "best_adapter"
    if shared_best.exists() or shared_best.is_symlink():
        if shared_best.is_symlink() or shared_best.is_file():
            shared_best.unlink()
        else:
            shutil.rmtree(shared_best)
    shutil.copytree(best_adapter, shared_best)

    history = pd.DataFrame(trainer.state.log_history)
    history.to_csv(run_dir / "trainer_log_history.csv", index=False, encoding="utf-8-sig")
    if (dataset_dir / "token_stats.csv").exists():
        shutil.copy2(dataset_dir / "token_stats.csv", run_dir / "token_stats.csv")
    if (dataset_dir / "dropped_by_length.csv").exists():
        shutil.copy2(dataset_dir / "dropped_by_length.csv", run_dir / "dropped_by_length.csv")
    if (dataset_dir / "dropped.csv").exists():
        shutil.copy2(dataset_dir / "dropped.csv", run_dir / "dropped.csv")

    train_manifest = {
        "created_at": now_utc_iso(),
        "config": cfg,
        "run_dir": str(run_dir),
        "model_name": cfg.get("model_name"),
        "prompt_mode": cfg.get("prompt_mode"),
        "max_seq_length": cfg.get("max_seq_length"),
        "train_rows": len(train_dataset),
        "val_rows": len(eval_dataset),
        "dataset_manifest": manifest,
        "runtime_dropped_overlength": dropped_overlength,
        "best_model_checkpoint": best_ckpt,
        "best_metric": best_metric,
    }
    json_dump(train_manifest, run_dir / "manifest.json")
    if tcfg.get("plot_after_train", True):
        try:
            from .plot_training import generate_plots

            generate_plots(run_dir)
        except Exception as exc:
            print(f"[train] plot generation failed: {exc}")
    print(f"[train] saved best adapter: {best_adapter}")
    print(f"[train] copied latest best adapter alias: {shared_best}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_from_config(cfg)


if __name__ == "__main__":
    main()
