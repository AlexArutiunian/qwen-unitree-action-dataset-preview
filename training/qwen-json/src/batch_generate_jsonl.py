from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .hf_utils import PROJECT_ROOT, resolve_path, set_runtime_env


def load_base_model_and_tokenizer(model_name: str, dtype_name: str):
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def load_lora_model_and_tokenizer(model_name: str, adapter: Path, dtype_name: str):
    model, tokenizer = load_base_model_and_tokenizer(model_name, dtype_name)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, system: str, user: str, max_new_tokens: int) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - start
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return text, latency


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch JSONL generation for robot eval cases.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--cuda-visible-devices", default="0")
    args = parser.parse_args()

    root = PROJECT_ROOT
    set_runtime_env(args.cuda_visible_devices)
    adapter = resolve_path(args.adapter, root) if args.adapter else None

    if adapter is not None and adapter.exists():
        print(f"[batch_generate] loading LoRA adapter: {adapter}", flush=True)
        model, tokenizer = load_lora_model_and_tokenizer(args.model_name, adapter, args.dtype)
    else:
        print(f"[batch_generate] loading base model: {args.model_name}", flush=True)
        model, tokenizer = load_base_model_and_tokenizer(args.model_name, args.dtype)

    cases_path = resolve_path(args.cases, root) or Path(args.cases)
    out_path = resolve_path(args.out, root) or Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(cases_path))
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, 1):
            error = None
            pred = ""
            latency = 0.0
            try:
                pred, latency = generate(
                    model,
                    tokenizer,
                    str(row["system"]),
                    str(row["user"]),
                    int(args.max_new_tokens),
                )
            except Exception as exc:
                error = str(exc)
            payload: dict[str, Any] = {
                **row,
                "predicted_text": pred,
                "generation_latency_sec": latency,
                "error": error,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[batch_generate] {idx}/{len(rows)} error={int(error is not None)} latency={latency:.3f}", flush=True)


if __name__ == "__main__":
    main()
