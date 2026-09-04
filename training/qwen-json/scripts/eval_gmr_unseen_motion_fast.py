#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_model(model_name: str, adapter: Path):
    dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        adapter if (adapter / "tokenizer_config.json").exists() else model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-time", type=float, default=45.0)
    args = parser.parse_args()

    rows = load_rows(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.model, args.adapter)

    done = 0
    with args.out.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            meta = row.get("meta", {})
            messages = row["messages"][:-1]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            start = time.perf_counter()
            status = "ok"
            error = ""
            try:
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        max_time=args.max_time,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            except Exception as exc:
                status = "error"
                error = repr(exc)
                pred = ""
            latency = time.perf_counter() - start
            result = {
                "idx": idx,
                "sample_id": row.get("sample_id"),
                "task": row.get("task"),
                "robot": meta.get("robot"),
                "scene": meta.get("scene"),
                "name": meta.get("name") or f"{meta.get('robot','robot')}_{idx}",
                "command": meta.get("unseen_command"),
                "pred": pred,
                "gold": row["messages"][-1].get("content", ""),
                "latency_sec": latency,
                "status": status,
                "error": error,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            print(f"[eval] {idx}/{len(rows)} robot={result['robot']} status={status} latency={latency:.2f}s chars={len(pred)}", flush=True)
    print(f"[eval] wrote {done}: {args.out}")


if __name__ == "__main__":
    main()
