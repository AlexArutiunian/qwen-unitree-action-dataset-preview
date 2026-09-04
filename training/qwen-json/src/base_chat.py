from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .hf_utils import PROJECT_ROOT, json_dump, resolve_path, set_runtime_env
from .prompts import parse_joint_ranges
from .validate_robot_json import stable_json_dumps, validate_motion_obj


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


def generate(model, tokenizer, system_prompt: str, command: str, max_new_tokens: int) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Command:\n{command.strip()}"},
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
    pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return pred, latency


def safe_name(text: str, max_len: int = 48) -> str:
    text = re.sub(r"\s+", "_", text.strip().lower())
    text = re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ_-]+", "", text)
    return text[:max_len].strip("_") or "command"


def save_response(out_dir: Path, command: str, pred: str, latency: float, errors: list[str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "_meta"
    raw_dir = out_dir / "_raw"
    meta_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name(command)}"
    json_path = out_dir / f"{stem}.json"
    raw_path = raw_dir / f"{stem}.raw.txt"
    meta_path = meta_dir / f"{stem}.meta.json"

    try:
        obj = json.loads(pred)
        json_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        parse_ok = True
        response_path = json_path
    except Exception:
        raw_path.write_text(pred.strip() + "\n", encoding="utf-8")
        parse_ok = False
        response_path = raw_path

    json_dump(
        {
            "command": command,
            "response_file": str(response_path),
            "parse_ok": parse_ok,
            "schema_ok": len(errors) == 0,
            "validation_errors": errors,
            "latency_sec": latency,
            "raw_response": pred,
            "mode": "base_model_no_lora",
        },
        meta_path,
    )
    return response_path


def run_command(model, tokenizer, system_prompt: str, joint_ranges: dict[str, tuple[float, float]], out_dir: Path, command: str, max_new_tokens: int) -> None:
    pred, latency = generate(model, tokenizer, system_prompt, command, max_new_tokens)
    try:
        obj = json.loads(pred)
        errors = validate_motion_obj(obj, joint_ranges)
        display = stable_json_dumps(obj)
    except Exception as exc:
        errors = [f"json_parse_error:{exc}"]
        display = pred.strip()
    saved = save_response(out_dir, command, pred, latency, errors)
    print(display)
    print(f"[base_chat] saved: {saved}")
    print(f"[base_chat] latency_sec={latency:.3f} schema_ok={int(len(errors) == 0)}")
    if errors:
        print("[base_chat] validation_errors:")
        for error in errors[:20]:
            print(f"- {error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="*", help="Optional one-shot command. Omit for interactive chat.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--system-prompt-file", default="full_robot_prompt.txt")
    parser.add_argument("--out-dir", default="outputs/robot_generations")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    root = PROJECT_ROOT
    prompt_path = resolve_path(args.system_prompt_file, root)
    if prompt_path is None or not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {args.system_prompt_file}")
    out_dir = resolve_path(args.out_dir, root) or (root / args.out_dir)

    set_runtime_env(args.cuda_visible_devices)
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    joint_ranges = parse_joint_ranges(system_prompt)

    print(f"[base_chat] loading base model without LoRA: {args.model_name}")
    print(f"[base_chat] system prompt: {prompt_path}")
    model, tokenizer = load_base_model_and_tokenizer(args.model_name, args.dtype)
    print(f"[base_chat] ready. outputs: {out_dir}")

    one_shot = " ".join(args.command).strip()
    if one_shot:
        run_command(model, tokenizer, system_prompt, joint_ranges, out_dir, one_shot, args.max_new_tokens)
        return

    while True:
        try:
            command = input("\ncommand> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if command.lower() in {"exit", "quit", "q"}:
            return
        if not command:
            continue
        run_command(model, tokenizer, system_prompt, joint_ranges, out_dir, command, args.max_new_tokens)


if __name__ == "__main__":
    main()
