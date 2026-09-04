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


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_model(model_name: str, adapter: Path, dtype_name: str = "float16"):
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
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


def generate(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int) -> tuple[str, float]:
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
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


def motion_checks(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "parse_ok": False,
        "compact_frame_ok": False,
        "final_hold_ok": False,
        "has_markdown": "```" in text,
        "errors": [],
    }
    try:
        obj = json.loads(text)
    except Exception as exc:
        result["errors"].append(f"json_parse_error:{exc}")
        return result
    result["parse_ok"] = isinstance(obj, list)
    if not isinstance(obj, list):
        result["errors"].append("top_level_not_array")
        return result
    list_frames = []
    for i, step in enumerate(obj):
        if isinstance(step, dict) and isinstance(step.get("frame"), list):
            list_frames.append(i)
    result["compact_frame_ok"] = len(list_frames) == 0
    if list_frames:
        result["errors"].append(f"legacy_list_frames:{list_frames}")
    if obj and isinstance(obj[-1], dict) and obj[-1].get("frame") == {}:
        try:
            dur = float(obj[-1].get("duration", 0))
            result["final_hold_ok"] = 3.0 <= dur <= 5.0
        except Exception:
            pass
    if not result["final_hold_ok"]:
        result["errors"].append("missing_or_bad_final_hold")
    return result


def write_report(rows: list[dict[str, Any]], out_md: Path) -> None:
    lines = ["# Multitask Eval Report", ""]
    for row in rows:
        lines.append(f"## {row['idx']}. {row['task']} sample_id={row.get('sample_id')}")
        lines.append("")
        lines.append("### User")
        lines.append("```text")
        lines.append(row["user"])
        lines.append("```")
        lines.append("")
        lines.append("### Gold")
        fence = "json" if row["task"] == "motion_json" else "text"
        lines.append(f"```{fence}")
        lines.append(row["gold"])
        lines.append("```")
        lines.append("")
        lines.append("### Model")
        lines.append(f"```{fence}")
        lines.append(row["pred"])
        lines.append("```")
        lines.append("")
        lines.append(f"latency_sec: `{row['latency_sec']:.3f}`")
        if row["task"] == "motion_json":
            lines.append("")
            lines.append("checks:")
            for key, value in row["checks"].items():
                lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "outputs" / "dataset_100_statis_multitask_robot_sft" / "val_compact_4096.jsonl")
    parser.add_argument("--adapter", type=Path, default=PROJECT_ROOT / "outputs" / "statis100_multitask_robot_qa_qwen25_7b_lora" / "best_adapter")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "multitask_eval_report")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer = load_model(args.model, args.adapter)

    results = []
    for idx, row in enumerate(rows, start=1):
        messages = row["messages"]
        pred, latency = generate(model, tokenizer, messages, args.max_new_tokens)
        task = row.get("task", "")
        result = {
            "idx": idx,
            "sample_id": row.get("sample_id"),
            "task": task,
            "user": messages[1]["content"],
            "gold": messages[2]["content"],
            "pred": pred,
            "latency_sec": latency,
            "checks": motion_checks(pred) if task == "motion_json" else {},
        }
        results.append(result)
        print(f"[eval] {idx}/{len(rows)} {task} latency={latency:.3f}s")

    jsonl_path = args.out_dir / "val_predictions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    md_path = args.out_dir / "val_report.md"
    write_report(results, md_path)
    print(f"[eval] saved jsonl: {jsonl_path}")
    print(f"[eval] saved md: {md_path}")


if __name__ == "__main__":
    main()
