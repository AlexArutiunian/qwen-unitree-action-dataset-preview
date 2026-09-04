from __future__ import annotations

import argparse
import json
import math
import time
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .hf_utils import PROJECT_ROOT, json_dump, load_config, resolve_path, set_runtime_env
from .prompts import load_spec, parse_joint_ranges
from .validate_robot_json import stable_json_dumps, validate_motion_obj


def normalize_json_text(text: str) -> str:
    try:
        return stable_json_dumps(json.loads(text))
    except Exception:
        return text.strip()


def load_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            messages = obj["messages"]
            rows.append(
                {
                    "sample_id": obj.get("sample_id"),
                    "source_id": obj.get("source_id"),
                    "class": obj.get("class"),
                    "augmentation_type": obj.get("augmentation_type"),
                    "messages": messages,
                    "command": messages[1]["content"].split("Command:\n", 1)[-1],
                    "gold": messages[2]["content"],
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def load_model_and_tokenizer(model_name: str, adapter: Path, dtype_name: str):
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter if (adapter / "tokenizer_config.json").exists() else model_name, trust_remote_code=True)
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


def summarize_errors(errors: list[str]) -> dict[str, float]:
    unknown = sum("unknown_joint" in e or "forbidden_joint" in e for e in errors)
    angle = sum("angle_out_of_range" in e for e in errors)
    duration_bad = sum("duration_below_min" in e or "bad_duration" in e or "missing_duration" in e for e in errors)
    return {
        "unknown_joint_count": unknown,
        "angle_out_of_range_count": angle,
        "duration_invalid_count": duration_bad,
    }


def evaluate_split(cfg: dict[str, Any], adapter: Path, split: str, limit: int | None = None) -> pd.DataFrame:
    root = resolve_path(cfg.get("project_root", PROJECT_ROOT)) or PROJECT_ROOT
    set_runtime_env(str(cfg.get("cuda_visible_devices", "0")))
    dataset_dir = resolve_path(cfg.get("dataset", {}).get("work_dir", "outputs/dataset"), root) or (root / "outputs" / "dataset")
    prefix = f"{cfg.get('prompt_mode', 'compact')}_{int(cfg.get('max_seq_length', 4096))}"
    data_path = dataset_dir / f"{split}_{prefix}.jsonl"
    if not data_path.exists():
        data_path = dataset_dir / f"{split}.jsonl"
    spec_path = dataset_dir / "hf_downloaded_dataset" / "robot_spec_joints-only.txt"
    if not spec_path.exists():
        spec_path = resolve_path(cfg.get("dataset", {}).get("spec_path", ""), root) or spec_path
    if not spec_path.exists():
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        spec_path = Path(manifest["spec_input"])
    joint_ranges = parse_joint_ranges(load_spec(spec_path))
    model, tokenizer = load_model_and_tokenizer(
        cfg.get("model_name", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        adapter,
        cfg.get("bnb_4bit_compute_dtype", "float16"),
    )
    rows = load_rows(data_path, limit=limit)
    results = []
    for idx, row in enumerate(rows, 1):
        pred, latency = generate(model, tokenizer, row["messages"], int(cfg.get("max_new_tokens", 768)))
        parse_ok = 0
        schema_ok = 0
        errors = []
        try:
            pred_obj = json.loads(pred)
            parse_ok = 1
            errors = validate_motion_obj(pred_obj, joint_ranges)
            schema_ok = int(len(errors) == 0)
        except Exception as exc:
            errors = [f"json_parse_error:{exc}"]
        exact = int(normalize_json_text(pred) == normalize_json_text(row["gold"]))
        sim = SequenceMatcher(None, normalize_json_text(pred), normalize_json_text(row["gold"])).ratio()
        err_counts = summarize_errors(errors)
        results.append(
            {
                **{k: row[k] for k in ["sample_id", "source_id", "class", "augmentation_type", "command", "gold"]},
                "pred": pred,
                "parse_ok": parse_ok,
                "schema_ok": schema_ok,
                "exact_match": exact,
                "sequence_similarity": sim,
                "generation_latency_sec": latency,
                "errors": ";".join(errors[:40]),
                **err_counts,
            }
        )
        if idx % 10 == 0:
            print(f"[eval] {split}: {idx}/{len(rows)}")
    return pd.DataFrame(results)


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"n": 0}
    lat = df["generation_latency_sec"].dropna().tolist()
    return {
        "n": int(len(df)),
        "parse_ok_rate": float(df["parse_ok"].mean()),
        "schema_ok_rate": float(df["schema_ok"].mean()),
        "exact_match": float(df["exact_match"].mean()),
        "sequence_similarity": float(df["sequence_similarity"].mean()),
        "unknown_joint_rate": float((df["unknown_joint_count"] > 0).mean()),
        "angle_out_of_range_rate": float((df["angle_out_of_range_count"] > 0).mean()),
        "duration_valid_rate": float((df["duration_invalid_count"] == 0).mean()),
        "avg_generation_latency_sec": float(sum(lat) / len(lat)) if lat else math.nan,
        "p50_latency_sec": float(median(lat)) if lat else math.nan,
        "p95_latency_sec": float(pd.Series(lat).quantile(0.95)) if lat else math.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_compact_4096.yaml")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--split", choices=["val", "reserved", "both"], default="reserved")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = resolve_path(cfg.get("project_root", PROJECT_ROOT)) or PROJECT_ROOT
    adapter = resolve_path(args.adapter, root)
    if adapter is None or not adapter.exists():
        raise FileNotFoundError(f"Adapter not found: {args.adapter}")
    eval_dir = root / "outputs" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    splits = ["val", "reserved"] if args.split == "both" else [args.split]
    summary = {}
    for split in splits:
        df = evaluate_split(cfg, adapter, split, args.limit)
        out = eval_dir / f"eval_{split}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        summary[split] = summarize(df)
        if split == "reserved":
            with (eval_dir / "predictions_reserved.jsonl").open("w", encoding="utf-8") as f:
                for _, row in df.iterrows():
                    f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        print(f"[eval] saved {out}")
    json_dump(summary, eval_dir / "eval_summary.json")
    print(f"[eval] summary: {eval_dir / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
