from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def chat_token_len(example: dict[str, Any], tokenizer) -> tuple[int, int, int]:
    full_text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(example["messages"][:-1], tokenize=False, add_generation_prompt=True)
    full = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
    prompt = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    return full, prompt, max(0, full - prompt)


def compute_token_stats(jsonl_path: Path, tokenizer) -> pd.DataFrame:
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            full, prompt, assistant = chat_token_len(ex, tokenizer)
            rows.append(
                {
                    "sample_id": ex.get("sample_id"),
                    "source_id": ex.get("source_id"),
                    "class": ex.get("class"),
                    "augmentation_type": ex.get("augmentation_type"),
                    "split": ex.get("split"),
                    "n_tokens": full,
                    "prompt_tokens": prompt,
                    "assistant_tokens": assistant,
                }
            )
    return pd.DataFrame(rows)


def describe_tokens(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "n_tokens" not in df:
        return {"n": 0}
    x = df["n_tokens"].astype(float)
    return {
        "n": int(len(x)),
        "min": int(x.min()),
        "p50": int(np.percentile(x, 50)),
        "p90": int(np.percentile(x, 90)),
        "p95": int(np.percentile(x, 95)),
        "p99": int(np.percentile(x, 99)),
        "max": int(x.max()),
        "mean": float(x.mean()),
    }


def filter_by_length(df: pd.DataFrame, max_seq_length: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept = df[df["n_tokens"] <= max_seq_length].copy()
    dropped = df[df["n_tokens"] > max_seq_length].copy()
    return kept, dropped
