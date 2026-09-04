from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path | None, base: Path | None = None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or PROJECT_ROOT) / p


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(config_path)
    cfg.setdefault("project_root", str(PROJECT_ROOT))
    return cfg


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def set_runtime_env(cuda_visible_devices: str = "0") -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", cuda_visible_devices)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_hf_token(required: bool = False) -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    if required:
        raise RuntimeError("HF_TOKEN is not set. Export it before downloading a private dataset.")
    return None


def download_hf_dataset_files(
    repo_id: str,
    repo_type: str,
    files: list[str],
    cache_dir: Path,
    revision: str = "main",
    token: str | None = None,
) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}
    for filename in files:
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            revision=revision,
            token=token,
            local_dir=cache_dir,
        )
        downloaded[filename] = Path(path)
        print(f"[hf] downloaded {filename}: {downloaded[filename]}")
    return downloaded


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
