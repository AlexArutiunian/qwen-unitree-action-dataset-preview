from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from .hf_utils import (
    PROJECT_ROOT,
    download_hf_dataset_files,
    get_hf_token,
    json_dump,
    load_config,
    load_tokenizer,
    now_utc_iso,
    resolve_path,
    safe_str,
)
from .prompts import build_system_prompt, load_spec, make_messages, parse_joint_ranges
from .token_stats import compute_token_stats, describe_tokens
from .validate_robot_json import normalize_motion_obj, stable_json_dumps, validate_motion_obj


REQUIRED_PARQUET_COLUMNS = {"sample_id", "source_id", "text", "target_json"}
VALID_SPLITS = {"train", "val", "reserved"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, frame: pd.DataFrame, system_prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, row in frame.iterrows():
            obj = {
                "sample_id": int(row["sample_id"]),
                "source_id": int(row["source_id"]),
                "class": safe_str(row.get("class", "")),
                "augmentation_type": safe_str(row.get("augmentation_type", "")),
                "split": safe_str(row.get("split", "")),
                "messages": make_messages(safe_str(row["text"]), safe_str(row["target_json"]), system_prompt),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _normalize_dataframe(data: pd.DataFrame, joint_ranges: dict[str, tuple[float, float]], out_dir: Path) -> pd.DataFrame:
    missing = REQUIRED_PARQUET_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Parquet is missing required columns: {sorted(missing)}")

    data = data.copy()
    data["sample_id"] = data["sample_id"].astype(int)
    data["source_id"] = data["source_id"].astype(int)
    data["text"] = data["text"].astype(str).str.strip()
    data["target_json"] = data["target_json"].astype(str).str.strip()
    for col in ["class", "augmentation_type", "json_name", "source_json", "target_sha256", "target_len_chars", "sanitize_issues", "split"]:
        if col not in data.columns:
            data[col] = ""

    invalid: list[dict[str, Any]] = []
    normalized_json: list[str] = []
    normalized_sha: list[str] = []
    normalized_len: list[int] = []
    sanitize_col: list[str] = []

    for i, row in data.iterrows():
        errors: list[str] = []
        issues: list[str] = []
        text = safe_str(row["target_json"])
        try:
            obj = json.loads(text)
            obj, issues = normalize_motion_obj(obj)
            errors = validate_motion_obj(obj, joint_ranges)
            target = stable_json_dumps(obj)
        except Exception as exc:
            errors = [f"parse_or_normalize_error:{exc}"]
            target = text
        normalized_json.append(target)
        normalized_sha.append(sha256_text(target))
        normalized_len.append(len(target))
        sanitize_col.append(";".join(issues))
        if errors:
            invalid.append(
                {
                    "row_idx": int(i),
                    "sample_id": row.get("sample_id"),
                    "source_id": row.get("source_id"),
                    "class": row.get("class", ""),
                    "error": ";".join(errors[:40]),
                }
            )

    data["target_json"] = normalized_json
    data["target_sha256"] = normalized_sha
    data["target_len_chars"] = normalized_len
    data["sanitize_issues"] = sanitize_col

    invalid_df = pd.DataFrame(invalid)
    invalid_path = out_dir / "invalid_targets.csv"
    invalid_df.to_csv(invalid_path, index=False, encoding="utf-8-sig")
    print(f"[dataset] invalid targets: {len(invalid_df)} -> {invalid_path}")
    if len(invalid_df):
        raise RuntimeError(f"Invalid target JSON rows found. See {invalid_path}")
    return data


def _ensure_split(data: pd.DataFrame, seed: int, val_ratio: float, reserved_source_ids: list[int]) -> tuple[pd.DataFrame, bool]:
    data = data.copy()
    use_existing = (
        "split" in data.columns
        and data["split"].notna().any()
        and set(data["split"].dropna().astype(str).unique()).issubset(VALID_SPLITS)
    )
    if use_existing:
        data["split"] = data["split"].astype(str)
        return data, True

    all_source_ids = sorted(map(int, data["source_id"].unique()))
    reserved = set(map(int, reserved_source_ids))
    train_val_ids = [x for x in all_source_ids if x not in reserved]
    rng = random.Random(seed)
    rng.shuffle(train_val_ids)
    val_count = max(1, round(len(train_val_ids) * val_ratio)) if len(train_val_ids) > 1 else 0
    val_ids = set(train_val_ids[:val_count])

    def assign(source_id: int) -> str:
        sid = int(source_id)
        if sid in reserved:
            return "reserved"
        if sid in val_ids:
            return "val"
        return "train"

    data["split"] = data["source_id"].map(assign)
    return data, False


def _filter_split_jsonl(
    raw_jsonl: Path,
    filtered_jsonl: Path,
    tokenizer,
    max_seq_length: int,
    dropped_rows: list[pd.DataFrame],
) -> pd.DataFrame:
    stats = compute_token_stats(raw_jsonl, tokenizer)
    if stats.empty:
        stats["keep_for_training"] = []
        stats["drop_reason"] = []
    else:
        stats["keep_for_training"] = stats["n_tokens"] <= max_seq_length
        stats["drop_reason"] = ""
        stats.loc[~stats["keep_for_training"], "drop_reason"] = f"n_tokens>{max_seq_length}"
    kept_ids = set(stats.loc[stats["n_tokens"] <= max_seq_length, "sample_id"].astype(int).tolist())
    dropped = stats[stats["n_tokens"] > max_seq_length].copy()
    if not dropped.empty:
        dropped_rows.append(dropped)
    filtered_jsonl.parent.mkdir(parents=True, exist_ok=True)
    kept_count = 0
    with raw_jsonl.open("r", encoding="utf-8") as src, filtered_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            ex = json.loads(line)
            if int(ex["sample_id"]) in kept_ids:
                n_tokens = int(stats.loc[stats["sample_id"].astype(int) == int(ex["sample_id"]), "n_tokens"].iloc[0])
                ex["n_tokens"] = n_tokens
                dst.write(json.dumps(ex, ensure_ascii=False) + "\n")
                kept_count += 1
    print(f"[dataset] {raw_jsonl.name}: kept {kept_count}, dropped {len(dropped)}, max_seq_length={max_seq_length}")
    return stats


def _metadata_for_reports(data: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sample_id",
        "source_id",
        "split",
        "class",
        "augmentation_type",
        "json_name",
        "source_json",
        "target_len_chars",
        "target_sha256",
        "text",
        "target_json",
    ]
    present = [c for c in cols if c in data.columns]
    meta = data[present].copy()
    if "text" in meta.columns:
        meta["command_preview"] = meta["text"].astype(str).str.slice(0, 240)
    if "target_json" in meta.columns:
        meta["target_preview"] = meta["target_json"].astype(str).str.slice(0, 240)
    return meta.drop(columns=[c for c in ["text", "target_json"] if c in meta.columns])


def _write_report_summaries(out_dir: Path, token_stats: pd.DataFrame, dropped: pd.DataFrame, max_seq_length: int) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    for split, part in token_stats.groupby("split", dropna=False):
        dropped_part = dropped[dropped["split"] == split] if "split" in dropped.columns else pd.DataFrame()
        summary_rows.append(
            {
                "split": split,
                "rows": int(len(part)),
                "kept": int((part["n_tokens"] <= max_seq_length).sum()) if "n_tokens" in part else 0,
                "dropped": int(len(dropped_part)),
                "max_seq_length": int(max_seq_length),
                "min_tokens": int(part["n_tokens"].min()) if len(part) else None,
                "p50_tokens": int(part["n_tokens"].quantile(0.50)) if len(part) else None,
                "p90_tokens": int(part["n_tokens"].quantile(0.90)) if len(part) else None,
                "p95_tokens": int(part["n_tokens"].quantile(0.95)) if len(part) else None,
                "p99_tokens": int(part["n_tokens"].quantile(0.99)) if len(part) else None,
                "max_tokens": int(part["n_tokens"].max()) if len(part) else None,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "token_stats_summary.csv", index=False, encoding="utf-8-sig")

    if not dropped.empty and "class" in dropped.columns:
        dropped["class"].fillna("").replace("", "unknown").value_counts().rename_axis("class").reset_index(name="dropped").to_csv(
            out_dir / "dropped_by_class.csv", index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame(columns=["class", "dropped"]).to_csv(out_dir / "dropped_by_class.csv", index=False, encoding="utf-8-sig")

    longest = token_stats.sort_values("n_tokens", ascending=False).head(100) if not token_stats.empty else token_stats
    longest.to_csv(out_dir / "longest_examples.csv", index=False, encoding="utf-8-sig")

    summary = {
        "max_seq_length": int(max_seq_length),
        "splits": summary_rows,
        "dropped_total": int(len(dropped)),
        "report_files": {
            "token_stats": str(out_dir / "token_stats.csv"),
            "token_stats_summary": str(out_dir / "token_stats_summary.csv"),
            "dropped": str(out_dir / "dropped.csv"),
            "dropped_by_length": str(out_dir / "dropped_by_length.csv"),
            "dropped_by_class": str(out_dir / "dropped_by_class.csv"),
            "longest_examples": str(out_dir / "longest_examples.csv"),
        },
    }
    json_dump(summary, out_dir / "dataset_report_summary.json")
    return summary


def prepare_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    root = resolve_path(cfg.get("project_root", PROJECT_ROOT)) or PROJECT_ROOT
    dataset_cfg = cfg.get("dataset", {})
    out_dir = resolve_path(dataset_cfg.get("work_dir", "outputs/dataset"), root) or (root / "outputs" / "dataset")
    hf_cache = resolve_path(dataset_cfg.get("hf_cache_dir", "outputs/hf_cache"), root) or (root / "outputs" / "hf_cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    token = get_hf_token(required=bool(dataset_cfg.get("hf_token_required", False)))
    files = download_hf_dataset_files(
        repo_id=dataset_cfg.get("repo_id", "AlexArutiunian/qwen-robot-action-dataset"),
        repo_type=dataset_cfg.get("repo_type", "dataset"),
        files=dataset_cfg.get("files", ["robot_sft.parquet", "robot_spec_joints-only.txt"]),
        revision=dataset_cfg.get("revision", "main"),
        cache_dir=hf_cache,
        token=token,
    )

    parquet_path = files[dataset_cfg.get("parquet_file", "robot_sft.parquet")]
    spec_path = files[dataset_cfg.get("spec_file", "robot_spec_joints-only.txt")]
    spec_text = load_spec(spec_path)
    joint_ranges = parse_joint_ranges(spec_text)
    if len(joint_ranges) < 10:
        raise RuntimeError(f"Too few joint ranges parsed from {spec_path}: {len(joint_ranges)}")

    prompt_mode = cfg.get("prompt_mode", "compact")
    system_prompt = build_system_prompt(spec_text, prompt_mode)
    (out_dir / f"system_prompt_{prompt_mode}.txt").write_text(system_prompt, encoding="utf-8")

    data = pd.read_parquet(parquet_path)
    data = _normalize_dataframe(data, joint_ranges, out_dir)
    data, used_existing_split = _ensure_split(
        data,
        seed=int(cfg.get("seed", 42)),
        val_ratio=float(dataset_cfg.get("val_ratio", 0.10)),
        reserved_source_ids=dataset_cfg.get("reserved_source_ids", list(range(1, 12))),
    )

    train_ids = set(data.loc[data["split"] == "train", "source_id"].astype(int))
    val_ids = set(data.loc[data["split"] == "val", "source_id"].astype(int))
    reserved_ids = set(data.loc[data["split"] == "reserved", "source_id"].astype(int))
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(reserved_ids)
    assert val_ids.isdisjoint(reserved_ids)

    normalized_parquet = out_dir / "robot_sft.parquet"
    data.to_parquet(normalized_parquet, index=False)
    for split in ["train", "val", "reserved"]:
        _write_jsonl(out_dir / f"{split}.jsonl", data[data["split"] == split].copy(), system_prompt)

    tokenizer = load_tokenizer(cfg.get("model_name", "Qwen/Qwen2.5-Coder-7B-Instruct"))
    max_seq_length = int(cfg.get("max_seq_length", 4096))
    prefix = f"{prompt_mode}_{max_seq_length}"
    dropped_parts: list[pd.DataFrame] = []
    all_stats = []
    for split in ["train", "val", "reserved"]:
        stats = _filter_split_jsonl(
            out_dir / f"{split}.jsonl",
            out_dir / f"{split}_{prefix}.jsonl",
            tokenizer,
            max_seq_length,
            dropped_parts,
        )
        stats["split"] = split
        all_stats.append(stats)

    token_stats = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    meta = _metadata_for_reports(data)
    if not token_stats.empty and not meta.empty:
        token_stats = token_stats.drop(columns=[c for c in ["source_id", "split", "class", "augmentation_type"] if c in token_stats.columns])
        token_stats = token_stats.merge(meta, on="sample_id", how="left")
    token_stats_path = out_dir / "token_stats.csv"
    token_stats.to_csv(token_stats_path, index=False, encoding="utf-8-sig")

    dropped = pd.concat(dropped_parts, ignore_index=True) if dropped_parts else pd.DataFrame(
        columns=["sample_id", "source_id", "class", "augmentation_type", "split", "n_tokens"]
    )
    if not dropped.empty and not meta.empty:
        dropped = dropped.drop(columns=[c for c in ["source_id", "split", "class", "augmentation_type"] if c in dropped.columns])
        dropped = dropped.merge(meta, on="sample_id", how="left")
        dropped = dropped.sort_values("n_tokens", ascending=False)
    dropped_path = out_dir / "dropped.csv"
    dropped_by_length_path = out_dir / "dropped_by_length.csv"
    dropped.to_csv(dropped_path, index=False, encoding="utf-8-sig")
    dropped.to_csv(dropped_by_length_path, index=False, encoding="utf-8-sig")
    report_summary = _write_report_summaries(out_dir, token_stats, dropped, max_seq_length)

    manifest = {
        "created_at": now_utc_iso(),
        "repo_id": dataset_cfg.get("repo_id"),
        "repo_type": dataset_cfg.get("repo_type", "dataset"),
        "parquet_input": str(parquet_path),
        "spec_input": str(spec_path),
        "normalized_parquet": str(normalized_parquet),
        "model_name": cfg.get("model_name"),
        "prompt_mode": prompt_mode,
        "max_seq_length": max_seq_length,
        "used_existing_split": used_existing_split,
        "rows": {split: int((data["split"] == split).sum()) for split in ["train", "val", "reserved"]},
        "filtered_rows": {
            split: int(sum(1 for _ in (out_dir / f"{split}_{prefix}.jsonl").open("r", encoding="utf-8")))
            for split in ["train", "val", "reserved"]
        },
        "token_stats": describe_tokens(token_stats),
        "dropped_rows": int(len(dropped)),
        "report_summary": report_summary,
        "joint_count": int(len(joint_ranges)),
    }
    json_dump(manifest, out_dir / "manifest.json")
    print(f"[dataset] manifest: {out_dir / 'manifest.json'}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    prepare_from_config(cfg)


if __name__ == "__main__":
    main()
