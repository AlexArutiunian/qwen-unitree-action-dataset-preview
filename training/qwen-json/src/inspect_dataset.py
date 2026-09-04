from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .hf_utils import PROJECT_ROOT, resolve_path


def print_table(df: pd.DataFrame, cols: list[str], n: int) -> None:
    if df.empty:
        print("(empty)")
        return
    present = [c for c in cols if c in df.columns]
    with pd.option_context("display.max_colwidth", 120, "display.width", 240):
        print(df[present].head(n).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="outputs/dataset_compact_4096")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    dataset_dir = resolve_path(args.dataset_dir, PROJECT_ROOT)
    if dataset_dir is None or not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {args.dataset_dir}")

    summary_path = dataset_dir / "dataset_report_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print("\nSUMMARY")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    token_path = dataset_dir / "token_stats.csv"
    dropped_path = dataset_dir / "dropped.csv"
    by_class_path = dataset_dir / "dropped_by_class.csv"

    if token_path.exists():
        stats = pd.read_csv(token_path)
        print("\nLONGEST EXAMPLES")
        print_table(
            stats.sort_values("n_tokens", ascending=False),
            [
                "sample_id",
                "source_id",
                "split",
                "class",
                "augmentation_type",
                "n_tokens",
                "prompt_tokens",
                "assistant_tokens",
                "json_name",
                "source_json",
                "command_preview",
            ],
            args.top,
        )

    if dropped_path.exists():
        dropped = pd.read_csv(dropped_path)
        print("\nDROPPED BY LENGTH")
        print_table(
            dropped.sort_values("n_tokens", ascending=False) if not dropped.empty else dropped,
            [
                "sample_id",
                "source_id",
                "split",
                "class",
                "augmentation_type",
                "n_tokens",
                "prompt_tokens",
                "assistant_tokens",
                "drop_reason",
                "json_name",
                "source_json",
                "command_preview",
            ],
            args.top,
        )

    if by_class_path.exists():
        by_class = pd.read_csv(by_class_path)
        print("\nDROPPED BY CLASS")
        print_table(by_class, list(by_class.columns), args.top)


if __name__ == "__main__":
    main()
