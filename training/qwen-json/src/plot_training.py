from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .hf_utils import PROJECT_ROOT, resolve_path


def save_plot(df: pd.DataFrame, x: str, y: str, title: str, out: Path, kind: str = "line") -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        print(f"[plots] skip {out.name}: missing {x}/{y}")
        return
    plot_df = df[[x, y]].copy()
    if kind != "bar":
        plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna()
    if plot_df.empty:
        print(f"[plots] skip {out.name}: empty")
        return
    plt.figure(figsize=(9, 5))
    if kind == "bar":
        plt.bar(plot_df[x].astype(str), plot_df[y])
        plt.xticks(rotation=30, ha="right")
    else:
        plt.plot(plot_df[x], plot_df[y], marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[plots] saved {out}")


def generate_plots(run_dir: Path) -> None:
    run_dir = resolve_path(run_dir, PROJECT_ROOT)
    if run_dir is None:
        raise RuntimeError("Bad run dir")
    plots = run_dir / "plots"
    hist_path = run_dir / "trainer_log_history.csv"
    live_path = run_dir / "training_metrics_live.jsonl"
    token_path = run_dir / "token_stats.csv"
    dropped_path = run_dir / "dropped_by_length.csv"

    hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    live = pd.read_json(live_path, lines=True) if live_path.exists() and live_path.stat().st_size else pd.DataFrame()
    save_plot(hist, "step", "loss", "Training loss by step", plots / "train_loss_by_step.png")
    save_plot(hist, "step", "eval_loss", "Validation loss by step", plots / "eval_loss_by_step.png")
    if "eval_loss" in hist.columns:
        pp = hist[["step", "eval_loss"]].dropna().copy()
        pp["step"] = pd.to_numeric(pp["step"], errors="coerce")
        pp["eval_loss"] = pd.to_numeric(pp["eval_loss"], errors="coerce")
        pp = pp.dropna()
        if not pp.empty:
            pp["eval_perplexity"] = pp["eval_loss"].apply(lambda x: math.exp(x) if x < 20 else float("inf"))
            save_plot(pp, "step", "eval_perplexity", "Validation perplexity by step", plots / "eval_perplexity_by_step.png")
    save_plot(hist, "step", "learning_rate", "Learning rate by step", plots / "learning_rate_by_step.png")
    save_plot(hist, "step", "grad_norm", "Gradient norm by step", plots / "grad_norm_by_step.png")
    if not live.empty:
        mem_cols = [c for c in ["gpu_allocated_gb", "gpu_reserved_gb", "gpu_max_allocated_gb"] if c in live.columns]
        if mem_cols:
            plt.figure(figsize=(9, 5))
            for col in mem_cols:
                plt.plot(live["step"], live[col], marker="o", linewidth=1, label=col)
            plt.title("GPU memory by step")
            plt.xlabel("step")
            plt.ylabel("GB")
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            out = plots / "gpu_memory_by_step.png"
            plt.savefig(out, dpi=200)
            plt.close()
            print(f"[plots] saved {out}")
    if token_path.exists():
        stats = pd.read_csv(token_path)
        if "n_tokens" in stats.columns and not stats.empty:
            plt.figure(figsize=(9, 5))
            plt.hist(stats["n_tokens"].dropna(), bins=40)
            plt.title("Token length distribution")
            plt.xlabel("n_tokens")
            plt.ylabel("examples")
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            out = plots / "token_length_distribution.png"
            plt.savefig(out, dpi=200)
            plt.close()
            print(f"[plots] saved {out}")
    if dropped_path.exists():
        dropped = pd.read_csv(dropped_path)
        if "class" in dropped.columns and not dropped.empty:
            by_class = dropped["class"].fillna("").replace("", "unknown").value_counts().reset_index()
            by_class.columns = ["class", "count"]
            save_plot(by_class, "class", "count", "Dropped by class", plots / "dropped_by_class.png", kind="bar")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    generate_plots(Path(args.run_dir))


if __name__ == "__main__":
    main()
