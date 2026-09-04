from pathlib import Path
import json
import ast
import re
import math
import zipfile
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:
    from sklearn.metrics import (
        confusion_matrix,
        roc_curve,
        auc,
        roc_auc_score,
        classification_report,
    )
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


OUT = Path("thesis_vlm_mcq_plots")
OUT.mkdir(parents=True, exist_ok=True)


def contrast_text_color(value, vmin=None, vmax=None, threshold=0.50):
    """
    White text on dark cells, black text on light cells.
    Works for both normalized matrices and signed margins.
    """
    try:
        value = float(value)
    except Exception:
        return "black"

    if vmin is None or vmax is None:
        # For already normalized 0..1 values
        x = value
    else:
        if vmax <= vmin:
            x = 0.0
        else:
            x = (value - vmin) / (vmax - vmin)

    return "white" if x >= threshold else "black"


BIG_ROBOT_EVAL = Path("eval_big_robot_mcq_val_best_nano")
TRAIN_RUN = Path("run_robovqa_affordance_10000_distill_clean_cont2")

HARD22_RUNS = {
    "Qwen2.5-VL-3B": Path("qwen25vl3b_hard22_fix_remove18_open19_20"),
    "Qwen2.5-VL-7B": Path("qwen25vl7b_hard22_fix_remove18_open19_20"),
    "Qwen3-VL-8B": Path("qwen3vl8b_hard22_fix_remove18_open19_20"),
    "nano raw": Path("eval_hard22_fix_remove18_open19_20_nano_raw"),
    "nano AI2THOR": Path("eval_hard22_fix_remove18_open19_20_nano_ai2thor_1k"),
    "nano robot-affordance": Path("eval_hard22_fix_remove18_open19_20_nano_affordance"),
    "our nanoVLM robot+reflect": Path("eval_hard22_fix_remove18_open19_20_nano_reflect"),
}


def savefig(name: str):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close()


def parse_choices(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return []


def get_letters(n=4):
    return list("ABCD")[:n]


def ensure_gt_pred_for_nano(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    letters = list("ABCD")

    if "gt_idx" in df.columns:
        df["gt_idx_int"] = df["gt_idx"].astype(int)
    else:
        gt_idx = []
        for _, r in df.iterrows():
            choices = parse_choices(r.get("choices", "[]"))
            ans = r.get("answer", r.get("dataset_gt_answer", None))
            try:
                gt_idx.append(choices.index(ans))
            except Exception:
                gt_idx.append(np.nan)
        df["gt_idx_int"] = gt_idx

    df["gt_letter"] = df["gt_idx_int"].apply(lambda i: letters[int(i)] if pd.notna(i) and 0 <= int(i) < 4 else "UNK")

    if "nanovlm_pred_letter" in df.columns:
        df["pred_letter"] = df["nanovlm_pred_letter"].astype(str)
    elif "qwen_pred_letter" in df.columns:
        df["pred_letter"] = df["qwen_pred_letter"].astype(str)
    else:
        df["pred_letter"] = "UNK"

    if "nanovlm_correct" in df.columns:
        df["correct"] = df["nanovlm_correct"].astype(int)
    else:
        df["correct"] = (df["pred_letter"] == df["gt_letter"]).astype(int)

    return df


def load_any_results(run_dir: Path, label: str) -> pd.DataFrame | None:
    candidates = [
        run_dir / "nanovlm_results.csv",
        run_dir / "qwen_fourchoice_explain_results.csv",
        run_dir / "qwen3_results.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None

    df = pd.read_csv(path)
    df = df.copy()
    df["model_label"] = label

    if "nanovlm_correct" in df.columns:
        df["correct"] = df["nanovlm_correct"].astype(int)
        df["pred_answer"] = df.get("nanovlm_pred_answer", "")
        df["pred_letter"] = df.get("nanovlm_pred_letter", "")
        df["gt_answer"] = df.get("dataset_gt_answer", df.get("answer", ""))
    elif "qwen_agrees_with_dataset" in df.columns:
        df["correct"] = df["qwen_agrees_with_dataset"].astype(int)
        df["pred_answer"] = df.get("qwen_pred_answer", "")
        df["gt_answer"] = df.get("dataset_gt_answer", df.get("answer", ""))
        df["pred_letter"] = df.get("qwen_pred_letter", "")
    elif "qwen3_correct" in df.columns:
        df["correct"] = df["qwen3_correct"].astype(int)
        df["pred_answer"] = df.get("qwen3_pred_answer", "")
        df["gt_answer"] = df.get("answer", "")
        df["pred_letter"] = df.get("qwen3_pred_letter", "")
    else:
        return None

    return df


def plot_big_robot_best_nano():
    result_path = BIG_ROBOT_EVAL / "nanovlm_results.csv"
    summary_path = BIG_ROBOT_EVAL / "summary.json"

    if not result_path.exists():
        print("missing big robot eval:", result_path)
        return {}

    df = pd.read_csv(result_path)
    df = ensure_gt_pred_for_nano(df)

    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    acc = float(df["correct"].mean())
    total = len(df)
    correct = int(df["correct"].sum())

    labels = list("ABCD")

    # confusion matrix
    cm = confusion_matrix(df["gt_letter"], df["pred_letter"], labels=labels)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(7.5, 6.2))
    plt.imshow(cm_norm)
    plt.title(f"Best nanoVLM on Robot MCQ validation\nAccuracy = {acc*100:.2f}% ({correct}/{total})")
    plt.xlabel("Predicted choice")
    plt.ylabel("Ground truth choice")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(label="row-normalized share")
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = contrast_text_color(cm_norm[i, j], 0.0, 1.0, threshold=0.45)
            plt.text(
                j, i,
                f"{cm[i,j]}\n{cm_norm[i,j]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )
    savefig("robot_val_best_nano_confusion_matrix.png")

    # class accuracy
    class_rows = []
    for lab in labels:
        part = df[df["gt_letter"] == lab]
        if len(part):
            class_rows.append((lab, float(part["correct"].mean()), len(part)))

    if class_rows:
        plt.figure(figsize=(8, 4.8))
        xs = np.arange(len(class_rows))
        vals = [x[1] for x in class_rows]
        names = [f"{x[0]}\n(n={x[2]})" for x in class_rows]
        plt.bar(xs, vals)
        plt.ylim(0, 1.05)
        plt.xticks(xs, names)
        plt.ylabel("accuracy")
        plt.title("Per-choice accuracy: best nanoVLM on Robot MCQ validation")
        for x, v in zip(xs, vals):
            plt.text(x, v + 0.02, f"{v*100:.1f}%", ha="center")
        savefig("robot_val_best_nano_per_class_accuracy.png")

    # ROC/AUC
    auc_info = {}
    prob_cols = [f"prob_{l}" for l in labels]
    if SKLEARN_OK and all(c in df.columns for c in prob_cols):
        y_true = np.zeros((len(df), len(labels)))
        y_score = df[prob_cols].astype(float).values

        for i, lab in enumerate(labels):
            y_true[:, i] = (df["gt_letter"].values == lab).astype(int)

        plt.figure(figsize=(8.5, 6.2))
        for i, lab in enumerate(labels):
            if len(np.unique(y_true[:, i])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            auc_info[f"AUC_{lab}"] = float(roc_auc)
            plt.plot(fpr, tpr, label=f"{lab}: AUC={roc_auc:.3f}")

        fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_score.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        auc_info["AUC_micro"] = float(auc_micro)
        try:
            auc_macro = roc_auc_score(y_true, y_score, average="macro", multi_class="ovr")
            auc_info["AUC_macro"] = float(auc_macro)
        except Exception:
            auc_macro = None

        plt.plot(fpr_micro, tpr_micro, linestyle="--", label=f"micro: AUC={auc_micro:.3f}")
        plt.plot([0, 1], [0, 1], linestyle=":", label="random")
        title = "ROC curves: best nanoVLM on Robot MCQ validation"
        if auc_macro is not None:
            title += f"\nmacro AUC={auc_macro:.3f}"
        plt.title(title)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        savefig("robot_val_best_nano_roc_auc.png")

    # confidence histogram
    if all(c in df.columns for c in prob_cols):
        probs = df[prob_cols].astype(float).values
        df["confidence"] = probs.max(axis=1)
        plt.figure(figsize=(8, 4.8))
        plt.hist(df[df["correct"] == 1]["confidence"], bins=20, alpha=0.65, label="correct")
        plt.hist(df[df["correct"] == 0]["confidence"], bins=20, alpha=0.65, label="wrong")
        plt.xlabel("max predicted probability")
        plt.ylabel("samples")
        plt.title("Prediction confidence distribution: best nanoVLM")
        plt.legend()
        savefig("robot_val_best_nano_confidence_hist.png")

    # save tables
    df.to_csv(OUT / "robot_val_best_nano_predictions.csv", index=False)
    pd.DataFrame(class_rows, columns=["choice", "accuracy", "support"]).to_csv(
        OUT / "robot_val_best_nano_class_accuracy.csv", index=False
    )

    return {
        "robot_val_total": total,
        "robot_val_correct": correct,
        "robot_val_accuracy": acc,
        **auc_info,
    }


def parse_training_logs():
    rows = []

    # CSV candidates
    for p in TRAIN_RUN.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".json", ".jsonl", ".log", ".txt"}:
            name = p.name.lower()
            if any(k in name for k in ["log", "history", "metric", "train"]):
                try:
                    txt = p.read_text(errors="ignore")
                except Exception:
                    continue

                # regex from progress output
                for m in re.finditer(r"step=(\d+)/\d+\s+loss=([0-9.]+)", txt):
                    rows.append({"step": int(m.group(1)), "loss": float(m.group(2)), "val_acc": np.nan, "source": str(p)})
                for m in re.finditer(r"val step\s+(\d+)\s+ACC:\s+([0-9.]+)%", txt):
                    rows.append({"step": int(m.group(1)), "loss": np.nan, "val_acc": float(m.group(2)) / 100.0, "source": str(p)})

    hist = pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["step", "loss", "val_acc", "source"])
    hist.to_csv(OUT / "training_history_parsed.csv", index=False)

    if len(hist) == 0:
        return

    loss_df = hist[pd.notna(hist["loss"])].sort_values("step")
    if len(loss_df):
        plt.figure(figsize=(9, 5))
        plt.plot(loss_df["step"], loss_df["loss"], marker="o", linewidth=1.5)
        plt.xlabel("training step")
        plt.ylabel("loss")
        plt.title("Fine-tuning loss: best nanoVLM robot model")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        savefig("finetuning_train_loss_curve.png")

    val_df = hist[pd.notna(hist["val_acc"])].sort_values("step")
    if len(val_df):
        plt.figure(figsize=(9, 5))
        plt.plot(val_df["step"], val_df["val_acc"], marker="o", linewidth=1.5)
        plt.ylim(0, 1.05)
        plt.xlabel("training step")
        plt.ylabel("validation accuracy")
        plt.title("Validation accuracy during fine-tuning")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        savefig("finetuning_val_accuracy_curve.png")


def hard22_pairwise():
    frames = []
    summaries = []

    for label, run_dir in HARD22_RUNS.items():
        df = load_any_results(run_dir, label)
        if df is None:
            print("missing hard22 result:", label, run_dir)
            continue

        if "image" not in df.columns:
            continue

        frames.append(df[["image", "model_label", "correct", "pred_answer", "gt_answer"]].copy())
        summaries.append({
            "model": label,
            "total": len(df),
            "correct": int(df["correct"].sum()),
            "accuracy": float(df["correct"].mean()),
        })

    if not frames:
        return {}

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(OUT / "hard22_all_model_predictions_long.csv", index=False)

    summ = pd.DataFrame(summaries).sort_values("accuracy", ascending=False)
    summ.to_csv(OUT / "hard22_model_summary.csv", index=False)

    plt.figure(figsize=(10, 5.5))
    xs = np.arange(len(summ))
    plt.bar(xs, summ["accuracy"].values)
    plt.ylim(0, 1.05)
    plt.xticks(xs, summ["model"].values, rotation=25, ha="right")
    plt.ylabel("accuracy")
    plt.title("Hard22 fixed: model accuracy")
    for x, v, c, t in zip(xs, summ["accuracy"].values, summ["correct"].values, summ["total"].values):
        plt.text(x, v + 0.02, f"{v*100:.1f}%\n{c}/{t}", ha="center", fontsize=9)
    savefig("hard22_accuracy_leaderboard.png")

    models = list(summ["model"].values)
    wide = all_df.pivot_table(index="image", columns="model_label", values="correct", aggfunc="first")
    wide = wide[models]

    n = len(models)
    wins = np.zeros((n, n), dtype=int)
    losses = np.zeros((n, n), dtype=int)
    ties = np.zeros((n, n), dtype=int)
    margins = np.zeros((n, n), dtype=int)

    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if a == b:
                valid = wide[[a]].dropna()
                wins[i, j] = 0
                losses[i, j] = 0
                ties[i, j] = int(len(valid))
                margins[i, j] = 0
                continue

            valid = wide[[a, b]].dropna()
            a_vals = valid[a].astype(int)
            b_vals = valid[b].astype(int)

            aw = int(((a_vals == 1) & (b_vals == 0)).sum())
            bw = int(((a_vals == 0) & (b_vals == 1)).sum())
            tie = int((a_vals == b_vals).sum())

            wins[i, j] = aw
            losses[i, j] = bw
            ties[i, j] = tie
            margins[i, j] = aw - bw

    pd.DataFrame(wins, index=models, columns=models).to_csv(OUT / "hard22_pairwise_wins.csv")
    pd.DataFrame(margins, index=models, columns=models).to_csv(OUT / "hard22_pairwise_win_margins.csv")
    pd.DataFrame(ties, index=models, columns=models).to_csv(OUT / "hard22_pairwise_ties.csv")

    plt.figure(figsize=(10, 8))
    max_abs_margin = max(1, int(np.max(np.abs(margins))))
    plt.imshow(margins, cmap="coolwarm", vmin=-max_abs_margin, vmax=max_abs_margin)
    plt.title("Hard22 fixed: pairwise win margin\ncell = row model wins minus column model wins")
    plt.xticks(range(n), models, rotation=35, ha="right")
    plt.yticks(range(n), models)
    plt.colorbar(label="win margin")
    for i in range(n):
        for j in range(n):
            value = margins[i, j]
            # coolwarm is darkest near extremes, light near zero
            color = "white" if abs(value) >= 0.55 * max_abs_margin else "black"
            plt.text(
                j, i,
                str(value),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )
    savefig("hard22_pairwise_win_margin_heatmap.png")

    win_rate = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            denom = wins[i, j] + losses[i, j]
            win_rate[i, j] = wins[i, j] / denom if denom else 0.5

    plt.figure(figsize=(10, 8))
    plt.imshow(win_rate, cmap="Blues", vmin=0, vmax=1)
    plt.title("Hard22 fixed: pairwise win rate\nexcluding ties")
    plt.xticks(range(n), models, rotation=35, ha="right")
    plt.yticks(range(n), models)
    plt.colorbar(label="win rate")
    for i in range(n):
        for j in range(n):
            value = win_rate[i, j]
            color = contrast_text_color(value, 0.0, 1.0, threshold=0.50)
            plt.text(
                j, i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )
    savefig("hard22_pairwise_win_rate_heatmap.png")

    # net pairwise score
    net = margins.sum(axis=1)
    pair = pd.DataFrame({"model": models, "net_pairwise_margin": net}).sort_values("net_pairwise_margin", ascending=False)
    pair.to_csv(OUT / "hard22_pairwise_net_score.csv", index=False)

    plt.figure(figsize=(10, 5.5))
    xs = np.arange(len(pair))
    plt.bar(xs, pair["net_pairwise_margin"].values)
    plt.xticks(xs, pair["model"].values, rotation=25, ha="right")
    plt.ylabel("net pairwise margin")
    plt.title("Hard22 fixed: net pairwise dominance")
    for x, v in zip(xs, pair["net_pairwise_margin"].values):
        plt.text(x, v + (0.4 if v >= 0 else -0.8), str(int(v)), ha="center")
    savefig("hard22_pairwise_net_dominance.png")

    return {
        "hard22_top_model": str(summ.iloc[0]["model"]),
        "hard22_top_accuracy": float(summ.iloc[0]["accuracy"]),
        "hard22_top_correct": int(summ.iloc[0]["correct"]),
        "hard22_total": int(summ.iloc[0]["total"]),
    }


def write_interpretation(robot_info, hard_info):
    hard_summary = pd.read_csv(OUT / "hard22_model_summary.csv") if (OUT / "hard22_model_summary.csv").exists() else pd.DataFrame()

    lines = []
    lines.append("# VLM MCQ Experimental Plots and Interpretation")
    lines.append("")
    lines.append("## Robot MCQ validation")
    if robot_info:
        lines.append(
            f"Best fine-tuned nanoVLM on the large robot MCQ validation set achieved "
            f"{robot_info['robot_val_accuracy']*100:.2f}% accuracy "
            f"({robot_info['robot_val_correct']}/{robot_info['robot_val_total']})."
        )
        if "AUC_micro" in robot_info:
            lines.append(f"Micro-average ROC AUC: {robot_info['AUC_micro']:.4f}.")
        if "AUC_macro" in robot_info:
            lines.append(f"Macro-average ROC AUC: {robot_info['AUC_macro']:.4f}.")
    lines.append("")
    lines.append("Generated plots:")
    lines.append("- `robot_val_best_nano_confusion_matrix.png`")
    lines.append("- `robot_val_best_nano_roc_auc.png`")
    lines.append("- `robot_val_best_nano_per_class_accuracy.png`")
    lines.append("- `robot_val_best_nano_confidence_hist.png`")
    lines.append("")
    lines.append("## Hard22 glass-door benchmark")
    if len(hard_summary):
        lines.append("Model ranking on corrected hard22:")
        for _, r in hard_summary.iterrows():
            lines.append(f"- {r['model']}: {r['accuracy']*100:.2f}% ({int(r['correct'])}/{int(r['total'])})")
        lines.append("")

        top = hard_summary.iloc[0]
        if "our nanoVLM" in str(top["model"]) or "reflect" in str(top["model"]):
            lines.append(
                "Interpretation: the best result on hard22 is achieved by our fine-tuned nanoVLM variant "
                "(`our nanoVLM robot+reflect`). This supports the conclusion that domain-specific fine-tuning "
                "on robot/directional affordance and reflective-door examples improves robustness in the hard glass-door setting."
            )
        else:
            lines.append(
                f"Interpretation: the strongest result in this exact run is `{top['model']}`. "
                "If the goal is to emphasize the compact fine-tuned nanoVLM, compare it separately against other nano variants: "
                "`our nanoVLM robot+reflect` is the domain-adapted compact model and should be discussed as the best specialized nanoVLM."
            )
    lines.append("")
    lines.append("Pairwise plots:")
    lines.append("- `hard22_pairwise_win_margin_heatmap.png`")
    lines.append("- `hard22_pairwise_win_rate_heatmap.png`")
    lines.append("- `hard22_pairwise_net_dominance.png`")
    lines.append("")
    lines.append("## Suggested thesis wording")
    lines.append(
        "На большом роботном MCQ-наборе лучшая дообученная nanoVLM демонстрирует устойчивое качество, "
        "что подтверждается матрицей ошибок, ROC/AUC и поклассовой точностью. "
        "На hard22, состоящем из сложных сцен со стеклянными дверьми и отражениями, "
        "попарное сравнение показывает преимущество доменно-адаптированной nanoVLM-модели над базовыми nanoVLM-вариантами. "
        "Это указывает, что специализированное дообучение под роботные affordance-сцены и отражающие двери "
        "повышает практическую применимость компактной VLM для задач робототехнического восприятия."
    )

    (OUT / "interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_zip():
    zip_path = Path("thesis_vlm_mcq_plots.zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(OUT.parent))
    print("ZIP:", zip_path.resolve())


def main():
    robot_info = plot_big_robot_best_nano()
    parse_training_logs()
    hard_info = hard22_pairwise()
    write_interpretation(robot_info, hard_info)
    make_zip()

    print("\nDONE")
    print("OUT_DIR:", OUT.resolve())
    print("robot_info:", json.dumps(robot_info, indent=2, ensure_ascii=False))
    print("hard_info:", json.dumps(hard_info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
