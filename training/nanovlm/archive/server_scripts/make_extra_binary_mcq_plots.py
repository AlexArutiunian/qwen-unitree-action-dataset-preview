from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import ast

OUT = Path("thesis_vlm_mcq_plots")
OUT.mkdir(parents=True, exist_ok=True)

ROBOT_EVAL = Path("eval_big_robot_mcq_val_best_nano/nanovlm_results.csv")
HARD22_SUMMARY = Path("thesis_vlm_mcq_plots/hard22_model_summary.csv")
HARD22_LONG = Path("thesis_vlm_mcq_plots/hard22_all_model_predictions_long.csv")

LETTERS = ["A", "B", "C", "D"]

def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

def parse_choices(x):
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return []

def prepare_robot_df():
    df = pd.read_csv(ROBOT_EVAL)

    if "gt_idx" in df.columns:
        df["gt_idx_int"] = df["gt_idx"].astype(int)
    else:
        gt = []
        for _, r in df.iterrows():
            choices = parse_choices(r.get("choices", "[]"))
            ans = r.get("answer", r.get("dataset_gt_answer", None))
            gt.append(choices.index(ans) if ans in choices else -1)
        df["gt_idx_int"] = gt

    df["gt_letter"] = df["gt_idx_int"].apply(lambda i: LETTERS[int(i)] if 0 <= int(i) < 4 else "UNK")
    df["pred_letter"] = df["nanovlm_pred_letter"].astype(str)
    df["correct"] = (df["gt_letter"] == df["pred_letter"]).astype(int)

    for l in LETTERS:
        col = f"prob_{l}"
        if col not in df.columns:
            raise RuntimeError(f"Missing probability column: {col}")

    return df

def draw_binary_cm(ax, cm, title, subtitle=""):
    # cm layout from sklearn labels=[1,0]:
    # [[TP, FN],
    #  [FP, TN]]
    total = cm.sum()
    norm = cm / max(total, 1)

    ax.imshow(norm, vmin=0, vmax=1)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Positive", "Negative"], fontsize=10)
    ax.set_yticklabels(["Positive", "Negative"], fontsize=10)

    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_title(title + ("\n" + subtitle if subtitle else ""), fontsize=12, pad=10)

    for i in range(2):
        for j in range(2):
            value = int(cm[i, j])
            share = norm[i, j]
            color = "white" if share > 0.45 else "black"
            ax.text(j, i, f"{value}\n{share:.2f}", ha="center", va="center", fontsize=12, color=color)

    ax.grid(False)

def robot_binary_confusions():
    df = prepare_robot_df()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    rows = []
    for k, letter in enumerate(LETTERS):
        y_true = (df["gt_letter"] == letter).astype(int)
        y_pred = (df["pred_letter"] == letter).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        tp, fn = cm[0, 0], cm[0, 1]
        fp, tn = cm[1, 0], cm[1, 1]

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        acc = (tp + tn) / max(cm.sum(), 1)

        rows.append({
            "class": letter,
            "TP": int(tp),
            "FN": int(fn),
            "FP": int(fp),
            "TN": int(tn),
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        subtitle = f"acc={acc:.3f}, F1={f1:.3f}"
        draw_binary_cm(axes[k], cm, f"Choice {letter} vs rest", subtitle)

        # отдельный крупный PNG для слайда
        fig_single, ax_single = plt.subplots(figsize=(6.2, 5.2))
        draw_binary_cm(ax_single, cm, f"Choice {letter} vs rest", subtitle)
        savefig(f"robot_val_binary_confusion_{letter}_vs_rest.png")

    fig.suptitle("Best nanoVLM on Robot MCQ validation: binary one-vs-rest confusion matrices", fontsize=14, y=1.02)
    savefig("robot_val_binary_confusions_one_vs_rest_grid.png")

    pd.DataFrame(rows).to_csv(OUT / "robot_val_binary_one_vs_rest_metrics.csv", index=False)

def robot_binary_roc_grid():
    df = prepare_robot_df()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    auc_rows = []
    for k, letter in enumerate(LETTERS):
        y_true = (df["gt_letter"] == letter).astype(int).values
        y_score = df[f"prob_{letter}"].astype(float).values

        ax = axes[k]
        if len(np.unique(y_true)) < 2:
            ax.set_title(f"Choice {letter}: ROC unavailable")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        auc_rows.append({"class": letter, "auc": roc_auc})

        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="random")
        ax.set_title(f"Choice {letter} vs rest", fontsize=12)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="lower right")

        fig_single, ax_single = plt.subplots(figsize=(6.2, 5.2))
        ax_single.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
        ax_single.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="random")
        ax_single.set_title(f"ROC: choice {letter} vs rest", fontsize=13)
        ax_single.set_xlabel("False Positive Rate")
        ax_single.set_ylabel("True Positive Rate")
        ax_single.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax_single.legend(loc="lower right")
        savefig(f"robot_val_binary_roc_{letter}_vs_rest.png")

    fig.suptitle("Best nanoVLM on Robot MCQ validation: binary ROC curves", fontsize=14, y=1.02)
    savefig("robot_val_binary_roc_one_vs_rest_grid.png")

    pd.DataFrame(auc_rows).to_csv(OUT / "robot_val_binary_one_vs_rest_auc.csv", index=False)

def hard22_pairwise_binary_per_model():
    if not HARD22_LONG.exists():
        print("skip hard22 pairwise binary: missing", HARD22_LONG)
        return

    df = pd.read_csv(HARD22_LONG)

    models = sorted(df["model_label"].unique().tolist())
    if "our nanoVLM robot+reflect" in models:
        baseline = "our nanoVLM robot+reflect"
    else:
        baseline = models[0]

    base = df[df["model_label"] == baseline][["image", "correct"]].rename(columns={"correct": "base_correct"})

    rows = []
    for model in models:
        if model == baseline:
            continue

        cur = df[df["model_label"] == model][["image", "correct"]].rename(columns={"correct": "other_correct"})
        m = base.merge(cur, on="image", how="inner")

        # Positive = our model correct. Negative = our model wrong.
        # Predicted positive = compared model correct. Это 2x2 согласия/разногласия.
        cm = confusion_matrix(m["base_correct"].astype(int), m["other_correct"].astype(int), labels=[1, 0])

        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        draw_binary_cm(
            ax,
            cm,
            f"Pairwise correctness: our nanoVLM vs {model}",
            "rows=our model, columns=compared model"
        )
        ax.set_xticklabels([f"{model}\ncorrect", f"{model}\nwrong"], fontsize=9)
        ax.set_yticklabels(["our\ncorrect", "our\nwrong"], fontsize=9)
        savefig(f"hard22_pairwise_binary_our_vs_{model.replace(' ', '_').replace('/', '_')}.png")

        both_correct = int(((m["base_correct"] == 1) & (m["other_correct"] == 1)).sum())
        our_only = int(((m["base_correct"] == 1) & (m["other_correct"] == 0)).sum())
        other_only = int(((m["base_correct"] == 0) & (m["other_correct"] == 1)).sum())
        both_wrong = int(((m["base_correct"] == 0) & (m["other_correct"] == 0)).sum())

        rows.append({
            "baseline": baseline,
            "compared_model": model,
            "both_correct": both_correct,
            "our_only_correct": our_only,
            "other_only_correct": other_only,
            "both_wrong": both_wrong,
            "our_pairwise_margin": our_only - other_only,
        })

    pd.DataFrame(rows).to_csv(OUT / "hard22_pairwise_binary_our_vs_others.csv", index=False)

def main():
    robot_binary_confusions()
    robot_binary_roc_grid()
    hard22_pairwise_binary_per_model()

    print("DONE extra binary plots")
    print("OUT:", OUT.resolve())
    print("Files:")
    for p in sorted(OUT.glob("*binary*")):
        print(" ", p)

if __name__ == "__main__":
    main()
