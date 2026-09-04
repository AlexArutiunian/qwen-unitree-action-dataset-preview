from pathlib import Path
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score

OUT = Path("thesis_vlm_mcq_plots")
OUT.mkdir(parents=True, exist_ok=True)

ROBOT_CSV = Path("eval_big_robot_mcq_val_best_nano/nanovlm_results.csv")
LETTERS = ["A", "B", "C", "D"]
THRESHOLD = 0.5

def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close()

def parse_choices(x):
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return []

def text_color(value):
    return "white" if float(value) >= 0.45 else "black"

def load_and_expand():
    df = pd.read_csv(ROBOT_CSV)

    if "gt_idx" in df.columns:
        df["gt_idx_int"] = df["gt_idx"].astype(int)
    else:
        gt = []
        for _, r in df.iterrows():
            choices = parse_choices(r.get("choices", "[]"))
            ans = r.get("answer", r.get("dataset_gt_answer", None))
            gt.append(choices.index(ans) if ans in choices else -1)
        df["gt_idx_int"] = gt

    rows = []
    for _, r in df.iterrows():
        image = r.get("image", "")
        gt_idx = int(r["gt_idx_int"])

        for i, letter in enumerate(LETTERS):
            prob_col = f"prob_{letter}"
            if prob_col not in df.columns:
                raise RuntimeError(f"Missing column: {prob_col}")

            rows.append({
                "image": image,
                "candidate_letter": letter,
                "is_gt_pair": int(i == gt_idx),
                "score": float(r[prob_col]),
                "pred_positive": int(float(r[prob_col]) >= THRESHOLD),
                "sample_pred_letter": r.get("nanovlm_pred_letter", ""),
                "sample_correct": int(r.get("nanovlm_correct", 0)),
            })

    pair_df = pd.DataFrame(rows)
    pair_df.to_csv(OUT / "robot_val_mcq_pairwise_binary_expanded.csv", index=False)
    return pair_df

def draw_pairwise_cm(pair_df):
    y_true = pair_df["is_gt_pair"].astype(int).values
    y_pred = pair_df["pred_positive"].astype(int).values

    # labels=[1,0] gives:
    # [[TP, FN],
    #  [FP, TN]]
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn = int(cm[0, 0]), int(cm[0, 1])
    fp, tn = int(cm[1, 0]), int(cm[1, 1])

    total = tp + fn + fp + tn
    acc = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    mat = np.array([[tp, fn], [fp, tn]], dtype=float)
    norm = mat / max(total, 1)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_title(
        f"nanoVLM MCQ pairwise confusion matrix\nthreshold={THRESHOLD}, acc={acc:.3f}, F1={f1:.3f}",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Positive", "Negative"], fontsize=13)
    ax.set_yticklabels(["Positive (GT pair)", "Negative"], fontsize=13)

    ax.set_xlabel("Predicted label", fontsize=13)
    ax.set_ylabel("True label", fontsize=13)

    names = [["TP", "FN"], ["FP", "TN"]]
    vals = [[tp, fn], [fp, tn]]

    for i in range(2):
        for j in range(2):
            share = vals[i][j] / max(total, 1)
            ax.text(
                j, i,
                f"{names[i][j]}\n{vals[i][j]}\n{share:.1%}",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color=text_color(norm[i, j]),
            )

    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")

    savefig("robot_val_mcq_pairwise_confusion_2x2.png")

    metrics = {
        "threshold": THRESHOLD,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_pairs": int((y_true == 1).sum()),
        "negative_pairs": int((y_true == 0).sum()),
        "total_pairs": int(total),
    }
    pd.DataFrame([metrics]).to_csv(OUT / "robot_val_mcq_pairwise_2x2_metrics.csv", index=False)
    print(metrics)

def draw_pairwise_roc(pair_df):
    y_true = pair_df["is_gt_pair"].astype(int).values
    scores = pair_df["score"].astype(float).values

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.plot(fpr, tpr, linewidth=2.5, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="random")

    ax.set_title(
        f"nanoVLM MCQ ROC, pairwise scoring\nAUC={roc_auc:.3f}, AP={ap:.3f}",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="lower right", fontsize=12)

    savefig("robot_val_mcq_pairwise_roc_auc.png")

    pd.DataFrame([{
        "roc_auc": roc_auc,
        "average_precision": ap,
        "threshold": THRESHOLD,
    }]).to_csv(OUT / "robot_val_mcq_pairwise_auc.csv", index=False)

def main():
    pair_df = load_and_expand()
    draw_pairwise_cm(pair_df)
    draw_pairwise_roc(pair_df)

    print("\nDONE")
    print("created:")
    for p in [
        "robot_val_mcq_pairwise_confusion_2x2.png",
        "robot_val_mcq_pairwise_roc_auc.png",
        "robot_val_mcq_pairwise_2x2_metrics.csv",
        "robot_val_mcq_pairwise_auc.csv",
        "robot_val_mcq_pairwise_binary_expanded.csv",
    ]:
        print(" ", OUT / p)

if __name__ == "__main__":
    main()
