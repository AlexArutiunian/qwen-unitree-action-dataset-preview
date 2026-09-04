import json
from pathlib import Path

import pandas as pd


RUNS = [
    {
        "name": "Qwen 3B | mixed v2",
        "dir": "qwen_validate_robovqa_100_v2",
        "type": "qwen",
        "dataset": "mixed_v2",
    },
    {
        "name": "Qwen 3B | affordance only",
        "dir": "qwen_validate_robovqa_100_affordance_only",
        "type": "qwen",
        "dataset": "affordance_only",
    },
    {
        "name": "nano A-OKVQA base | mixed v2",
        "dir": "nanovlm_eval_robovqa_100_v2_aokvqa_base",
        "type": "nano",
        "dataset": "mixed_v2",
    },
    {
        "name": "nano A-OKVQA base | affordance only",
        "dir": "nanovlm_eval_robovqa_100_affordance_only_aokvqa_base",
        "type": "nano",
        "dataset": "affordance_only",
    },
    {
        "name": "nano AI2-THOR best | mixed v2",
        "dir": "nanovlm_eval_robovqa_100_v2_ai2thor_best",
        "type": "nano",
        "dataset": "mixed_v2",
    },
]


def read_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_task_accuracy(run_dir):
    p = run_dir / "task_accuracy.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


rows = []
task_rows = []

for r in RUNS:
    run_dir = Path(r["dir"])
    summary = read_json(run_dir / "summary.json")
    task_df = read_task_accuracy(run_dir)

    if not summary:
        print(f"MISS summary: {run_dir}")
        continue

    # Qwen uses agreement_percent; nano uses accuracy_percent
    overall = summary.get("agreement_percent", summary.get("accuracy_percent", None))
    total = summary.get("total_validated", summary.get("total", None))
    correct = summary.get("qwen_agrees", summary.get("correct", None))
    invalid = summary.get("invalid_or_errors", summary.get("invalid", None))

    rows.append({
        "run": r["name"],
        "model_type": r["type"],
        "dataset": r["dataset"],
        "total": total,
        "correct_or_agree": correct,
        "invalid": invalid,
        "overall_percent": overall,
        "latency_avg_sec": summary.get("latency_avg_sec", None),
        "latency_p50_sec": summary.get("latency_p50_sec", None),
        "latency_p95_sec": summary.get("latency_p95_sec", None),
        "dir": str(run_dir),
    })

    if len(task_df):
        for _, tr in task_df.iterrows():
            # Qwen column: agreement_percent; nano column: accuracy_percent
            val = tr.get("agreement_percent", tr.get("accuracy_percent", None))
            c = tr.get("agree", tr.get("correct", None))
            task_rows.append({
                "run": r["name"],
                "model_type": r["type"],
                "dataset": r["dataset"],
                "task": tr.get("task", ""),
                "total": int(tr.get("total", 0)),
                "correct_or_agree": int(c) if pd.notna(c) else None,
                "percent": float(val) if pd.notna(val) else None,
            })


summary_df = pd.DataFrame(rows)
task_summary_df = pd.DataFrame(task_rows)

print("\n" + "=" * 100)
print("OVERALL COMPARISON")
print("=" * 100)
if len(summary_df):
    print(summary_df[[
        "run", "dataset", "total", "correct_or_agree", "invalid",
        "overall_percent", "latency_avg_sec", "latency_p50_sec"
    ]].to_string(index=False))

print("\n" + "=" * 100)
print("TASK COMPARISON")
print("=" * 100)
if len(task_summary_df):
    print(task_summary_df[[
        "run", "dataset", "task", "total", "correct_or_agree", "percent"
    ]].to_string(index=False))

    pivot = task_summary_df.pivot_table(
        index=["dataset", "task"],
        columns="run",
        values="percent",
        aggfunc="first",
    )

    print("\n" + "=" * 100)
    print("PIVOT: percent by task")
    print("=" * 100)
    print(pivot.round(2).to_string())

summary_df.to_csv("robovqa_eval_overall_comparison.csv", index=False)
task_summary_df.to_csv("robovqa_eval_task_comparison.csv", index=False)

print("\nSaved:")
print("  robovqa_eval_overall_comparison.csv")
print("  robovqa_eval_task_comparison.csv")
