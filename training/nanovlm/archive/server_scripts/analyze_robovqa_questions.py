#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import pandas as pd


def normalize_text(s):
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[?.!]+$", "", s)
    return s


def question_template(q):
    q = str(q).strip()

    # Для наших affordance-вопросов:
    # Is this action possible right now: INSTRUCTION?
    if ":" in q:
        prefix = q.split(":", 1)[0].strip()
        return prefix + ": {INSTRUCTION}"

    return re.sub(r"\s+", " ", q)


def extract_instruction_from_question(q):
    q = str(q).strip()
    if ":" in q:
        return q.split(":", 1)[1].strip().rstrip("?").strip()
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--metadata", default="metadata.csv")
    parser.add_argument("--topn", type=int, default=30)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    path = dataset_dir / args.metadata

    df = pd.read_csv(path)

    print("=" * 100)
    print("DATASET:", dataset_dir)
    print("METADATA:", path)
    print("=" * 100)

    print("\nBASIC")
    print("rows:", len(df))
    print("columns:", list(df.columns))

    for col in ["task", "answer", "gt_idx"]:
        if col in df.columns:
            print(f"\n{col} counts:")
            print(df[col].value_counts(dropna=False).to_string())

    if "question" not in df.columns:
        print("No question column")
        return

    df["q_norm"] = df["question"].map(normalize_text)
    df["q_template"] = df["question"].map(question_template)

    if "instruction" in df.columns:
        df["instr_norm"] = df["instruction"].map(normalize_text)
    else:
        df["instruction"] = df["question"].map(extract_instruction_from_question)
        df["instr_norm"] = df["instruction"].map(normalize_text)

    print("\nUNIQUE COUNTS")
    print("unique question raw:", df["question"].nunique())
    print("unique question normalized:", df["q_norm"].nunique())
    print("unique instruction normalized:", df["instr_norm"].nunique())
    print("unique question templates:", df["q_template"].nunique())

    print("\nDUPLICATION")
    print("duplicate question rows:", int(df.duplicated("q_norm").sum()))
    print("duplicate instruction rows:", int(df.duplicated("instr_norm").sum()))

    print("\nQUESTION TEMPLATE COUNTS")
    print(df["q_template"].value_counts().head(args.topn).to_string())

    print("\nTOP REPEATED QUESTIONS")
    q_counts = df["q_norm"].value_counts()
    print(q_counts.head(args.topn).to_string())

    print("\nTOP REPEATED INSTRUCTIONS")
    instr_counts = df["instr_norm"].value_counts()
    print(instr_counts.head(args.topn).to_string())

    print("\nEXAMPLES OF DUPLICATED INSTRUCTIONS")
    repeated_instr = instr_counts[instr_counts > 1].head(10).index.tolist()
    for instr in repeated_instr:
        sub = df[df["instr_norm"] == instr].head(5)
        print("\n" + "-" * 100)
        print("INSTRUCTION:", instr)
        print("count:", instr_counts[instr])
        print(sub[["id", "question", "answer", "choices", "gt_idx"]].to_string(index=False))

    print("\nQUESTION LENGTH STATS")
    q_len = df["question"].astype(str).str.split().map(len)
    print(q_len.describe().to_string())

    print("\nINSTRUCTION LENGTH STATS")
    instr_len = df["instruction"].astype(str).str.split().map(len)
    print(instr_len.describe().to_string())

    out_csv = dataset_dir / "question_uniqueness_report.csv"
    report = pd.DataFrame({
        "metric": [
            "rows",
            "unique_question_raw",
            "unique_question_normalized",
            "unique_instruction_normalized",
            "unique_question_templates",
            "duplicate_question_rows",
            "duplicate_instruction_rows",
        ],
        "value": [
            len(df),
            df["question"].nunique(),
            df["q_norm"].nunique(),
            df["instr_norm"].nunique(),
            df["q_template"].nunique(),
            int(df.duplicated("q_norm").sum()),
            int(df.duplicated("instr_norm").sum()),
        ],
    })
    report.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()
