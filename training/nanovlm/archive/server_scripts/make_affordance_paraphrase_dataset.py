#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd


SAFE_TEMPLATES = [
    # 0. Original-like canonical form
    "Is this action possible right now: {instr}?",

    # 1. Direct robot affordance
    "Can the robot do this right now: {instr}?",

    # 2. General possibility
    "Is it possible to do this now: {instr}?",

    # 3. Action performance in current scene
    "Can this action be performed in the current scene: {instr}?",

    # 4. Current visual situation
    "Does the current scene allow this action: {instr}?",
]


def clean_instruction(x: str) -> str:
    s = str(x).strip()

    # If instruction accidentally includes full question, cut after colon.
    if s.lower().startswith("is this action possible") and ":" in s:
        s = s.split(":", 1)[1]

    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(" ?.!").strip()

    # Keep original capitalization if present, but remove weird trailing punctuation.
    return s


def infer_instruction(row):
    if "instruction" in row and pd.notna(row["instruction"]) and str(row["instruction"]).strip():
        return clean_instruction(row["instruction"])

    q = str(row["question"]).strip()
    if ":" in q:
        return clean_instruction(q.split(":", 1)[1])

    return clean_instruction(q)


def validate_row(row):
    required = ["image", "question", "choices", "gt_idx", "answer"]
    for c in required:
        if c not in row:
            return False, f"missing {c}"

    instr = infer_instruction(row)
    if not instr:
        return False, "empty instruction"

    answer = str(row["answer"]).strip().lower()
    if answer not in {"yes", "no"}:
        return False, f"bad answer {answer}"

    try:
        gt_idx = int(row["gt_idx"])
        if gt_idx < 0 or gt_idx > 3:
            return False, f"bad gt_idx {gt_idx}"
    except Exception:
        return False, "bad gt_idx"

    return True, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--metadata", default="metadata.csv")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--max-templates", type=int, default=5)
    parser.add_argument("--keep-original-id", action="store_true")
    args = parser.parse_args()

    src = Path(args.source_dir)
    out = Path(args.out_dir)

    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src / args.metadata)

    templates = SAFE_TEMPLATES[:args.max_templates]

    rows = []
    skipped = []

    for idx, row in df.iterrows():
        row = row.to_dict()

        ok, reason = validate_row(row)
        if not ok:
            skipped.append((idx, reason))
            continue

        instr = infer_instruction(row)

        for t_i, tmpl in enumerate(templates):
            new_row = dict(row)

            new_row["question"] = tmpl.format(instr=instr)
            new_row["instruction"] = instr
            new_row["task"] = "affordance_possible"
            new_row["paraphrase_template_id"] = t_i
            new_row["paraphrase_template"] = tmpl
            new_row["source_row_index"] = idx
            new_row["source_id"] = row.get("id", f"src_{idx:06d}")

            if args.keep_original_id and t_i == 0:
                new_row["id"] = row.get("id", f"src_{idx:06d}")
            else:
                base_id = row.get("id", f"src_{idx:06d}")
                new_row["id"] = f"{base_id}_para{t_i}"

            rows.append(new_row)

    out_df = pd.DataFrame(rows)

    # Remove exact duplicates just in case.
    before = len(out_df)
    out_df = out_df.drop_duplicates(
        subset=["image", "question", "choices", "gt_idx", "answer"]
    ).reset_index(drop=True)
    after = len(out_df)

    out_df.to_csv(out / "metadata.csv", index=False)

    with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in out_df.to_dict(orient="records"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    copied = 0
    if args.copy_images:
        for p in out_df["image"].dropna().unique():
            sp = src / p
            dp = out / p
            dp.parent.mkdir(parents=True, exist_ok=True)
            if sp.exists() and not dp.exists():
                shutil.copy2(sp, dp)
                copied += 1
    else:
        # symlink images dir when possible, avoids copying 5k images again.
        # If symlink already exists or fails, user can pass --copy-images.
        link = out / "images"
        if link.exists() and not any(link.iterdir()):
            try:
                link.rmdir()
                link.symlink_to((src / "images").resolve(), target_is_directory=True)
                print("Symlinked images:", link, "->", (src / "images").resolve())
            except Exception as e:
                print("Could not symlink images:", e)
                print("Use --copy-images if needed.")

    print("\nDONE")
    print("source rows:", len(df))
    print("templates:", len(templates))
    print("augmented rows before dedup:", before)
    print("augmented rows after dedup:", after)
    print("skipped:", len(skipped))
    print("copied images:", copied)
    print("out:", out)

    print("\nAnswer counts:")
    print(out_df["answer"].value_counts().to_string())

    print("\nTemplate counts:")
    print(out_df["paraphrase_template_id"].value_counts().sort_index().to_string())

    print("\nUnique questions:", out_df["question"].nunique())
    print("Unique instructions:", out_df["instruction"].astype(str).str.lower().str.strip().nunique())

    print("\nPreview:")
    cols = ["id", "image", "question", "answer", "gt_idx", "paraphrase_template_id"]
    print(out_df[cols].head(15).to_string(index=False))

    if skipped[:10]:
        print("\nSkipped examples:")
        for x in skipped[:10]:
            print(x)


if __name__ == "__main__":
    main()
