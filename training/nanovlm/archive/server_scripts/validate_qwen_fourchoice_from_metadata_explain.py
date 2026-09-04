#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def parse_choices(x):
    if isinstance(x, list):
        return x
    return ast.literal_eval(str(x))


def strip_code_fence(s):
    s = str(s).strip()
    s = re.sub(r"^```json\s*", "", s)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def parse_raw(raw):
    clean = strip_code_fence(raw)
    try:
        obj = json.loads(clean)
        ans = str(obj.get("answer", "")).strip().upper()
        exp = str(obj.get("explanation", "")).strip()
        if ans in ["A", "B", "C", "D"]:
            return ans, exp
    except Exception:
        pass

    t = clean.upper()
    m = re.search(r'"ANSWER"\s*:\s*"([ABCD])"', t)
    if m:
        return m.group(1), clean

    m = re.search(r"\b([ABCD])\b", t)
    if m:
        return m.group(1), clean

    return "", clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_dir / "metadata.csv")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print("dataset:", dataset_dir)
    print("rows:", len(df))

    print("Loading processor:", args.model)
    processor = AutoProcessor.from_pretrained(args.model)

    print("Loading model:", args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    rows = []
    correct = 0
    errors = 0

    for i, r in tqdm(df.iterrows(), total=len(df), desc="Qwen 4-choice explain"):
        try:
            image = Image.open(dataset_dir / str(r["image"])).convert("RGB")
            choices = parse_choices(r["choices"])
            assert len(choices) == 4, f"Expected 4 choices, got {len(choices)}"

            gt_idx = int(r["gt_idx"])
            gt_letter = "ABCD"[gt_idx]
            gt_answer = str(r["answer"])

            prompt = f"""
You are analyzing a camera image of a doorway.

Question:
{r["question"]}

Choices:
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Important rules:
- Choose only one letter: A, B, C, or D.
- Focus on the physical state of the actual door.
- A glass, transparent, or mirror-like door may reflect the room or show a visible room behind it.
- Reflections or seeing a room behind glass are not enough to call the door open.
- If a glass/reflective door leaf or panel physically closes the doorway, answer No.
- If there is a real open doorway/gap, answer Yes.
- Return only valid JSON.

Required JSON format:
{{"answer":"A/B/C/D", "explanation":"one short sentence explaining the visible evidence"}}
""".strip()

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            raw = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            pred_letter, explanation = parse_raw(raw)
            pred_idx = "ABCD".find(pred_letter)
            pred_answer = choices[pred_idx] if pred_idx >= 0 else ""

            ok = int(pred_letter == gt_letter)
            correct += ok

            row = r.to_dict()
            row.update({
                "dataset_gt_letter": gt_letter,
                "dataset_gt_answer": gt_answer,
                "qwen_pred_letter": pred_letter,
                "qwen_pred_answer": pred_answer,
                "qwen_explanation": explanation,
                "qwen_raw_output": raw,
                "qwen_agrees_with_dataset": ok,
            })
            rows.append(row)

        except Exception as e:
            errors += 1
            row = r.to_dict()
            row.update({
                "dataset_gt_letter": "",
                "dataset_gt_answer": str(r.get("answer", "")),
                "qwen_pred_letter": "",
                "qwen_pred_answer": "",
                "qwen_explanation": "",
                "qwen_raw_output": f"ERROR: {e}",
                "qwen_agrees_with_dataset": 0,
            })
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "qwen_fourchoice_explain_results.csv", index=False)

    summary = {
        "total_validated": int(len(df)),
        "qwen_agrees": int(correct),
        "invalid_or_errors": int(errors),
        "agreement_percent": float(correct / len(df) * 100 if len(df) else 0.0),
        "dataset_dir": str(dataset_dir),
        "model": args.model,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDONE")
    print(f"agreement={summary['agreement_percent']:.2f}% {correct}/{len(df)}")
    print("errors:", errors)
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
