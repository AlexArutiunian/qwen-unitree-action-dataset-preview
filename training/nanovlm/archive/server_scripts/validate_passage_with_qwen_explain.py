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


def extract_letter(text):
    t = str(text).strip()

    # JSON style: {"answer":"A", ...}
    try:
        obj = json.loads(t)
        for k in ["answer", "letter", "choice"]:
            if k in obj:
                s = str(obj[k]).strip().upper()
                if s in ["A", "B", "C", "D"]:
                    return s
    except Exception:
        pass

    m = re.search(r'\b([ABCD])\b', t.upper())
    if m:
        return m.group(1)

    m = re.search(r'"answer"\s*:\s*"([ABCD])"', t.upper())
    if m:
        return m.group(1)

    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_dir / "metadata.csv")

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

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

    for i, r in tqdm(df.iterrows(), total=len(df), desc="Qwen explain"):
        try:
            image_path = dataset_dir / str(r["image"])
            image = Image.open(image_path).convert("RGB")

            choices = parse_choices(r["choices"])
            gt_idx = int(r["gt_idx"])
            gt_letter = "ABCD"[gt_idx]
            gt_answer = str(r["answer"])

            prompt = f"""
You are judging physical passability for a robot from a single camera image.

Important rules:
- The robot can pass only if there is a real open physical gap.
- A transparent glass door, mirror-like door, reflective panel, closed door leaf, person, or obstacle can block the passage.
- Visible space behind glass or reflections does not by itself mean the passage is clear.
- Do not over-assume: if the physical opening is unclear, choose C.

Question:
{r["question"]}

Choices:
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Return strictly valid JSON with keys:
{{"answer":"A/B/C/D", "explanation":"one short sentence explaining the visible evidence"}}
""".strip()

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=96,
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

            pred_letter = extract_letter(raw)
            pred_idx = "ABCD".find(pred_letter) if pred_letter else -1
            pred_answer = choices[pred_idx] if 0 <= pred_idx < len(choices) else ""

            explanation = ""
            try:
                obj = json.loads(raw)
                explanation = str(obj.get("explanation", ""))
            except Exception:
                explanation = raw

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
    out.to_csv(out_dir / "qwen_explain_results.csv", index=False)

    total = len(df)
    acc = correct / total * 100 if total else 0.0

    summary = {
        "total_validated": total,
        "qwen_agrees": int(correct),
        "invalid_or_errors": int(errors),
        "agreement_percent": acc,
        "dataset_dir": str(dataset_dir),
        "model": args.model,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDONE")
    print(f"agreement={acc:.2f}% qwen_agrees={correct}/{total}")
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
