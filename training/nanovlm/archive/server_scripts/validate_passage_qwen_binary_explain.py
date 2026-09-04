#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def extract_json_or_text(raw):
    t = str(raw).strip()
    t = re.sub(r"^```json\s*", "", t)
    t = re.sub(r"^```\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        return None


def extract_letter(raw):
    obj = extract_json_or_text(raw)
    if isinstance(obj, dict):
        s = str(obj.get("answer", "")).strip().upper()
        if s in ["A", "B"]:
            return s

    t = str(raw).upper()
    m = re.search(r'"ANSWER"\s*:\s*"([AB])"', t)
    if m:
        return m.group(1)
    m = re.search(r'\b([AB])\b', t)
    if m:
        return m.group(1)
    return ""


def extract_expl(raw):
    obj = extract_json_or_text(raw)
    if isinstance(obj, dict):
        return str(obj.get("explanation", ""))
    return str(raw).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_dir / "metadata.csv")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

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

    for i, r in tqdm(df.iterrows(), total=len(df), desc="Qwen binary explain"):
        image = Image.open(dataset_dir / str(r["image"])).convert("RGB")

        label = str(r.get("manual_label", r.get("gt_label", ""))).strip().lower()
        if label == "clear":
            gt_letter = "A"
            gt_answer = "Yes, the passage is clear."
        elif label == "blocked":
            gt_letter = "B"
            gt_answer = "No, the passage is blocked."
        else:
            # fallback from gt_idx
            gt_idx = int(r["gt_idx"])
            gt_letter = "A" if gt_idx == 0 else "B"
            gt_answer = "Yes, the passage is clear." if gt_letter == "A" else "No, the passage is blocked."

        prompt = """
You are judging physical passability for a robot from a single image.

Task:
Decide whether the robot can move straight through the visible passage right now without opening a door, moving a person, or moving any object.

Important:
- If a closed glass door, transparent panel, mirror-like surface, closed door leaf, person, or obstacle physically blocks the way, choose B.
- If there is a real open gap wide enough for the robot to pass, choose A.
- Visible space behind glass or a reflection does NOT by itself mean the passage is clear.
- You MUST choose A or B. Do not answer "cannot tell".

Choices:
A. Yes, the passage is clear.
B. No, the passage is blocked.

Return only valid JSON:
{"answer":"A or B", "explanation":"one short sentence with the visible evidence"}
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
        pred_answer = (
            "Yes, the passage is clear." if pred_letter == "A"
            else "No, the passage is blocked." if pred_letter == "B"
            else ""
        )
        ok = int(pred_letter == gt_letter)
        correct += ok

        row = r.to_dict()
        row.update({
            "dataset_gt_letter": gt_letter,
            "dataset_gt_answer": gt_answer,
            "qwen_pred_letter": pred_letter,
            "qwen_pred_answer": pred_answer,
            "qwen_explanation": extract_expl(raw),
            "qwen_raw_output": raw,
            "qwen_agrees_with_dataset": ok,
        })
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "qwen_binary_explain_results.csv", index=False)

    summary = {
        "total_validated": len(df),
        "qwen_agrees": int(correct),
        "agreement_percent": correct / len(df) * 100 if len(df) else 0,
        "dataset_dir": str(dataset_dir),
        "mode": "binary_A_B_forced",
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), ensure_ascii=False, indent=2)

    print("\nDONE")
    print(f"agreement={summary['agreement_percent']:.2f}% {correct}/{len(df)}")
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
