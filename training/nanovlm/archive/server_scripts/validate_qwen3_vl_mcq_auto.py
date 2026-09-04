from pathlib import Path
import argparse, ast, json, re
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

def parse_answer(text):
    raw = str(text).strip()
    m = re.search(r'"answer"\s*:\s*"([ABCD])"', raw, flags=re.I)
    if m:
        return m.group(1).upper(), raw
    m = re.search(r"\b([ABCD])\b", raw.upper())
    if m:
        return m.group(1).upper(), raw
    return "INVALID", raw

ap = argparse.ArgumentParser()
ap.add_argument("--dataset-dir", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
ap.add_argument("--max-new-tokens", type=int, default=128)
args = ap.parse_args()

dataset = Path(args.dataset_dir)
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(dataset / "metadata.csv")

dtype = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}[args.dtype]

print("Loading processor:", args.model)
processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

print("Loading model:", args.model)
model = AutoModelForImageTextToText.from_pretrained(
    args.model,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

letters = ["A", "B", "C", "D"]
rows = []

for _, r in tqdm(df.iterrows(), total=len(df), desc="Qwen3-VL MCQ"):
    img = Image.open(dataset / r["image"]).convert("RGB")
    choices = ast.literal_eval(r["choices"])

    choices_text = "\n".join([f"{letters[i]}. {choices[i]}" for i in range(4)])

    prompt = f"""Answer the multiple-choice question about the image.

Question:
{r['question']}

Choices:
{choices_text}

Return only valid JSON:
{{"answer":"A","explanation":"short visual reason"}}
"""

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=[img],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    raw = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    pred_letter, raw = parse_answer(raw)

    gt_idx = int(r["gt_idx"])
    gt_letter = letters[gt_idx]
    pred_answer = choices[letters.index(pred_letter)] if pred_letter in letters else "INVALID"

    rr = r.to_dict()
    rr.update({
        "gt_letter": gt_letter,
        "qwen3_pred_letter": pred_letter,
        "qwen3_pred_answer": pred_answer,
        "qwen3_correct": int(pred_letter == gt_letter),
        "qwen3_raw_output": raw,
    })
    rows.append(rr)

res = pd.DataFrame(rows)
res.to_csv(out / "qwen3_results.csv", index=False)

summary = {
    "model": args.model,
    "total_validated": int(len(res)),
    "correct": int(res["qwen3_correct"].sum()),
    "accuracy_percent": float(res["qwen3_correct"].mean() * 100 if len(res) else 0),
}

json.dump(summary, open(out / "summary.json", "w"), ensure_ascii=False, indent=2)

print("\nDONE")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\nDETAILS:")
print(res[[
    "image",
    "question",
    "answer",
    "qwen3_pred_answer",
    "qwen3_pred_letter",
    "qwen3_correct",
    "qwen3_raw_output",
]].to_string(index=False, max_colwidth=240))
