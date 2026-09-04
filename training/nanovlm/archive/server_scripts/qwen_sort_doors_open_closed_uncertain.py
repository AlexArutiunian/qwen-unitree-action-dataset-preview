from pathlib import Path
import argparse
import json
import re
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

QUESTION = """Classify the door state in this image.

Important:
- Answer "A" only if the door is visibly closed.
- Answer "B" only if the door is visibly open.
- Answer "C" if the image is ambiguous, the door is reflective/glass/mirror-like and the state is not clear, or there is not enough visual evidence.

Choices:
A. The door is closed.
B. The door is open.
C. Cannot tell from this image.

Return only JSON:
{"answer":"A","label":"closed","explanation":"short reason"}"""

LABEL_BY_ANSWER = {
    "A": "closed",
    "B": "open",
    "C": "cannot_tell",
}

def extract_json_or_answer(text: str):
    text = text.strip()

    # Try JSON block
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            ans = str(obj.get("answer", "")).strip().upper()[:1]
            label = str(obj.get("label", "")).strip().lower()
            explanation = str(obj.get("explanation", "")).strip()

            if ans in LABEL_BY_ANSWER:
                canonical = LABEL_BY_ANSWER[ans]
                if label not in {"closed", "open", "cannot_tell"}:
                    label = canonical
                return ans, label, explanation, text
        except Exception:
            pass

    # Fallback: first A/B/C
    m = re.search(r"\b([ABC])\b", text.upper())
    if m:
        ans = m.group(1)
        return ans, LABEL_BY_ANSWER[ans], "", text

    return "INVALID", "cannot_tell", "Could not parse model output.", text

def safe_copy(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        stem = src.stem
        suffix = src.suffix
        k = 1
        while True:
            cand = dst_dir / f"{stem}__dup{k}{suffix}"
            if not cand.exists():
                dst = cand
                break
            k += 1
    shutil.copy2(src, dst)
    return dst

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    if out_dir.exists():
        print("WARNING: output dir exists, appending:", out_dir)
    for sub in ["closed", "open", "cannot_tell", "invalid"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    if args.recursive:
        images = [p for p in sorted(input_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        images = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]

    if args.limit > 0:
        images = images[:args.limit]

    print("images:", len(images))
    print("model:", args.model)
    print("input:", input_dir)
    print("out:", out_dir)

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    print("Loading processor:", args.model)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    print("Loading model:", args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    rows = []

    for img_path in tqdm(images, desc="Qwen sorting doors"):
        try:
            image = Image.open(img_path).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": QUESTION},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=96,
                    do_sample=False,
                    temperature=0.0,
                )

            generated_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
            output_text = processor.batch_decode(
                generated_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            ans, label, explanation, raw = extract_json_or_answer(output_text)

            if ans == "INVALID":
                folder = "invalid"
            else:
                folder = label

            saved_path = safe_copy(img_path, out_dir / folder)

            rows.append({
                "source_path": str(img_path),
                "saved_path": str(saved_path),
                "answer": ans,
                "label": label,
                "explanation": explanation,
                "raw_output": raw,
            })

        except Exception as e:
            saved_path = safe_copy(img_path, out_dir / "invalid")
            rows.append({
                "source_path": str(img_path),
                "saved_path": str(saved_path),
                "answer": "ERROR",
                "label": "invalid",
                "explanation": repr(e),
                "raw_output": "",
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "report.csv", index=False)

    print("\nDONE")
    print("report:", out_dir / "report.csv")
    print("\nLabel counts:")
    print(df["label"].value_counts().to_string())
    print("\nAnswer counts:")
    print(df["answer"].value_counts().to_string())

if __name__ == "__main__":
    main()
