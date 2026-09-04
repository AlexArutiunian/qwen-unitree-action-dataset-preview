from pathlib import Path
import argparse
import re
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

QUESTION = """Look at the image and classify the visible door state.

Choose exactly one option:

A. OPEN — the door is visibly open, moved aside, or there is a real visible gap/passage through the doorway.
B. CLOSED — the door is visibly closed, covering the doorway, including glass/transparent/reflective doors that are closed.
C. UNCERTAIN — cannot tell from the image, no clear door is visible, or the image is too ambiguous.

Important:
- Do not assume that a reflective or glass surface means open.
- If a glass or mirror-like panel blocks the doorway, choose B.
- If the door leaf is clearly moved aside and there is a real open gap, choose A.
- If there is not enough evidence, choose C.

Answer format:
First line: only A, B, or C.
Second line: short visual reason.
"""

LABEL_BY_ANSWER = {
    "A": "open",
    "B": "closed",
    "C": "cannot_tell",
}

def parse_answer(text: str):
    raw = text.strip()

    # first non-empty line
    lines = [x.strip() for x in raw.splitlines() if x.strip()]
    first = lines[0] if lines else raw

    m = re.search(r"\b([ABC])\b", first.upper())
    if not m:
        m = re.search(r"\b([ABC])\b", raw.upper())

    if not m:
        return "INVALID", "cannot_tell", raw, raw

    ans = m.group(1)
    label = LABEL_BY_ANSWER[ans]

    explanation = ""
    if len(lines) >= 2:
        explanation = " ".join(lines[1:])
    else:
        explanation = raw

    return ans, label, explanation, raw

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

    for sub in ["open", "closed", "cannot_tell", "invalid"]:
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

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    rows = []

    for img_path in tqdm(images, desc="Qwen v2 sorting doors"):
        try:
            image = Image.open(img_path).convert("RGB")

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": QUESTION},
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
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False,
                )

            generated_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
            output_text = processor.batch_decode(
                generated_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            ans, label, explanation, raw = parse_answer(output_text)

            folder = label if ans != "INVALID" else "invalid"
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

    print("\nExamples:")
    print(df[["source_path", "answer", "label", "explanation"]].head(30).to_string(index=False, max_colwidth=180))

if __name__ == "__main__":
    main()
