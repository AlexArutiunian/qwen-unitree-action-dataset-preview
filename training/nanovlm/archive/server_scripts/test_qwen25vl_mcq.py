import argparse
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info


def extract_letter(text):
    text = text.strip()

    # Ищем A/B/C/D
    m = re.search(r"\b([ABCD])\b", text.upper())
    if m:
        return m.group(1)

    # fallback по словам
    low = text.lower()
    if "yes" in low or "open" in low:
        return "A"
    if "closed" in low or "close" in low or "no" in low:
        return "B"
    if "cannot" in low or "can't" in low or "not visible" in low:
        return "C"

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default="Is the door open?")
    parser.add_argument("--a", default="Yes, the door is open.")
    parser.add_argument("--b", default="No, the door is closed.")
    parser.add_argument("--c", default="Cannot tell from the image.")
    parser.add_argument("--d", default="The door is not visible.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    image_path = str(Path(args.image).resolve())

    print("Loading processor:", args.model)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    print("Loading model:", args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    prompt = f"""You are a visual robot-state judge.

Look at the image carefully and answer the multiple-choice question.

Question: {args.question}

A) {args.a}
B) {args.b}
C) {args.c}
D) {args.d}

Return only one letter: A, B, C, or D.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    out = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    pred = extract_letter(out)

    print("\nMODEL RAW OUTPUT:")
    print(out)

    print("\nPRED LETTER:", pred)

    mapping = {
        "A": args.a,
        "B": args.b,
        "C": args.c,
        "D": args.d,
    }

    if pred in mapping:
        print("PRED ANSWER:", mapping[pred])


if __name__ == "__main__":
    main()
