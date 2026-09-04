#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw
import random

ROOT = Path("osdd_data")
OUT = Path("osdd_class_samples")
OUT.mkdir(exist_ok=True)

IMG_EXTS = [".jpg", ".jpeg", ".png"]

def find_image_for_txt(txt_path: Path):
    for ext in IMG_EXTS:
        p = txt_path.with_suffix(ext)
        if p.exists():
            return p
    return None

by_class = defaultdict(list)

for txt in ROOT.rglob("*.txt"):
    img = find_image_for_txt(txt)
    if img is None:
        continue

    for li, line in enumerate(txt.read_text(errors="ignore").splitlines()):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            x, y, bw, bh = map(float, parts[1:5])
        except Exception:
            continue
        by_class[cid].append((img, txt, li, (x, y, bw, bh)))

print("classes:", sorted(by_class))

for cid in sorted(by_class):
    samples = by_class[cid]
    random.Random(123).shuffle(samples)

    class_dir = OUT / f"class_{cid}"
    class_dir.mkdir(parents=True, exist_ok=True)

    for j, (img_path, txt_path, li, box) in enumerate(samples[:40]):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        x, y, bw, bh = box
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
        draw.rectangle([0, 0, 320, 42], fill=(255, 255, 255))
        draw.text((10, 10), f"class_id={cid} | {img_path.name}", fill=(255, 0, 0))

        img.save(class_dir / f"{j:03d}_{img_path.stem}.jpg", quality=95)

    print(f"class {cid}: total={len(samples)}, saved={min(40, len(samples))}")

print("saved:", OUT.resolve())
