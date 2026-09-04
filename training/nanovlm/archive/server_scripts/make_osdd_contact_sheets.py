#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path("osdd_class_samples")
OUT = Path("osdd_class_contact_sheets")
OUT.mkdir(exist_ok=True)

thumb_w, thumb_h = 240, 180
cols = 5
rows = 8

for class_dir in sorted(ROOT.glob("class_*")):
    imgs = sorted(class_dir.glob("*.jpg"))[:cols * rows]
    if not imgs:
        continue

    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, p in enumerate(imgs):
        img = Image.open(p).convert("RGB")
        img.thumbnail((thumb_w, thumb_h - 22))

        x = (idx % cols) * thumb_w
        y = (idx // cols) * thumb_h

        sheet.paste(img, (x, y + 22))
        draw.text((x + 5, y + 4), p.name[:32], fill=(0, 0, 0))

    out = OUT / f"{class_dir.name}.jpg"
    sheet.save(out, quality=95)
    print("saved", out)

print("done:", OUT.resolve())
