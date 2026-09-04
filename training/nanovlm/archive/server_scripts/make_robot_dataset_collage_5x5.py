from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import textwrap
import random
import ast

DATASET = Path("robovqa_affordance_10000_accepted_split_train")
OUT_DIR = Path("thesis_vlm_mcq_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "robot_dataset_train_collage_5x5.png"

N = 25
GRID = 5
TILE_W = 360
TILE_H = 300
IMG_H = 225
PAD = 12
SEED = 42

random.seed(SEED)

meta = DATASET / "metadata.csv"
df = pd.read_csv(meta)

# берём валидные строки, где картинка реально существует
valid = []
for _, r in df.iterrows():
    img_path = DATASET / str(r["image"])
    if img_path.exists():
        valid.append(r)

if len(valid) < N:
    raise RuntimeError(f"Need at least {N} valid images, found {len(valid)}")

sample = random.sample(valid, N)

try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
    font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except Exception:
    font_title = ImageFont.load_default()
    font_text = ImageFont.load_default()
    font_small = ImageFont.load_default()

canvas_w = GRID * TILE_W
canvas_h = GRID * TILE_H
canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
draw = ImageDraw.Draw(canvas)

def fit_image(img, max_w, max_h):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(max_w / w, max_h / h)
    nw, nh = int(w * scale), int(h * scale)
    return img.resize((nw, nh), Image.LANCZOS)

def short_text(s, width=38, lines=2):
    s = str(s).replace("\n", " ").strip()
    wrapped = textwrap.wrap(s, width=width)
    if len(wrapped) > lines:
        wrapped = wrapped[:lines]
        wrapped[-1] = wrapped[-1][:max(0, len(wrapped[-1]) - 3)] + "..."
    return "\n".join(wrapped)

for idx, r in enumerate(sample):
    row = idx // GRID
    col = idx % GRID
    x0 = col * TILE_W
    y0 = row * TILE_H

    # рамка карточки
    draw.rectangle([x0 + 4, y0 + 4, x0 + TILE_W - 4, y0 + TILE_H - 4], outline=(190, 190, 190), width=2)

    img_path = DATASET / str(r["image"])
    img = Image.open(img_path)
    img_fit = fit_image(img, TILE_W - 2 * PAD, IMG_H)

    ix = x0 + (TILE_W - img_fit.width) // 2
    iy = y0 + PAD
    canvas.paste(img_fit, (ix, iy))

    q = r.get("question", "")
    ans = r.get("answer", r.get("dataset_gt_answer", ""))
    task = r.get("task", "")

    text_y = y0 + IMG_H + PAD + 6

    draw.text((x0 + PAD, text_y), f"#{idx+1}", fill=(20, 40, 70), font=font_title)
    draw.text((x0 + PAD + 42, text_y), short_text(task, width=32, lines=1), fill=(20, 40, 70), font=font_small)

    draw.text((x0 + PAD, text_y + 26), short_text(q, width=42, lines=2), fill=(0, 0, 0), font=font_small)
    draw.text((x0 + PAD, text_y + 62), "GT: " + short_text(ans, width=36, lines=1), fill=(0, 90, 0), font=font_text)

canvas.save(OUT, quality=95)
print("saved:", OUT.resolve())
print("rows:", len(df), "valid_images:", len(valid))
