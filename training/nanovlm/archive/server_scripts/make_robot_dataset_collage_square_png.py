from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
import textwrap

DATASET = Path("robovqa_affordance_10000_accepted_split_train")
OUT_DIR = Path("thesis_vlm_mcq_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "robot_dataset_train_collage_5x5_SQUARE.png"

N = 25
GRID = 5
SIZE = 3000
TILE = SIZE // GRID
PAD = 18
IMG_BOX_H = 365
SEED = 42

random.seed(SEED)

df = pd.read_csv(DATASET / "metadata.csv")

valid = []
for _, r in df.iterrows():
    p = DATASET / str(r["image"])
    if p.exists():
        valid.append(r)

if len(valid) < N:
    raise RuntimeError(f"Need {N} images, found {len(valid)}")

sample = random.sample(valid, N)

def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

FONT_BOLD = font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 25)
FONT_TASK = font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
FONT_Q = font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
FONT_GT = font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 21)

canvas = Image.new("RGB", (SIZE, SIZE), (245, 247, 250))
draw = ImageDraw.Draw(canvas)

def wrap(s, width, lines):
    s = str(s).replace("\n", " ").strip()
    arr = textwrap.wrap(s, width=width)
    if len(arr) > lines:
        arr = arr[:lines]
        arr[-1] = arr[-1][:-3] + "..."
    return arr

def fit_img(img, max_w, max_h):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(max_w / w, max_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), Image.LANCZOS)

for idx, r in enumerate(sample, start=1):
    row = (idx - 1) // GRID
    col = (idx - 1) % GRID

    x0 = col * TILE
    y0 = row * TILE
    x1 = x0 + TILE
    y1 = y0 + TILE

    # card
    draw.rounded_rectangle(
        [x0 + 10, y0 + 10, x1 - 10, y1 - 10],
        radius=18,
        fill=(255, 255, 255),
        outline=(205, 213, 222),
        width=2,
    )

    task = str(r.get("task", "")).strip()
    q = str(r.get("question", "")).strip()
    gt = str(r.get("answer", r.get("dataset_gt_answer", ""))).strip()

    tx = x0 + PAD + 10
    ty = y0 + PAD + 6

    # header
    draw.text((tx, ty), f"#{idx}", font=FONT_BOLD, fill=(28, 55, 92))
    draw.text((tx + 58, ty + 5), task[:42], font=FONT_TASK, fill=(85, 96, 110))

    # question
    q_lines = wrap(q, 48, 3)
    q_y = ty + 48
    for line in q_lines:
        draw.text((tx, q_y), line, font=FONT_Q, fill=(20, 20, 20))
        q_y += 27

    # GT fixed block
    gt_y = ty + 140
    draw.rounded_rectangle(
        [tx, gt_y - 5, x1 - PAD - 10, gt_y + 37],
        radius=8,
        fill=(235, 248, 237),
        outline=(190, 225, 195),
        width=1,
    )
    draw.text((tx + 10, gt_y + 4), f"GT: {gt}", font=FONT_GT, fill=(20, 110, 35))

    # image box
    box_x0 = x0 + PAD + 10
    box_y0 = y0 + TILE - IMG_BOX_H - PAD
    box_x1 = x1 - PAD - 10
    box_y1 = y1 - PAD - 10

    draw.rectangle(
        [box_x0, box_y0, box_x1, box_y1],
        fill=(250, 250, 250),
        outline=(225, 228, 232),
        width=1,
    )

    img_path = DATASET / str(r["image"])
    img = Image.open(img_path)
    img_fit = fit_img(img, box_x1 - box_x0 - 10, box_y1 - box_y0 - 10)

    ix = box_x0 + ((box_x1 - box_x0) - img_fit.width) // 2
    iy = box_y0 + ((box_y1 - box_y0) - img_fit.height) // 2
    canvas.paste(img_fit, (ix, iy))

canvas.save(OUT, quality=95)
print("saved:", OUT.resolve())
print("size:", canvas.size)
