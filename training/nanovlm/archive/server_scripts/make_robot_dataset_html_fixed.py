from pathlib import Path
import pandas as pd
import random
import html

DATASET_DIR = Path("robovqa_affordance_10000_accepted_split_train")
OUT_DIR = Path("thesis_vlm_mcq_plots/robot_dataset_html_fixed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

META = DATASET_DIR / "metadata.csv"
OUT_HTML = OUT_DIR / "index.html"

N = 25
SEED = 42

random.seed(SEED)

df = pd.read_csv(META)

# только строки, где картинка реально существует
rows = []
for _, r in df.iterrows():
    img_rel = str(r["image"])
    img_abs = DATASET_DIR / img_rel
    if img_abs.exists():
        rows.append(r)

if len(rows) < N:
    raise RuntimeError(f"Need at least {N} rows with existing images, found {len(rows)}")

sample = random.sample(rows, N)

cards = []

for i, r in enumerate(sample, start=1):
    img_rel = str(r["image"])
    q = str(r.get("question", "")).strip()
    gt = str(r.get("answer", r.get("dataset_gt_answer", ""))).strip()
    task = str(r.get("task", "")).strip()

    card = f"""
    <div class="card">
      <div class="card-head">
        <div class="idx">#{i}</div>
        <div class="task">{html.escape(task)}</div>
      </div>

      <div class="question">{html.escape(q)}</div>
      <div class="gt"><span class="gt-label">GT:</span> {html.escape(gt)}</div>

      <div class="img-wrap">
        <img src="../../{DATASET_DIR.name}/{html.escape(img_rel)}" alt="sample_{i}">
      </div>
    </div>
    """
    cards.append(card)

html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Robot dataset samples (fixed)</title>
<style>
  * {{
    box-sizing: border-box;
  }}

  body {{
    margin: 0;
    padding: 24px;
    font-family: Arial, Helvetica, sans-serif;
    background: #f4f6f8;
    color: #111;
  }}

  h1 {{
    margin: 0 0 8px 0;
    font-size: 28px;
  }}

  .sub {{
    margin-bottom: 20px;
    color: #444;
    font-size: 15px;
  }}

  .grid {{
    display: grid;
    grid-template-columns: repeat(5, minmax(250px, 1fr));
    gap: 16px;
    align-items: start;
  }}

  .card {{
    background: #fff;
    border: 1px solid #cfd6dd;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    display: flex;
    flex-direction: column;
    min-height: 420px;
  }}

  .card-head {{
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 12px 12px 6px 12px;
    border-bottom: 1px solid #eef1f4;
  }}

  .idx {{
    font-size: 16px;
    font-weight: 700;
    color: #1f3b63;
    white-space: nowrap;
  }}

  .task {{
    font-size: 13px;
    color: #5c6b7a;
    line-height: 1.3;
    word-break: break-word;
  }}

  .question {{
    padding: 10px 12px 6px 12px;
    font-size: 14px;
    line-height: 1.4;
    min-height: 64px;
  }}

  .gt {{
    padding: 0 12px 10px 12px;
    font-size: 14px;
    line-height: 1.45;
    color: #156b1f;
    font-weight: 600;
    min-height: 42px;
    word-break: break-word;
  }}

  .gt-label {{
    color: #156b1f;
    font-weight: 700;
  }}

  .img-wrap {{
    margin-top: auto;
    width: 100%;
    height: 220px;
    padding: 10px 12px 14px 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #fff;
  }}

  .img-wrap img {{
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
    display: block;
    object-fit: contain;
    border: 1px solid #e5e7eb;
  }}

  @media (max-width: 1600px) {{
    .grid {{
      grid-template-columns: repeat(4, minmax(250px, 1fr));
    }}
  }}

  @media (max-width: 1280px) {{
    .grid {{
      grid-template-columns: repeat(3, minmax(250px, 1fr));
    }}
  }}
</style>
</head>
<body>
  <h1>Robot dataset sample cards — fixed layout</h1>
  <div class="sub">25 examples from robovqa_affordance_10000_accepted_split_train. GT text is placed in a separate block above the image.</div>

  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""

OUT_HTML.write_text(html_text, encoding="utf-8")

print("saved:", OUT_HTML.resolve())
