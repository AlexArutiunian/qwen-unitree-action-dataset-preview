from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import os
import math
import shutil

CSV = Path("eval_yolo_deepdoors2_on_hard22/results.csv")
OUT_DIR = Path("yolo_hard22_pdf_review")
PDF = OUT_DIR / "yolo_hard22_review.pdf"
IMG_DIR = OUT_DIR / "images"

OUT_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV)

# inverted check
df["pred_inv"] = df["pred"].map({"open": "closed", "closed": "open"})
df["correct_inv"] = (df["pred_inv"] == df["gt"]).astype(int)

normal_acc = df["correct"].mean() * 100
inv_acc = df["correct_inv"].mean() * 100

# Fonts
font_name = "Helvetica"
font_bold = "Helvetica-Bold"
for fpath in [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]:
    if Path(fpath).exists():
        try:
            if "Bold" in fpath:
                pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", fpath))
                font_bold = "DejaVuSans-Bold"
            else:
                pdfmetrics.registerFont(TTFont("DejaVuSans", fpath))
                font_name = "DejaVuSans"
        except Exception:
            pass

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="Small",
    parent=styles["Normal"],
    fontName=font_name,
    fontSize=8,
    leading=10,
))
styles.add(ParagraphStyle(
    name="NormalCustom",
    parent=styles["Normal"],
    fontName=font_name,
    fontSize=10,
    leading=12,
))
styles.add(ParagraphStyle(
    name="TitleCustom",
    parent=styles["Title"],
    fontName=font_bold,
    fontSize=18,
    leading=22,
))
styles.add(ParagraphStyle(
    name="HeaderCustom",
    parent=styles["Heading2"],
    fontName=font_bold,
    fontSize=13,
    leading=16,
))

doc = SimpleDocTemplate(
    str(PDF),
    pagesize=A4,
    rightMargin=12*mm,
    leftMargin=12*mm,
    topMargin=12*mm,
    bottomMargin=12*mm,
)

story = []

story.append(Paragraph("YOLO door classifier review on hard22", styles["TitleCustom"]))
story.append(Spacer(1, 5*mm))

summary_data = [
    ["Metric", "Value"],
    ["Dataset", "door_cls_hard22/test"],
    ["Rows", str(len(df))],
    ["Normal accuracy", f"{normal_acc:.2f}% ({int(df['correct'].sum())}/{len(df)})"],
    ["Inverted accuracy", f"{inv_acc:.2f}% ({int(df['correct_inv'].sum())}/{len(df)})"],
    ["GT closed", str((df["gt"] == "closed").sum())],
    ["GT open", str((df["gt"] == "open").sum())],
    ["Pred closed", str((df["pred"] == "closed").sum())],
    ["Pred open", str((df["pred"] == "open").sum())],
]
t = Table(summary_data, colWidths=[55*mm, 105*mm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("FONTNAME", (0,0), (-1,0), font_bold),
    ("FONTNAME", (0,1), (-1,-1), font_name),
    ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("LEFTPADDING", (0,0), (-1,-1), 5),
    ("RIGHTPADDING", (0,0), (-1,-1), 5),
]))
story.append(t)
story.append(Spacer(1, 6*mm))

story.append(Paragraph(
    "Note: if inverted accuracy is high, the visual classes learned from DeepDoors2 may not match the manual hard22 labels.",
    styles["NormalCustom"]
))
story.append(PageBreak())

def make_preview(src_path, out_path, max_w=900, max_h=650):
    img = Image.open(src_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img.thumbnail((max_w, max_h))
    img.save(out_path, quality=92)
    return out_path

for i, r in df.reset_index(drop=True).iterrows():
    img_path = Path(str(r["image"]))
    if not img_path.exists():
        story.append(Paragraph(f"Missing image: {img_path}", styles["HeaderCustom"]))
        story.append(PageBreak())
        continue

    preview_path = IMG_DIR / f"preview_{i:03d}{img_path.suffix.lower() or '.jpg'}"
    make_preview(img_path, preview_path)

    ok = int(r["correct"]) == 1
    inv_ok = int(r["correct_inv"]) == 1

    status = "OK" if ok else "ERROR"
    status_color = colors.green if ok else colors.red

    story.append(Paragraph(
        f"#{i:02d} - {status}",
        ParagraphStyle(
            name=f"Status{i}",
            parent=styles["HeaderCustom"],
            textColor=status_color,
            fontName=font_bold,
        )
    ))

    meta = [
        ["GT", str(r["gt"])],
        ["YOLO pred", str(r["pred"])],
        ["Correct", str(int(r["correct"]))],
        ["Inverted pred", str(r["pred_inv"])],
        ["Correct if inverted", str(int(r["correct_inv"]))],
        ["Confidence", f"{float(r['conf']):.4f}"],
        ["prob_closed", f"{float(r['prob_closed']):.6f}"],
        ["prob_open", f"{float(r['prob_open']):.6f}"],
        ["Image", str(img_path)],
    ]

    mt = Table(meta, colWidths=[40*mm, 130*mm])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.whitesmoke),
        ("FONTNAME", (0,0), (-1,-1), font_name),
        ("FONTNAME", (0,0), (0,-1), font_bold),
        ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(mt)
    story.append(Spacer(1, 4*mm))

    # image scaled to page width
    rl_img = RLImage(str(preview_path))
    max_width = 170 * mm
    max_height = 150 * mm
    w, h = Image.open(preview_path).size
    scale = min(max_width / w, max_height / h)
    rl_img.drawWidth = w * scale
    rl_img.drawHeight = h * scale
    story.append(rl_img)

    if i != len(df) - 1:
        story.append(PageBreak())

doc.build(story)

print("PDF saved:", PDF.resolve())
print("CSV:", CSV.resolve())
print("normal_acc:", normal_acc)
print("inverted_acc:", inv_acc)
