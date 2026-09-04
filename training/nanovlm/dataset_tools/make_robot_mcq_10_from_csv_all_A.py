from pathlib import Path
import pandas as pd
import shutil
import csv
import re

SRC = Path("robotpct_final")
OUT = Path("robot_mcq_10_all_A")

CSV_PATH = SRC / "qe.csv"
if not CSV_PATH.exists():
    CSV_PATH = SRC / "q.csv"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def numeric_key(p: Path):
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 10**9

def try_read_with_delim(path: Path, delim: str):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=delim)
        for row in reader:
            row = [x.strip() for x in row]
            if not row or all(not x for x in row):
                continue
            rows.append(row)
    return rows

def read_questions(path: Path):
    # пробуем разные разделители вручную
    candidates = [",", ";", "\t", "|"]
    best_rows = None
    best_delim = None
    best_score = -1

    for delim in candidates:
        rows = try_read_with_delim(path, delim)
        if not rows:
            continue

        # score = сколько строк имеют хотя бы 5 колонок
        score = sum(1 for r in rows if len(r) >= 5)
        if score > best_score:
            best_score = score
            best_rows = rows
            best_delim = delim

    if best_rows is None or best_score <= 0:
        print("RAW CSV CONTENT:")
        print(path.read_text(encoding="utf-8-sig", errors="replace")[:2000])
        raise RuntimeError(f"Could not parse CSV with delimiters: {candidates}")

    print("CSV delimiter selected:", repr(best_delim))

    parsed = []
    for row in best_rows:
        # header skip
        first = row[0].strip().lower() if row else ""
        if first in {"idx", "id", "номер", "n", "num"}:
            continue

        # expected: idx, question, A, B, C, D
        if len(row) >= 6:
            idx = row[0]
            question = row[1]
            choices = row[2:6]
        # fallback: question, A, B, C, D
        elif len(row) == 5:
            idx = str(len(parsed) + 1)
            question = row[0]
            choices = row[1:5]
        else:
            print("BAD ROW:", row)
            raise RuntimeError(f"Bad CSV row, expected 5 or 6 columns, got {len(row)}")

        parsed.append({
            "idx": idx,
            "question": question,
            "choices": choices,
            "answer": choices[0],
            "gt_idx": 0,
        })

    return parsed

imgs = [
    p for p in SRC.iterdir()
    if p.is_file() and p.suffix.lower() in IMG_EXTS
]
imgs = sorted(imgs, key=numeric_key)

questions = read_questions(CSV_PATH)

print("CSV:", CSV_PATH)
print("images:", len(imgs))
for i, p in enumerate(imgs, 1):
    print(f"{i:02d}: {p}")

print("\nquestions:", len(questions))
for i, q in enumerate(questions, 1):
    print(f"{i:02d}: {q['question']} | A={q['choices'][0]}")

if len(imgs) != 10:
    raise RuntimeError(f"Need exactly 10 images, found {len(imgs)}")

if len(questions) != 10:
    raise RuntimeError(f"Need exactly 10 questions in {CSV_PATH}, found {len(questions)}")

if OUT.exists():
    shutil.rmtree(OUT)

(OUT / "images").mkdir(parents=True, exist_ok=True)

out_rows = []

for i, (img, q) in enumerate(zip(imgs, questions), 1):
    ext = img.suffix.lower()
    dst_rel = f"images/robot_{i:02d}{ext}"
    shutil.copy2(img, OUT / dst_rel)

    out_rows.append({
        "id": f"robot_{i:02d}",
        "image": dst_rel,
        "question": q["question"],
        "choices": str(q["choices"]),
        "gt_idx": 0,
        "answer": q["answer"],
        "task": "robot_mcq_10_all_A",
        "source_path": str(img),
        "csv_source": str(CSV_PATH),
    })

df = pd.DataFrame(out_rows)
df.to_csv(OUT / "metadata.csv", index=False)

print("\nDONE:", OUT)
print(df[["id", "image", "question", "answer", "gt_idx", "source_path"]].to_string(index=False, max_colwidth=220))

print("\ngt_idx counts:")
print(df["gt_idx"].value_counts().sort_index().to_string())
