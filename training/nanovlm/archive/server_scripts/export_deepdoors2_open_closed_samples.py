#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import pandas as pd

META = Path("deepdoors2_door_state_mcq/test/metadata.csv")
DATASET = Path("deepdoors2_door_state_mcq/test")
OUT = Path("deepdoors2_visual_check_100")

if OUT.exists():
    shutil.rmtree(OUT)

(OUT / "closed").mkdir(parents=True, exist_ok=True)
(OUT / "open").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(META)

# Берем уникальные изображения, чтобы не было размножения вопросами
df = df.drop_duplicates("image").copy()

for state in ["closed", "open"]:
    sub = df[df["door_state"] == state].copy()
    sub = sub.sample(n=min(100, len(sub)), random_state=42).reset_index(drop=True)

    print(state, "available:", len(df[df["door_state"] == state]), "export:", len(sub))

    rows = []
    for i, r in sub.iterrows():
        src = DATASET / str(r["image"])
        if not src.exists():
            # fallback: поиск по имени
            matches = list(DATASET.rglob(Path(str(r["image"])).name))
            if not matches:
                print("missing:", r["image"])
                continue
            src = matches[0]

        dst_name = f"{state}_{i:03d}__{Path(str(r['image'])).name}"
        dst = OUT / state / dst_name
        shutil.copy2(src, dst)

        rows.append({
            "state": state,
            "export_file": str(dst.relative_to(OUT)),
            "source_image": r["image"],
        })

    pd.DataFrame(rows).to_csv(OUT / f"{state}_manifest.csv", index=False)

print("DONE:", OUT.resolve())
