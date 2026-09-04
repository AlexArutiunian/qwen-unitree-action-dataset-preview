#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datasets import load_dataset
import json

DATASET = "keplerccc/ManipulationVQA-60k"

print("Loading:", DATASET)
ds = load_dataset(DATASET)

print("\nSPLITS:")
print(ds)

for split in ds:
    print("\n" + "=" * 100)
    print("SPLIT:", split)
    d = ds[split]
    print("rows:", len(d))
    print("features:")
    print(d.features)

    n = min(3, len(d))
    for i in range(n):
        print("\n--- sample", i, "---")
        row = d[i]
        for k, v in row.items():
            if k.lower() in {"image", "images", "rgb"}:
                print(k, type(v), getattr(v, "size", None))
            else:
                s = str(v)
                print(k, type(v), s[:1000])
