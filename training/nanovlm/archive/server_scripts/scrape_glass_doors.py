#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from icrawler.builtin import BingImageCrawler
import time

OUT = Path("web_glass_door_raw")
OUT.mkdir(exist_ok=True)

# passable = проход свободен / дверь распахнута
PASSABLE_QUERIES = [
    "open glass office door",
    "open glass entrance door",
    "open glass door corridor",
    "glass office door wide open",
    "open transparent door office",
    "open glass storefront door",
    "open office entrance glass door",
    "open glass lobby door",
    "open glass door passage",
    "opened glass door hallway",
]

# not_passable = проход заблокирован / стеклянная дверь закрыта
NOT_PASSABLE_QUERIES = [
    "closed glass office door",
    "closed glass entrance door",
    "closed glass door corridor",
    "transparent glass door closed",
    "closed office entrance glass door",
    "closed glass lobby door",
    "glass door blocking passage",
    "closed glass door hallway",
    "closed glass storefront door",
    "closed transparent office door",
]

def crawl_group(group_name, queries, per_query=60):
    group_dir = OUT / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    for qi, q in enumerate(queries):
        subdir = group_dir / f"q{qi:02d}"
        subdir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 100)
        print("GROUP:", group_name)
        print("QUERY:", q)
        print("OUT:", subdir)

        crawler = BingImageCrawler(
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=4,
            storage={"root_dir": str(subdir)},
        )

        try:
            crawler.crawl(
                keyword=q,
                max_num=per_query,
                min_size=(160, 160),
                file_idx_offset="auto",
            )
        except Exception as e:
            print("ERROR:", repr(e))

        time.sleep(2)

if __name__ == "__main__":
    crawl_group("passable", PASSABLE_QUERIES, per_query=60)
    crawl_group("not_passable", NOT_PASSABLE_QUERIES, per_query=60)

    print("\nDONE")
    print("Saved to:", OUT.resolve())
