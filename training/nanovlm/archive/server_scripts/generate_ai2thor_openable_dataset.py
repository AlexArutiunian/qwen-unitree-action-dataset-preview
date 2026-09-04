#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate AI2-THOR open/closed object-state MCQ dataset.

Output:
  <out_dir>/images/*.jpg
  <out_dir>/metadata.jsonl
  <out_dir>/metadata.csv

Recommended first test:
  xvfb-run -s "-screen 0 1024x768x24" python generate_ai2thor_openable_dataset.py --target-images 10

For real generation:
  python generate_ai2thor_openable_dataset.py --platform cloud --target-images 5000 --width 640 --height 480
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="ai2thor_openable_dataset")
    p.add_argument("--target-images", type=int, default=10, help="Number of RGB frames to save. Each frame creates 3 MCQ examples.")
    p.add_argument("--max-scenes", type=int, default=120)
    p.add_argument("--views-per-scene", type=int, default=40)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--quality", type=str, default="Low")
    p.add_argument("--grid-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--platform", type=str, default="auto", choices=["auto", "normal", "cloud"],
                   help="normal = regular Unity build, usually with xvfb; cloud = CloudRendering Vulkan; auto tries cloud then normal")
    p.add_argument("--exclude-doors", action="store_true", default=True)
    p.add_argument("--include-doors", action="store_true", help="Override and include Door objects in generation.")
    p.add_argument("--allow-types", nargs="*", default=None,
                   help="Optional list of AI2-THOR object types, e.g. Drawer Cabinet Fridge Microwave Safe Box")
    p.add_argument("--jpg-quality", type=int, default=92)
    p.add_argument("--force-action", action="store_true", default=True)
    return p.parse_args()


def humanize_obj_type(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", str(name)).lower().strip()


def shuffled_mcq(correct_text, distractors, rng):
    choices = [correct_text] + list(distractors)
    # unique, keep first occurrence
    out = []
    for x in choices:
        if x not in out:
            out.append(x)
    if len(out) != 4:
        raise ValueError(f"Bad choices: {out}")
    rng.shuffle(out)
    return out, out.index(correct_text)


def make_examples_for_frame(scene, obj, is_open, rel_image_path, image_idx, rng):
    obj_type = obj.get("objectType", "Object")
    obj_name = humanize_obj_type(obj_type)
    state_word = "open" if is_open else "closed"

    examples = []

    # 1) Direct state MCQ
    q = rng.choice([
        f"What is the state of the {obj_name}?",
        f"Which state best describes the {obj_name}?",
        f"Is the {obj_name} open or closed?",
    ])
    choices, gt_idx = shuffled_mcq(state_word, ["closed" if is_open else "open", "not visible", "cannot tell"], rng)
    examples.append({
        "id": f"img{image_idx:06d}_state",
        "image": rel_image_path,
        "scene": scene,
        "object_id": obj.get("objectId"),
        "object_type": obj_type,
        "question": q,
        "choices": choices,
        "gt_idx": gt_idx,
        "answer": state_word,
        "task": "openable_state",
    })

    # 2) Is open? yes/no
    q = rng.choice([
        f"Is the {obj_name} open?",
        f"Can you see that the {obj_name} is open?",
    ])
    correct = "yes" if is_open else "no"
    choices, gt_idx = shuffled_mcq(correct, ["no" if correct == "yes" else "yes", "not visible", "cannot tell"], rng)
    examples.append({
        "id": f"img{image_idx:06d}_isopen",
        "image": rel_image_path,
        "scene": scene,
        "object_id": obj.get("objectId"),
        "object_type": obj_type,
        "question": q,
        "choices": choices,
        "gt_idx": gt_idx,
        "answer": correct,
        "task": "is_open_yesno",
    })

    # 3) Is closed? yes/no
    q = rng.choice([
        f"Is the {obj_name} closed?",
        f"Can you see that the {obj_name} is closed?",
    ])
    correct = "yes" if not is_open else "no"
    choices, gt_idx = shuffled_mcq(correct, ["no" if correct == "yes" else "yes", "not visible", "cannot tell"], rng)
    examples.append({
        "id": f"img{image_idx:06d}_isclosed",
        "image": rel_image_path,
        "scene": scene,
        "object_id": obj.get("objectId"),
        "object_type": obj_type,
        "question": q,
        "choices": choices,
        "gt_idx": gt_idx,
        "answer": correct,
        "task": "is_closed_yesno",
    })

    return examples


def build_controller(args):
    from ai2thor.controller import Controller

    common = dict(
        width=args.width,
        height=args.height,
        gridSize=args.grid_size,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        quality=args.quality,
        visibilityDistance=2.0,
        fieldOfView=90,
    )

    if args.platform in ("auto", "cloud"):
        try:
            from ai2thor.platform import CloudRendering
            print("Trying AI2-THOR CloudRendering...")
            return Controller(platform=CloudRendering, server_timeout=300, server_start_timeout=300, **common)
        except Exception as e:
            print("CloudRendering failed:", repr(e))
            if args.platform == "cloud":
                raise

    print("Trying normal AI2-THOR Controller. If headless, run with xvfb-run.")
    return Controller(server_timeout=300, server_start_timeout=300, **common)


def get_scenes():
    return (
        [f"FloorPlan{i}" for i in range(1, 31)] +       # kitchens
        [f"FloorPlan{i}" for i in range(201, 231)] +    # living rooms
        [f"FloorPlan{i}" for i in range(301, 331)] +    # bedrooms
        [f"FloorPlan{i}" for i in range(401, 431)]      # bathrooms
    )


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_jsonl = out_dir / "metadata.jsonl"
    meta_csv = out_dir / "metadata.csv"

    exclude_types = set() if args.include_doors else {"Door", "Doorway"}
    allow_types = set(args.allow_types) if args.allow_types else None

    def should_use_object(o):
        t = o.get("objectType", "")
        if not o.get("openable", False):
            return False
        if args.exclude_doors and not args.include_doors:
            if t in exclude_types or "door" in t.lower():
                return False
        if allow_types is not None and t not in allow_types:
            return False
        return True

    controller = build_controller(args)
    all_examples = []
    images_saved = 0
    attempts = 0

    scenes = get_scenes()
    rng.shuffle(scenes)
    scenes = scenes[: min(args.max_scenes, len(scenes))]

    pbar = tqdm(total=args.target_images, desc="saved images")
    try:
        for scene in scenes:
            if images_saved >= args.target_images:
                break
            try:
                controller.reset(scene)
                controller.step(action="Initialize", gridSize=args.grid_size)
                ev_pos = controller.step(action="GetReachablePositions")
                reachable = ev_pos.metadata.get("actionReturn", []) or []
                if not reachable:
                    print("No reachable positions:", scene)
                    continue
            except Exception as e:
                print("Scene failed", scene, repr(e))
                continue

            for _view in range(args.views_per_scene):
                if images_saved >= args.target_images:
                    break

                attempts += 1
                pos = rng.choice(reachable)
                rot = rng.choice([0, 45, 90, 135, 180, 225, 270, 315])
                hor = rng.choice([-30, -15, 0, 15, 30, 45])
                try:
                    event = controller.step(
                        action="TeleportFull",
                        x=pos["x"], y=pos["y"], z=pos["z"],
                        rotation={"x": 0, "y": rot, "z": 0},
                        horizon=hor,
                        standing=True,
                        forceAction=True,
                    )
                except Exception:
                    continue

                objs = event.metadata.get("objects", [])
                visible_openables = [o for o in objs if should_use_object(o) and o.get("visible", False)]
                rng.shuffle(visible_openables)

                for obj in visible_openables[:4]:
                    if images_saved >= args.target_images:
                        break
                    oid = obj.get("objectId")
                    if not oid:
                        continue

                    # Balance open/closed roughly by alternating target state.
                    target_open = (images_saved % 2 == 0)
                    action = "OpenObject" if target_open else "CloseObject"
                    try:
                        if action == "OpenObject":
                            ev2 = controller.step(action=action, objectId=oid, openness=1.0, forceAction=args.force_action)
                        else:
                            ev2 = controller.step(action=action, objectId=oid, forceAction=args.force_action)
                    except Exception:
                        continue

                    if not ev2.metadata.get("lastActionSuccess", False):
                        continue

                    new_obj = None
                    for oo in ev2.metadata.get("objects", []):
                        if oo.get("objectId") == oid:
                            new_obj = oo
                            break
                    if new_obj is None or not new_obj.get("visible", False):
                        continue

                    is_open = bool(new_obj.get("isOpen", target_open))
                    # Keep only if requested state actually happened.
                    if is_open != target_open:
                        continue

                    obj_type = new_obj.get("objectType", "Object")
                    state_word = "open" if is_open else "closed"
                    img_name = f"img_{images_saved:06d}_{scene}_{obj_type}_{state_word}.jpg"
                    img_path = img_dir / img_name
                    rel_path = str(Path("images") / img_name)

                    frame = ev2.frame
                    Image.fromarray(frame).save(img_path, quality=args.jpg_quality)

                    exs = make_examples_for_frame(scene, new_obj, is_open, rel_path, images_saved, rng)
                    all_examples.extend(exs)
                    images_saved += 1
                    pbar.update(1)
    finally:
        pbar.close()
        try:
            controller.stop()
        except Exception:
            pass

    with open(meta_jsonl, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    pd.DataFrame(all_examples).to_csv(meta_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    print("\nDONE")
    print("images_saved:", images_saved)
    print("mcq_examples:", len(all_examples))
    print("attempts:", attempts)
    print("out_dir:", out_dir.resolve())
    print("metadata:", meta_jsonl.resolve())
    if all_examples:
        print("\nPreview:")
        print(pd.DataFrame(all_examples).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
