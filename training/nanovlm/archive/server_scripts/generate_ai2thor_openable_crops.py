import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ai2thor.controller import Controller


def humanize_obj_type(name: str) -> str:
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower().strip()


def get_controller(platform, width, height, quality):
    if platform == "cloud":
        from ai2thor.platform import CloudRendering
        return Controller(
            platform=CloudRendering,
            width=width,
            height=height,
            quality=quality,
            gridSize=0.25,
            renderDepthImage=False,
            renderInstanceSegmentation=True,
            visibilityDistance=2.5,
            fieldOfView=90,
        )

    return Controller(
        width=width,
        height=height,
        quality=quality,
        gridSize=0.25,
        renderDepthImage=False,
        renderInstanceSegmentation=True,
        visibilityDistance=2.5,
        fieldOfView=90,
    )


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def pad_bbox(bbox, width, height, pad_ratio=0.20):
    x1, y1, x2, y2 = bbox
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    pad = int(max(bw, bh) * pad_ratio)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width - 1, x2 + pad)
    y2 = min(height - 1, y2 + pad)

    return x1, y1, x2, y2


def crop_and_square(image: Image.Image, bbox, out_size=384):
    w, h = image.size
    x1, y1, x2, y2 = bbox

    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    side = max(bw, bh)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    nx1 = max(0, cx - side // 2)
    ny1 = max(0, cy - side // 2)
    nx2 = min(w - 1, nx1 + side)
    ny2 = min(h - 1, ny1 + side)

    # если упёрлись в границу — сдвигаем назад
    nx1 = max(0, nx2 - side)
    ny1 = max(0, ny2 - side)

    crop = image.crop((nx1, ny1, nx2 + 1, ny2 + 1))
    crop = crop.resize((out_size, out_size), Image.BICUBIC)
    return crop


def choose_openable_objects(event, allow_types, exclude_types):
    objects = []
    for obj in event.metadata["objects"]:
        if not obj.get("openable", False):
            continue

        obj_type = obj.get("objectType", "")
        if allow_types and obj_type not in allow_types:
            continue

        if obj_type in exclude_types:
            continue

        if not obj.get("visible", False):
            continue

        objects.append(obj)

    return objects


def set_object_state(controller, object_id, open_state: bool):
    if open_state:
        event = controller.step(
            action="OpenObject",
            objectId=object_id,
            openness=1.0,
            forceAction=True,
        )
    else:
        event = controller.step(
            action="CloseObject",
            objectId=object_id,
            forceAction=True,
        )
    return event


def get_object_by_id(event, object_id):
    for obj in event.metadata["objects"]:
        if obj["objectId"] == object_id:
            return obj
    return None


def get_instance_mask(event, object_id):
    masks = getattr(event, "instance_masks", None)
    if masks is None:
        return None

    if object_id not in masks:
        return None

    return masks[object_id]


def make_mcq_rows(base_id, image_path, scene, object_id, object_type, state):
    obj_name = humanize_obj_type(object_type)

    rows = []

    # 1) main state
    choices = ["open", "closed", "not visible", "cannot tell"]
    random.shuffle(choices)
    gt = choices.index(state)
    rows.append({
        "id": f"{base_id}_state",
        "image": image_path,
        "scene": scene,
        "object_id": object_id,
        "object_type": object_type,
        "question": "What is the state of this object?",
        "choices": str(choices),
        "gt_idx": gt,
        "answer": state,
        "task": "crop_openable_state",
    })

    # 2) object-specific question
    choices = ["open", "closed", "not visible", "cannot tell"]
    random.shuffle(choices)
    gt = choices.index(state)
    rows.append({
        "id": f"{base_id}_state_named",
        "image": image_path,
        "scene": scene,
        "object_id": object_id,
        "object_type": object_type,
        "question": f"What is the state of the {obj_name}?",
        "choices": str(choices),
        "gt_idx": gt,
        "answer": state,
        "task": "crop_openable_state_named",
    })

    # 3) yes/no open
    answer = "yes" if state == "open" else "no"
    choices = ["yes", "no", "not visible", "cannot tell"]
    random.shuffle(choices)
    gt = choices.index(answer)
    rows.append({
        "id": f"{base_id}_isopen",
        "image": image_path,
        "scene": scene,
        "object_id": object_id,
        "object_type": object_type,
        "question": "Is this object open?",
        "choices": str(choices),
        "gt_idx": gt,
        "answer": answer,
        "task": "crop_is_open_yesno",
    })

    # 4) yes/no closed
    answer = "yes" if state == "closed" else "no"
    choices = ["yes", "no", "not visible", "cannot tell"]
    random.shuffle(choices)
    gt = choices.index(answer)
    rows.append({
        "id": f"{base_id}_isclosed",
        "image": image_path,
        "scene": scene,
        "object_id": object_id,
        "object_type": object_type,
        "question": "Is this object closed?",
        "choices": str(choices),
        "gt_idx": gt,
        "answer": answer,
        "task": "crop_is_closed_yesno",
    })

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out-dir", type=str, default="ai2thor_openable_crops")
    parser.add_argument("--target-images", type=int, default=1000)
    parser.add_argument("--platform", type=str, default="cloud", choices=["cloud", "normal"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--quality", type=str, default="Low")

    parser.add_argument("--max-scenes", type=int, default=120)
    parser.add_argument("--views-per-scene", type=int, default=80)
    parser.add_argument("--min-bbox-area", type=int, default=3500)
    parser.add_argument("--min-bbox-side", type=int, default=45)

    parser.add_argument(
        "--allow-types",
        nargs="*",
        default=["Drawer", "Cabinet", "Fridge", "Microwave", "Safe", "Box"],
    )
    parser.add_argument(
        "--exclude-types",
        nargs="*",
        default=["Door", "Doorway"],
    )

    args = parser.parse_args()

    random.seed(42)

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    full_dir = out_dir / "full_images"

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)

    scenes = (
        [f"FloorPlan{i}" for i in range(1, 31)] +
        [f"FloorPlan{i}" for i in range(201, 231)] +
        [f"FloorPlan{i}" for i in range(301, 331)] +
        [f"FloorPlan{i}" for i in range(401, 431)]
    )
    scenes = scenes[:args.max_scenes]

    controller = get_controller(args.platform, args.width, args.height, args.quality)

    rows = []
    saved_images = 0
    state_counter = {"open": 0, "closed": 0}
    type_counter = {}

    try:
        pbar = tqdm(total=args.target_images, desc="Generating object crops")

        for scene in scenes:
            if saved_images >= args.target_images:
                break

            try:
                controller.reset(scene)
            except Exception as e:
                print("skip scene", scene, e)
                continue

            event = controller.step(action="GetReachablePositions")
            positions = event.metadata.get("actionReturn", [])
            if not positions:
                continue

            for _ in range(args.views_per_scene):
                if saved_images >= args.target_images:
                    break

                pos = random.choice(positions)
                rot = random.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
                horizon = random.choice([-15, 0, 15, 30])

                event = controller.step(
                    action="Teleport",
                    position=pos,
                    rotation={"x": 0, "y": rot, "z": 0},
                    horizon=horizon,
                    standing=True,
                )

                candidates = choose_openable_objects(
                    event,
                    allow_types=set(args.allow_types),
                    exclude_types=set(args.exclude_types),
                )

                random.shuffle(candidates)

                for obj in candidates:
                    if saved_images >= args.target_images:
                        break

                    object_id = obj["objectId"]
                    object_type = obj["objectType"]

                    # баланс open/closed
                    target_open = state_counter["open"] <= state_counter["closed"]
                    if random.random() < 0.25:
                        target_open = not target_open

                    event2 = set_object_state(controller, object_id, target_open)
                    obj2 = get_object_by_id(event2, object_id)
                    if obj2 is None or not obj2.get("visible", False):
                        continue

                    actual_state = "open" if obj2.get("isOpen", False) else "closed"

                    mask = get_instance_mask(event2, object_id)
                    if mask is None:
                        continue

                    bbox = bbox_from_mask(mask)
                    if bbox is None:
                        continue

                    x1, y1, x2, y2 = bbox
                    bw = x2 - x1 + 1
                    bh = y2 - y1 + 1
                    area = bw * bh

                    if area < args.min_bbox_area:
                        continue
                    if min(bw, bh) < args.min_bbox_side:
                        continue

                    full_img = Image.fromarray(event2.frame).convert("RGB")

                    padded_bbox = pad_bbox(bbox, args.width, args.height, pad_ratio=0.25)
                    crop_img = crop_and_square(full_img, padded_bbox, out_size=args.crop_size)

                    base = f"img_{saved_images:06d}_{scene}_{object_type}_{actual_state}"

                    crop_name = f"{base}_crop.jpg"
                    full_name = f"{base}_full.jpg"

                    crop_rel = f"images/{crop_name}"
                    full_rel = f"full_images/{full_name}"

                    crop_img.save(img_dir / crop_name, quality=95)
                    full_img.save(full_dir / full_name, quality=90)

                    base_id = f"img{saved_images:06d}"

                    new_rows = make_mcq_rows(
                        base_id=base_id,
                        image_path=crop_rel,
                        scene=scene,
                        object_id=object_id,
                        object_type=object_type,
                        state=actual_state,
                    )

                    for r in new_rows:
                        r["full_image"] = full_rel
                        r["bbox"] = str([int(v) for v in bbox])
                        r["bbox_padded"] = str([int(v) for v in padded_bbox])
                        r["bbox_area"] = int(area)
                        r["bbox_w"] = int(bw)
                        r["bbox_h"] = int(bh)

                    rows.extend(new_rows)

                    saved_images += 1
                    state_counter[actual_state] += 1
                    type_counter[object_type] = type_counter.get(object_type, 0) + 1

                    pbar.update(1)
                    break

        pbar.close()

    finally:
        try:
            controller.stop()
        except Exception:
            pass

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metadata.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nDONE")
    print("out_dir:", out_dir)
    print("saved crop images:", saved_images)
    print("rows:", len(df))
    print("state_counter:", state_counter)
    print("type_counter:", type_counter)

    if len(df):
        print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
