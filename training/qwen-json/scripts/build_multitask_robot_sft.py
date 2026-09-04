#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUMANOID_ROOT = PROJECT_ROOT.parent


MOTION_SYSTEM = """You are a Unitree G1 robot-control assistant.
The user message starts with /motion_json or /robot_qa.
For /motion_json return only a valid compact JSON array, no markdown and no explanations.
Compact motion format:
{"frame":{"<joint_alias>":<degrees>},"duration":<seconds>}
{"repeat":[<steps>],"times":<integer>}
Use short aliases: L_ for left, R_ for right, omit _joint. Every motion starts from neutral pose; omitted joints keep their previous value. End every motion with {"frame":{},"duration":3.0 to 5.0}.
For /robot_qa answer in Russian, briefly and technically; do not output motion JSON unless the user explicitly asks for /motion_json."""


QA_SYSTEM = """You are a robotics assistant for Unitree G1.
The user message starts with /robot_qa or /motion_json.
For /robot_qa answer in Russian, briefly and technically. Use URDF/FK facts when provided. Be careful: URDF axis alone gives the mathematical rotation axis; FK observations tell how the palm/end-effector moves in robot coordinates.
For /motion_json return only compact robot JSON."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def make_motion_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        messages = row["messages"]
        command = messages[1]["content"]
        if not command.startswith("/motion_json"):
            command = "/motion_json\n" + command
        out.append(
            {
                "sample_id": row.get("sample_id"),
                "source_id": row.get("source_id"),
                "task": "motion_json",
                "split": split,
                "messages": [
                    {"role": "system", "content": MOTION_SYSTEM},
                    {"role": "user", "content": command},
                    {"role": "assistant", "content": messages[2]["content"]},
                ],
            }
        )
    return out


def parse_urdf_joints(path: Path) -> dict[str, dict[str, str]]:
    root = ET.parse(path).getroot()
    joints: dict[str, dict[str, str]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        limit = joint.find("limit")
        joints[name] = {
            "type": joint.attrib.get("type", ""),
            "parent": parent.attrib.get("link", "") if parent is not None else "",
            "child": child.attrib.get("link", "") if child is not None else "",
            "axis": axis.attrib.get("xyz", "") if axis is not None else "",
            "lower": limit.attrib.get("lower", "") if limit is not None else "",
            "upper": limit.attrib.get("upper", "") if limit is not None else "",
        }
    return joints


def deg_from_rad_text(value: str) -> str:
    try:
        return f"{float(value) * 180.0 / 3.141592653589793:.1f}"
    except Exception:
        return value


def joint_alias(name: str) -> str:
    if name.startswith("left_") and name.endswith("_joint"):
        return "L_" + name[len("left_") : -len("_joint")]
    if name.startswith("right_") and name.endswith("_joint"):
        return "R_" + name[len("right_") : -len("_joint")]
    if name.startswith("waist_") and name.endswith("_joint"):
        return name[: -len("_joint")]
    return name


def human_joint_role(joint: str) -> str:
    side = "левую" if joint.startswith("left_") else ("правую" if joint.startswith("right_") else "")
    if "shoulder_pitch" in joint:
        return f"двигает {side} руку в плоскости вперед-назад/вверх-вниз через плечо".strip()
    if "shoulder_roll" in joint:
        return f"отводит или приводит {side} руку вбок через плечо".strip()
    if "shoulder_yaw" in joint:
        return f"вращает {side} руку вокруг плечевой оси".strip()
    if "elbow" in joint:
        return f"сгибает или разгибает {side} локоть".strip()
    if "wrist_roll" in joint:
        return f"вращает {side} кисть по roll".strip()
    if "wrist_pitch" in joint:
        return f"наклоняет {side} кисть по pitch".strip()
    if "wrist_yaw" in joint:
        return f"поворачивает {side} кисть по yaw".strip()
    if "waist_yaw" in joint:
        return "поворачивает корпус вокруг вертикальной оси"
    if "waist_roll" in joint:
        return "наклоняет корпус влево-вправо"
    if "waist_pitch" in joint:
        return "наклоняет корпус вперед-назад"
    return "управляет соответствующим звеном робота"


def load_fk_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def qa_item(sample_id: int, user: str, assistant: str, split: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "source_id": sample_id,
        "task": "robot_qa",
        "split": split,
        "messages": [
            {"role": "system", "content": QA_SYSTEM},
            {"role": "user", "content": "/robot_qa\n" + user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
    }


def build_qa_rows(direction_csv: Path, urdf: Path) -> list[dict[str, Any]]:
    fk_rows = load_fk_rows(direction_csv)
    joints = parse_urdf_joints(urdf)
    rows: list[dict[str, Any]] = []
    sid = 1_000_000

    rows.append(
        qa_item(
            sid,
            "Что такое FK и чем он отличается от IK?",
            (
                "FK, или прямая кинематика, считает положение звена или end-effector по заданным углам суставов. "
                "IK, или обратная кинематика, решает обратную задачу: по желаемой позиции кисти подбирает углы суставов."
            ),
            "train",
        )
    )
    sid += 1
    rows.append(
        qa_item(
            sid,
            "Можно ли по одному axis из URDF точно сказать, куда уйдет кисть робота?",
            (
                "Не всегда. Axis из URDF говорит, вокруг какой локальной оси вращается сустав и как определяется знак угла. "
                "Куда уйдет кисть в координатах робота надежно показывает FK, потому что нужен весь кинематический chain и текущая поза."
            ),
            "train",
        )
    )
    sid += 1
    rows.append(
        qa_item(
            sid,
            "Какие оси робота используются в FK-наблюдениях?",
            "В этих FK-наблюдениях принято: +X вперед, +Y влево, +Z вверх.",
            "train",
        )
    )
    sid += 1
    rows.append(
        qa_item(
            sid,
            "Почему для JSON-режима нельзя отвечать объяснением вместо массива?",
            (
                "Потому что JSON-режим исполняется плеером робота. В режиме /motion_json ответ должен быть только валидным JSON-массивом; "
                "пояснения допустимы только в режиме /robot_qa."
            ),
            "train",
        )
    )
    sid += 1

    for r in fk_rows:
        joint = r["joint"]
        ee = r["end_effector"]
        meta = joints.get(joint, {})
        alias = joint_alias(joint)
        role = human_joint_role(joint)
        axis = r["axis"]
        parent = meta.get("parent", "")
        child = meta.get("child", "")
        lo = deg_from_rad_text(meta.get("lower", ""))
        hi = deg_from_rad_text(meta.get("upper", ""))
        plus = r["plus_direction"]
        minus = r["minus_direction"]
        plus_delta = r["plus_delta"]
        minus_delta = r["minus_delta"]

        rows.append(
            qa_item(
                sid,
                f"Что делает сустав {joint}?",
                (
                    f"{joint} ({alias}) {role}. В URDF это сустав типа {r['type']} между parent link {parent} "
                    f"и child link {child}; локальная ось вращения axis=[{axis}]."
                ),
                "train",
            )
        )
        sid += 1
        rows.append(
            qa_item(
                sid,
                f"Вокруг какой оси вращается {joint} и какой compact alias использовать?",
                f"{joint} вращается вокруг локальной оси axis=[{axis}]. В compact JSON его alias: {alias}.",
                "train",
            )
        )
        sid += 1
        if lo and hi:
            rows.append(
                qa_item(
                    sid,
                    f"Какие лимиты у {joint}?",
                    f"По URDF лимиты {joint}: примерно от {lo} до {hi} градусов. В JSON нельзя выходить за эти пределы.",
                    "train",
                )
            )
            sid += 1
        rows.append(
            qa_item(
                sid,
                f"Куда сместится {ee}, если увеличить {joint} на +10 градусов?",
                f"По FK-наблюдению {ee} при {joint}=+10 градусов смещается: {plus}; delta примерно {plus_delta}.",
                "train",
            )
        )
        sid += 1
        rows.append(
            qa_item(
                sid,
                f"Куда сместится {ee}, если поставить {joint} на -10 градусов?",
                f"По FK-наблюдению {ee} при {joint}=-10 градусов смещается: {minus}; delta примерно {minus_delta}.",
                "train",
            )
        )
        sid += 1
        rows.append(
            qa_item(
                sid,
                f"Какой знак угла выбрать для {joint}, если нужно направление '{minus}' для {ee}?",
                f"Для направления '{minus}' по этому FK-наблюдению нужен отрицательный угол {joint}.",
                "train",
            )
        )
        sid += 1

    # Reserve a small deterministic QA validation slice.
    for i, row in enumerate(rows):
        if i % 10 == 0:
            row["split"] = "val"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-sft", type=Path, default=PROJECT_ROOT / "outputs" / "dataset_100_statis_compact_slow_sft")
    parser.add_argument("--direction-csv", type=Path, default=HUMANOID_ROOT / "prompt_engineering" / "g1_upper_body_joint_direction.csv")
    parser.add_argument("--urdf", type=Path, default=HUMANOID_ROOT / "prompt_engineering" / "g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "dataset_100_statis_multitask_robot_sft")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_motion = make_motion_rows(read_jsonl(args.motion_sft / "train_compact_4096.jsonl"), "train")
    val_motion = make_motion_rows(read_jsonl(args.motion_sft / "val_compact_4096.jsonl"), "val")
    qa_rows = build_qa_rows(args.direction_csv, args.urdf)
    train = train_motion + [r for r in qa_rows if r["split"] == "train"]
    val = val_motion + [r for r in qa_rows if r["split"] == "val"]

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(val)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        write_jsonl(args.out_dir / f"{name}.jsonl", rows)
        write_jsonl(args.out_dir / f"{name}_compact_4096.jsonl", rows)
        write_jsonl(args.out_dir / f"{name}_compact_8192.jsonl", rows)

    manifest = {
        "dataset_kind": "statis100_motion_json_plus_robot_qa_multitask",
        "motion_sft": str(args.motion_sft),
        "direction_csv": str(args.direction_csv),
        "urdf": str(args.urdf),
        "out_dir": str(args.out_dir),
        "train_rows": len(train),
        "val_rows": len(val),
        "train_motion_rows": len(train_motion),
        "val_motion_rows": len(val_motion),
        "qa_rows": len(qa_rows),
        "mode_prefixes": ["/motion_json", "/robot_qa"],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
