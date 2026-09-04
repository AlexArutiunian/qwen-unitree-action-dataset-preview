#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUMANOID_ROOT = PROJECT_ROOT.parent


UPPER_BODY_JOINTS = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


QA_SYSTEM = """You are a robotics assistant for Unitree G1 FTP.
The user message starts with /robot_qa or /motion_json.
For /robot_qa answer in Russian, briefly and technically.
Use the provided Unitree G1 FTP facts exactly: env_FTP.yaml uses resources/g1/scene_FTP_clean.xml -> resources/g1/g1_inspire_clean.xml. The real robot URDF is resources/g1/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf.
URDF/MJCF axis gives the local rotation axis. FK observations give how the palm moves in robot coordinates (+X forward, +Y left, +Z up).
Do not invent limits or directions. If asked for motion JSON, use /motion_json compact JSON mode."""


def alias(name: str) -> str:
    if name.startswith("left_") and name.endswith("_joint"):
        return "L_" + name[len("left_") : -len("_joint")]
    if name.startswith("right_") and name.endswith("_joint"):
        return "R_" + name[len("right_") : -len("_joint")]
    if name.startswith("waist_") and name.endswith("_joint"):
        return name[: -len("_joint")]
    if (name.startswith("L_") or name.startswith("R_")) and name.endswith("_joint"):
        return name[: -len("_joint")]
    return name


def deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def fmt_deg(x: float) -> str:
    # Stable one-decimal formatting matches the rest of the project prompts.
    return f"{x:.1f}"


def parse_mjcf_joints(path: Path) -> dict[str, dict[str, Any]]:
    root = ET.parse(path).getroot()
    out: dict[str, dict[str, Any]] = {}

    def walk_body(body: ET.Element, parent_body: str | None) -> None:
        body_name = body.attrib.get("name", "")
        for joint in body.findall("joint"):
            name = joint.attrib.get("name", "")
            if not name:
                continue
            rng = joint.attrib.get("range", "").split()
            lo_hi = None
            if len(rng) == 2:
                lo_hi = (deg(float(rng[0])), deg(float(rng[1])))
            out[name] = {
                "source": str(path),
                "type": joint.attrib.get("type", "hinge"),
                "axis": joint.attrib.get("axis", ""),
                "range_deg": lo_hi,
                "parent_body": parent_body or "",
                "child_body": body_name,
            }
        for child in body.findall("body"):
            walk_body(child, body_name)

    world = root.find("worldbody")
    if world is not None:
        for body in world.findall("body"):
            walk_body(body, None)
    return out


def parse_urdf_joints(path: Path) -> dict[str, dict[str, Any]]:
    root = ET.parse(path).getroot()
    out: dict[str, dict[str, Any]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        if not name:
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        limit = joint.find("limit")
        lo_hi = None
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            lo_hi = (deg(float(limit.attrib["lower"])), deg(float(limit.attrib["upper"])))
        out[name] = {
            "source": str(path),
            "type": joint.attrib.get("type", ""),
            "axis": axis.attrib.get("xyz", "") if axis is not None else "",
            "range_deg": lo_hi,
            "parent_link": parent.attrib.get("link", "") if parent is not None else "",
            "child_link": child.attrib.get("link", "") if child is not None else "",
        }
    return out


def load_fk(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {(r["joint"], r["end_effector"]): r for r in rows}


def role(joint: str) -> str:
    side = "левого" if joint.startswith("left_") else ("правого" if joint.startswith("right_") else "")
    if "shoulder_pitch" in joint:
        return f"управляет pitch {side} плеча: движение руки вперед/назад и вверх/вниз".strip()
    if "shoulder_roll" in joint:
        return f"управляет roll {side} плеча: отведение/приведение руки вбок".strip()
    if "shoulder_yaw" in joint:
        return f"управляет yaw {side} плеча: вращение руки вокруг плечевой оси".strip()
    if "elbow" in joint:
        return f"сгибает и разгибает {side} локтя".strip()
    if "wrist_roll" in joint:
        return f"вращает {side} запястье по roll".strip()
    if "wrist_pitch" in joint:
        return f"наклоняет {side} запястье по pitch".strip()
    if "wrist_yaw" in joint:
        return f"поворачивает {side} запястье по yaw".strip()
    if joint == "waist_yaw_joint":
        return "поворачивает корпус вокруг вертикальной оси"
    if joint == "waist_roll_joint":
        return "наклоняет корпус влево-вправо"
    if joint == "waist_pitch_joint":
        return "наклоняет корпус вперед-назад"
    return "управляет соответствующим звеном робота"


def qa(sample_id: int, user: str, assistant: str, split: str = "train") -> dict[str, Any]:
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


def compare_sources(mjcf: dict[str, dict[str, Any]], urdf: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name in UPPER_BODY_JOINTS:
        m = mjcf.get(name, {})
        u = urdf.get(name, {})
        m_range = m.get("range_deg")
        u_range = u.get("range_deg")
        range_diff = ""
        if m_range and u_range:
            range_diff = f"{abs(m_range[0]-u_range[0]):.4f},{abs(m_range[1]-u_range[1]):.4f}"
        rows.append(
            {
                "joint": name,
                "alias": alias(name),
                "mjcf_axis": m.get("axis", ""),
                "urdf_axis": u.get("axis", ""),
                "mjcf_range_deg": f"{fmt_deg(m_range[0])}..{fmt_deg(m_range[1])}" if m_range else "",
                "urdf_range_deg": f"{fmt_deg(u_range[0])}..{fmt_deg(u_range[1])}" if u_range else "",
                "range_abs_diff_deg": range_diff,
                "urdf_parent": u.get("parent_link", ""),
                "urdf_child": u.get("child_link", ""),
            }
        )
    return rows


def build_rows(mjcf_path: Path, urdf_path: Path, fk_csv: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mjcf = parse_mjcf_joints(mjcf_path)
    urdf = parse_urdf_joints(urdf_path)
    fk = load_fk(fk_csv)
    compare = compare_sources(mjcf, urdf)
    rows: list[dict[str, Any]] = []
    sid = 2_000_000

    rows.extend(
        [
            qa(
                sid,
                "Какие файлы являются источниками правды для G1 FTP в этом проекте?",
                (
                    "Для симулятора env_FTP.yaml источник правды: resources/g1/scene_FTP_clean.xml, который подключает "
                    "resources/g1/g1_inspire_clean.xml. Для реального робота используется "
                    "resources/g1/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf. Для QA я ориентируюсь на env_FTP/MJCF, "
                    "а URDF использую как описание реального робота; по верхней части тела их оси и лимиты совпадают практически полностью."
                ),
            ),
            qa(
                sid + 1,
                "Что значат оси FK-наблюдений для G1 FTP?",
                "В FK-наблюдениях для G1 FTP принято: +X вперед, +Y влево, +Z вверх.",
            ),
            qa(
                sid + 2,
                "Почему нельзя просто по axis из URDF сказать, куда уйдет ладонь?",
                (
                    "Axis из URDF/MJCF задает локальную математическую ось вращения сустава. Куда уйдет ладонь в координатах робота, "
                    "надежно показывает FK, потому что учитываются вся цепочка звеньев, origin/rpy и текущая поза."
                ),
            ),
            qa(
                sid + 3,
                "Какой формат использовать в motion JSON для G1 FTP?",
                (
                    "В motion JSON используется compact alias формат: {\"frame\":{\"L_shoulder_pitch\":-120},\"duration\":2.0}. "
                    "Обычные joint frames не должны использовать name/angle объекты. Финальный hold: {\"frame\":{},\"duration\":3.0..5.0}."
                ),
            ),
            qa(
                sid + 4,
                "Что означает elbow=0 и elbow=90 у G1 FTP?",
                "Для локтей G1 FTP: 0 градусов означает сгиб примерно на 90 градусов, а +90 градусов означает почти прямой локоть.",
            ),
        ]
    )
    sid += 10

    for item in compare:
        name = item["joint"]
        a = item["alias"]
        axis = item["mjcf_axis"] or item["urdf_axis"]
        rng = item["mjcf_range_deg"] or item["urdf_range_deg"]
        parent = item["urdf_parent"]
        child = item["urdf_child"]
        lo, hi = rng.split("..")
        eff = "left_palm" if name.startswith("left_") else ("right_palm" if name.startswith("right_") else None)
        fk_row = fk.get((name, eff), {}) if eff else None
        if name.startswith("waist_"):
            fk_left = fk.get((name, "left_palm"), {})
            fk_right = fk.get((name, "right_palm"), {})
        else:
            fk_left = fk_right = {}

        fact = (
            f"{name} ({a}) в env_FTP/MJCF имеет axis=[{axis}], лимиты {lo}..{hi} градусов. "
            f"В URDF parent={parent}, child={child}. Роль: {role(name)}."
        )
        rows.append(qa(sid, f"Дай точную карточку сустава {name} для G1 FTP.", fact))
        sid += 1
        rows.append(qa(sid, f"Какие лимиты у {name}?", f"У {name} ({a}) лимиты: от {lo} до {hi} градусов. Это значение из env_FTP/MJCF; в JSON нельзя выходить за эти пределы."))
        sid += 1
        rows.append(qa(sid, f"Какой compact alias у {name} и вокруг какой оси он вращается?", f"Compact alias для {name}: {a}. Локальная ось вращения: axis=[{axis}]."))
        sid += 1
        rows.append(qa(sid, f"Между какими звеньями находится {name}?", f"{name} расположен между URDF parent link {parent} и child link {child}."))
        sid += 1
        rows.append(qa(sid, f"За что отвечает {name}?", f"{name} ({a}) {role(name)}."))
        sid += 1
        # Repeated exact-limit variants help reduce hallucinated limits.
        rows.append(qa(sid, f"Проверь утверждение: у {name} лимиты -92.5..92.5 градусов.", f"Не принимай это автоматически. Точные лимиты {name} для G1 FTP: {lo}..{hi} градусов. -92.5..92.5 подходит только для некоторых wrist pitch/yaw, но не для всех суставов."))
        sid += 1

        if fk_row:
            plus = fk_row["plus_direction"]
            minus = fk_row["minus_direction"]
            plus_delta = fk_row["plus_delta"]
            minus_delta = fk_row["minus_delta"]
            rows.append(qa(sid, f"Куда сместится {eff}, если увеличить {name} на +10 градусов?", f"По FK для G1 FTP: при {name}=+10 градусов {eff} смещается {plus}; delta примерно {plus_delta}."))
            sid += 1
            rows.append(qa(sid, f"Куда сместится {eff}, если поставить {name} на -10 градусов?", f"По FK для G1 FTP: при {name}=-10 градусов {eff} смещается {minus}; delta примерно {minus_delta}."))
            sid += 1
            rows.append(qa(sid, f"Какой знак {name} выбрать, чтобы {eff} сместилась '{plus}'?", f"Для направления '{plus}' у {eff} по FK нужен положительный угол {name}."))
            sid += 1
            rows.append(qa(sid, f"Какой знак {name} выбрать, чтобы {eff} сместилась '{minus}'?", f"Для направления '{minus}' у {eff} по FK нужен отрицательный угол {name}."))
            sid += 1
        elif name.startswith("waist_"):
            for palm, rr in [("left_palm", fk_left), ("right_palm", fk_right)]:
                if not rr:
                    continue
                rows.append(qa(sid, f"Куда сместится {palm}, если увеличить {name} на +10 градусов?", f"По FK для G1 FTP: при {name}=+10 градусов {palm} смещается {rr['plus_direction']}; delta примерно {rr['plus_delta']}."))
                sid += 1
                rows.append(qa(sid, f"Куда сместится {palm}, если поставить {name} на -10 градусов?", f"По FK для G1 FTP: при {name}=-10 градусов {palm} смещается {rr['minus_direction']}; delta примерно {rr['minus_delta']}."))
                sid += 1

    # Fixed validation examples include previously failed classes.
    val_questions = {
        "Какие лимиты у right_shoulder_roll_joint?",
        "Куда сместится right_palm, если поставить right_wrist_roll_joint на -10 градусов?",
        "Куда сместится left_palm, если поставить left_wrist_yaw_joint на -10 градусов?",
        "Что делает сустав right_elbow_joint?",
        "Что означает elbow=0 и elbow=90 у G1 FTP?",
        "Какие файлы являются источниками правды для G1 FTP в этом проекте?",
    }
    for row in rows:
        user = row["messages"][1]["content"].replace("/robot_qa\n", "", 1)
        if user in val_questions or row["sample_id"] % 17 == 0:
            row["split"] = "val"
    return rows, compare


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=Path, default=HUMANOID_ROOT / "resources" / "g1" / "g1_inspire_clean.xml")
    parser.add_argument("--urdf", type=Path, default=HUMANOID_ROOT / "resources" / "g1" / "g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf")
    parser.add_argument("--fk-csv", type=Path, default=HUMANOID_ROOT / "prompt_engineering" / "g1_upper_body_joint_direction.csv")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "robot_qa_g1_ftp_v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows, compare = build_rows(args.mjcf, args.urdf, args.fk_csv)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    random.Random(args.seed).shuffle(train)
    random.Random(args.seed).shuffle(val)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train_robot_qa.jsonl", train)
    write_jsonl(args.out_dir / "val_robot_qa.jsonl", val)
    with (args.out_dir / "joint_source_compare.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(compare[0]))
        writer.writeheader()
        writer.writerows(compare)
    manifest = {
        "dataset_kind": "robot_qa_g1_ftp_v2_env_ftp_oriented",
        "mjcf": str(args.mjcf),
        "urdf": str(args.urdf),
        "fk_csv": str(args.fk_csv),
        "train_rows": len(train),
        "val_rows": len(val),
        "total_rows": len(rows),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
