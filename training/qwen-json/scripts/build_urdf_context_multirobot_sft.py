#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SYSTEM = """Ты робототехнический ассистент.
Отвечай строго по URDF_CONTEXT, который дан в сообщении пользователя.
Если в URDF_CONTEXT нет факта, не выдумывай: скажи, какого фрагмента URDF/FK не хватает.
Для /motion_json возвращай только валидный compact JSON-массив без markdown и пояснений.
Compact motion format: [{"frame":{"joint_name_or_alias":angle_deg},"duration":seconds},{"frame":{},"duration":3.0}].
Углы в JSON всегда в градусах. URDF limits обычно в радианах, перевод: degrees = radians * 180/pi."""


def elem_xml(elem: ET.Element) -> str:
    return ET.tostring(elem, encoding="unicode", short_empty_elements=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def joint_info(j: ET.Element) -> dict[str, Any]:
    parent = j.find("parent")
    child = j.find("child")
    axis = j.find("axis")
    limit = j.find("limit")
    lower = upper = None
    if limit is not None:
        lower = limit.get("lower")
        upper = limit.get("upper")
    return {
        "name": j.get("name", ""),
        "type": j.get("type", ""),
        "parent": parent.get("link", "") if parent is not None else "",
        "child": child.get("link", "") if child is not None else "",
        "axis": axis.get("xyz", "0 0 0") if axis is not None else "0 0 0",
        "lower": lower,
        "upper": upper,
        "xml": elem_xml(j),
    }


def deg(x: str | None) -> str:
    if x is None:
        return "нет"
    return f"{math.degrees(float(x)):.3f}"


def robot_profile_context(robot_name: str, joints: list[dict[str, Any]], max_joints: int = 60) -> str:
    lines = [f"robot_name={robot_name}", "movable_joints:"]
    for info in joints[:max_joints]:
        lines.append(
            f"- {info['name']}: type={info['type']}; parent={info['parent']}; child={info['child']}; "
            f"axis={info['axis']}; limits_rad=[{info['lower']},{info['upper']}]; "
            f"limits_deg=[{deg(info['lower'])},{deg(info['upper'])}]"
        )
    return "\n".join(lines)


def make_qa_rows(robot_name: str, urdf_path: Path, sample_offset: int) -> list[dict[str, Any]]:
    root = ET.parse(urdf_path).getroot()
    joints = [joint_info(j) for j in root.findall("joint") if j.get("type") not in ("fixed", None)]
    rows: list[dict[str, Any]] = []

    profile = robot_profile_context(robot_name, joints)
    rows.append({
        "sample_id": sample_offset,
        "task": "robot_qa",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"/robot_qa\nURDF_CONTEXT:\n{profile}\n\nВопрос: Какие movable joints есть у робота {robot_name}? Ответь кратко группами."},
            {"role": "assistant", "content": f"По URDF_CONTEXT у {robot_name} перечислены movable joints: " + ", ".join(i["name"] for i in joints) + "."},
        ],
        "meta": {"robot": robot_name, "category": "urdf_profile"},
    })

    idx = sample_offset + 1
    for info in joints:
        ctx = f"robot_name={robot_name}\nURDF joint XML:\n{info['xml']}"
        name = info["name"]
        rows.extend([
            {
                "sample_id": idx,
                "task": "robot_qa",
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"/robot_qa\nURDF_CONTEXT:\n{ctx}\n\nВопрос: parent, child, type и axis сустава {name}?"},
                    {"role": "assistant", "content": f"По URDF_CONTEXT: {name} имеет type={info['type']}, parent={info['parent']}, child={info['child']}, axis={info['axis']}."},
                ],
                "meta": {"robot": robot_name, "category": "joint_topology"},
            },
            {
                "sample_id": idx + 1,
                "task": "robot_qa",
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"/robot_qa\nURDF_CONTEXT:\n{ctx}\n\nВопрос: Какие лимиты у {name} в радианах и градусах?"},
                    {"role": "assistant", "content": f"По URDF_CONTEXT лимиты {name}: radians=[{info['lower']},{info['upper']}], degrees=[{deg(info['lower'])},{deg(info['upper'])}]. Если limit отсутствует, сустав может быть continuous или требует отдельной проверки."},
                ],
                "meta": {"robot": robot_name, "category": "limits"},
            },
            {
                "sample_id": idx + 2,
                "task": "robot_qa",
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"/robot_qa\nURDF_CONTEXT:\n{ctx}\n\nВопрос: Можно ли по одному этому URDF-фрагменту уверенно сказать, куда визуально пойдет end-effector при +10 градусов?"},
                    {"role": "assistant", "content": f"Нет. По этому URDF-фрагменту видно axis={info['axis']} и связи {info['parent']} -> {info['child']}, но визуальное направление end-effector при +10 градусов требует FK с полной цепочкой и текущей позой."},
                ],
                "meta": {"robot": robot_name, "category": "anti_hallucination"},
            },
        ])
        idx += 3
    return rows


def make_seer_motion_rows(sample_offset: int) -> list[dict[str, Any]]:
    context = """robot_name=seer
Use exact SEER joint names from this URDF-derived arm list:
left arm: arm_left_1, arm_left_2, arm_left_3, arm_left_4, arm_left_5, arm_left_6, arm_left_7
right arm: arm_right_1, arm_right_2, arm_right_3, arm_right_4, arm_right_5, arm_right_6, arm_right_7
main safe limits in degrees:
arm_*_1 [-178,178], arm_*_2 [-120,120], arm_*_3 [-178,178], arm_*_4 [-60,145], arm_*_5 [-178,178], arm_*_6 [-60,60], arm_*_7 [-90,90].
Omitted joints keep previous values. End every motion with {"frame":{},"duration":3.0}."""
    examples = [
        ("Подними обе руки в простую верхнюю позу", [{"frame": {"arm_left_2": -55, "arm_right_2": 55, "arm_left_4": 35, "arm_right_4": 35}, "duration": 1.8}, {"frame": {}, "duration": 3.0}]),
        ("Сделай Т-позу руками", [{"frame": {"arm_left_1": 70, "arm_right_1": -70, "arm_left_4": 20, "arm_right_4": 20}, "duration": 1.8}, {"frame": {}, "duration": 3.0}]),
        ("Опусти обе руки в нейтраль", [{"frame": {"arm_left_1": 0, "arm_left_2": 0, "arm_left_3": 0, "arm_left_4": 0, "arm_left_5": 0, "arm_left_6": 0, "arm_left_7": 0, "arm_right_1": 0, "arm_right_2": 0, "arm_right_3": 0, "arm_right_4": 0, "arm_right_5": 0, "arm_right_6": 0, "arm_right_7": 0}, "duration": 2.0}, {"frame": {}, "duration": 3.0}]),
        ("Помаши правой рукой три раза", [{"frame": {"arm_right_1": -45, "arm_right_2": 45, "arm_right_4": 80}, "duration": 1.2}, {"repeat": [{"frame": {"arm_right_5": -35}, "duration": 0.6}, {"frame": {"arm_right_5": 35}, "duration": 0.6}], "times": 3}, {"frame": {}, "duration": 3.0}]),
        ("Покажи левой рукой в сторону", [{"frame": {"arm_left_1": 65, "arm_left_4": 20, "arm_left_6": 15}, "duration": 1.5}, {"frame": {}, "duration": 3.0}]),
        ("Согни обе руки в локтях", [{"frame": {"arm_left_4": 90, "arm_right_4": 90}, "duration": 1.4}, {"frame": {}, "duration": 3.0}]),
    ]
    rows = []
    for i, (cmd, answer) in enumerate(examples):
        rows.append({
            "sample_id": sample_offset + i,
            "task": "motion_json",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"/motion_json\nURDF_CONTEXT:\n{context}\n\nCommand:\n{cmd}"},
                {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))},
            ],
            "meta": {"robot": "seer", "category": "seer_motion_json"},
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1-urdf", type=Path, default=Path("/home/alpc/humanoid/resources/g1/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"))
    parser.add_argument("--seer-urdf", type=Path, default=Path("/home/alpc/RWB/simulators/isaac_demo_pack/urdf_seer/urdf/robot_view_obj_collision_mesh.urdf"))
    parser.add_argument("--motion-sft", type=Path, default=PROJECT_ROOT / "outputs" / "dataset_100_statis_multitask_strict_profile_v4_sft")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "dataset_urdf_context_g1_seer_sft")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    # Keep the best current G1 motion/strict-profile behavior as base.
    rows.extend(load_jsonl(args.motion_sft / "train_compact_4096.jsonl"))
    rows.extend(make_qa_rows("unitree_g1_ftp", args.g1_urdf, 3_000_000))
    rows.extend(make_qa_rows("seer", args.seer_urdf, 4_000_000))
    rows.extend(make_seer_motion_rows(5_000_000))

    val = []
    val.extend(load_jsonl(args.motion_sft / "val_compact_4096.jsonl"))
    # deterministic small holdout from generated rows
    extra = make_qa_rows("seer", args.seer_urdf, 6_000_000)[:20] + make_seer_motion_rows(7_000_000)[:3]
    val.extend(extra)

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rng.shuffle(val)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in [("train", rows), ("val", val)]:
        write_jsonl(args.out_dir / f"{split}.jsonl", split_rows)
        write_jsonl(args.out_dir / f"{split}_compact_4096.jsonl", split_rows)
        write_jsonl(args.out_dir / f"{split}_compact_8192.jsonl", split_rows)

    manifest = {
        "dataset_kind": "g1_plus_seer_urdf_context_sft",
        "g1_urdf": str(args.g1_urdf),
        "seer_urdf": str(args.seer_urdf),
        "base_motion_sft": str(args.motion_sft),
        "train_rows": len(rows),
        "val_rows": len(val),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
