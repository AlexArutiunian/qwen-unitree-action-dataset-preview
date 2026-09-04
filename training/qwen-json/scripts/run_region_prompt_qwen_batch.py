#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


REGION_SYSTEM = """You are a robot semantic motion planner.
You do not output numeric coordinates.
You choose semantic regions and intensity values for each wrist from the robot-specific region catalog.

Use only the provided robot context and region catalog.
Do not use examples. Do not use memorized poses. Do not invent region names.
Do not calculate x/y/z coordinates.
Do not rely on hidden templates for named poses. Infer spatial intent from the command and the catalog meanings.
The command may be in Russian or English. First understand the ordinary spatial meaning of the command, then map that meaning to catalog regions.

Region language:
- depth_region controls x: keep / forward / backward
- body_side_plane depth means the wrist should be near the shoulder/torso side plane rather than in front or behind.
- keep depth means preserve the current wrist depth even if it is already forward/backward.
- forward/backward depth means deliberately move away from the torso side plane.
- lateral_region controls y: keep / shoulder_lateral / same_side_outward / toward_body_center / cross_body
- height_region controls z: keep / low / chest_height / shoulder_height / above_shoulder / over_head
- limb_orientation controls the visible limb direction: none / vertical_up / horizontal_side / forward_diagonal
- intensity values are 0.0 to 1.0 positions inside the chosen region. 0.5 means middle, 0.8 means strong movement in that region.
- active=false means leave that hand at keep/keep/keep.
- gesture can be none or wave. Use wave only if the command asks greeting/waving.

Planning process:
1. Read the robot name, coordinate frame, current wrist positions, shoulder positions, and reachable intervals.
2. Interpret the command as a desired spatial relation of the wrists relative to the body.
3. Select the region names whose meanings and numeric intervals best match that spatial relation.
4. Use active=false only for a hand that should intentionally stay in its current place.
5. Use limb_orientation only as a visible arm-direction constraint; it must be consistent with the selected wrist target.
6. Keep the plan minimal: move only hands needed by the command.
7. If the command describes symmetric use of both arms, choose mirror-consistent left/right regions unless the robot's catalog makes that impossible.
8. Use toward_body_center only when the wrist should move inward toward the torso centerline. Use same_side_outward only when the wrist should move away from the torso centerline on its own side.
9. Explain the spatial reason briefly in each hand's reason field.

Return only valid JSON, no markdown."""


def load_model(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int) -> tuple[str, float]:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return text, time.perf_counter() - start


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        obj = json.loads(cleaned)
    except Exception as exc:
        return None, f"json_parse_error:{exc}"
    return obj if isinstance(obj, dict) else None, "" if isinstance(obj, dict) else "json_not_object"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def lerp(interval: list[Any], t: float) -> float:
    lo, hi = float(interval[0]), float(interval[1])
    t = max(0.0, min(1.0, float(t)))
    return lo + (hi - lo) * t


def hand_point(catalog: dict[str, Any], hand: str, choice: dict[str, Any]) -> dict[str, float]:
    hand_cat = catalog["hands"][hand]
    regions = hand_cat["regions"]
    depth_name = str(choice.get("depth_region", "keep"))
    lateral_name = str(choice.get("lateral_region", "keep"))
    height_name = str(choice.get("height_region", "keep"))
    intens = choice.get("intensity", {}) if isinstance(choice.get("intensity"), dict) else {}
    orientation = str(choice.get("limb_orientation", "none"))
    x = lerp(regions["depth"][depth_name]["x"], float(intens.get("depth", 0.5)))
    y = lerp(regions["lateral"][lateral_name]["y"], float(intens.get("lateral", 0.5)))
    z = lerp(regions["height"][height_name]["z"], float(intens.get("height", 0.5)))
    return {"x": round(x, 3), "y": round(y, 3), "z": round(z, 3), "limb_orientation": orientation}


def keep_choice() -> dict[str, Any]:
    return {
        "active": False,
        "depth_region": "keep",
        "lateral_region": "keep",
        "height_region": "keep",
        "intensity": {"depth": 0.5, "lateral": 0.5, "height": 0.5},
        "limb_orientation": "none",
        "gesture": "none",
    }


def resolve_region_plan(catalog: dict[str, Any], parsed: dict[str, Any] | None) -> tuple[list[dict[str, Any]] | None, list[str]]:
    if parsed is None:
        return None, ["missing parsed region plan"]
    hands = parsed.get("hands")
    if not isinstance(hands, dict):
        return None, ["missing hands object"]
    ik: dict[str, dict[str, float]] = {}
    errors: list[str] = []
    gestures: list[tuple[str, dict[str, Any]]] = []
    for hand in ("left_hand", "right_hand"):
        choice = hands.get(hand) if isinstance(hands.get(hand), dict) else keep_choice()
        try:
            ik[hand] = hand_point(catalog, hand, choice)
            if str(choice.get("gesture", "none")) == "wave" and bool(choice.get("active", True)):
                gestures.append((hand, choice))
        except Exception as exc:
            errors.append(f"{hand}: {exc}")
    if errors:
        return None, errors
    steps: list[dict[str, Any]] = [{"ik": ik, "duration": 1.4}]
    for hand, choice in gestures:
        target = "left_wave" if hand == "left_hand" else "right_wave"
        steps.append({"target": target, "duration": 1.0})
    steps.append({"target": "hold", "duration": 2.0})
    return steps, []


def user_prompt(row: dict[str, Any]) -> str:
    catalog = row["region_catalog"]
    small_catalog = {
        "robot": catalog["robot"],
        "coordinate_frame": catalog["coordinate_frame"],
        "hands": catalog["hands"],
    }
    return f"""Robot-specific region catalog:
{json.dumps(small_catalog, ensure_ascii=False, indent=2)}

Command: {row["command"]}

Choose semantic regions and intensities, not coordinates.
Return exactly valid JSON:
{{
  "robot_used": "{row["robot"]}",
  "motion_intent": "...",
  "spatial_interpretation": "Describe in words where each wrist should move relative to torso, without coordinates.",
  "hands": {{
    "left_hand": {{
      "active": true,
      "depth_region": "keep|body_side_plane|forward|backward",
      "lateral_region": "keep|shoulder_lateral|same_side_outward|toward_body_center|cross_body",
      "height_region": "keep|low|chest_height|shoulder_height|above_shoulder|over_head",
      "intensity": {{"depth":0.5,"lateral":0.5,"height":0.5}},
      "limb_orientation": "none|vertical_up|horizontal_side|forward_diagonal",
      "gesture": "none|wave",
      "reason": "..."
    }},
    "right_hand": {{
      "active": true,
      "depth_region": "keep|body_side_plane|forward|backward",
      "lateral_region": "keep|shoulder_lateral|same_side_outward|toward_body_center|cross_body",
      "height_region": "keep|low|chest_height|shoulder_height|above_shoulder|over_head",
      "intensity": {{"depth":0.5,"lateral":0.5,"height":0.5}},
      "limb_orientation": "none|vertical_up|horizontal_side|forward_diagonal",
      "gesture": "none|wave",
      "reason": "..."
    }}
  }},
  "self_check": ["..."]
}}"""


def safe_name(text: str) -> str:
    out = []
    for ch in text.lower().strip():
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    return "".join(out).strip("_")[:80] or "motion"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plans-out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=360)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.max_rows:
        rows = rows[: args.max_rows]
    model, tokenizer = load_model(args.model)
    result_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    for row in rows:
        raw, latency = generate(
            model,
            tokenizer,
            [{"role": "system", "content": REGION_SYSTEM}, {"role": "user", "content": user_prompt(row)}],
            max_new_tokens=args.max_new_tokens,
        )
        parsed, parse_error = extract_json_object(raw)
        plan, resolve_errors = resolve_region_plan(row["region_catalog"], parsed)
        result = {
            "idx": row["idx"],
            "robot": row["robot"],
            "scene": row["scene"],
            "command": row["command"],
            "raw": raw,
            "latency_sec": latency,
            "parsed": parsed,
            "parse_ok": parsed is not None,
            "parse_error": parse_error,
            "plan": plan,
            "resolve_errors": resolve_errors,
        }
        result_rows.append(result)
        if plan is not None:
            plan_rows.append(
                {
                    "idx": row["idx"],
                    "robot": row["robot"],
                    "scene": row["scene"],
                    "command": row["command"],
                    "name": f"{row['robot']}_region_{row['idx']:03d}_{safe_name(row['command'])}",
                    "plan": plan,
                    "source": "qwen_region_intensity_plus_deterministic_resolver",
                }
            )
        print(json.dumps({"idx": row["idx"], "robot": row["robot"], "command": row["command"], "ok": plan is not None, "latency": round(latency, 3)}, ensure_ascii=False))
    write_jsonl(args.out, result_rows)
    write_jsonl(args.plans_out, plan_rows)
    print(f"[region-batch] wrote {args.out}")
    print(f"[region-batch] wrote {args.plans_out}")


if __name__ == "__main__":
    main()
