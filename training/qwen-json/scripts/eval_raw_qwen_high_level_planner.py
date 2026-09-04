#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM = """You are a robot motion planner.
Return only valid JSON, no markdown, no comments.

You do not output joint angles. You output a high-level IK waypoint plan.

You will receive ROBOT_IK_CONTEXT for the current robot.
Use that context to choose reachable waypoints.

Coordinates are normalized body-frame coordinates relative to the robot pelvis/torso origin:
- x: front/back, positive in front of the chest
- y: left/right, positive to robot-left, negative to robot-right
- z: up/down, positive upward

The context gives reachable min/max ranges for each hand. Stay inside them.
Prefer values slightly inside the reachable range, not exactly outside or beyond rounded endpoints.
Do not invent joint names. Do not output angles. Do not output explanations.

Allowed IK waypoint format:
[
  {"ik":{"right_hand":{"x":0.20,"y":-0.25,"z":0.65}},"duration":1.2},
  {
    "repeat":[
      {"ik":{"right_hand":{"x":0.20,"y":-0.40,"z":0.65}},"duration":0.45},
      {"ik":{"right_hand":{"x":0.20,"y":-0.10,"z":0.65}},"duration":0.45}
    ],
    "times":3
  },
  {"target":"hold","duration":3.0}
]

Planning rules:
- If the command says both hands / обе руки / руки, the first motion step MUST contain both left_hand and right_hand in the same ik object.
- For any left_hand point, use the left_hand reachable fields only. For any right_hand point, use the right_hand reachable fields only.
- For ordinary motions, left_hand y must stay positive and right_hand y must stay negative. Never swap signs unless asked to cross arms.
- For hands up: choose z near each hand's own z_max, x near that hand's initial/middle x, y inside that hand's natural_y_band.
- For T-pose / arms to sides: choose left_hand y near its y_max and right_hand y near its y_min; z near each hand's initial z; x near each hand's initial x.
- For hands forward: choose x near each hand's own x_max; y MUST stay inside each hand's natural_y_band so hands remain shoulder-width. Do not move both hands toward y=0 unless the user asks to bring hands together. Keep z near each hand's initial z.
- For greeting/wave: use right_hand unless user asks left. First waypoint goes to a reachable upper-side point with right_hand y negative. Then repeat two nearby waypoints with different negative y values.
- Use repeat ONLY for greeting/wave commands such as привет, помаши, wave, greeting.
- For T-pose, hands-up, and hands-forward commands output exactly one ik waypoint plus hold. Never output repeat for these commands.
- For greeting output exactly one setup ik waypoint, one repeat block with two different ik waypoints, then hold.

Always end with {"target":"hold","duration":3.0}.
Use the smallest sufficient plan."""


def load_model(model_name: str):
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


def parse_json_plan(text: str) -> tuple[list[dict[str, Any]] | None, str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        obj = json.loads(cleaned)
    except Exception as exc:
        return None, f"json_parse_error:{exc}"
    if not isinstance(obj, list) or not all(isinstance(x, dict) for x in obj):
        return None, "plan_not_list_of_objects"
    return obj, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    commands = [json.loads(line) for line in args.commands.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.model)
    with args.out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(commands, start=1):
            command = row["command"]
            context = str(row.get("context") or "").strip()
            user_text = f"Command: {command}\n"
            if context:
                user_text += f"\nROBOT_IK_CONTEXT:\n{context}\n"
            user_text += "\nReturn the high-level IK waypoint plan JSON."
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_text},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            start = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            plan, error = parse_json_plan(pred)
            result = {
                "idx": i,
                "command": command,
                "robot": row.get("robot"),
                "scene": row.get("scene"),
                "pred": pred,
                "plan": plan,
                "parse_ok": plan is not None,
                "error": error,
                "latency_sec": time.perf_counter() - start,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[plan] {i}/{len(commands)} parse={result['parse_ok']} latency={result['latency_sec']:.2f}s command={command}", flush=True)
    print(f"[plan] wrote {args.out}")


if __name__ == "__main__":
    main()
