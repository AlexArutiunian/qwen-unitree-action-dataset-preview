from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluate import generate, load_model_and_tokenizer
from .hf_utils import PROJECT_ROOT, load_config, resolve_path, set_runtime_env
from .prompts import build_system_prompt, load_spec, parse_joint_ranges
from .validate_robot_json import stable_json_dumps, validate_motion_obj


def resolve_spec_path(cfg: dict[str, Any], root: Path) -> Path:
    dataset_dir = resolve_path(cfg.get("dataset", {}).get("work_dir", "outputs/dataset"), root) or (root / "outputs" / "dataset")
    candidates = [
        dataset_dir / "robot_spec_joints-only.txt",
        dataset_dir / "hf_downloaded_dataset" / "robot_spec_joints-only.txt",
        root / "outputs" / "hf_downloaded_dataset" / "robot_spec_joints-only.txt",
        root / "outputs" / "dataset_viewer" / "robot_spec_joints-only.txt",
        root / "outputs" / "dataset_viewer" / "hf_files" / "robot_spec_joints-only.txt",
    ]
    spec_file = cfg.get("dataset", {}).get("spec_file")
    if spec_file:
        candidates.append(dataset_dir / str(spec_file))
    spec_path = cfg.get("dataset", {}).get("spec_path")
    if spec_path:
        resolved = resolve_path(spec_path, root)
        if resolved is not None:
            candidates.append(resolved)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Robot spec was not found. Run training/evaluation once so the dataset files are prepared.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", help="Russian robot motion command.")
    parser.add_argument("--config", default="configs/train_compact_8192.yaml")
    parser.add_argument("--adapter", default="outputs/best_adapter")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    if not args.command:
        raise SystemExit('Pass a command, for example: python -m src.infer "подними правую руку"')

    cfg = load_config(args.config)
    root = resolve_path(cfg.get("project_root", PROJECT_ROOT)) or PROJECT_ROOT
    adapter = resolve_path(args.adapter, root)
    if adapter is None or not adapter.exists():
        raise FileNotFoundError(f"Adapter not found: {args.adapter}")

    set_runtime_env(str(cfg.get("cuda_visible_devices", "0")))
    spec_text = load_spec(resolve_spec_path(cfg, root))
    messages = [
        {"role": "system", "content": build_system_prompt(spec_text, cfg.get("prompt_mode", "compact"))},
        {"role": "user", "content": f"Command:\n{args.command.strip()}"},
        {"role": "assistant", "content": ""},
    ]
    model, tokenizer = load_model_and_tokenizer(
        cfg.get("model_name", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        adapter,
        cfg.get("bnb_4bit_compute_dtype", "float16"),
    )
    pred, latency = generate(model, tokenizer, messages, args.max_new_tokens or int(cfg.get("max_new_tokens", 768)))
    print(pred)
    print(f"\n[infer] latency_sec={latency:.3f}")

    if args.no_validate:
        return
    try:
        pred_obj = json.loads(pred)
    except Exception as exc:
        print(f"[infer] json_parse_error: {exc}")
        return
    errors = validate_motion_obj(pred_obj, parse_joint_ranges(spec_text))
    if errors:
        print("[infer] validation_errors:")
        for error in errors[:40]:
            print(f"- {error}")
    else:
        print("[infer] valid_robot_json=1")
        print("[infer] normalized_json:")
        print(stable_json_dumps(pred_obj))


if __name__ == "__main__":
    main()
