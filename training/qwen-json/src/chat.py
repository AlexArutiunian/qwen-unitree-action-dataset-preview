from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from .evaluate import generate, load_model_and_tokenizer
from .hf_utils import PROJECT_ROOT, json_dump, load_config, resolve_path, set_runtime_env
from .infer import resolve_spec_path
from .prompts import build_system_prompt, load_spec, parse_joint_ranges
from .validate_robot_json import stable_json_dumps, validate_motion_obj


def make_messages(command: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Command:\n{command.strip()}"},
        {"role": "assistant", "content": ""},
    ]


def safe_name(text: str, max_len: int = 48) -> str:
    text = re.sub(r"\s+", "_", text.strip().lower())
    text = re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ_-]+", "", text)
    return text[:max_len].strip("_") or "command"


def save_response(out_dir: Path, command: str, pred: str, latency: float, errors: list[str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "_meta"
    raw_dir = out_dir / "_raw"
    meta_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name(command)}"
    json_path = out_dir / f"{stem}.json"
    raw_path = raw_dir / f"{stem}.raw.txt"
    meta_path = meta_dir / f"{stem}.meta.json"

    try:
        obj = json.loads(pred)
        json_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        parse_ok = True
        response_path = json_path
    except Exception:
        raw_path.write_text(pred.strip() + "\n", encoding="utf-8")
        parse_ok = False
        response_path = raw_path

    json_dump(
        {
            "command": command,
            "response_file": str(response_path),
            "parse_ok": parse_ok,
            "schema_ok": len(errors) == 0,
            "validation_errors": errors,
            "latency_sec": latency,
            "raw_response": pred,
        },
        meta_path,
    )
    return response_path


def run_command(
    model,
    tokenizer,
    command: str,
    system_prompt: str,
    joint_ranges: dict[str, tuple[float, float]],
    out_dir: Path,
    max_new_tokens: int,
    display_command: str | None = None,
) -> None:
    pred, latency = generate(model, tokenizer, make_messages(command, system_prompt), max_new_tokens)
    errors: list[str] = []
    try:
        obj = json.loads(pred)
        errors = validate_motion_obj(obj, joint_ranges)
        display = stable_json_dumps(obj)
    except Exception as exc:
        errors = [f"json_parse_error:{exc}"]
        display = pred.strip()
    saved = save_response(out_dir, display_command or command, pred, latency, errors)
    print(display)
    print(f"[chat] saved: {saved}")
    print(f"[chat] latency_sec={latency:.3f} schema_ok={int(len(errors) == 0)}")
    if errors:
        print("[chat] validation_errors:")
        for error in errors[:20]:
            print(f"- {error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="*", help="Optional one-shot command. Omit for interactive chat.")
    parser.add_argument("--config", default="configs/train_compact_8192.yaml")
    parser.add_argument("--adapter", default="outputs/best_adapter")
    parser.add_argument("--out-dir", default="outputs/robot_generations")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--jsonl-stdin",
        action="store_true",
        help="Read JSON lines from stdin. Each line: {'command': augmented_prompt, 'display_command': original_command}.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = resolve_path(cfg.get("project_root", PROJECT_ROOT)) or PROJECT_ROOT
    adapter = resolve_path(args.adapter, root)
    if adapter is None or not adapter.exists():
        raise FileNotFoundError(f"Adapter not found: {args.adapter}")
    out_dir = resolve_path(args.out_dir, root) or (root / args.out_dir)

    set_runtime_env(str(cfg.get("cuda_visible_devices", "0")))
    spec_text = load_spec(resolve_spec_path(cfg, root))
    system_prompt = build_system_prompt(spec_text, cfg.get("prompt_mode", "compact"))
    joint_ranges = parse_joint_ranges(spec_text)
    max_new_tokens = args.max_new_tokens or int(cfg.get("max_new_tokens", 768))

    print("[chat] loading model...")
    model, tokenizer = load_model_and_tokenizer(
        cfg.get("model_name", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        adapter,
        cfg.get("bnb_4bit_compute_dtype", "float16"),
    )
    print(f"[chat] ready. outputs: {out_dir}")

    if args.jsonl_stdin:
        print("[chat] jsonl stdin mode is ready.")
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                command = str(payload["command"]).strip()
                display_command = str(payload.get("display_command") or command).strip()
            except Exception as exc:
                print(f"[chat] bad_jsonl_input: {exc}", flush=True)
                continue
            if display_command.lower() in {"exit", "quit", "q"}:
                return
            run_command(model, tokenizer, command, system_prompt, joint_ranges, out_dir, max_new_tokens, display_command=display_command)
        return

    one_shot = " ".join(args.command).strip()
    if one_shot:
        run_command(model, tokenizer, one_shot, system_prompt, joint_ranges, out_dir, max_new_tokens)
        return

    while True:
        try:
            command = input("\ncommand> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if command.lower() in {"exit", "quit", "q"}:
            return
        if not command:
            continue
        run_command(model, tokenizer, command, system_prompt, joint_ranges, out_dir, max_new_tokens)


if __name__ == "__main__":
    main()
