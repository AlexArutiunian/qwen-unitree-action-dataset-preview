from __future__ import annotations

import re
from pathlib import Path


def parse_joint_ranges(spec_text: str) -> dict[str, tuple[float, float]]:
    pattern = re.compile(
        r"^\s*([A-Za-z0-9_]+_joint):\s*([-+]?\d+(?:\.\d+)?)\s*(?:to|\.\.)\s*([-+]?\d+(?:\.\d+)?)",
        re.MULTILINE,
    )
    ranges: dict[str, tuple[float, float]] = {}
    for name, lo, hi in pattern.findall(spec_text):
        if name == "floating_base_joint":
            continue
        ranges[name] = (float(lo), float(hi))
    return ranges


def load_spec(spec_path: str | Path) -> str:
    return Path(spec_path).read_text(encoding="utf-8").strip()


def build_system_prompt(spec_text: str, mode: str = "compact") -> str:
    mode = mode.lower()
    joint_ranges = parse_joint_ranges(spec_text)
    joint_lines = "\n".join(
        f"{name}: {lo:g} to {hi:g}" for name, (lo, hi) in sorted(joint_ranges.items())
    )
    header = (
        "You convert Russian natural-language robot motion commands into Unitree G1 joint-control JSON.\n"
        "Return only a valid JSON array, with no comments or explanations.\n"
        "Allowed step forms:\n"
        "{\"frame\":[{\"name\":\"<joint>\",\"angle\":<degrees>}],\"duration\":<seconds>}\n"
        "{\"repeat\":[<steps>],\"times\":<integer>}\n"
        "Rules:\n"
        "Use only allowed joints. floating_base_joint is forbidden.\n"
        "Angles are degrees and must stay within joint limits.\n"
        "frame: [] is valid as a pause or hold step.\n"
        "Use duration >= 0.75 seconds for active movement steps.\n"
        "Prefer one frame for simultaneous joint movement.\n"
        "Use 1 second for fast, 2 seconds for medium, and 3 seconds for slow movement.\n"
    )
    semantics = (
        "Critical joint semantics:\n"
        "- shoulder_pitch: negative angle moves the arm forward/up; positive angle moves it backward.\n"
        "- left_shoulder_roll: positive moves the left arm outward; negative moves it inward.\n"
        "- right_shoulder_roll: negative moves the right arm outward; positive moves it inward.\n"
        "- elbow joints: 0 degrees means about a 90-degree bend; +90 degrees means a straight elbow.\n"
        "- For straight arms, use elbow angles near +90, not 0.\n"
        "- For raised arms with shoulder_pitch below -100, compensate shoulder roll: about +40 for left, -40 for right.\n"
        "- When both arms act together, put left and right joints in the same frame so they move simultaneously.\n"
        "- Do not command finger joints unless the user asks for fingers, fist, open hand, grasping, or palm/finger gestures.\n"
    )
    if mode == "compact":
        return (header + semantics + "Allowed joints and limits:\n" + joint_lines).strip()
    if mode == "full":
        return (header + "Full robot joint specification:\n" + spec_text).strip()
    raise ValueError(f"Unknown prompt mode: {mode}")


def make_messages(command: str, target_json: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Command:\n{command.strip()}"},
        {"role": "assistant", "content": target_json.strip()},
    ]
