# HANDOFF: interactive MuJoCo dataset viewer

## Goal

Turn this repository from a static table into an interactive Unitree G1 motion
showcase inspired by [wbc-demo](https://wbc-mjlab.github.io/wbc-demo/):

- choose any natural-language action from the dataset;
- execute its full motion JSON on the real G1 kinematic model in the browser;
- orbit, zoom and pan around the moving robot at any time;
- play, pause, reset, scrub and change playback speed;
- retain high-quality MuJoCo screenshots/GIF/MP4 as lightweight card previews;
- keep the robot framed throughout every generated preview.

The primary experience must be a real browser 3D engine, not a prerecorded
video or an AI-generated illustration.

## Current state (2026-09-07)

The existing static preview remains intentionally unchanged. A local proof of
concept established that the official `@mujoco/mujoco` WASM package and
Three.js are the right stack, and that all 1,626 full motions can be exported to
a roughly 108 KB gzip index. It also exposed the first engineering blocker:

- a 53-DoF G1 model with original CAD meshes compiles to roughly 185 MB;
- a 120k-triangle web LOD compiles to roughly 31 MB;
- a first headless Chrome smoke test remained at the MJB loading/compilation
  screen, even after aligning Python and WASM on MuJoCo 3.12.0.

The unfinished prototype is deliberately not wired into `main`. Web GPT/Codex
should implement the viewer from this brief, validate it in a real browser and
only then replace the static page.

## Exact example to validate first

Dataset row:

```text
sample_id: 91
json_name: quality_00091_src_13_intensity_large.json
command: Выполни шире и выразительнее: левую руку вверх, правую руку вбок
duration: 1.5 s
```

Target joints in degrees:

```json
{
  "left_shoulder_pitch_joint": -134.4,
  "left_shoulder_roll_joint": 39.2,
  "left_shoulder_yaw_joint": 0.0,
  "left_elbow_joint": 70.0,
  "right_shoulder_pitch_joint": 16.8,
  "right_shoulder_roll_joint": -95.0,
  "right_shoulder_yaw_joint": 0.0,
  "right_elbow_joint": 70.0
}
```

Expected result: anatomical left arm high above the head, anatomical right arm
extended sideways, with the torso and legs neutral. A native desktop MuJoCo
reference is stored at `assets/sample-91-mujoco-comparison.png`.

## Authoritative source locations

The original work is on:

```text
ssh alpc@192.168.0.109
/home/alpc/humanoid/
```

Important paths:

```text
# Full dataset (small, authoritative)
/home/alpc/humanoid/dataset_panto/hf_robot_json_sft/robot_sft.parquet
/home/alpc/humanoid/dataset_panto/hf_robot_json_sft/robot_spec_joints-only.txt

# Original complete per-action JSON files
/home/alpc/humanoid/dataset_panto/dataset_approved_augmented_quality_fixed/

# Existing desktop MuJoCo render implementation
/home/alpc/humanoid/deploy/render_model_motion_screens.py
/home/alpc/humanoid/deploy/render_seer_motion_screens.py

# G1 MJCF and meshes
/home/alpc/humanoid/resources/g1/g1_inspire_clean.xml
/home/alpc/humanoid/resources/g1/scene_FTP_clean.xml
/home/alpc/humanoid/resources/g1/meshes/

# Other useful code families
/home/alpc/humanoid/lora-robot/
/home/alpc/humanoid/nanoVLM/
/home/alpc/humanoid/deploy/
```

Do not copy `.venv`, checkpoints, model weights, caches, raw datasets, logs or
secret-bearing `.env` files into Git. Keep paths to heavy artifacts in prose.

## Dataset facts

The full parquet has 1,626 rows:

- train: 1,401;
- validation: 148;
- reserved: 77;
- basic pose control: 622;
- objectless pantomime: 371;
- communicative gesture: 368;
- expressive pose: 175;
- sport/exercise motion: 90.

The old `robot_sft_preview.csv` is not authoritative for rendering: 691 target
JSON strings are truncated at 1,200 characters. Generate the compact web
payload from the full parquet, not from this preview file.

## Motion semantics

- angles in dataset JSON are degrees;
- MuJoCo `qpos` values are radians;
- each frame updates only listed joints; all other joints retain their previous
  values;
- interpolate each block from its current pose to its target pose over the
  block duration;
- use smoothstep `a²(3−2a)` unless parity testing shows another required curve;
- support nested `{ "repeat": [...], "times": N }` blocks;
- an empty frame is a hold, not a reset;
- reset starts from model `qpos0`;
- call `mj_forward` after writing `qpos` before rendering.

## Web engine architecture

Reference implementation: `wbc-demo` uses official MuJoCo WASM, Three.js and a
MuJoCo-geometry renderer. Follow the same high-level architecture without
depending on its policy/ONNX stack:

```text
actions.json.gz
      │
      ▼
motion timeline ──► qpos interpolation ──► mj_forward
                                             │
                                             ▼
Three.js meshes ◄── geom_xpos / geom_xmat ─ MuJoCo WASM
      │
      ▼
OrbitControls: rotate / zoom / pan while playback continues
```

The dataset contains prescribed joint trajectories, so no neural policy and no
physics stepping are required for the basic viewer. MuJoCo remains the source
of truth for forward kinematics and compiled geometry.

## Model packaging

The unmodified CAD model compiles to roughly 185 MB, which is unsuitable for
normal GitHub Pages delivery. The tested local web LOD used decimated visual meshes:

- original geometry: about 947k triangles;
- current web LOD: about 120k triangles;
- MJB: about 34 MB;
- joint names, ranges, transforms, dimensions and kinematic tree are preserved.

The source meshes and private parquet are deliberately not committed. A future
cleanup should add a deterministic `tools/build_web_model.py` that rebuilds the
MJB from an authorized local G1 asset directory and documents asset licensing.

## Required next work for Web GPT / Codex

1. Scaffold a minimal Vite project with `@mujoco/mujoco` 3.12.0 and Three.js.
2. First make one small model and `sample_id=91` work end to end; do not start
   with the whole gallery.
3. Diagnose the MJB loading stall. Compare binary loading with loading an MJCF
   plus meshes into MEMFS, as done by `wbc-demo/src/engine/mujoco.ts`.
4. Serve locally and smoke-test in a real Chrome session; confirm both
   left/right semantics and final joint values.
5. Inspect console for WASM, MJB, WebGL and memory errors.
6. Confirm the robot is fully visible at desktop and mobile aspect ratios.
7. Verify play/pause/reset, scrubbing, speed, keyboard shortcuts and orbit
   controls while a motion is playing.
8. Test long/repeated programs and hand/finger actions, not just single frames.
9. Add loading fallback and a static MuJoCo image when WebGL/WASM is unavailable.
10. Add virtual scrolling for all 1,626 actions if list performance requires it.
11. Add optional camera modes: orbit, follow and front/side/back presets.
12. Generate optimized card previews from desktop MuJoCo (poster WebP first;
    hover MP4 only) and measure repository/page weight before committing them.
13. Add automated tests for timeline expansion, hold frames, degree-to-radian
    conversion, partial joint updates and `sample_id=91`.
14. Check GitHub Pages deployment and ensure Vite `base: "./"` works from the
    repository subpath.
15. Review Unitree/G1 mesh licensing before public distribution of the MJB.

## Visual/UX direction

- dark technical lab rather than a generic admin dashboard;
- motion library on the left, large uninterrupted 3D stage in the center,
  selected sample/JSON inspector on the right;
- compact WBC-like live status and telemetry;
- 3D viewport must dominate the page;
- actions selectable by search, category and split;
- no huge HTML table in the primary experience;
- no generated robot imagery presented as simulation output.

## Local non-repository artifacts

Current desktop exports are kept outside Git:

```text
/home/al/humanoid/mujoco_pose_renders/sample_00091/
/home/al/humanoid/mujoco_pose_renders/all_exact_preview_rows/
/home/al/humanoid/mujoco_pose_renders/source_alpc/
```

The earlier preview-only pass rendered 935 exact rows and correctly rejected
691 truncated rows. With the recovered parquet, a new full export can render
all 1,626 programs.
