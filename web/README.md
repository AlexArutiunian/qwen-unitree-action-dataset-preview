# Sample 91: real MuJoCo in the browser

This is deliberately a **single-action milestone**, not a gallery or dataset export.
The existing repository-root preview remains intact. `../demo/` contains its standalone GitHub Pages build.

## Run

```sh
cd web
npm ci
npm test
npm run dev
npm run build
```

Vite `base: './'` supports `/qwen-unitree-action-dataset-preview/demo/`.
The generated `demo/` directory is committed so the existing branch-based Pages setup needs no configuration changes.
After editing the application, run `npm run release` to refresh it.

## What actually runs

- Official `@mujoco/mujoco` **3.12.0**, single-threaded WASM inside a dedicated Web Worker.
- Real official Unitree `g1_29dof.xml`: joint names, ranges, body transforms, inertials and qpos0 are retained. This is the public 29-DoF G1, **not** the private 53-DoF Inspire-hand variant in the handoff.
- Eight target joint angles from HANDOFF sample 91, degrees → radians, smoothstep over 1.5 seconds.
- Only qpos is prescribed; `mj_forward` supplies every visual geometry position/rotation. No physics stepping, learned controller, or prerecorded playback.
- Anatomical left is robot +Y and appears on the viewer's right in front view. Anatomical right is -Y.
- OrbitControls remains active during playback. Space toggles playback, R resets qpos0, arrows seek ±0.1 seconds; focused controls retain normal keyboard behavior.
- Camera fitting uses a conservative union of MuJoCo geometry bounds over the entire motion, including extended arms.
- WebGL uses decimated CAD visuals. If WebGL is absent, Three.js SVGRenderer projects a lower-detail convex-shell G1 from the same MuJoCo kinematic tree. This remains interactive 3D, with reduced visual detail and speed. A native MuJoCo image is the final fallback if WASM or both renderers fail.
- Loading has a 60-second external worker deadline and 30-second network deadlines. A stuck synchronous model compilation can be terminated without freezing the UI.

## Model source and rebuild

The model and derived visual meshes come from:
https://github.com/unitreerobotics/unitree_mujoco/tree/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/g1

Unitree's BSD-3-Clause notice is retained under each model directory and distributed with the demo. Changes are visual mesh simplification only. The software-rendered fallback uses convex visual shells; neither version is a replacement for the private Inspire model.

```sh
python -m pip install -r tools/requirements-web-model.txt
# Use a checkout at the pinned source revision above:
python tools/build_web_model.py /path/to/unitree_mujoco
python tools/build_web_model.py /path/to/unitree_mujoco --output web/public/model-lite --faces-per-mesh 200 --hull
```

The manifests include source hashes, file hashes, triangle counts and byte sizes. The builder does not fetch private files, weights or datasets.

## MJB diagnosis

Default: MJCF + mesh files loaded into Emscripten MEMFS, then `MjModel.from_xml_path`.
MJB diagnostics are stored losslessly as `.mjb.gz` and decompressed before MEMFS loading.
For an explicit binary comparison, open the demo with `?format=mjb`. Python and WASM must both use 3.12.0.

Two binding details are covered by the actual WASM integration test:

1. `mj_id2name` and `mj_name2id` take numeric object IDs: use `mj.mjtObj.mjOBJ_JOINT.value`, not the enum wrapper object.
2. `MjModel.from_binary_path` requires an `MjVFS` instance; `null` raises `BindingError: null is not a valid MjVFS` in this package.

Both current MJCF and MJB load and produce matching geometry transforms at the target pose. The inaccessible previous 31 MB MJB was **not reproduced**, so these findings are not a proven explanation for that historical stall. A worker boundary and external deadline prevent its failure mode from trapping the page.

## Validation

`npm test` runs motion semantics tests and loads the actual G1 model through the real MuJoCo WASM package. The integration test checks all eight target qpos values, every unchanged qpos, left/right anatomical direction, finite transforms and MJCF/MJB parity.

Browser observations and current limitations are recorded in `VALIDATION.md`.
