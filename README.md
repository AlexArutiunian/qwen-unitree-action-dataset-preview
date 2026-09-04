# nanoVLM training and visual verification

This branch preserves the nanoVLM training, MCQ evaluation and plotting code associated with visual verification in the DCNA 2026 Unitree G1 pipeline.

The dataset preview inherited from `main` remains in place. The recovered server code is organized under [`training/nanovlm`](training/nanovlm).

## Layout

```text
training/nanovlm/
├── train/            # three recovered MCQ fine-tuning entrypoints
├── evaluation/       # inference, comparison and paper-plot code
├── dataset_tools/    # RobotVQA/robot MCQ preparation utilities
├── upstream/nanoVLM # model architecture and upstream training code
└── archive/          # remaining server experiment scripts, kept for provenance
```

No images, videos, datasets, virtual environments or model weights are committed.

## Typical invocation

Run from `training/nanovlm` and point the wrapper at the bundled model implementation:

```bash
python train/train_nanovlm_image_mcq.py \
  --train-dataset-dir /path/to/train \
  --val-dataset-dir /path/to/val \
  --model-dir /path/to/base-or-checkpoint \
  --repo-dir upstream/nanoVLM \
  --work-dir runs/robot-mcq
```

## Freeze-policy provenance note

The recovered `train_nanovlm_image_mcq.py` first freezes every parameter, then enables the MCQ head, modality projector and the requested final decoder layers. The vision encoder stays frozen. `train_nanovlm_image_mcq_fuller.py` additionally allows final vision layers to be enabled, but defaults to zero such layers.

This does **not** exactly match the revised paper wording that both the vision encoder and modality projector were frozen. No separate server script implementing that exact combination was found during the 2026-09-04 inventory. The recovered code is kept unchanged so the repository records what actually existed on the server.

The reported `74.85% (857/1,145)` artifact was also not found verbatim. The closest paper-plot source currently points to an 861-example evaluation (`613/861`, 71.20%). This discrepancy should be resolved before claiming full reproduction.

See [`REMOTE_ARTIFACTS.md`](training/nanovlm/REMOTE_ARTIFACTS.md) for paths to the uncommitted checkpoints.
