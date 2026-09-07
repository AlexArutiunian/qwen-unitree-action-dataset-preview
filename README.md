# nanoVLM training and visual verification

This branch preserves the nanoVLM training, MCQ evaluation and plotting code associated with visual verification in the DCNA 2026 Unitree G1 pipeline.

The dataset preview inherited from `main` remains in place. The recovered server code is organized under [`training/nanovlm`](training/nanovlm).

## Layout

```text
training/nanovlm/
├── train/            # three recovered MCQ fine-tuning entrypoints
├── evaluation/       # inference, comparison and paper-plot code
├── dataset_tools/    # RobotVQA/robot MCQ preparation utilities
├── notebooks/        # sanitized Kaggle experiment notebooks
├── artifacts/        # compact experiment metrics and plots
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

## Paper result provenance

The Google Drive archive added on 2026-09-07 contains the missing original paper experiment. [`experiment4-5-change-architect.ipynb`](training/nanovlm/notebooks/experiment4-5-change-architect.ipynb) and its extracted training entrypoint [`paper_single_gpu_aokvqa_mcq.py`](training/nanovlm/train/paper_single_gpu_aokvqa_mcq.py) establish the reported single-GPU configuration:

- vision encoder frozen;
- modality projector frozen;
- four-way MLP MCQ head trained;
- final four decoder layers unfrozen;
- A-OKVQA validation: `857/1,145 = 74.85%`.

This is distinct from the later RobotVQA server scripts, which train the projector and report `613/861` on another validation set. See [`PAPER_RUN.md`](training/nanovlm/PAPER_RUN.md) for the exact command and interpretation.

Notebook outputs and execution counters were removed before commit. The originals contained large embedded logs/images and plaintext Kaggle credentials; source cells now read `KAGGLE_USERNAME` and `KAGGLE_KEY` from the environment.

See [`REMOTE_ARTIFACTS.md`](training/nanovlm/REMOTE_ARTIFACTS.md) for paths to the uncommitted checkpoints.
