# Qwen training for Unitree motion JSON

This branch preserves the code used to fine-tune and evaluate a local Qwen model for Russian natural-language command to Unitree G1 motion JSON generation.

The static dataset preview inherited from `main` is intentionally kept unchanged. Training code lives under [`training/qwen-json`](training/qwen-json).

## Reproduced paper run

The run matching the DCNA 2026 paper is configured by:

```text
training/qwen-json/configs/train_compact_8192.yaml
```

Its preserved manifest reports:

- base model: `Qwen/Qwen2.5-Coder-7B-Instruct`;
- QLoRA rank/alpha: `4/16`;
- target modules: `q_proj`, `v_proj`;
- dataset splits: 1,401 train, 148 validation, 77 reserved;
- best checkpoint: step 150;
- best validation loss: `0.32654744386672974`.

## Layout

```text
training/qwen-json/
├── src/          # dataset preparation, training, inference and evaluation
├── configs/      # reproducible QLoRA run configurations
├── scripts/      # train/eval helpers and dataset builders
├── artifacts/    # small manifests and training histories; no weights
├── benchmarks/   # lightweight benchmark outputs
└── notebooks/    # compact launcher notebook preserved from the server
```

## Run

```bash
cd training/qwen-json
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --config configs/train_compact_8192.yaml
```

The dataset is downloaded from `AlexArutiunian/qwen-robot-action-dataset`. Set `HF_TOKEN` only in the environment when access requires it. No token or model weight is committed.

See [`REMOTE_ARTIFACTS.md`](training/qwen-json/REMOTE_ARTIFACTS.md) for the original server paths.
