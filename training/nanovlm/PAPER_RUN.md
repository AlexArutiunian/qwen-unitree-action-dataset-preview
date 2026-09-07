# DCNA 2026 nanoVLM paper run

## Source

The original experiment was recovered from `drive-download-20260907T120538Z-1-001.zip`:

- notebook: `experiment4-5-change-architect.ipynb`;
- extracted train script: `train/paper_single_gpu_aokvqa_mcq.py`;
- base model: `lusxvr/nanoVLM`;
- validation dataset: `HuggingFaceM4/A-OKVQA`, validation split.

The notebook was executed on one NVIDIA Tesla T4. It records 1,145 validation questions and 857 correct predictions.

## Exact training command

Run from the bundled nanoVLM source directory, or otherwise make its `models` and `data` packages importable:

```bash
python ../../train/paper_single_gpu_aokvqa_mcq.py \
  --model_id lusxvr/nanoVLM \
  --output_dir checkpoints_mcq_single \
  --epochs 1 \
  --lr 1e-4 \
  --grad_accum 16 \
  --label_smoothing 0.1 \
  --head_dropout 0.1 \
  --unfreeze_decoder_layers 4 \
  --freeze_vision \
  --freeze_proj
```

The code attaches a four-choice MLP head and trains it together with the last four decoder layers. Both the vision encoder and modality projector are frozen by the explicit flags above.

## Recorded result

```text
step 500:  68.73%
step 1000: 74.85%
final:     74.85%
correct:   857 / 1,145
P50 E2E:   269.62 ms
```

The paper's binary confusion matrix expands every four-way question into one positive and three negative option records, producing 4,580 binary records. It is a diagnostic derived from the same MCQ predictions, not a separate binary test set.

## Other recovered experiments

- Experiment 1: baseline and revised single-GPU training, best final result 69.96%.
- Experiments 2–3: two-GPU DDP training/evaluation, approximately 67% accuracy and worse efficiency for this workload.
- Experiments 4–5: stronger MCQ architecture and partial decoder unfreezing; this is the 74.85% paper run.
- Experiments 9–10: preprocessing and DDP variants; recorded result 72.40% on the full validation pass.

The notebooks in `notebooks/` retain code and markdown but have outputs stripped. This file preserves the relevant recorded results without committing credentials or megabytes of progress-bar output.
