# Qwen Robot Action Dataset Preview

![Unitree G1 action preview](assets/robot-action-preview.gif)

Static HTML preview for the Unitree G1 Russian command → robot motion JSON SFT dataset.

The full dataset is intended for LoRA/QLoRA fine-tuning of Qwen-family models on structured robot motion JSON generation.

## Preview

Open the GitHub Pages site to inspect dataset samples in table form.

## Dataset summary

- Total samples: 1626
- Train: 1401
- Validation: 148
- Reserved: 77

We use the new format for JSON:

<img width="660" height="585" alt="image" src="https://github.com/user-attachments/assets/a46fe294-f153-4a69-9f52-1655a4fd82a5" />


## Main task

Russian natural-language robot command → Unitree G1 motion JSON plan.

## Files

- `index.html` — static dataset preview.
- `robot_sft_preview.csv` — CSV preview table.

The full training files should be hosted separately on Hugging Face.
https://huggingface.co/datasets/AlexArutiunian/g1-json-to-action/
