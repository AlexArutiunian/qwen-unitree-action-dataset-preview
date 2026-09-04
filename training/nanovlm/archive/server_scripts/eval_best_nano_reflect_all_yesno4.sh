#!/usr/bin/env bash
set -e

DATASET="reflect_doors_all_yesno4_eval"

declare -A MODELS

MODELS["nano_raw"]="/home/arutiunyan_ag/alut/qwen_lora/nanovlm/nanovlm_eval_work/kaggle_model_dataset"
MODELS["nano_deepdoors2_best"]="/home/arutiunyan_ag/alut/qwen_lora/nanovlm/run_deepdoors2_door_state_from_affordance_cont2/runs/best_model"
MODELS["nano_yesno4_before_boost"]="/home/arutiunyan_ag/alut/qwen_lora/nanovlm/run_affordance_to_deepdoors2_reflect_yesno4_no_vision/runs/best_model"
MODELS["nano_reflect_closed_boost"]="/home/arutiunyan_ag/alut/qwen_lora/nanovlm/run_affordance_to_deepdoors2_reflect_yesno4_reflect_closed_boost/runs/best_model"

for name in "${!MODELS[@]}"; do
  model="${MODELS[$name]}"
  out="eval_reflect_all_yesno4_${name}"

  echo
  echo "================================================================================"
  echo "$name"
  echo "$model"

  if [ ! -f "$model/model.safetensors" ]; then
    echo "SKIP missing model: $model"
    continue
  fi

  python eval_nanovlm_robovqa_mcq.py \
    --dataset-dir "$DATASET" \
    --out-dir "$out" \
    --mode normal \
    --model-dir "$model"
done
