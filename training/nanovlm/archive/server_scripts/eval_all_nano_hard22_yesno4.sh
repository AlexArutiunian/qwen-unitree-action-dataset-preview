#!/usr/bin/env bash
set -e

DATASET="real_glass_hard22_yesno4_eval"

declare -A MODELS

MODELS["nano_raw"]="/home/arutiunyan_ag/alut/qwen_lora/nanovlm/nanovlm_eval_work/kaggle_model_dataset"
MODELS["nano_affordance"]="/home/arutiunyan_ag/alut/qwen_lora/nanovlm/run_robovqa_affordance_10000_distill_clean_cont2/runs/best_model"
MODELS["nano_deepdoors2_best"]="/home/arutiunyan_ag/alut/qwen_lora/nanovlm/run_deepdoors2_door_state_from_affordance_cont2/runs/best_model"

# до reflect-boost, первый yesno4
MODELS["nano_yesno4_before_boost"]="/home/arutiunyan_ag/alut/qwen_lora/nanovlm/run_affordance_to_deepdoors2_reflect_yesno4_no_vision/runs/best_model"

# continuation, где val поднялся до ~68.57
MODELS["nano_yesno4_cont"]="/home/arutiunyan_ag/alut/qwen_lora/nanovlm/run_affordance_to_deepdoors2_reflect_yesno4_no_vision_cont/runs/best_model"

# boost reflect closed
MODELS["nano_reflect_closed_boost"]="/home/arutiunyan_ag/alut/qwen_lora/nanovlm/run_affordance_to_deepdoors2_reflect_yesno4_reflect_closed_boost/runs/best_model"

# overnight 25k
MODELS["nano_overnight_25k"]="/home/arutiunyan_ag/alut/qwen_lora/nanovlm/run_reflect_closed_boost_overnight_25k_from_boost_best/runs/best_model"

for name in "${!MODELS[@]}"; do
  model="${MODELS[$name]}"
  out="eval_hard22_yesno4_${name}"

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
