# Remote artifact inventory

Source server: `arutiunyan_ag@93.175.18.15:33322`  
Source directory: `/home/arutiunyan_ag/alut/qwen_lora/nanovlm`

The following large model directories remain on the server and were not copied into Git:

| Description | Remote path | Approximate size |
|---|---|---:|
| Raw/evaluation nanoVLM model | `/home/arutiunyan_ag/alut/qwen_lora/nanovlm/nanovlm_eval_work/kaggle_model_dataset` | 1.8 GB |
| Robot affordance best model | `/home/arutiunyan_ag/alut/qwen_lora/nanovlm/run_robovqa_affordance_10000_distill_clean/runs/best_model` | 1.8 GB |
| Continuation best model | `/home/arutiunyan_ag/alut/qwen_lora/nanovlm/run_robovqa_affordance_10000_distill_clean_cont/runs/best_model` | 1.8 GB |
| Continuation 2 best model used by current paper-plot code | `/home/arutiunyan_ag/alut/qwen_lora/nanovlm/run_robovqa_affordance_10000_distill_clean_cont2/runs/best_model` | 1.8 GB |
| Paraphrase-quality best model | `/home/arutiunyan_ag/alut/qwen_lora/nanovlm/run_robovqa_affordance_10000_para_quality/runs/best_model` | 1.8 GB |
| AI2-THOR Kaggle model | `/home/arutiunyan_ag/alut/qwen_lora/nanovlm/run_ai2thor_1k/kaggle_model_dataset` | 1.8 GB |
| Qwen3-VL-8B local model | `/home/arutiunyan_ag/models/qwen3-vl-8b` | 17 GB |

Additional historical nanoVLM checkpoints follow the same layout under `/home/arutiunyan_ag/alut/qwen_lora/nanovlm/run_*/runs/{best_model,last_model,final_model}`. Most individual `model.safetensors` files are about 1.84 GB.

Evaluation and paper-plot inputs:

- `/home/arutiunyan_ag/alut/qwen_lora/nanovlm/eval_big_robot_mcq_val_best_nano`
- `/home/arutiunyan_ag/alut/qwen_lora/nanovlm/thesis_vlm_mcq_plots`
- `/home/arutiunyan_ag/alut/qwen_lora/nanovlm/robovqa_affordance_10000_accepted_split_val`
