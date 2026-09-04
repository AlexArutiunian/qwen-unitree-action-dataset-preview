# Remote artifact inventory

Source server: `arutiunyan_ag@93.175.18.15:33322`  
Source project: `/home/arutiunyan_ag/alut/qwen_lora`

These artifacts were deliberately not copied into Git.

## Paper-matching QLoRA run

| Artifact | Remote path | Approximate size |
|---|---|---:|
| Best adapter copy | `/home/arutiunyan_ag/alut/qwen_lora/outputs/compact_8192/best_adapter` | 16 MB |
| Best checkpoint (step 150) | `/home/arutiunyan_ag/alut/qwen_lora/outputs/compact_8192/checkpoints/checkpoint-150` | 7.8 MB |
| Checkpoint 300 | `/home/arutiunyan_ag/alut/qwen_lora/outputs/compact_8192/checkpoints/checkpoint-300` | 7.9 MB |
| Checkpoint 350 | `/home/arutiunyan_ag/alut/qwen_lora/outputs/compact_8192/checkpoints/checkpoint-350` | 7.9 MB |

The server also contains `/home/arutiunyan_ag/alut/qwen_lora/outputs/best_adapter` (about 165 MB), but this appears to belong to a different/later run and must not be confused with the `compact_8192` adapter.

The base `Qwen/Qwen2.5-Coder-7B-Instruct` cache was not present as a resolved model directory during the 2026-09-04 inventory. Hugging Face lock entries remained under `/home/arutiunyan_ag/.cache/huggingface/hub/.locks/`.

## Dataset and generated outputs

- Download/cache directory: `/home/arutiunyan_ag/alut/qwen_lora/outputs/hf_downloaded_dataset`
- Prepared 8k-token dataset: `/home/arutiunyan_ag/alut/qwen_lora/outputs/dataset_compact_8192`
- Paper-matching run directory: `/home/arutiunyan_ag/alut/qwen_lora/outputs/compact_8192`
- RAG benchmark: `/home/arutiunyan_ag/alut/qwen_lora/outputs/bench_multitask_motion_20260517_121341`
