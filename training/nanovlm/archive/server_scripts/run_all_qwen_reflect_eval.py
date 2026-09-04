from pathlib import Path
import subprocess
import shlex

DATASET = "reflect_doors_all_yesno4_eval"

cache = Path.home() / ".cache/huggingface/hub"
models = sorted(cache.glob("models--Qwen--*VL*"))

model_ids = []
for p in models:
    model_id = p.name.replace("models--", "").replace("--", "/")
    # Берём только Instruct VL, чтобы не хватать лишнее
    if "Instruct" in model_id and "VL" in model_id:
        model_ids.append(model_id)

# На случай если cache-поиск ничего не нашёл, но 3B точно доступна
if "Qwen/Qwen2.5-VL-3B-Instruct" not in model_ids:
    model_ids.append("Qwen/Qwen2.5-VL-3B-Instruct")

print("MODELS TO EVAL:")
for m in model_ids:
    print(" -", m)

for model in model_ids:
    safe = model.replace("/", "__").replace(".", "_").replace("-", "_")
    out_dir = f"qwen_reflect_all_yesno4__{safe}"

    print("\n" + "=" * 100)
    print("MODEL:", model)
    print("OUT:", out_dir)

    cmd = [
        "python",
        "validate_qwen_fourchoice_from_metadata_explain.py",
        "--dataset-dir", DATASET,
        "--out-dir", out_dir,
        "--model", model,
        "--dtype", "bf16",
    ]

    print("CMD:", " ".join(shlex.quote(x) for x in cmd))
    r = subprocess.run(cmd)

    if r.returncode != 0:
        print("FAILED:", model, "returncode:", r.returncode)
    else:
        print("DONE:", model)
