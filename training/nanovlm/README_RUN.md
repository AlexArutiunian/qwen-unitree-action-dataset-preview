# Local AI2-THOR -> nanoVLM pipeline

## 1) Install

```bash
sudo apt-get update
sudo apt-get install -y xvfb libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 libvulkan1 vulkan-tools
pip install ai2thor pillow pandas tqdm kaggle
```

Check Vulkan/GPU:

```bash
nvidia-smi
vulkaninfo --summary || true
ls /usr/share/vulkan/icd.d
```

## 2) Generate 10 samples

If normal headless Unity:

```bash
xvfb-run -s "-screen 0 1024x768x24" python generate_ai2thor_openable_dataset.py --target-images 10 --platform normal
```

If NVIDIA Vulkan works:

```bash
python generate_ai2thor_openable_dataset.py --target-images 10 --platform cloud
```

## 3) Generate training set

Start moderate:

```bash
python generate_ai2thor_openable_dataset.py \
  --out-dir ai2thor_openable_dataset_5k \
  --platform cloud \
  --target-images 5000 \
  --max-scenes 120 \
  --views-per-scene 80 \
  --width 640 --height 480 \
  --allow-types Drawer Cabinet Fridge Microwave Safe Box
```

5000 images => about 15000 MCQ examples.

## 4) Kaggle credentials

Put kaggle.json into ~/.kaggle/kaggle.json:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## 5) Train nanoVLM

```bash
python train_nanovlm_ai2thor_local.py \
  --dataset-dir ai2thor_openable_dataset_5k \
  --kaggle-dataset alexandrdixsept/nanovlm-mcq-upd-arch-exp4-5 \
  --work-dir run_ai2thor_5k \
  --max-steps 8000 \
  --eval-every 500 \
  --val-max-examples 500 \
  --grad-accum 8 \
  --amp-dtype bf16
```

Best model will be in:

```text
run_ai2thor_5k/runs/best_model
```
