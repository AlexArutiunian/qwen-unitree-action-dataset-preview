#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CUDA_LIBS="$(
  find "$PWD/.venv/lib" \
    -type d \
    -path '*/site-packages/nvidia/*/lib' \
    -print | paste -sd: -
)"

if [ -z "$CUDA_LIBS" ]; then
  echo "ОШИБКА: CUDA-библиотеки внутри .venv не найдены"
  exit 1
fi

echo "[CUDA] $CUDA_LIBS"

tmux kill-session -t g1_voice_server 2>/dev/null || true

tmux new-session -d -s g1_voice_server \
  "cd '$PWD' && export LD_LIBRARY_PATH='$CUDA_LIBS':\${LD_LIBRARY_PATH:-} && exec .venv/bin/python -u server.py"

sleep 3
tmux capture-pane -pt g1_voice_server -S -100
