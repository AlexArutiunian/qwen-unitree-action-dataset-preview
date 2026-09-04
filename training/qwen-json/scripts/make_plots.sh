#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR="${1:-outputs/compact_4096}"
python -m src.plot_training --run-dir "$RUN_DIR"
