#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${D4RL_DATASET_ROOT:?Set D4RL_DATASET_ROOT to the directory containing the nine D4RL-v2 HDF5 files.}"

RUNSPEC="${E7_FIG1_RUNSPEC:-configs/e7_figure1_d4rl9_runspec.yaml}"
WORK_DIR="${E7_FIG1_WORK_DIR:-outputs/e7_figure1_d4rl9/run_001}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" scripts/run_e7_bench.py figure1-d4rl9 \
  --runspec "$RUNSPEC" \
  --dataset-root "$D4RL_DATASET_ROOT" \
  --work-dir "$WORK_DIR" \
  "$@"
