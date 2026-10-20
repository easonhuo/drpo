#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNSPEC="${E7_FIG1_RUNSPEC:-${ROOT}/configs/e7_figure1_d4rl9_runspec.yaml}"
DATASET_ROOT="${D4RL_DATASET_ROOT:-}"
WORK_DIR="${E7_FIG1_WORK_DIR:-${ROOT}/outputs/e7_figure1_d4rl9/run_001}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -z "${DATASET_ROOT}" ]]; then
  cat >&2 <<'EOF'
D4RL_DATASET_ROOT is required and must contain these nine exact files:
  halfcheetah_medium-v2.hdf5
  halfcheetah_medium_replay-v2.hdf5
  halfcheetah_medium_expert-v2.hdf5
  hopper_medium-v2.hdf5
  hopper_medium_replay-v2.hdf5
  hopper_medium_expert-v2.hdf5
  walker2d_medium-v2.hdf5
  walker2d_medium_replay-v2.hdf5
  walker2d_medium_expert-v2.hdf5

Example:
  D4RL_DATASET_ROOT=/data/d4rl-v2 bash scripts/run_e7_figure1_d4rl9.sh
EOF
  exit 2
fi

exec "${PYTHON_BIN}" "${ROOT}/scripts/run_e7_bench.py" figure1-d4rl9 \
  --runspec "${RUNSPEC}" \
  --dataset-root "${DATASET_ROOT}" \
  --work-dir "${WORK_DIR}" \
  "$@"
