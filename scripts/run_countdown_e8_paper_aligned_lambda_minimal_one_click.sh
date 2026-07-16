#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MODEL_PATH="${E8_PAPER_MODEL_PATH:-/root/models/Qwen2.5-0.5B-Instruct}"
WORK_DIR="${E8_PAPER_WORK_DIR:-/root/experiment_output/e8_paper_aligned_lambda_minimal}"
BANK="${E8_PAPER_BANK:-/root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl}"
VAL="${E8_PAPER_VAL:-/root/experiment_output/e8_oracle_bank_v2/data/val.jsonl}"
BASE_CONFIG="${E8_PAPER_BASE_CONFIG:-configs/countdown_e8_base_rl_replay_0p5b.yaml}"
GRID_CONFIG="${E8_PAPER_GRID_CONFIG:-configs/countdown_e8_paper_aligned_lambda_minimal_0p5b.yaml}"
AUTO="scripts/run_countdown_e8_paper_aligned_lambda_minimal_auto.py"
CALIBRATE="scripts/calibrate_countdown_e8_paper_aligned_lambda.py"

mkdir -p "${WORK_DIR}"
COMMON_ARGS=(
  --model_path "${MODEL_PATH}"
  --work_dir "${WORK_DIR}"
  --bank "${BANK}"
  --val "${VAL}"
  --base_config "${BASE_CONFIG}"
  --grid_config "${GRID_CONFIG}"
  --allow-dev-unregistered
)

python "${AUTO}" plan "${COMMON_ARGS[@]}"
FIRST_GPU="$(python - "${WORK_DIR}/RUNTIME_SELECTION.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding='utf-8'))
print(payload['selection']['selected_device_ids'][0])
PY
)"
CUDA_VISIBLE_DEVICES="${FIRST_GPU}" LOCAL_RANK=0 \
  python "${CALIBRATE}" \
    --model_path "${MODEL_PATH}" \
    --bank "${BANK}" \
    --base_config "${BASE_CONFIG}" \
    --grid_config "${GRID_CONFIG}" \
    --work_dir "${WORK_DIR}"

export DRPO_E8_PAPER_CALIBRATION="${WORK_DIR}/TAPER_CALIBRATION.json"
python "${AUTO}" run "${COMMON_ARGS[@]}"
