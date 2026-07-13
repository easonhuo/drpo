#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MODEL_PATH="${E8_CONTINUOUS_EXP_MODEL_PATH:-/root/models/Qwen2.5-0.5B-Instruct}"
WORK_DIR="${E8_CONTINUOUS_EXP_WORK_DIR:-/root/experiment_output/e8_v2_continuous_exp_grid_dev}"
BANK="${E8_CONTINUOUS_EXP_BANK:-/root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl}"
VAL="${E8_CONTINUOUS_EXP_VAL:-/root/experiment_output/e8_oracle_bank_v2/data/val.jsonl}"
BASE_CONFIG="${E8_CONTINUOUS_EXP_BASE_CONFIG:-configs/countdown_e8_base_rl_replay_0p5b.yaml}"
GRID_CONFIG="${E8_CONTINUOUS_EXP_GRID_CONFIG:-configs/countdown_e8_oracle_offline_v2_continuous_exp_grid_0p5b.yaml}"
CANDIDATE_GPUS="${E8_CONTINUOUS_EXP_GPUS:-}"
MAX_DEVICES="${E8_CONTINUOUS_EXP_MAX_DEVICES:-}"

for required in "${MODEL_PATH}" "${BANK}" "${VAL}" "${BASE_CONFIG}" "${GRID_CONFIG}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required input: ${required}" >&2
    exit 2
  fi
done

ARGS=(
  run
  --model_path "${MODEL_PATH}"
  --work_dir "${WORK_DIR}"
  --bank "${BANK}"
  --val "${VAL}"
  --base_config "${BASE_CONFIG}"
  --grid_config "${GRID_CONFIG}"
  --allow-dev-unregistered
)
if [[ -n "${CANDIDATE_GPUS}" ]]; then
  ARGS+=(--gpus "${CANDIDATE_GPUS}")
fi
if [[ -n "${MAX_DEVICES}" ]]; then
  ARGS+=(--max-devices "${MAX_DEVICES}")
fi

python scripts/run_countdown_e8_oracle_offline_v2_continuous_exp_auto.py "${ARGS[@]}"
