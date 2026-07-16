#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MODEL_PATH="${E8_PAPER_ALIGNED_LAMBDA_MODEL_PATH:-/root/models/Qwen2.5-0.5B-Instruct}"
WORK_DIR="${E8_PAPER_ALIGNED_LAMBDA_WORK_DIR:-outputs/e8/paper_aligned_lambda_round1_001}"
BANK="${E8_PAPER_ALIGNED_LAMBDA_BANK:-/root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl}"
VAL="${E8_PAPER_ALIGNED_LAMBDA_VAL:-/root/experiment_output/e8_oracle_bank_v2/data/val.jsonl}"
BASE_CONFIG="${E8_PAPER_ALIGNED_LAMBDA_BASE_CONFIG:-configs/countdown_e8_base_rl_replay_0p5b.yaml}"
GRID_CONFIG="${E8_PAPER_ALIGNED_LAMBDA_GRID_CONFIG:-configs/countdown_e8_paper_aligned_lambda_round1_0p5b.yaml}"
CANDIDATE_GPUS="${E8_PAPER_ALIGNED_LAMBDA_GPUS:-}"
MAX_DEVICES="${E8_PAPER_ALIGNED_LAMBDA_MAX_DEVICES:-}"

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
)
if [[ -n "${CANDIDATE_GPUS}" ]]; then
  ARGS+=(--gpus "${CANDIDATE_GPUS}")
fi
if [[ -n "${MAX_DEVICES}" ]]; then
  ARGS+=(--max-devices "${MAX_DEVICES}")
fi

python scripts/run_countdown_e8_paper_aligned_lambda_auto.py "${ARGS[@]}"
