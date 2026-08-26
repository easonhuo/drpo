#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export E8_COLDSTART_EXPERIMENT_ID="EXT-C-E8-MULTITASK-EXP-LAMBDA-COMPLETION-01"
export E8_COLDSTART_CONFIG="${ROOT_DIR}/configs/e8_multitask_exp_lambda_completion.yaml"
export E8_COLDSTART_RUN_ID="${E8_COLDSTART_RUN_ID:-E8_MULTITASK_EXP_LAMBDA_COMPLETION_01}"
exec bash "${ROOT_DIR}/scripts/run_e8_multitask_exp_coldstart.sh" "$@"
