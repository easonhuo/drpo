#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export E8_COLDSTART_EXPERIMENT_ID="EXT-C-E8-MULTITASK-EXP-LAMBDA-COMPLETION-01"
export E8_COLDSTART_CONFIG="${ROOT_DIR}/configs/e8_multitask_exp_lambda_completion.yaml"
export E8_COLDSTART_RUN_ID="${E8_COLDSTART_RUN_ID:-E8_MULTITASK_EXP_LAMBDA_COMPLETION_01}"
export E8_COLDSTART_RUN_CLASS="${E8_COLDSTART_RUN_CLASS:-pilot}"
export E8_COLDSTART_REQUIRE_ORIGIN_MAIN="${E8_COLDSTART_REQUIRE_ORIGIN_MAIN:-0}"
if [[ -z "${E8_COLDSTART_TARGET_REF:-}" ]]; then
  branch="$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD)"
  [[ "${branch}" != "HEAD" ]] || {
    echo "ERROR: set E8_COLDSTART_TARGET_REF from a detached checkout" >&2
    exit 2
  }
  export E8_COLDSTART_TARGET_REF="refs/heads/${branch}"
fi
export E8_COLDSTART_BOOTSTRAP_ROOT="${E8_COLDSTART_BOOTSTRAP_ROOT:-${ROOT_DIR}/../drpo-e8-lambda-completion-${1:-full}}"
exec bash "${ROOT_DIR}/scripts/bootstrap_e8_multitask_exp_coldstart.sh" "$@"
