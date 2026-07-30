#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${E8_DENSE_CONFIG:-${ROOT_DIR}/configs/e8_multitask_exp_lambda_dense.yaml}"
PARENT_CONFIG_PATH="${E8_DENSE_PARENT_CONFIG:-${ROOT_DIR}/configs/e8_multitask_exp_tuning.yaml}"
OUTPUT_ROOT="${E8_DENSE_OUTPUT_ROOT:-${ROOT_DIR}/outputs/e8/e8_multitask_exp_lambda_dense_001}"
PARENT_OUTPUT_ROOT="${E8_DENSE_PARENT_OUTPUT_ROOT:-}"
BASE_MODEL_PATH="${E8_DENSE_BASE_MODEL_PATH:-}"
EXPECTED_COMMIT="${E8_DENSE_EXPECTED_COMMIT:-}"
MODE="${1:-full}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

run_module() {
  python3 -m drpo.e8_multitask_exp_tuning \
    --config "${CONFIG_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    "$@"
}

check_source() {
  if [[ -z "${EXPECTED_COMMIT}" ]]; then
    echo "E8_DENSE_EXPECTED_COMMIT is required for ${MODE}" >&2
    exit 2
  fi
  local current_commit
  current_commit="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
  if [[ "${current_commit}" != "${EXPECTED_COMMIT}" ]]; then
    echo "Source commit mismatch: expected ${EXPECTED_COMMIT}, found ${current_commit}" >&2
    exit 3
  fi
  if ! git -C "${ROOT_DIR}" diff --quiet || ! git -C "${ROOT_DIR}" diff --cached --quiet; then
    echo "Dense refinement requires a clean source checkout" >&2
    exit 4
  fi
}

require_runtime_inputs() {
  if [[ -z "${PARENT_OUTPUT_ROOT}" ]]; then
    echo "E8_DENSE_PARENT_OUTPUT_ROOT must point to the completed 72-cell predecessor output" >&2
    exit 5
  fi
  if [[ -z "${BASE_MODEL_PATH}" ]]; then
    echo "E8_DENSE_BASE_MODEL_PATH must point to Qwen2.5-0.5B-Instruct" >&2
    exit 6
  fi
}

inherit_inputs() {
  run_module inherit \
    --parent-output-root "${PARENT_OUTPUT_ROOT}" \
    --parent-config "${PARENT_CONFIG_PATH}" \
    --base-model-path "${BASE_MODEL_PATH}"
}

calibrate() {
  run_module calibrate --base-model-path "${BASE_MODEL_PATH}"
}

liveness() {
  run_module liveness \
    --task word_sorting \
    --lambda 1.3862943611198906 \
    --base-model-path "${BASE_MODEL_PATH}"
}

run_all() {
  run_module run-all --base-model-path "${BASE_MODEL_PATH}"
}

case "${MODE}" in
  plan)
    run_module plan
    ;;
  inherit)
    check_source
    require_runtime_inputs
    inherit_inputs
    ;;
  calibrate)
    check_source
    require_runtime_inputs
    calibrate
    ;;
  liveness)
    check_source
    require_runtime_inputs
    liveness
    ;;
  run)
    check_source
    require_runtime_inputs
    run_all
    ;;
  aggregate)
    check_source
    run_module aggregate
    ;;
  audit)
    check_source
    run_module audit
    ;;
  full)
    check_source
    require_runtime_inputs
    inherit_inputs
    calibrate
    liveness
    run_all
    run_module aggregate
    run_module audit
    ;;
  *)
    echo "Usage: $0 {plan|inherit|calibrate|liveness|run|aggregate|audit|full}" >&2
    exit 2
    ;;
esac
