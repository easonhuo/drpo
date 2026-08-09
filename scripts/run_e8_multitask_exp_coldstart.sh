#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${E8_COLDSTART_CONFIG:-${ROOT_DIR}/configs/e8_multitask_exp_coldstart.yaml}"
P0_CONFIG_PATH="${E8_COLDSTART_P0_CONFIG:-${ROOT_DIR}/configs/e8_multitask_p0.yaml}"
RUN_ID="${E8_COLDSTART_RUN_ID:-E8_MULTITASK_EXP_COLDSTART_20260808_01}"
OUTPUT_ROOT="${E8_COLDSTART_OUTPUT_ROOT:-${ROOT_DIR}/outputs/e8/${RUN_ID}}"
P0_WORK_DIR="${E8_COLDSTART_P0_WORK_DIR:-${OUTPUT_ROOT}/p0_inputs}"
COUNTDOWN_WORK_DIR="${E8_COLDSTART_COUNTDOWN_WORK_DIR:-${OUTPUT_ROOT}/countdown_inputs}"
VENV_DIR="${E8_COLDSTART_VENV_DIR:-${ROOT_DIR}/.venv-e8-coldstart}"
MODEL_DIR="${E8_COLDSTART_MODEL_DIR:-${ROOT_DIR}/models/Qwen2.5-0.5B-Instruct-7ae5576}"
EXPECTED_COMMIT="${E8_COLDSTART_EXPECTED_COMMIT:-}"
MODEL_REPO="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION="7ae557604adf67be50417f59c2c2f167def9a775"
MODE="${1:-full}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

check_source() {
  [[ -n "${EXPECTED_COMMIT}" ]] || fail "set E8_COLDSTART_EXPECTED_COMMIT to the reviewed implementation commit"
  local current_commit
  current_commit="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
  [[ "${current_commit}" == "${EXPECTED_COMMIT}" ]] || \
    fail "source commit mismatch: expected ${EXPECTED_COMMIT}, found ${current_commit}"
  git -C "${ROOT_DIR}" diff --quiet || fail "tracked source files are modified"
  git -C "${ROOT_DIR}" diff --cached --quiet || fail "tracked staged source files are modified"
}

activate_runtime() {
  [[ -x "${VENV_DIR}/bin/python" ]] || fail "runtime is absent; run '$0 setup' first"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
}

preflight_gpu() {
  command -v nvidia-smi >/dev/null || fail "nvidia-smi is unavailable"
  local gpu_count
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  [[ "${gpu_count}" -ge 8 ]] || fail "eight visible GPUs are required; found ${gpu_count}"
  local small_gpu
  small_gpu="$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits | awk -F, '$2+0 < 12000 {print $1":"$2}')"
  [[ -z "${small_gpu}" ]] || fail "each GPU needs at least 12 GiB; undersized: ${small_gpu}"
  python - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see CUDA"
assert torch.cuda.device_count() >= 8, f"PyTorch sees {torch.cuda.device_count()} GPUs"
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpus": torch.cuda.device_count()})
PY
  local free_kb
  free_kb="$(df -Pk "${ROOT_DIR}" | awk 'NR==2 {print $4}')"
  [[ "${free_kb}" -ge 83886080 ]] || fail "at least 80 GiB free disk is required"
}

run_module() {
  python -m drpo.e8_multitask_exp_tuning \
    --config "${CONFIG_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    "$@"
}

setup() {
  check_source
  command -v python3 >/dev/null || fail "python3 is unavailable"
  python3 - <<'PY'
import sys
assert sys.version_info >= (3, 10), sys.version
try:
    import torch
except ImportError as exc:
    raise SystemExit("Install a CUDA-compatible PyTorch build before running setup") from exc
assert torch.cuda.is_available(), "The system PyTorch build cannot see CUDA"
PY
  python3 -m venv --system-site-packages "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1"
  python -m pip install -r "${ROOT_DIR}/requirements/e8_multitask_exp_coldstart.txt"
  python -m pip install --no-deps -e "${ROOT_DIR}"
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${MODEL_REPO}",
    revision="${MODEL_REVISION}",
    local_dir="${MODEL_DIR}",
)
PY
  preflight_gpu
  python -m pytest -q \
    "${ROOT_DIR}/tests/test_e8_multitask_p0.py" \
    "${ROOT_DIR}/tests/test_countdown_e8_oracle_offline_v2_taper_sweep.py"
  python - <<PY
from pathlib import Path
from drpo.e8_multitask_exp_tuning import (
    audit_canonical_coldstart_sources,
    load_config,
)
config = load_config(Path("${CONFIG_PATH}"))
audit = audit_canonical_coldstart_sources(config)
assert audit["verified"], audit
print(audit)
PY
}

prepare() {
  check_source
  activate_runtime
  preflight_gpu
  "${ROOT_DIR}/scripts/run_e8_multitask_p0.sh" \
    --work-dir "${P0_WORK_DIR}" prepare
  "${ROOT_DIR}/scripts/run_e8_multitask_p0.sh" \
    --work-dir "${P0_WORK_DIR}" qualify
  python "${ROOT_DIR}/scripts/run_countdown_e8_oracle_bank_v2.py" \
    --config "${ROOT_DIR}/configs/countdown_e8_oracle_offline_bank_v2_0p5b.yaml" \
    --work_dir "${COUNTDOWN_WORK_DIR}"
  python "${ROOT_DIR}/scripts/v2_bank_convert.py" \
    --input "${COUNTDOWN_WORK_DIR}/data/oracle_offline_bank_v2_train.jsonl" \
    --output "${COUNTDOWN_WORK_DIR}/data/offline_bank_v2.jsonl" \
    --manifest "${COUNTDOWN_WORK_DIR}/data/offline_bank_v2.convert_manifest.json" \
    --model "${MODEL_DIR}"
  run_module prepare \
    --p0-work-dir "${P0_WORK_DIR}" \
    --p0-config "${P0_CONFIG_PATH}" \
    --countdown-bank "${COUNTDOWN_WORK_DIR}/data/offline_bank_v2.jsonl" \
    --countdown-validation "${COUNTDOWN_WORK_DIR}/data/val.jsonl"
  python - <<PY
import json
from pathlib import Path
value = {
    "schema_version": 1,
    "run_id": "${RUN_ID}",
    "source_commit": "${EXPECTED_COMMIT}",
    "model_repo": "${MODEL_REPO}",
    "model_revision": "${MODEL_REVISION}",
    "model_path": str(Path("${MODEL_DIR}").resolve()),
    "test_partition_accessed": False,
}
path = Path("${OUTPUT_ROOT}") / "source_provenance.json"
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

calibrate() {
  check_source
  activate_runtime
  mkdir -p "${OUTPUT_ROOT}/logs/calibration"
  local tasks=(
    countdown word_sorting mini_sudoku maze word_ladder knights_knaves graph_color wikisql
  )
  local pids=()
  local task
  local gpu
  for gpu in "${!tasks[@]}"; do
    task="${tasks[$gpu]}"
    CUDA_VISIBLE_DEVICES="${gpu}" LOCAL_RANK=0 run_module calibrate-task \
      --base-model-path "${MODEL_DIR}" \
      --task "${task}" \
      >"${OUTPUT_ROOT}/logs/calibration/${task}.log" 2>&1 &
    pids+=("$!")
  done
  local status=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  [[ "${status}" -eq 0 ]] || fail "one or more task calibrations failed; inspect logs/calibration"
  CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 run_module calibrate --base-model-path "${MODEL_DIR}"
}

liveness() {
  check_source
  activate_runtime
  CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 run_module liveness \
    --task countdown \
    --lambda 1.0498221244986778 \
    --base-model-path "${MODEL_DIR}"
}

run_queue() {
  check_source
  activate_runtime
  preflight_gpu
  run_module run-all --base-model-path "${MODEL_DIR}" --retry-incomplete
}

finish() {
  check_source
  activate_runtime
  run_module aggregate
  run_module audit
  run_module package
  python - <<PY
import json
from pathlib import Path
root = Path("${OUTPUT_ROOT}")
audit = json.loads((root / "terminal_audit.json").read_text())
package = json.loads((root / "packages/package_manifest.json").read_text())
assert audit["all_training_and_evaluation_complete"], audit
print("FULL_RESULTS_ZIP=" + package["full_results_zip"])
print("FULL_RESULTS_SHA256=" + package["full_results_zip_sha256"])
print("PLOT_CSV=" + package["plot_curve_points_csv"])
print("PLOT_CSV_SHA256=" + package["plot_curve_points_csv_sha256"])
PY
}

case "${MODE}" in
  setup) setup ;;
  prepare) prepare ;;
  plan) check_source; activate_runtime; run_module plan ;;
  calibrate) calibrate ;;
  liveness) liveness ;;
  run|resume) run_queue ;;
  finish) finish ;;
  full)
    setup
    prepare
    calibrate
    liveness
    run_queue
    finish
    ;;
  *) fail "usage: $0 {setup|prepare|plan|calibrate|liveness|run|resume|finish|full}" ;;
esac
