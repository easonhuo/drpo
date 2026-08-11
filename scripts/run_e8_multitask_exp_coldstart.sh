#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ID="EXT-C-E8-MULTITASK-EXP-COLDSTART-01"
CONFIG_PATH="${E8_COLDSTART_CONFIG:-${ROOT_DIR}/configs/e8_multitask_exp_coldstart.yaml}"
P0_CONFIG_PATH="${E8_COLDSTART_P0_CONFIG:-${ROOT_DIR}/configs/e8_multitask_p0.yaml}"
RUN_ID="${E8_COLDSTART_RUN_ID:-E8_MULTITASK_EXP_COLDSTART_20260808_01}"
RUNTIME_ROOT="${E8_COLDSTART_RUNTIME_ROOT:-${ROOT_DIR}/../drpo-e8-coldstart-runtime}"
GUARD_ROOT="${E8_COLDSTART_GUARD_ROOT:-${RUNTIME_ROOT}/guard/${RUN_ID}}"
OUTPUT_ROOT="${E8_COLDSTART_OUTPUT_ROOT:-${GUARD_ROOT}/workload}"
P0_WORK_DIR="${E8_COLDSTART_P0_WORK_DIR:-${OUTPUT_ROOT}/p0_inputs}"
COUNTDOWN_WORK_DIR="${E8_COLDSTART_COUNTDOWN_WORK_DIR:-${OUTPUT_ROOT}/countdown_inputs}"
VENV_DIR="${E8_COLDSTART_VENV_DIR:-${RUNTIME_ROOT}/venv}"
SELFTEST_VENV_DIR="${E8_COLDSTART_SELFTEST_VENV_DIR:-${RUNTIME_ROOT}/selftest-venv}"
MODEL_DIR="${E8_COLDSTART_MODEL_DIR:-${RUNTIME_ROOT}/models/Qwen2.5-0.5B-Instruct-7ae5576}"
GUARD_ARTIFACT="${E8_COLDSTART_GUARD_ARTIFACT:-${RUNTIME_ROOT}/packages/${RUN_ID}_guarded.zip}"
EXPECTED_COMMIT="${E8_COLDSTART_EXPECTED_COMMIT:-}"
MODEL_REPO="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION="7ae557604adf67be50417f59c2c2f167def9a775"
MODE="${1:-full}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

is_expected_origin() {
  local url="$1"
  [[ "${url}" =~ ^https://([^/@]+(:[^/@]*)?@)?github\.com/easonhuo/drpo(\.git)?/?$ ]] ||
    [[ "${url}" =~ ^git@github\.com:easonhuo/drpo(\.git)?$ ]] ||
    [[ "${url}" =~ ^ssh://git@github\.com/easonhuo/drpo(\.git)?/?$ ]]
}

check_source() {
  [[ -n "${EXPECTED_COMMIT}" ]] || fail "set E8_COLDSTART_EXPECTED_COMMIT to the reviewed implementation commit"
  local current_commit
  current_commit="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
  [[ "${current_commit}" == "${EXPECTED_COMMIT}" ]] || \
    fail "source commit mismatch: expected ${EXPECTED_COMMIT}, found ${current_commit}"
  [[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "expected commit must be a full lowercase SHA"
  local origin_url
  origin_url="$(git -C "${ROOT_DIR}" remote get-url origin)"
  is_expected_origin "${origin_url}" || fail "origin is not the canonical easonhuo/drpo repository"
  [[ -z "$(git -C "${ROOT_DIR}" status --porcelain=v1 --untracked-files=all)" ]] || \
    fail "source checkout must be fully clean; keep runtime files outside the repository"
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
  free_kb="$(df -Pk "${RUNTIME_ROOT}" | awk 'NR==2 {print $4}')"
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
  mkdir -p "${RUNTIME_ROOT}"
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

self_test_setup() {
  check_source
  command -v python3 >/dev/null || fail "python3 is unavailable"
  mkdir -p "${RUNTIME_ROOT}"
  python3 - <<'PY'
import sys
assert sys.version_info >= (3, 10), sys.version
PY
  python3 -m venv --system-site-packages "${SELFTEST_VENV_DIR}"
  # shellcheck disable=SC1091
  source "${SELFTEST_VENV_DIR}/bin/activate"
  if ! python - <<'PY'
import numpy
import yaml
PY
  then
    python -m pip install --upgrade "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1"
    python -m pip install "numpy==1.26.4" "PyYAML==6.0.2"
  fi
}

engineering_self_test() {
  self_test_setup
  local selftest_parent
  if [[ -n "${E8_COLDSTART_SELFTEST_OUTPUT_ROOT:-}" ]]; then
    selftest_parent="${E8_COLDSTART_SELFTEST_OUTPUT_ROOT}"
    [[ ! -e "${selftest_parent}" ]] || fail "self-test output already exists: ${selftest_parent}"
    mkdir -p "${selftest_parent}"
  else
    selftest_parent="$(mktemp -d "${TMPDIR:-/tmp}/e8-coldstart-selftest.XXXXXX")"
  fi
  local selftest_guard_root="${selftest_parent}/guard"
  local selftest_output_root="${selftest_guard_root}/workload"
  local selftest_artifact="${selftest_parent}/guarded_engineering_self_test.zip"
  python "${ROOT_DIR}/scripts/run_experiment_guard_hardened.py" \
    --experiment-id "${EXPERIMENT_ID}" \
    --repo-root "${ROOT_DIR}" \
    --output-root "${selftest_guard_root}" \
    --artifact-output "${selftest_artifact}" \
    --run-class pilot \
    --expected-commit "${EXPECTED_COMMIT}" \
    --large-file-persistence persistent_local \
    --required-output workload/ENGINEERING_SELF_TEST_REPORT.json \
    --required-output workload/RUN_COMPLETE.json \
    --required-output workload/terminal_audit.json \
    --required-output workload/run_manifest.json \
    --required-output workload/scheduler/dynamic_run.json \
    --required-output workload/aggregate/plot_curve_points.csv \
    --source-file scripts/run_e8_multitask_exp_coldstart.sh \
    --source-file scripts/bootstrap_e8_multitask_exp_coldstart.sh \
    --source-file src/drpo/e8_multitask_exp_tuning.py \
    --source-file configs/e8_multitask_exp_coldstart.yaml \
    --source-file docs/experiments/EXT-C-E8-MULTITASK-EXP-COLDSTART-01_RUNBOOK.md \
    --progress-glob 'workload/scheduler/queue_events.jsonl' \
    --progress-glob 'workload/logs/*.log' \
    -- \
    python -m drpo.e8_multitask_exp_tuning \
      --config "${CONFIG_PATH}" \
      --output-root "${selftest_output_root}" \
      engineering-self-test \
      --source-commit "${EXPECTED_COMMIT}"
  python "${ROOT_DIR}/scripts/verify_experiment_package_hardened.py" \
    --repo-root "${ROOT_DIR}" \
    "${selftest_artifact}"
  echo "ENGINEERING_SELF_TEST_ROOT=${selftest_output_root}"
  echo "ENGINEERING_SELF_TEST_GUARD_ARTIFACT=${selftest_artifact}"
}

require_registered_ready() {
  grep -Fq "${EXPERIMENT_ID}" "${ROOT_DIR}/docs/handoff.md" || \
    fail "${EXPERIMENT_ID} is absent from docs/handoff.md"
  grep -Fq "${EXPERIMENT_ID}" "${ROOT_DIR}/experiments/registry.yaml" || \
    fail "${EXPERIMENT_ID} is absent from experiments/registry.yaml"
  python3 - <<PY
from pathlib import Path
import re

experiment_id = "${EXPERIMENT_ID}"
registry_path = Path("${ROOT_DIR}/experiments/registry.yaml")
lines = registry_path.read_text(encoding="utf-8").splitlines()
starts = [index for index, line in enumerate(lines) if line == f"- id: {experiment_id}"]
assert len(starts) == 1, f"expected exactly one registry entry for {experiment_id}, found {len(starts)}"
start = starts[0]
end = next(
    (index for index in range(start + 1, len(lines)) if re.fullmatch(r"- id: .+", lines[index])),
    len(lines),
)
block = lines[start:end]
implementation = next(
    (line.split(":", 1)[1].strip() for line in block if line.startswith("  implementation_state:")),
    "",
)
assert implementation.startswith("implemented"), f"implementation_state is not implemented: {implementation!r}"
gate_index = next((index for index, line in enumerate(block) if line == "  execution_gate:"), None)
assert gate_index is not None, "execution_gate is absent"
gate_state = next(
    (
        line.split(":", 1)[1].strip()
        for line in block[gate_index + 1 :]
        if line.startswith("    state:")
    ),
    "",
)
assert gate_state == "ready", f"{experiment_id} execution_gate is not ready: {gate_state!r}"
print({"experiment_id": experiment_id, "execution_gate": "ready"})
PY
  local runspec_matches=()
  mapfile -t runspec_matches < <(
    grep -l -F "experiment_id: ${EXPERIMENT_ID}" "${ROOT_DIR}"/runspecs/ready/*.yaml || true
  )
  [[ "${#runspec_matches[@]}" -eq 1 ]] || \
    fail "expected exactly one READY RunSpec for ${EXPERIMENT_ID}; found ${#runspec_matches[@]}"
  grep -Eq "^repo_commit:[[:space:]]*${EXPECTED_COMMIT}[[:space:]]*$" "${runspec_matches[0]}" || \
    fail "READY RunSpec does not bind reviewed commit ${EXPECTED_COMMIT}: ${runspec_matches[0]}"
  grep -Fq "run_e8_multitask_exp_coldstart.sh full" "${runspec_matches[0]}" || \
    fail "READY RunSpec does not call the reviewed full entrypoint: ${runspec_matches[0]}"
}

validate_registered_channel() {
  python "${ROOT_DIR}/scripts/validate_formal_execution_channel.py" --repo-root "${ROOT_DIR}"
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

run_queue_with_retry() {
  local attempt=1
  local maximum_attempts="${E8_COLDSTART_QUEUE_ATTEMPTS:-3}"
  [[ "${maximum_attempts}" =~ ^[1-9][0-9]*$ ]] || fail "queue attempts must be a positive integer"
  while ! run_queue; do
    if [[ "${attempt}" -ge "${maximum_attempts}" ]]; then
      fail "dynamic queue still failed after ${attempt} guarded attempts"
    fi
    attempt=$((attempt + 1))
    echo "Retrying only incomplete/unscheduled cells inside the same guard: attempt ${attempt}"
  done
}

finish() {
  check_source
  activate_runtime
  run_module aggregate
  run_module audit
  run_module finalize
  python - <<PY
import json
from pathlib import Path
root = Path("${OUTPUT_ROOT}")
audit = json.loads((root / "terminal_audit.json").read_text())
assert audit["all_training_and_evaluation_complete"], audit
plot = root / "aggregate" / "plot_curve_points.csv"
print("RAW_RESULTS_ROOT=" + str(root.resolve()))
print("PLOT_CSV=" + str(plot.resolve()))
PY
}

guarded_full() {
  require_registered_ready
  setup
  validate_registered_channel
  [[ "${OUTPUT_ROOT}" == "${GUARD_ROOT}/workload" ]] || \
    fail "formal output root must be the guard workload child: ${GUARD_ROOT}/workload"
  [[ ! -e "${GUARD_ROOT}" ]] || fail "formal guard root must be new: ${GUARD_ROOT}"
  [[ ! -e "${GUARD_ARTIFACT}" ]] || fail "guard artifact already exists: ${GUARD_ARTIFACT}"
  mkdir -p "$(dirname "${GUARD_ARTIFACT}")"
  python "${ROOT_DIR}/scripts/run_experiment_guard_hardened.py" \
    --experiment-id "${EXPERIMENT_ID}" \
    --repo-root "${ROOT_DIR}" \
    --output-root "${GUARD_ROOT}" \
    --artifact-output "${GUARD_ARTIFACT}" \
    --run-class formal \
    --expected-commit "${EXPECTED_COMMIT}" \
    --require-origin-main-match \
    --large-file-persistence persistent_local \
    --required-output workload/RUN_COMPLETE.json \
    --required-output workload/terminal_audit.json \
    --required-output workload/run_manifest.json \
    --required-output workload/scientific_run_manifest.json \
    --required-output workload/scheduler/dynamic_run.json \
    --required-output workload/aggregate/plot_curve_points.csv \
    --source-file scripts/run_e8_multitask_exp_coldstart.sh \
    --source-file scripts/bootstrap_e8_multitask_exp_coldstart.sh \
    --source-file src/drpo/e8_multitask_exp_tuning.py \
    --source-file configs/e8_multitask_exp_coldstart.yaml \
    --source-file requirements/e8_multitask_exp_coldstart.txt \
    --source-file src/drpo/countdown_qwen_arena_onefile.py \
    --source-file src/drpo/countdown_e8_base_rl_replay.py \
    --source-file src/drpo/countdown_e8_oracle_offline_v2_taper_sweep.py \
    --source-file src/drpo/countdown_e8_oracle_offline_v2_taper_runtime.py \
    --source-file docs/handoff.md \
    --source-file experiments/registry.yaml \
    --progress-glob 'workload/scheduler/queue_events.jsonl' \
    --progress-glob 'workload/logs/*.log' \
    -- \
    bash "${ROOT_DIR}/scripts/run_e8_multitask_exp_coldstart.sh" guarded-full-internal
  python "${ROOT_DIR}/scripts/verify_experiment_package_hardened.py" \
    --repo-root "${ROOT_DIR}" \
    "${GUARD_ARTIFACT}"
  echo "RAW_COMPLETE_RESULTS_ZIP=${GUARD_ARTIFACT}"
  sha256sum "${GUARD_ARTIFACT}"
  echo "PLOT_CSV=${OUTPUT_ROOT}/aggregate/plot_curve_points.csv"
  sha256sum "${OUTPUT_ROOT}/aggregate/plot_curve_points.csv"
}

case "${MODE}" in
  self-test) engineering_self_test ;;
  setup) setup ;;
  prepare) prepare ;;
  plan) check_source; activate_runtime; run_module plan ;;
  calibrate) calibrate ;;
  liveness) liveness ;;
  run|resume) run_queue ;;
  finish) finish ;;
  guarded-full-internal)
    prepare
    calibrate
    liveness
    run_queue_with_retry
    finish
    ;;
  full) guarded_full ;;
  *) fail "usage: $0 {self-test|setup|prepare|plan|calibrate|liveness|run|resume|finish|full}" ;;
esac
