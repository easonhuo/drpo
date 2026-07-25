#!/usr/bin/env bash
set -euo pipefail

COMMAND="${1:-run}"
case "${COMMAND}" in
  validate|plan|run) ;;
  *)
    echo "usage: $0 [validate|plan|run]" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

GRID="${E7_HR_POSDRPO_GRID:-configs/e7_hopper_replay_posbase_drpo_cgrid_500k.json}"
CONTRACT="${E7_CANONICAL_CONTRACT:-/root/d4rl2/configs/e7_canonical_contract_9task.json}"
RUN_SPEC="${E7_CANONICAL_RUN_SPEC:-/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json}"
WORK_DIR="${E7_HR_POSDRPO_WORK_DIR:-outputs/e7/hopper_replay_posbase_drpo_cgrid_500k_001}"
MAX_WORKERS="${E7_HR_POSDRPO_MAX_WORKERS:-180}"
RUNTIME_SOURCE_DIR="${REPO_ROOT}/scripts/e7_hopper_replay_posbase_drpo_cgrid_500k_runtime"
RUNTIME_DIR="${WORK_DIR}/.posbase_drpo_cgrid_runtime"
BOOTSTRAP_WRAPPER="${RUNTIME_DIR}/bootstrap_wrapper.py"
DRIVER="${RUNTIME_DIR}/driver.py"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to run from a dirty checkout" >&2
  exit 2
fi
if [[ ! -f "${GRID}" ]]; then
  echo "missing Hopper replay Positive-only/DRPO grid: ${GRID}" >&2
  exit 2
fi
if [[ "${COMMAND}" != "validate" ]]; then
  for required in "${CONTRACT}" "${RUN_SPEC}"; do
    if [[ ! -f "${required}" ]]; then
      echo "missing required file: ${required}" >&2
      exit 2
    fi
  done
fi
if ! [[ "${MAX_WORKERS}" =~ ^[1-9][0-9]*$ ]] || (( MAX_WORKERS > 180 )); then
  echo "E7_HR_POSDRPO_MAX_WORKERS must be an integer in [1,180]" >&2
  exit 2
fi

shopt -s nullglob
bootstrap_parts=("${RUNTIME_SOURCE_DIR}"/bootstrap_*.inc)
driver_parts=("${RUNTIME_SOURCE_DIR}"/driver_*.inc)
if (( ${#bootstrap_parts[@]} == 0 || ${#driver_parts[@]} == 0 )); then
  echo "missing governed runtime source fragments below ${RUNTIME_SOURCE_DIR}" >&2
  exit 2
fi
mkdir -p "${RUNTIME_DIR}"
cat "${bootstrap_parts[@]}" > "${BOOTSTRAP_WRAPPER}"
cat "${driver_parts[@]}" > "${DRIVER}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
python -m py_compile "${BOOTSTRAP_WRAPPER}" "${DRIVER}"
python "${BOOTSTRAP_WRAPPER}" --self-test >/dev/null
python "${DRIVER}" \
  "${COMMAND}" \
  "${CONTRACT}" \
  "${RUN_SPEC}" \
  "${GRID}" \
  "${WORK_DIR}" \
  "${MAX_WORKERS}" \
  "${BOOTSTRAP_WRAPPER}"
