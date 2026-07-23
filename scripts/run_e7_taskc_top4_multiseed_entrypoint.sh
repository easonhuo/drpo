#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

WORK_DIR="${E7_TASKC_WORK_DIR:-outputs/e7/taskc_top4_multiseed_001}"
PROFILE_SHIM_DIR="${WORK_DIR}/.taskc_python_profile"
mkdir -p "${PROFILE_SHIM_DIR}"
cat >"${PROFILE_SHIM_DIR}/sitecustomize.py" <<'PY'
from drpo import e7_squared_exp_night as _suite

_suite.TUNING_PROFILE_ID = "d4rl9_task_specific_c_top4_multiseed"
PY

export PYTHONPATH="${PROFILE_SHIM_DIR}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'PY'
from drpo import e7_squared_exp_night as suite

expected = "d4rl9_task_specific_c_top4_multiseed"
if suite.TUNING_PROFILE_ID != expected:
    raise SystemExit(
        f"task-specific bootstrap profile shim failed: {suite.TUNING_PROFILE_ID!r}"
    )
PY

exec bash scripts/run_e7_taskc_top4_multiseed.sh "${1:-run}"
