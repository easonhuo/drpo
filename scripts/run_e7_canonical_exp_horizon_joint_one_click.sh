#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CONTRACT="/root/d4rl2/configs/e7_canonical_contract_9task.json"
RUN_SPEC="/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json"
GRID="configs/e7_canonical_exp_horizon_joint_grid_v1.json"
WORK_DIR="outputs/e7/exp_horizon_joint_run_001"

for required in "${CONTRACT}" "${RUN_SPEC}" "${GRID}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required file: ${required}" >&2
    exit 2
  fi
done

python scripts/run_e7_canonical_exp_horizon_joint.py plan \
  --contract "${CONTRACT}" \
  --run-spec "${RUN_SPEC}" \
  --grid "${GRID}" \
  --work-dir "${WORK_DIR}" \
  --max-workers 60

python scripts/run_e7_canonical_exp_horizon_joint.py run \
  --contract "${CONTRACT}" \
  --run-spec "${RUN_SPEC}" \
  --grid "${GRID}" \
  --work-dir "${WORK_DIR}" \
  --max-workers 60
