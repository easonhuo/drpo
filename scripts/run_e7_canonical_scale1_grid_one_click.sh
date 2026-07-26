#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CONTRACT="/root/d4rl2/configs/e7_canonical_contract_9task.json"
RUN_SPEC="/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json"
GRID="configs/e7_canonical_d4rl9_glq_taskwise_tuning_v1.json"
WORK_DIR="outputs/e7/d4rl9_glq_taskwise_tuning_run_001"
MAX_WORKERS=60

for required in "${CONTRACT}" "${RUN_SPEC}" "${GRID}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required file: ${required}" >&2
    exit 2
  fi
done

python scripts/run_e7_canonical_scale1_grid.py plan \
  --contract "${CONTRACT}" \
  --run-spec "${RUN_SPEC}" \
  --grid "${GRID}" \
  --work-dir "${WORK_DIR}" \
  --max-workers "${MAX_WORKERS}"

python scripts/run_e7_canonical_scale1_grid.py run \
  --contract "${CONTRACT}" \
  --run-spec "${RUN_SPEC}" \
  --grid "${GRID}" \
  --work-dir "${WORK_DIR}" \
  --max-workers "${MAX_WORKERS}" \
  --resume

# The run command performs the same audit automatically. Re-running audit is
# identity-preserving and provides a convenient recovery path after file copies.
python scripts/run_e7_canonical_scale1_grid.py audit \
  --contract "${CONTRACT}" \
  --run-spec "${RUN_SPEC}" \
  --grid "${GRID}" \
  --work-dir "${WORK_DIR}" \
  --max-workers "${MAX_WORKERS}"
