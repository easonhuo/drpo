#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CONTRACT="/root/d4rl2/configs/e7_canonical_contract_9task.json"
RUN_SPEC="/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json"
GRID="configs/e7_canonical_d4rl9_glq_taskwise_refinement_v1.json"
PARENT_AUDIT="outputs/e7/d4rl9_glq_taskwise_tuning_run_001/TERMINAL_AUDIT.json"
WORK_DIR="outputs/e7/d4rl9_glq_taskwise_refinement_run_001"
MAX_WORKERS=60

for required in "${CONTRACT}" "${RUN_SPEC}" "${GRID}" "${PARENT_AUDIT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required file: ${required}" >&2
    exit 2
  fi
done

EXPECTED_PARENT_SHA="8775edcb436ba759a52eb6b2ae9cdb2cbce966852fd5e8e3739798134523234b"
ACTUAL_PARENT_SHA="$(sha256sum "${PARENT_AUDIT}" | awk '{print $1}')"
if [[ "${ACTUAL_PARENT_SHA}" != "${EXPECTED_PARENT_SHA}" ]]; then
  echo "parent terminal audit SHA-256 mismatch" >&2
  echo "expected: ${EXPECTED_PARENT_SHA}" >&2
  echo "actual:   ${ACTUAL_PARENT_SHA}" >&2
  exit 3
fi

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

# The run command audits automatically. Re-running audit is identity-preserving
# and is useful after restoring a completed output directory.
python scripts/run_e7_canonical_scale1_grid.py audit \
  --contract "${CONTRACT}" \
  --run-spec "${RUN_SPEC}" \
  --grid "${GRID}" \
  --work-dir "${WORK_DIR}" \
  --max-workers "${MAX_WORKERS}"
