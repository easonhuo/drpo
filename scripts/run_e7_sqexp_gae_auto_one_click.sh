#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CONTRACT="${E7_CANONICAL_CONTRACT:-/root/d4rl2/configs/e7_canonical_contract_9task.json}"
RUN_SPEC="${E7_CANONICAL_RUN_SPEC:-/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json}"
GRID="${E7_SQEXP_GAE_GRID:-configs/e7_sqexp_gae_v1.json}"
WORK_DIR="${E7_SQEXP_GAE_WORK_DIR:-outputs/e7/sqexp_gae_001}"
MAX_WORKERS="${E7_SQEXP_GAE_MAX_WORKERS:-40}"

for required in "${CONTRACT}" "${RUN_SPEC}" "${GRID}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required file: ${required}" >&2
    exit 2
  fi
done

python -m drpo.e7_sqexp_gae prepare \
  --contract "${CONTRACT}" \
  --run-spec "${RUN_SPEC}" \
  --grid "${GRID}" \
  --work-dir "${WORK_DIR}" \
  --resume

COMMON_ARGS=(
  --contract "${CONTRACT}"
  --run-spec "${RUN_SPEC}"
  --grid "${GRID}"
  --work-dir "${WORK_DIR}"
  --max-workers "${MAX_WORKERS}"
)
python -m drpo.e7_sqexp_gae plan "${COMMON_ARGS[@]}"
python -m drpo.e7_sqexp_gae run "${COMMON_ARGS[@]}" --resume
