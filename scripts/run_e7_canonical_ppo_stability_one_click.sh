#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CONTRACT="${E7_CANONICAL_CONTRACT:-/root/d4rl2/configs/e7_canonical_contract_9task.json}"
RUN_SPEC="${E7_CANONICAL_RUN_SPEC:-/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json}"
GRID="${E7_PPO_GRID:-configs/e7_canonical_ppo_stability_v1.json}"
WORK_DIR="${E7_PPO_WORK_DIR:-outputs/e7/ppo_stability_run_001}"
MAX_WORKERS="${E7_PPO_MAX_WORKERS:-96}"

for required in "${CONTRACT}" "${RUN_SPEC}" "${GRID}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required file: ${required}" >&2
    exit 2
  fi
done

python scripts/run_e7_canonical_ppo_stability.py plan \
  --contract "${CONTRACT}" \
  --run-spec "${RUN_SPEC}" \
  --grid "${GRID}" \
  --work-dir "${WORK_DIR}" \
  --max-workers "${MAX_WORKERS}"

python scripts/run_e7_canonical_ppo_stability.py run \
  --contract "${CONTRACT}" \
  --run-spec "${RUN_SPEC}" \
  --grid "${GRID}" \
  --work-dir "${WORK_DIR}" \
  --max-workers "${MAX_WORKERS}" \
  --resume
