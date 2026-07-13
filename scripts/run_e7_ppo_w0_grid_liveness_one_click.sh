#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CONTRACT="${E7_CANONICAL_CONTRACT:-/root/d4rl2/configs/e7_canonical_contract_9task.json}"
RUN_SPEC="${E7_CANONICAL_RUN_SPEC:-/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json}"
GRID="${E7_PPO_W0_GRID:-configs/e7_ppo_w0_exp_grid_pilot_v1.json}"
WORK_DIR="${E7_PPO_W0_LIVENESS_WORK_DIR:-outputs/e7/ppo_w0_exp_grid_liveness_001}"
PROBE_STEPS="${E7_PPO_W0_LIVENESS_PROBE_STEPS:-500}"
PROBE_SECONDS="${E7_PPO_W0_LIVENESS_PROBE_SECONDS:-120}"
MAX_WORKERS="${E7_PPO_W0_LIVENESS_MAX_WORKERS:-2}"

for required in "${CONTRACT}" "${RUN_SPEC}" "${GRID}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required file: ${required}" >&2
    exit 2
  fi
done

python scripts/run_e7_ppo_w0_grid_pilot_auto.py plan \
  --repo-root "${REPO_ROOT}" \
  --contract "${CONTRACT}" \
  --run-spec "${RUN_SPEC}" \
  --grid "${GRID}" \
  --work-dir "${WORK_DIR}" \
  --max-workers "${MAX_WORKERS}" \
  --probe-steps "${PROBE_STEPS}" \
  --probe-seconds "${PROBE_SECONDS}"
