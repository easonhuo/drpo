#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CONTRACT="${E7_CANONICAL_CONTRACT:-/root/d4rl2/configs/e7_canonical_contract_9task.json}"
RUN_SPEC="${E7_CANONICAL_RUN_SPEC:-/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json}"
GRID="${E7_PPO_GRID:-configs/e7_canonical_ppo_stability_v1.json}"
SMOKE_DIR="${E7_PPO_SMOKE_WORK_DIR:-outputs/e7/ppo_stability_smoke_001}"
WORK_DIR="${E7_PPO_WORK_DIR:-outputs/e7/ppo_stability_run_001}"
FALLBACK_WORKERS="${E7_PPO_FALLBACK_WORKERS:-60}"
PROBE_STEPS="${E7_PPO_PROBE_STEPS:-20000}"
PROBE_SECONDS="${E7_PPO_PROBE_SECONDS:-120}"
MAX_WORKERS="${E7_PPO_MAX_WORKERS:-}"

for required in "${CONTRACT}" "${RUN_SPEC}" "${GRID}" "${SMOKE_DIR}/SMOKE_GATE.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required file: ${required}" >&2
    exit 2
  fi
done

COMMON_ARGS=(
  --repo-root "${REPO_ROOT}"
  --contract "${CONTRACT}"
  --run-spec "${RUN_SPEC}"
  --grid "${GRID}"
  --smoke-dir "${SMOKE_DIR}"
  --work-dir "${WORK_DIR}"
  --fallback-workers "${FALLBACK_WORKERS}"
  --probe-steps "${PROBE_STEPS}"
  --probe-seconds "${PROBE_SECONDS}"
)
if [[ -n "${MAX_WORKERS}" ]]; then
  COMMON_ARGS+=(--max-workers "${MAX_WORKERS}")
fi

python scripts/run_e7_ppo_stability_pilot_auto.py plan "${COMMON_ARGS[@]}"
python scripts/run_e7_ppo_stability_pilot_auto.py run "${COMMON_ARGS[@]}" --resume
