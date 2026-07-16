#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CONTRACT="${E7_CANONICAL_CONTRACT:-/root/d4rl2/configs/e7_canonical_contract_9task.json}"
RUN_SPEC="${E7_CANONICAL_RUN_SPEC:-/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json}"
GRID="${E7_PPO_W0_GRID:-configs/e7_ppo_w0_exp_grid_pilot_v1.json}"
WORK_DIR="${E7_PPO_W0_WORK_DIR:-outputs/e7/ppo_w0_exp_grid_pilot_001}"
FALLBACK_WORKERS="${E7_PPO_W0_FALLBACK_WORKERS:-60}"
PROBE_STEPS="${E7_PPO_W0_PROBE_STEPS:-5000}"
PROBE_SECONDS="${E7_PPO_W0_PROBE_SECONDS:-120}"
THROUGHPUT_RETENTION="${E7_PPO_W0_THROUGHPUT_RETENTION:-0.97}"
MAX_WORKERS="${E7_PPO_W0_MAX_WORKERS:-}"
CAPACITY_WAIT_TIMEOUT_SECONDS="${E7_PPO_W0_CAPACITY_WAIT_TIMEOUT_SECONDS:--1}"
CAPACITY_POLL_SECONDS="${E7_PPO_W0_CAPACITY_POLL_SECONDS:-300}"
MINIMUM_ADMITTED_WORKERS="${E7_PPO_W0_MINIMUM_ADMITTED_WORKERS:-}"

for required in \
  "${CONTRACT}" \
  "${RUN_SPEC}" \
  "${GRID}" \
  "${WORK_DIR}/RUNTIME_SELECTION.json" \
  "${WORK_DIR}/RUN_IDENTITY.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required resume file: ${required}" >&2
    exit 2
  fi
done

COMMON_ARGS=(
  --repo-root "${REPO_ROOT}"
  --contract "${CONTRACT}"
  --run-spec "${RUN_SPEC}"
  --grid "${GRID}"
  --work-dir "${WORK_DIR}"
  --fallback-workers "${FALLBACK_WORKERS}"
  --probe-steps "${PROBE_STEPS}"
  --probe-seconds "${PROBE_SECONDS}"
  --throughput-retention-fraction "${THROUGHPUT_RETENTION}"
  --capacity-wait-timeout-seconds "${CAPACITY_WAIT_TIMEOUT_SECONDS}"
  --capacity-poll-seconds "${CAPACITY_POLL_SECONDS}"
)
if [[ -n "${MAX_WORKERS}" ]]; then
  COMMON_ARGS+=(--max-workers "${MAX_WORKERS}")
fi
if [[ -n "${MINIMUM_ADMITTED_WORKERS}" ]]; then
  COMMON_ARGS+=(--minimum-admitted-workers "${MINIMUM_ADMITTED_WORKERS}")
fi

python scripts/run_e7_ppo_w0_grid_pilot_auto.py run "${COMMON_ARGS[@]}" --resume
