#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CONTRACT="${E7_CANONICAL_CONTRACT:-/root/d4rl2/configs/e7_canonical_contract_9task.json}"
RUN_SPEC="${E7_CANONICAL_RUN_SPEC:-/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json}"
GRID="${E7_W0_HIGHC_GRID:-configs/e7_w0_highc_actor_ablation_v1.json}"
WORK_DIR="${E7_W0_HIGHC_WORK_DIR:-outputs/e7/w0_highc_actor_ablation_001}"
FALLBACK_WORKERS="${E7_W0_HIGHC_FALLBACK_WORKERS:-60}"
PROBE_STEPS="${E7_W0_HIGHC_PROBE_STEPS:-5000}"
PROBE_SECONDS="${E7_W0_HIGHC_PROBE_SECONDS:-120}"
THROUGHPUT_RETENTION="${E7_W0_HIGHC_THROUGHPUT_RETENTION:-0.97}"
MAX_WORKERS="${E7_W0_HIGHC_MAX_WORKERS:-}"

for required in "${CONTRACT}" "${RUN_SPEC}" "${GRID}"; do
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
  --work-dir "${WORK_DIR}"
  --fallback-workers "${FALLBACK_WORKERS}"
  --probe-steps "${PROBE_STEPS}"
  --probe-seconds "${PROBE_SECONDS}"
  --throughput-retention-fraction "${THROUGHPUT_RETENTION}"
)
if [[ -n "${MAX_WORKERS}" ]]; then
  COMMON_ARGS+=(--max-workers "${MAX_WORKERS}")
fi

python scripts/run_e7_w0_highc_actor_auto.py plan "${COMMON_ARGS[@]}"
python scripts/run_e7_w0_highc_actor_auto.py run "${COMMON_ARGS[@]}" --resume
