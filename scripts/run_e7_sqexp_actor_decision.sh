#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
if [[ "${MODE}" != "liveness" && "${MODE}" != "run" && "${MODE}" != "resume" ]]; then
  echo "usage: $0 {liveness|run|resume}" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CONTRACT="${E7_CANONICAL_CONTRACT:-/root/d4rl2/configs/e7_canonical_contract_9task.json}"
RUN_SPEC="${E7_CANONICAL_RUN_SPEC:-/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json}"
GRID="${E7_ACTOR_DECISION_GRID:-configs/e7_sqexp_actor_decision_v1.json}"
WORK_DIR="${E7_ACTOR_DECISION_WORK_DIR:-outputs/e7/sqexp_actor_decision_001}"
FALLBACK_WORKERS="${E7_ACTOR_DECISION_FALLBACK_WORKERS:-96}"
PROBE_STEPS="${E7_ACTOR_DECISION_PROBE_STEPS:-5000}"
PROBE_SECONDS="${E7_ACTOR_DECISION_PROBE_SECONDS:-120}"
THROUGHPUT_RETENTION="${E7_ACTOR_DECISION_THROUGHPUT_RETENTION:-0.97}"
MAX_WORKERS="${E7_ACTOR_DECISION_MAX_WORKERS:-}"

if [[ "${MODE}" == "liveness" ]]; then
  WORK_DIR="${E7_ACTOR_DECISION_LIVENESS_WORK_DIR:-outputs/e7/sqexp_actor_decision_liveness_001}"
  PROBE_STEPS="${E7_ACTOR_DECISION_LIVENESS_PROBE_STEPS:-500}"
  MAX_WORKERS="${E7_ACTOR_DECISION_LIVENESS_MAX_WORKERS:-2}"
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to run actor-decision pilot from a dirty checkout" >&2
  exit 2
fi
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

if [[ "${MODE}" == "liveness" ]]; then
  python scripts/run_e7_sqexp_actor_decision_auto.py plan "${COMMON_ARGS[@]}"
  exit 0
fi

if [[ "${MODE}" == "run" ]]; then
  python scripts/run_e7_sqexp_actor_decision_auto.py plan "${COMMON_ARGS[@]}"
fi

python scripts/run_e7_sqexp_actor_decision_auto.py run "${COMMON_ARGS[@]}" --resume
