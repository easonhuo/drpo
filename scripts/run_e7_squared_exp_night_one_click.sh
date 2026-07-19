#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CONTRACT="${E7_CANONICAL_CONTRACT:-/root/d4rl2/configs/e7_canonical_contract_9task.json}"
RUN_SPEC="${E7_CANONICAL_RUN_SPEC:-/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json}"
GRID="${E7_SQUARED_EXP_GRID:-configs/e7_squared_exp_night_v1.json}"
WORK_DIR="${E7_SQUARED_EXP_WORK_DIR:-outputs/e7/squared_exp_night_1m_001}"
FALLBACK_WORKERS="${E7_SQUARED_EXP_FALLBACK_WORKERS:-60}"
PROBE_STEPS="${E7_SQUARED_EXP_PROBE_STEPS:-5000}"
PROBE_SECONDS="${E7_SQUARED_EXP_PROBE_SECONDS:-120}"
THROUGHPUT_RETENTION="${E7_SQUARED_EXP_THROUGHPUT_RETENTION:-0.97}"
MAX_WORKERS="${E7_SQUARED_EXP_MAX_WORKERS:-}"
MODE="${E7_SQUARED_EXP_MODE:-historical}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to run from a dirty checkout" >&2
  exit 2
fi
for required in "${CONTRACT}" "${RUN_SPEC}" "${GRID}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required file: ${required}" >&2
    exit 2
  fi
done

case "${MODE}" in
  historical)
    ;;
  p1)
    if [[ "${GRID}" != "configs/e7_bench_joint_gae_tuning_p1_c.json" ]]; then
      echo "P1 mode requires the registered P1 grid" >&2
      exit 2
    fi
    export DRPO_E7_P1_FULL_RUN=1
    ;;
  *)
    echo "unsupported E7_SQUARED_EXP_MODE=${MODE}" >&2
    exit 2
    ;;
esac

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

python scripts/run_e7_squared_exp_night_auto.py plan "${COMMON_ARGS[@]}"
python scripts/run_e7_squared_exp_night_auto.py run "${COMMON_ARGS[@]}" --resume
