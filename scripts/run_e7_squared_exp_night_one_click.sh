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
MAX_WORKERS="${E7_SQUARED_EXP_MAX_WORKERS:-${DRPO_RUNTIME_MAX_WORKERS:-}}"
MAX_WORKERS_APPROVAL_FILE="${E7_SQUARED_EXP_MAX_WORKERS_APPROVAL_FILE:-${DRPO_RUNTIME_MAX_WORKERS_APPROVAL_FILE:-}}"
MODE="${E7_SQUARED_EXP_MODE:-historical}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to run from a dirty checkout" >&2
  exit 2
fi
for required in \
  "${CONTRACT}" \
  "${RUN_SPEC}" \
  "${GRID}" \
  "scripts/validate_user_approved_worker_cap.sh"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required file: ${required}" >&2
    exit 2
  fi
done

case "${MODE}" in
  historical|gae)
    ;;
  p3_left_saturation)
    if [[ "${GRID}" != "configs/e7_bench_joint_gae_p3_left_saturation.json" ]]; then
      echo "P3-left-saturation mode requires the frozen P3 grid" >&2
      exit 2
    fi
    if [[ "${DRPO_E7_P3_LEFT_SATURATION_FULL_RUN:-0}" != "1" ]]; then
      echo "P3-left-saturation mode is authorized only by the standard RunSpec entrypoint" >&2
      exit 2
    fi
    ;;
  *)
    echo "unsupported E7_SQUARED_EXP_MODE=${MODE}" >&2
    exit 2
    ;;
esac

bash scripts/validate_user_approved_worker_cap.sh \
  "${REPO_ROOT}" \
  "${WORK_DIR}" \
  "${MAX_WORKERS}" \
  "${MAX_WORKERS_APPROVAL_FILE}" \
  "${CONTRACT}" \
  "${RUN_SPEC}" \
  "${GRID}" \
  >/dev/null

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

SELECTION_PATH="${WORK_DIR}/RUNTIME_SELECTION.json"
IDENTITY_PATH="${WORK_DIR}/RUN_IDENTITY.json"
if [[ -f "${SELECTION_PATH}" && -f "${IDENTITY_PATH}" ]]; then
  python scripts/run_e7_squared_exp_night_auto.py run "${COMMON_ARGS[@]}" --resume
elif [[ -e "${SELECTION_PATH}" || -e "${IDENTITY_PATH}" ]]; then
  echo "partial runtime identity: selection and run identity must either both exist or both be absent" >&2
  exit 2
else
  python scripts/run_e7_squared_exp_night_auto.py plan "${COMMON_ARGS[@]}"
  python scripts/run_e7_squared_exp_night_auto.py run "${COMMON_ARGS[@]}" --resume
fi
