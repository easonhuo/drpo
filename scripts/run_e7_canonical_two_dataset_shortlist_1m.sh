#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_e7_canonical_two_dataset_shortlist_1m.sh \
    --data-dir /ABS/D4RL_HDF5_DIR \
    --work-dir /ABS/OUTPUT_DIR

Alternative explicit dataset inputs:
  --hopper-medium-replay-hdf5 /ABS/hopper-medium-replay-v2.hdf5
  --hopper-medium-replay-sha256 SHA256
  --hopper-medium-expert-hdf5 /ABS/hopper-medium-expert-v2.hdf5
  --hopper-medium-expert-sha256 SHA256

Optional:
  --resume

Environment:
  E7_MAX_WORKERS  Parallel branch workers. Default: 40.

All scientific variables are frozen by the registered protocol. Unknown flags,
including --seeds, --steps, --alpha, --tau, --temp, --lr, --batch, --grid,
--canonical-root, and --target-class, are rejected before Python is invoked.
The full pilot must run from a clean commit equal to authoritative origin/main.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

MAX_WORKERS="${E7_MAX_WORKERS:-40}"
if ! [[ "$MAX_WORKERS" =~ ^[0-9]+$ ]] || (( MAX_WORKERS < 2 )); then
  echo "E7_MAX_WORKERS must be an integer >= 2; got: $MAX_WORKERS" >&2
  exit 2
fi

FORWARD_ARGS=()
WORK_DIR=""
RESUME=0
declare -A SEEN=()

need_value() {
  local flag="$1"
  local value="${2:-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    echo "$flag requires a value" >&2
    exit 2
  fi
}

record_once() {
  local flag="$1"
  if [[ -n "${SEEN[$flag]:-}" ]]; then
    echo "duplicate argument is not allowed: $flag" >&2
    exit 2
  fi
  SEEN[$flag]=1
}

while (( $# > 0 )); do
  case "$1" in
    --data-dir|--hopper-medium-replay-hdf5|--hopper-medium-replay-sha256|--hopper-medium-expert-hdf5|--hopper-medium-expert-sha256|--work-dir)
      flag="$1"
      need_value "$flag" "${2:-}"
      record_once "$flag"
      value="$2"
      FORWARD_ARGS+=("$flag" "$value")
      if [[ "$flag" == "--work-dir" ]]; then
        WORK_DIR="$value"
      fi
      shift 2
      ;;
    --data-dir=*|--hopper-medium-replay-hdf5=*|--hopper-medium-replay-sha256=*|--hopper-medium-expert-hdf5=*|--hopper-medium-expert-sha256=*|--work-dir=*)
      flag="${1%%=*}"
      value="${1#*=}"
      need_value "$flag" "$value"
      record_once "$flag"
      FORWARD_ARGS+=("$flag" "$value")
      if [[ "$flag" == "--work-dir" ]]; then
        WORK_DIR="$value"
      fi
      shift
      ;;
    --resume)
      record_once "--resume"
      RESUME=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "argument is not allowed by the frozen shortlist protocol: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$WORK_DIR" ]]; then
  echo "--work-dir is required" >&2
  exit 2
fi
if [[ "$WORK_DIR" != /* ]]; then
  echo "--work-dir must be an absolute path: $WORK_DIR" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python scripts/run_e7_canonical_two_dataset.py prepare \
  "${FORWARD_ARGS[@]}" \
  --profile taper-pilot \
  --grid configs/e7_canonical_two_dataset_shortlist_1m_v1.json \
  --steps 1000000 \
  --eval-interval 50000 \
  --eval-episodes 10 \
  --ckpt-interval 50000 \
  --last-pct 0.10

RUN_ARGS=(
  run
  --contract "$WORK_DIR/canonical_contract.json"
  --run-spec "$WORK_DIR/run_spec.json"
  --grid configs/e7_canonical_two_dataset_shortlist_1m_v1.json
  --work-dir "$WORK_DIR"
  --max-workers "$MAX_WORKERS"
  --require-clean-main
)
if (( RESUME == 1 )); then
  RUN_ARGS+=(--resume)
fi
python scripts/run_e7_canonical_shortlist_1m.py "${RUN_ARGS[@]}"
