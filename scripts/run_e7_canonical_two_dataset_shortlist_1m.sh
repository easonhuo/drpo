#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_e7_canonical_two_dataset_shortlist_1m.sh \
    --data-dir /ABS/D4RL_HDF5_DIR \
    --work-dir /ABS/OUTPUT_DIR \
    [extra dataset/source args]

Environment:
  E7_MAX_WORKERS  Parallel branch workers. Default: 40.

Frozen protocol:
  - experiment: EXT-H-E7-BENCH-01
  - datasets: hopper-medium-replay-v2, hopper-medium-expert-v2
  - seeds: 200, 201, 202, 203
  - methods: 7 fixed shortlist methods
  - updates per branch: 1,000,000
  - evaluation: every 50,000 updates, 10 episodes
  - checkpoint cadence: every 50,000 updates in the final 10% window

This remains a two-dataset pilot. It is not the formal D4RL-9 benchmark and
must not be used for task-specific retuning.
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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python scripts/run_e7_canonical_two_dataset.py run \
  "$@" \
  --profile taper-pilot \
  --grid configs/e7_canonical_two_dataset_shortlist_1m_v1.json \
  --steps 1000000 \
  --eval-interval 50000 \
  --eval-episodes 10 \
  --ckpt-interval 50000 \
  --last-pct 0.10 \
  --max-workers "$MAX_WORKERS"
