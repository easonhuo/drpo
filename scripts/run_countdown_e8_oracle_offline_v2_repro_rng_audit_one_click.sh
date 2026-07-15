#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${DRPO_EXPECTED_COMMIT:?Set DRPO_EXPECTED_COMMIT to the reviewed frozen implementation SHA}"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$DRPO_EXPECTED_COMMIT" ]]; then
  echo "Expected HEAD $DRPO_EXPECTED_COMMIT but found $ACTUAL_COMMIT" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Repro RNG audit requires a clean checkout" >&2
  exit 2
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 scripts/run_countdown_e8_oracle_offline_v2_repro_rng_audit.py "$@"
