#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PINNED_IMPLEMENTATION_COMMIT="163a83ca92800fd011c652f3d8ba268df1846f89"
ACTUAL_COMMIT="$(git rev-parse HEAD)"

if [[ -n "${DRPO_EXPECTED_COMMIT:-}" ]]; then
  if [[ "$ACTUAL_COMMIT" != "$DRPO_EXPECTED_COMMIT" ]]; then
    echo "Expected HEAD $DRPO_EXPECTED_COMMIT but found $ACTUAL_COMMIT" >&2
    exit 2
  fi
else
  if ! git merge-base --is-ancestor "$PINNED_IMPLEMENTATION_COMMIT" "$ACTUAL_COMMIT"; then
    echo "Pinned implementation $PINNED_IMPLEMENTATION_COMMIT is not an ancestor of HEAD $ACTUAL_COMMIT" >&2
    exit 2
  fi
  PROTECTED_PATHS=(
    configs/countdown_e8_base_rl_replay_0p5b.yaml
    configs/countdown_e8_oracle_offline_v2_repro_rng_audit_0p5b.yaml
    scripts/run_countdown_e8_oracle_offline_v2_repro_rng_audit.py
    src/drpo/countdown_qwen_arena_onefile.py
    src/drpo/countdown_e8_alpha1_c_scan_common.py
    src/drpo/countdown_e8_alpha1_c_scan_runtime.py
    src/drpo/countdown_e8_alpha1_c_scan_trainer.py
    src/drpo/countdown_e8_repro_contract.py
    src/drpo/countdown_e8_repro_legacy_runtime.py
    src/drpo/countdown_e8_repro_rng_audit_common.py
    src/drpo/countdown_e8_repro_rng_isolated_runtime.py
    src/drpo/countdown_e8_rng_isolation.py
  )
  if ! git diff --quiet "$PINNED_IMPLEMENTATION_COMMIT" "$ACTUAL_COMMIT" -- "${PROTECTED_PATHS[@]}"; then
    echo "Protected E8 RNG-audit implementation files changed after $PINNED_IMPLEMENTATION_COMMIT" >&2
    git diff --name-only "$PINNED_IMPLEMENTATION_COMMIT" "$ACTUAL_COMMIT" -- "${PROTECTED_PATHS[@]}" >&2
    exit 2
  fi
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Repro RNG audit requires a clean checkout" >&2
  exit 2
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 scripts/run_countdown_e8_oracle_offline_v2_repro_rng_audit.py "$@"
