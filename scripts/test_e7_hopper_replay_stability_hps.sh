#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

python -m compileall -q \
  src/drpo/e7_squared_exp_kernel.py \
  src/drpo/e7_squared_exp_night.py \
  src/drpo/e7_squared_exp_night_bootstrap.py \
  src/drpo/e7_squared_exp_night_aggregate.py
bash -n scripts/run_e7_hopper_replay_stability_hps.sh
E7_HR_STAB_WORK_DIR=outputs/e7/hopper_replay_stability_hps_validation \
  bash scripts/run_e7_hopper_replay_stability_hps.sh validate

# The validation path is intentionally training-free.
