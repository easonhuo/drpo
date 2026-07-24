#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
python -m compileall -q \
  src/drpo/e7_squared_exp_kernel.py \
  src/drpo/e7_squared_exp_night.py \
  src/drpo/e7_squared_exp_night_bootstrap.py \
  src/drpo/e7_squared_exp_night_aggregate.py
bash -n scripts/run_e7_taskc_top4_multiseed.sh
bash -n scripts/run_e7_taskc_top4_multiseed_entrypoint.sh
bash scripts/run_e7_taskc_top4_multiseed_entrypoint.sh validate
