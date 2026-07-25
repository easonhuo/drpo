#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

python -m compileall -q \
  src/drpo/e7_squared_exp_kernel.py \
  src/drpo/e7_squared_exp_night.py \
  src/drpo/e7_squared_exp_night_bootstrap.py \
  src/drpo/e7_squared_exp_night_aggregate.py
bash -n scripts/run_e7_hopper_replay_positive_only_baseline_500k.sh
python -m json.tool configs/e7_hopper_replay_positive_only_baseline_500k.json >/dev/null

VALIDATION_JSON="$(
  E7_HR_POSBASE_WORK_DIR=outputs/e7/hopper_replay_positive_only_baseline_500k_validation \
    bash scripts/run_e7_hopper_replay_positive_only_baseline_500k.sh validate
)"
python - "${VALIDATION_JSON}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
assert payload["status"] == "PASS"
assert payload["experiment_id"] == "EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-POSBASE-500K-01"
assert payload["expected_branches"] == 66
assert payload["steps"] == 500000
assert payload["evaluation_interval"] == 20000
assert payload["baselines"] == [float(value) for value in range(11)]
assert payload["controls"] == [f"pos_b{value}" for value in range(11)]
assert payload["seeds"] == [200, 201, 202, 203, 208, 209]
assert payload["all_replay_samples_retained"] is True
assert payload["negative_shifted_advantage_actor_contribution"] == 0.0
assert payload["held_out_seeds_touched"] is False
PY
