#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

python -m compileall -q \
  src/drpo/e7_squared_exp_kernel.py \
  src/drpo/e7_squared_exp_night.py \
  src/drpo/e7_squared_exp_night_bootstrap.py \
  src/drpo/e7_squared_exp_night_aggregate.py
bash -n scripts/run_e7_hopper_replay_advshift_200k.sh
python -m json.tool configs/e7_hopper_replay_advshift_200k.json >/dev/null

VALIDATION_JSON="$(
  E7_HR_ADVSHIFT_WORK_DIR=outputs/e7/hopper_replay_advshift_200k_validation \
    bash scripts/run_e7_hopper_replay_advshift_200k.sh validate
)"
python - "${VALIDATION_JSON}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
assert payload["status"] == "PASS"
assert payload["experiment_id"] == "EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-ADVSHIFT-200K-01"
assert payload["expected_branches"] == 20
assert payload["steps"] == 200000
assert payload["evaluation_interval"] == 20000
assert payload["all_samples_retained"] is True
assert payload["held_out_seeds_touched"] is False
assert payload["controls"] == [
    "positive_only",
    "drpo_b0",
    "drpo_b0p25",
    "drpo_b0p5",
    "drpo_b1",
]
assert payload["seeds"] == [200, 202, 203, 208]
PY

# This path is training-free. It validates identity, matrix, shift semantics,
# embedded-Python syntax, and the no-sample-deletion self-test.
