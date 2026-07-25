#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

python -m compileall -q \
  src/drpo/e7_squared_exp_kernel.py \
  src/drpo/e7_squared_exp_night.py \
  src/drpo/e7_squared_exp_night_bootstrap.py \
  src/drpo/e7_squared_exp_night_aggregate.py \
  src/drpo/e7_w0_geometry_diagnostics.py
bash -n scripts/run_e7_hopper_replay_posbase_drpo_cgrid_500k.sh
python -m json.tool configs/e7_hopper_replay_posbase_drpo_cgrid_500k.json >/dev/null

RUNTIME_SOURCE_DIR="scripts/e7_hopper_replay_posbase_drpo_cgrid_500k_runtime"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
cat "${RUNTIME_SOURCE_DIR}"/bootstrap_*.inc > "${TMP_DIR}/bootstrap_wrapper.py"
cat "${RUNTIME_SOURCE_DIR}"/driver_*.inc > "${TMP_DIR}/driver.py"
python -m py_compile "${TMP_DIR}/bootstrap_wrapper.py" "${TMP_DIR}/driver.py"
python "${TMP_DIR}/bootstrap_wrapper.py" --self-test >/dev/null

VALIDATION_JSON="$(
  E7_HR_POSDRPO_WORK_DIR=outputs/e7/hopper_replay_posbase_drpo_cgrid_500k_validation \
    bash scripts/run_e7_hopper_replay_posbase_drpo_cgrid_500k.sh validate
)"
python - "${VALIDATION_JSON}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
assert payload["status"] == "PASS"
assert payload["experiment_id"] == "EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-POSBASE-DRPO-CGRID-500K-01"
assert payload["expected_controls"] == 20
assert payload["expected_branches"] == 180
assert payload["steps"] == 500000
assert payload["evaluation_interval"] == 20000
assert payload["positive_baselines"] == [0.0, 4.0, 5.0, 6.0]
assert payload["drpo_remoteness_scales"] == [0.06, 0.08, 0.1, 0.125]
assert payload["seeds"] == [200, 201, 202, 203, 208, 209, 210, 211, 212]
assert len(payload["controls"]) == 20
assert payload["all_replay_samples_retained"] is True
assert payload["advantage_shift_before_sign_split"] is True
assert payload["paired_drpo_vs_positive_only_same_baseline"] is True
assert payload["paired_shifted_drpo_vs_b0_same_c"] is True
assert payload["raw_actor_gradient_and_adam_update_diagnostics"] is True
assert payload["near_far_effective_negative_mass_diagnostics"] is True
assert payload["held_out_seeds_touched"] is False
PY
