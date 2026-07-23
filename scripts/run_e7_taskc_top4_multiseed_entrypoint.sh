#!/usr/bin/env bash
set -euo pipefail

COMMAND="${1:-run}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

WORK_DIR="${E7_TASKC_WORK_DIR:-outputs/e7/taskc_top4_multiseed_001}"
PROFILE_SHIM_DIR="${WORK_DIR}/.taskc_python_profile"
mkdir -p "${PROFILE_SHIM_DIR}"
cat >"${PROFILE_SHIM_DIR}/sitecustomize.py" <<'PY'
from drpo import e7_squared_exp_night as _suite

_suite.TUNING_PROFILE_ID = "d4rl9_task_specific_c_top4_multiseed"
PY

export PYTHONPATH="${PROFILE_SHIM_DIR}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'PY'
from drpo import e7_squared_exp_night as suite

expected = "d4rl9_task_specific_c_top4_multiseed"
if suite.TUNING_PROFILE_ID != expected:
    raise SystemExit(
        f"task-specific bootstrap profile shim failed: {suite.TUNING_PROFILE_ID!r}"
    )
PY

bash scripts/run_e7_taskc_top4_multiseed.sh "${COMMAND}"

if [[ "${COMMAND}" == "run" ]]; then
  python - "${WORK_DIR}" <<'PY'
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

work_dir = Path(os.sys.argv[1])
aggregate_dir = work_dir / "aggregate"
rows_path = aggregate_dir / "branch_results.csv"
audit_path = aggregate_dir / "terminal_audit.json"
if not rows_path.is_file() or not audit_path.is_file():
    raise SystemExit("completed run is missing branch_results.csv or terminal_audit.json")


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


with rows_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 180:
    raise SystemExit(f"expected 180 branch rows, found {len(rows)}")

rollout_count = sum(as_bool(row.get("rollout_failure_event")) for row in rows)
nan_inf_count = sum(as_bool(row.get("nan_inf_numerical_failure")) for row in rows)
audit = json.loads(audit_path.read_text(encoding="utf-8"))
audit["rollout_failure_count"] = rollout_count
audit["nan_inf_numerical_failure_count"] = nan_inf_count
audit["rollout_failure_count_source"] = "branch_results.csv"
audit["nan_inf_numerical_failure_count_source"] = "branch_results.csv"

with tempfile.NamedTemporaryFile(
    mode="w",
    encoding="utf-8",
    dir=audit_path.parent,
    prefix=audit_path.name + ".",
    suffix=".tmp",
    delete=False,
) as handle:
    json.dump(audit, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temp_path = Path(handle.name)
os.replace(temp_path, audit_path)
PY
fi
