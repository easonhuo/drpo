#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 - "${ROOT}" <<'PY'
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

from drpo import countdown_e8_alpha1_highc_scan_common as scan
from drpo import countdown_e8_alpha1_highc_scan_runtime as runtime

root = Path(sys.argv[1])
cases = {
    "scan": (
        root / "configs/countdown_e8_oracle_offline_v2_asymre_deltav_scan_0p5b.yaml",
        (-1.0, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.1),
    ),
    "boundary_dense": (
        root / "configs/countdown_e8_oracle_offline_v2_asymre_deltav_boundary_dense_0p5b.yaml",
        (-1.0, -0.95, -0.9, -0.85, -0.8, -0.7, -0.6, -0.5),
    ),
}
summary: dict[str, object] = {"schema_version": 1, "status": "PASS", "cases": {}}
for name, (path, legacy_points) in cases.items():
    profile = runtime._install_config_driven_asymre_profile(path)  # noqa: SLF001
    assert profile is not None
    scan.activate_for_grid_config(path)
    config = scan.load_yaml(path)
    points = scan.parameter_points(config)
    cells = scan.build_cells(config)
    expected_rows = tuple(
        {"delta_v": delta_v, "seed_offset": seed}
        for delta_v in legacy_points
        for seed in (4000, 5000)
    )
    actual_rows = tuple(
        {"delta_v": cell.delta_v, "seed_offset": cell.seed_offset}
        for cell in cells
    )
    assert points == legacy_points
    assert actual_rows == expected_rows
    assert len(cells) == 16
    assert len({cell.name for cell in cells}) == 16
    summary["cases"][name] = {
        "points": len(points),
        "cells": len(cells),
        "matrix_digest": profile["matrix_digest"],
    }

# Candidate B proves that a new reviewed delta-v grid can execute through the
# same runtime without adding an experiment-specific Python profile.
base_path, _ = cases["boundary_dense"]
candidate = yaml.safe_load(base_path.read_text(encoding="utf-8"))
candidate["experiment_id"] = "REPLAYAB-E8-ASYMRE-CONFIG-DRIVEN-CANDIDATE"
candidate["sweep"]["parameter_points"] = [
    {"delta_v": -1.0, "role": "anchor"},
    {"delta_v": -0.975, "role": "interior"},
    {"delta_v": -0.95, "role": "boundary"},
]
candidate["sweep"]["unique_parameter_points"] = 3
candidate["sweep"]["cells"] = 6
candidate_path = root / ".replayab_e8_asymre_candidate.yaml"
candidate_path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
try:
    profile = runtime._install_config_driven_asymre_profile(candidate_path)  # noqa: SLF001
    assert profile is not None
    scan.activate_for_grid_config(candidate_path)
    points = scan.parameter_points(candidate)
    cells = scan.build_cells(candidate)
    assert points == (-1.0, -0.975, -0.95)
    assert len(cells) == 6
    assert {cell.seed_offset for cell in cells} == {4000, 5000}
    summary["candidate_b"] = {
        "experiment_id": candidate["experiment_id"],
        "points": len(points),
        "cells": len(cells),
        "matrix_digest": profile["matrix_digest"],
        "python_profile_edit_required": False,
    }
finally:
    candidate_path.unlink(missing_ok=True)

print(json.dumps(summary, indent=2, sort_keys=True))
PY
