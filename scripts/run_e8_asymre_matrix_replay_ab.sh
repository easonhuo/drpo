#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 - "${ROOT}" <<'PY'
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from drpo import countdown_e8_alpha1_highc_scan_common as scan
from drpo import countdown_e8_alpha1_highc_scan_runtime as runtime

root = Path(sys.argv[1])
runtime_path = (
    root / "src" / "drpo" / "countdown_e8_alpha1_highc_scan_runtime.py"
)
base_config = root / "configs" / "countdown_e8_base_rl_replay_0p5b.yaml"
auto_path = (
    root
    / "scripts"
    / "run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto.py"
)
auto_spec = importlib.util.spec_from_file_location("_e8_config_driven_auto", auto_path)
if auto_spec is None or auto_spec.loader is None:
    raise RuntimeError(f"Unable to load E8 auto launcher: {auto_path}")
auto = importlib.util.module_from_spec(auto_spec)
auto_spec.loader.exec_module(auto)

historical_paths = (
    root / "configs/countdown_e8_oracle_offline_v2_alpha1_highc_scan_0p5b.yaml",
    root / "configs/countdown_e8_oracle_offline_v2_linear_c_extension_0p5b.yaml",
    root / "configs/countdown_e8_oracle_offline_v2_reciprocal_shape_screen_0p5b.yaml",
    root
    / "configs/countdown_e8_oracle_offline_v2_reciprocal_high_lambda_extension_0p5b.yaml",
    root
    / "configs/countdown_e8_oracle_offline_v2_reciprocal_quadratic_dense_lambda_curve_0p5b.yaml",
    root / "configs/countdown_e8_oracle_offline_v2_asymre_deltav_scan_0p5b.yaml",
    root
    / "configs/countdown_e8_oracle_offline_v2_asymre_deltav_boundary_dense_0p5b.yaml",
)


def cell_payload(cell) -> dict[str, object]:
    return {
        "name": cell.name,
        "method": cell.method,
        "family": cell.family,
        "alpha": float(cell.alpha),
        "coefficient": float(cell.coefficient),
        "delta_v": float(cell.delta_v),
        "seed_offset": int(cell.seed_offset),
    }


summary: dict[str, object] = {
    "schema_version": 2,
    "status": "PASS",
    "historical_replays": {},
    "new_grid_candidates": {},
}

for path in historical_paths:
    config = scan.load_yaml(path)
    experiment_id = str(config["experiment_id"])
    legacy_profile = copy.deepcopy(scan._PROFILES[experiment_id])  # noqa: SLF001
    scan.activate(legacy_profile)
    legacy_points = scan.parameter_points(config)
    legacy_cells = scan.build_cells(config)
    legacy_payload = tuple(cell_payload(cell) for cell in legacy_cells)

    profile = runtime.install_config_driven_profile(path)
    scan.activate_for_grid_config(path)
    actual_points = scan.parameter_points(config)
    actual_cells = scan.build_cells(config)
    actual_payload = tuple(cell_payload(cell) for cell in actual_cells)

    assert profile["kind"] == legacy_profile["kind"]
    assert actual_points == legacy_points
    assert tuple(profile["seed_offsets"]) == tuple(legacy_profile["seed_offsets"])
    assert actual_payload == legacy_payload
    assert len(actual_cells) == int(config["sweep"]["cells"])
    assert len({cell.name for cell in actual_cells}) == len(actual_cells)

    summary["historical_replays"][experiment_id] = {
        "kind": profile["kind"],
        "points": len(actual_points),
        "cells": len(actual_cells),
        "matrix_digest": profile["matrix_digest"],
        "exact_point_order": True,
        "exact_cell_order_and_identity": True,
    }


candidate_specs = (
    {
        "label": "legacy_exp",
        "source": historical_paths[1],
        "experiment_id": "REPLAYAB-E8-CONFIG-DRIVEN-EXP-CANDIDATE",
        "points": [
            {"alpha": 1.0, "c": 0.02, "role": "candidate_0p02"},
            {"alpha": 1.0, "c": 0.03, "role": "candidate_0p03"},
            {"alpha": 1.0, "c": 0.05, "role": "candidate_0p05"},
        ],
    },
    {
        "label": "reciprocal_screen",
        "source": historical_paths[3],
        "experiment_id": "REPLAYAB-E8-CONFIG-DRIVEN-RECIPROCAL-CANDIDATE",
        "points": [
            {
                "family": "reciprocal_linear",
                "alpha": 1.0,
                "coefficient": 23.0,
                "role": "candidate_linear_23",
            },
            {
                "family": "reciprocal_quadratic",
                "alpha": 1.0,
                "coefficient": 23.0,
                "role": "candidate_quadratic_23",
            },
        ],
    },
    {
        "label": "asymre_scan",
        "source": historical_paths[6],
        "experiment_id": "REPLAYAB-E8-CONFIG-DRIVEN-ASYMRE-CANDIDATE",
        "points": [
            {"delta_v": -1.0, "role": "anchor"},
            {"delta_v": -0.975, "role": "interior"},
            {"delta_v": -0.95, "role": "boundary"},
        ],
    },
)

with tempfile.TemporaryDirectory(prefix="e8-config-driven-replay-") as temporary:
    temp = Path(temporary)
    model = temp / "model"
    model.mkdir()
    bank = temp / "bank.jsonl"
    val = temp / "val.jsonl"
    bank.write_text("{}\n", encoding="utf-8")
    val.write_text("{}\n", encoding="utf-8")

    for spec in candidate_specs:
        candidate = yaml.safe_load(
            Path(spec["source"]).read_text(encoding="utf-8")
        )
        candidate["experiment_id"] = spec["experiment_id"]
        candidate["sweep"]["parameter_points"] = spec["points"]
        candidate["sweep"]["unique_parameter_points"] = len(spec["points"])
        candidate["sweep"]["cells"] = len(spec["points"]) * len(
            candidate["sweep"]["seed_offsets"]
        )
        candidate_path = temp / f"{spec['label']}.yaml"
        candidate_path.write_text(
            yaml.safe_dump(candidate, sort_keys=False),
            encoding="utf-8",
        )

        assert candidate["experiment_id"] not in scan._PROFILES  # noqa: SLF001
        profile = auto.activate_config_driven_profile(candidate_path)
        points = scan.parameter_points(candidate)
        cells = scan.build_cells(candidate)
        assert profile["kind"] == spec["label"]
        assert len(points) == len(spec["points"])
        assert len(cells) == candidate["sweep"]["cells"]
        assert len({cell.name for cell in cells}) == len(cells)

        work = temp / f"plan-{spec['label']}"
        subprocess.run(
            [
                sys.executable,
                str(runtime_path),
                "plan",
                "--model_path",
                str(model),
                "--bank",
                str(bank),
                "--val",
                str(val),
                "--base_config",
                str(base_config),
                "--grid_config",
                str(candidate_path),
                "--work_dir",
                str(work),
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        plan = json.loads(
            (work / "SWEEP_PLAN.json").read_text(encoding="utf-8")
        )
        assert plan["experiment_id"] == candidate["experiment_id"]
        assert plan["parameter_points"] == len(spec["points"])
        assert plan["cell_count"] == candidate["sweep"]["cells"]

        summary["new_grid_candidates"][spec["label"]] = {
            "experiment_id": candidate["experiment_id"],
            "points": len(points),
            "cells": len(cells),
            "matrix_digest": profile["matrix_digest"],
            "python_profile_edit_required": False,
            "runtime_plan_passed": True,
            "auto_launcher_adapter_id": auto._adapter_id(profile),  # noqa: SLF001
        }

print(json.dumps(summary, indent=2, sort_keys=True))
PY
