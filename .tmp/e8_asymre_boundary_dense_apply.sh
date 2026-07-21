#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path

common_path = Path("src/drpo/countdown_e8_alpha1_highc_scan_common.py")
common = common_path.read_text(encoding="utf-8")

id_old = '''ASYMRE_DELTAV_EXPERIMENT_ID = (
    "EXT-C-E8-ORACLE-OFFLINE-V2-ASYMRE-DELTAV-SCAN-0.5B-01"
)
ROUND1_PARAMETER_POINTS = (
'''
id_new = '''ASYMRE_DELTAV_EXPERIMENT_ID = (
    "EXT-C-E8-ORACLE-OFFLINE-V2-ASYMRE-DELTAV-SCAN-0.5B-01"
)
ASYMRE_DELTAV_BOUNDARY_DENSE_EXPERIMENT_ID = (
    "EXT-C-E8-ORACLE-OFFLINE-V2-ASYMRE-DELTAV-BOUNDARY-DENSE-0.5B-01"
)
ROUND1_PARAMETER_POINTS = (
'''
if common.count(id_old) != 1:
    raise SystemExit("unexpected AsymRE experiment-id anchor count")
common = common.replace(id_old, id_new, 1)

points_old = '''ASYMRE_DELTA_VS = (-1.0, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.1)
SEED_OFFSETS = (4000, 5000)
'''
points_new = '''ASYMRE_DELTA_VS = (-1.0, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.1)
ASYMRE_BOUNDARY_DENSE_DELTA_VS = (
    -1.0,
    -0.95,
    -0.9,
    -0.85,
    -0.8,
    -0.7,
    -0.6,
    -0.5,
)
SEED_OFFSETS = (4000, 5000)
'''
if common.count(points_old) != 1:
    raise SystemExit("unexpected AsymRE parameter anchor count")
common = common.replace(points_old, points_new, 1)

profile_old = '''    ASYMRE_DELTAV_EXPERIMENT_ID: {
        "experiment_id": ASYMRE_DELTAV_EXPERIMENT_ID,
        "version": "0.1.0-dev-code-first-asymre-deltav-scan",
        "default_grid_config": (
            "configs/countdown_e8_oracle_offline_v2_asymre_deltav_scan_0p5b.yaml"
        ),
        "parameter_points": ASYMRE_DELTA_VS,
        "seed_offsets": SEED_OFFSETS,
        "expected_points": 8,
        "expected_cells": 16,
        "requires_positive_only": False,
        "kind": "asymre_scan",
    },
}
'''
profile_new = '''    ASYMRE_DELTAV_EXPERIMENT_ID: {
        "experiment_id": ASYMRE_DELTAV_EXPERIMENT_ID,
        "version": "0.1.0-dev-code-first-asymre-deltav-scan",
        "default_grid_config": (
            "configs/countdown_e8_oracle_offline_v2_asymre_deltav_scan_0p5b.yaml"
        ),
        "parameter_points": ASYMRE_DELTA_VS,
        "seed_offsets": SEED_OFFSETS,
        "expected_points": 8,
        "expected_cells": 16,
        "requires_positive_only": False,
        "kind": "asymre_scan",
    },
    ASYMRE_DELTAV_BOUNDARY_DENSE_EXPERIMENT_ID: {
        "experiment_id": ASYMRE_DELTAV_BOUNDARY_DENSE_EXPERIMENT_ID,
        "version": "0.2.0-dev-code-first-asymre-deltav-boundary-dense",
        "default_grid_config": (
            "configs/countdown_e8_oracle_offline_v2_"
            "asymre_deltav_boundary_dense_0p5b.yaml"
        ),
        "parameter_points": ASYMRE_BOUNDARY_DENSE_DELTA_VS,
        "seed_offsets": SEED_OFFSETS,
        "expected_points": 8,
        "expected_cells": 16,
        "requires_positive_only": False,
        "kind": "asymre_scan",
    },
}
'''
if common.count(profile_old) != 1:
    raise SystemExit("unexpected AsymRE profile anchor count")
common = common.replace(profile_old, profile_new, 1)
common_path.write_text(common, encoding="utf-8")

test_path = Path("tests/test_countdown_e8_oracle_offline_v2_alpha1_highc_scan.py")
test = test_path.read_text(encoding="utf-8")
marker = "ASYMRE_BOUNDARY_DENSE_CONFIG = ("
if marker in test:
    raise SystemExit("dense AsymRE tests already present")
append = r'''


ASYMRE_BOUNDARY_DENSE_CONFIG = (
    ROOT
    / "configs"
    / "countdown_e8_oracle_offline_v2_asymre_deltav_boundary_dense_0p5b.yaml"
)


def test_asymre_boundary_dense_has_eight_points_and_sixteen_paired_cells() -> None:
    config = _load_path(ASYMRE_BOUNDARY_DENSE_CONFIG)
    points = scan.parameter_points(config)
    cells = scan.build_cells(config)
    assert points == scan.ASYMRE_BOUNDARY_DENSE_DELTA_VS
    assert points == (-1.0, -0.95, -0.9, -0.85, -0.8, -0.7, -0.6, -0.5)
    assert len(cells) == 16
    assert len({cell.name for cell in cells}) == 16
    assert {cell.seed_offset for cell in cells} == {4000, 5000}
    assert {cell.method for cell in cells} == {"asymre"}
    assert {cell.family for cell in cells} == {"asymre"}
    assert {cell.coefficient for cell in cells} == {0.0}
    assert {cell.delta_v for cell in cells} == set(points)
    assert {cell.alpha for cell in cells} == {
        0.0,
        0.05,
        0.1,
        0.15,
        0.2,
        0.3,
        0.4,
        0.5,
    }
    assert config["sweep"]["cells"] == 16
    assert config["execution"]["expected_full_waves"] == 1
    assert config["predecessor"]["rerun_boundary_points_as_internal_anchors"] is True


def test_asymre_boundary_dense_rejects_parameter_or_seed_drift() -> None:
    config = _load_path(ASYMRE_BOUNDARY_DENSE_CONFIG)
    config["sweep"]["parameter_points"][1]["delta_v"] = -0.94
    with pytest.raises(ValueError, match="parameter points changed"):
        scan.validate_grid_config(config)

    config = _load_path(ASYMRE_BOUNDARY_DENSE_CONFIG)
    config["sweep"]["seed_offsets"] = [4000]
    with pytest.raises(ValueError, match="seed offsets changed"):
        scan.validate_grid_config(config)


def test_runtime_plan_selects_asymre_boundary_dense_profile(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    bank = tmp_path / "bank.jsonl"
    val = tmp_path / "val.jsonl"
    bank.write_text("{}\n", encoding="utf-8")
    val.write_text("{}\n", encoding="utf-8")
    work = tmp_path / "work"
    subprocess.run(
        [
            sys.executable,
            str(RUNTIME),
            "plan",
            "--model_path",
            str(model),
            "--bank",
            str(bank),
            "--val",
            str(val),
            "--base_config",
            str(ROOT / "configs/countdown_e8_base_rl_replay_0p5b.yaml"),
            "--grid_config",
            str(ASYMRE_BOUNDARY_DENSE_CONFIG),
            "--work_dir",
            str(work),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads((work / "SWEEP_PLAN.json").read_text(encoding="utf-8"))
    assert plan["experiment_id"] == scan.ASYMRE_DELTAV_BOUNDARY_DENSE_EXPERIMENT_ID
    assert plan["parameter_points"] == 8
    assert plan["cell_count"] == 16
    assert {row["method"] for row in plan["cells"]} == {"asymre"}
    assert {row["seed_offset"] for row in plan["cells"]} == {4000, 5000}
'''
test_path.write_text(test.rstrip() + append + "\n", encoding="utf-8")
PY

python -m py_compile \
  src/drpo/countdown_e8_alpha1_highc_scan_common.py \
  tests/test_countdown_e8_oracle_offline_v2_alpha1_highc_scan.py

git diff --check
