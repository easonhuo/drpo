from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from drpo import countdown_e8_alpha1_c_scan_common as predecessor
from drpo import countdown_e8_alpha1_highc_scan_common as scan

ROOT = Path(__file__).resolve().parents[1]
ROUND1_CONFIG = (
    ROOT
    / "configs"
    / "countdown_e8_oracle_offline_v2_alpha1_highc_scan_0p5b.yaml"
)
EXTENSION_CONFIG = (
    ROOT
    / "configs"
    / "countdown_e8_oracle_offline_v2_linear_c_extension_0p5b.yaml"
)
RUNTIME = ROOT / "src" / "drpo" / "countdown_e8_alpha1_highc_scan_runtime.py"
AUTO = (
    ROOT
    / "scripts"
    / "run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto.py"
)
RUNSPEC = (
    ROOT
    / "runspecs"
    / "ready"
    / "E8_PAPER_ALIGNED_LINEAR_C_EXTENSION_20260717_01.yaml"
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_extension_has_eight_new_c_points_and_sixteen_cells() -> None:
    config = _load(EXTENSION_CONFIG)
    points = scan.parameter_points(config)
    cells = scan.build_cells(config)
    assert points == scan.C_EXTENSION_PARAMETER_POINTS
    assert len(cells) == 16
    assert len({cell.name for cell in cells}) == 16
    assert {cell.seed_offset for cell in cells} == {4000, 5000}
    assert {cell.alpha for cell in cells} == {1.0}
    assert {cell.method for cell in cells} == {"continuous_exp"}
    assert not any(cell.c in {0.0, 0.051293294, 2.995732274} for cell in cells)
    assert config["sweep"]["positive_only_same_seed_control"] is False
    assert config["historical_reference"]["positive_only_rerun_in_this_round"] is False


def test_round1_profile_remains_unchanged_after_extension_validation() -> None:
    original = (
        scan.EXPERIMENT_ID,
        scan.PARAMETER_POINTS,
        scan.SEED_OFFSETS,
        scan.EXPECTED_POINTS,
        scan.EXPECTED_CELLS,
    )
    scan.validate_grid_config(_load(EXTENSION_CONFIG))
    assert (
        scan.EXPERIMENT_ID,
        scan.PARAMETER_POINTS,
        scan.SEED_OFFSETS,
        scan.EXPECTED_POINTS,
        scan.EXPECTED_CELLS,
    ) == original
    assert scan.parameter_points(_load(ROUND1_CONFIG)) == scan.ROUND1_PARAMETER_POINTS


def test_extension_activation_updates_base_identity_only_inside_context() -> None:
    config = _load(EXTENSION_CONFIG)
    profile = scan._profile_for_config(config)
    original_id = predecessor.EXPERIMENT_ID
    with scan.activated(profile):
        assert predecessor.EXPERIMENT_ID == scan.C_EXTENSION_EXPERIMENT_ID
        assert predecessor.PARAMETER_POINTS == scan.C_EXTENSION_PARAMETER_POINTS
        assert predecessor.SEED_OFFSETS == (4000, 5000)
        assert predecessor.EXPECTED_POINTS == 8
        assert predecessor.EXPECTED_CELLS == 16
    assert predecessor.EXPERIMENT_ID == original_id


def test_extension_rejects_alpha_positive_only_and_reference_drift() -> None:
    config = _load(EXTENSION_CONFIG)
    config["sweep"]["parameter_points"][0]["alpha"] = 0.5
    with pytest.raises(ValueError, match="alpha fixed at 1"):
        scan.validate_grid_config(config)

    config = _load(EXTENSION_CONFIG)
    config["sweep"]["positive_only_same_seed_control"] = True
    with pytest.raises(ValueError, match="must not rerun Positive-only"):
        scan.validate_grid_config(config)

    config = _load(EXTENSION_CONFIG)
    config["historical_reference"]["result_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Historical result manifest"):
        scan.validate_grid_config(config)


def test_runtime_plan_selects_extension_profile(tmp_path: Path) -> None:
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
            str(EXTENSION_CONFIG),
            "--work_dir",
            str(work),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads((work / "SWEEP_PLAN.json").read_text(encoding="utf-8"))
    assert plan["experiment_id"] == scan.C_EXTENSION_EXPERIMENT_ID
    assert plan["parameter_points"] == 8
    assert plan["cell_count"] == 16
    assert {row["method"] for row in plan["cells"]} == {"continuous_exp"}


def test_runspec_enables_default_e8_results_delivery() -> None:
    spec = _load(RUNSPEC)
    assert spec["registration"] == {"mode": "deferred", "closure_required": True}
    assert spec["delivery"] == {
        "enabled": True,
        "auto": True,
        "mode": "results_repo",
        "repository": "easonhuo/drpo-results",
        "branch": "ingest/e8",
        "export_profile": "manifest_text_v1",
        "max_total_size_mb": 30,
        "max_file_size_mb": 10,
    }
    assert "E8_ALPHA1_HIGHC_GRID_CONFIG=" in spec["entrypoint"]["command"]
    assert "16 parameter points" not in "\n".join(spec["success_criteria"])
    assert any("8 parameter points and 16 cells" in item for item in spec["success_criteria"])


def test_existing_launcher_is_reused_without_new_training_stack() -> None:
    source = AUTO.read_text(encoding="utf-8")
    assert "activate_for_grid_config" in source
    assert "countdown_e8_alpha1_highc_scan_runtime.py" in source
    assert "e8_linear_c_extension_cuda_dev_v1" in source
    assert "run_countdown_e8_oracle_offline_v2_alpha1_c_scan_auto.py" in source
