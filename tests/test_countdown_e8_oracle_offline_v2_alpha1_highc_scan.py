from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from drpo import countdown_e8_alpha1_c_scan_common as predecessor
from drpo import countdown_e8_alpha1_highc_scan_common as scan


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "countdown_e8_oracle_offline_v2_alpha1_highc_scan_0p5b.yaml"
)
RUNTIME = ROOT / "src" / "drpo" / "countdown_e8_alpha1_highc_scan_runtime.py"
AUTO = (
    ROOT
    / "scripts"
    / "run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto.py"
)
ONE_CLICK = (
    ROOT
    / "scripts"
    / "run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto_one_click.sh"
)


def _config() -> dict:
    value = yaml.safe_load(CONFIG.read_text())
    assert isinstance(value, dict)
    return value


def test_linear_scan_has_16_points_and_32_cells() -> None:
    config = _config()
    points = scan.parameter_points(config)
    cells = scan.build_cells(config)
    assert points == (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 0.051293294),
        (1.0, 0.105360516),
        (1.0, 0.162518929),
        (1.0, 0.223143551),
        (1.0, 0.287682072),
        (1.0, 0.430782916),
        (1.0, 0.693147181),
        (1.0, 0.916290732),
        (1.0, 1.203972804),
        (1.0, 1.386294361),
        (1.0, 1.609437912),
        (1.0, 1.897119985),
        (1.0, 2.302585093),
        (1.0, 2.995732274),
    )
    assert len(cells) == 32
    assert len({cell.name for cell in cells}) == 32
    assert {cell.seed_offset for cell in cells} == {4000, 5000}
    assert sum(cell.method == "positive_only" for cell in cells) == 2
    assert sum(cell.method == "global" for cell in cells) == 2
    assert sum(cell.method == "continuous_exp" for cell in cells) == 28
    assert config["execution"]["default_gpus"] == list(range(8))
    assert config["execution"]["parallel_cells_per_gpu"] == 2


def test_predecessor_module_is_restored_after_scoped_validation() -> None:
    original = (
        predecessor.EXPERIMENT_ID,
        predecessor.PARAMETER_POINTS,
        predecessor.SEED_OFFSETS,
        predecessor.continuous_exp_weights,
    )
    scan.validate_grid_config(_config())
    assert (
        predecessor.EXPERIMENT_ID,
        predecessor.PARAMETER_POINTS,
        predecessor.SEED_OFFSETS,
        predecessor.continuous_exp_weights,
    ) == original


def test_activation_exposes_new_identity_only_inside_context() -> None:
    original_id = predecessor.EXPERIMENT_ID
    with scan.activated():
        assert predecessor.EXPERIMENT_ID == scan.EXPERIMENT_ID
        assert predecessor.PARAMETER_POINTS == scan.PARAMETER_POINTS
        assert predecessor.SEED_OFFSETS == scan.SEED_OFFSETS
        assert predecessor.continuous_exp_weights is scan.continuous_exp_weights
        assert predecessor.validate_grid_config is scan.validate_grid_config
        assert predecessor.build_cells is scan.build_cells
    assert predecessor.EXPERIMENT_ID == original_id


def test_weight_formula_removes_only_the_extra_square() -> None:
    seq_lp = torch.tensor([-2.0, -4.0])
    actual = scan.continuous_exp_weights(seq_lp, alpha=1.0, c=1.0)
    u = torch.tensor([1.0, 2.0])
    assert torch.allclose(actual, torch.exp(-u))
    assert not torch.allclose(actual, torch.exp(-u.square()))
    assert actual.requires_grad is False


def test_real_runtime_plan_uses_linear_validation(tmp_path: Path) -> None:
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
            str(CONFIG),
            "--work_dir",
            str(work),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads((work / "SWEEP_PLAN.json").read_text(encoding="utf-8"))
    assert plan["experiment_id"] == scan.EXPERIMENT_ID
    assert plan["parameter_points"] == 16
    assert plan["cell_count"] == 32


def test_config_rejects_positive_only_seed_gpu_or_metric_drift() -> None:
    config = _config()
    config["sweep"]["parameter_points"][0] = {
        "alpha": 1.0,
        "c": 0.0,
        "role": "unauthorized",
    }
    with pytest.raises(ValueError, match="Exactly one Positive-only"):
        scan.validate_grid_config(config)

    config = _config()
    config["sweep"]["seed_offsets"][0] = 3000
    with pytest.raises(ValueError, match="development seed offsets changed"):
        scan.validate_grid_config(config)

    config = _config()
    config["execution"]["default_gpus"] = list(range(7))
    with pytest.raises(ValueError, match="requires GPU 0-7"):
        scan.validate_grid_config(config)

    config = _config()
    config["evaluation"]["primary_selection_metric"] = "terminal_pass_at_8"
    with pytest.raises(ValueError, match="Primary selection metric"):
        scan.validate_grid_config(config)


def test_runtime_and_auto_are_thin_predecessor_adapters() -> None:
    runtime_source = RUNTIME.read_text()
    auto_source = AUTO.read_text()
    assert "highc.activate()" in runtime_source
    assert "countdown_e8_alpha1_c_scan_runtime" in runtime_source
    assert "countdown_e8_alpha1_highc_scan_runtime.py" in auto_source
    assert "e8_alpha1_highc_scan_cuda_dev_v1" in auto_source
    assert "requires all eight configured GPUs" in auto_source


def test_one_click_uses_highc_config_and_unregistered_acknowledgement() -> None:
    source = ONE_CLICK.read_text()
    assert "countdown_e8_oracle_offline_v2_alpha1_highc_scan_0p5b.yaml" in source
    assert "--allow-dev-unregistered" in source
    assert "E8_ALPHA1_HIGHC_MAX_DEVICES:-8" in source
