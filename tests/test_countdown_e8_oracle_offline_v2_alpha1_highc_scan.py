from __future__ import annotations

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


def test_highc_scan_has_8_points_and_32_cells_without_positive_only() -> None:
    points = scan.parameter_points(_config())
    cells = scan.build_cells(_config())
    assert points == (
        (0.5, 1.0),
        (1.0, 3.0),
        (1.0, 4.0),
        (1.0, 5.0),
        (1.0, 6.0),
        (1.0, 8.0),
        (1.0, 10.0),
        (1.0, 12.0),
    )
    assert len(cells) == 32
    assert len({cell.name for cell in cells}) == 32
    assert {cell.seed_offset for cell in cells} == {9000, 10000, 11000, 12000}
    assert all(cell.method == "continuous_exp" for cell in cells)
    assert all(cell.alpha > 0.0 for cell in cells)


def test_predecessor_module_is_restored_after_scoped_validation() -> None:
    original = (
        predecessor.EXPERIMENT_ID,
        predecessor.PARAMETER_POINTS,
        predecessor.SEED_OFFSETS,
    )
    scan.validate_grid_config(_config())
    assert (
        predecessor.EXPERIMENT_ID,
        predecessor.PARAMETER_POINTS,
        predecessor.SEED_OFFSETS,
    ) == original


def test_activation_exposes_new_identity_only_inside_context() -> None:
    original_id = predecessor.EXPERIMENT_ID
    with scan.activated():
        assert predecessor.EXPERIMENT_ID == scan.EXPERIMENT_ID
        assert predecessor.PARAMETER_POINTS == scan.PARAMETER_POINTS
        assert predecessor.SEED_OFFSETS == scan.SEED_OFFSETS
    assert predecessor.EXPERIMENT_ID == original_id


def test_weight_formula_is_inherited_unchanged() -> None:
    seq_lp = torch.tensor([-2.0, -4.0])
    actual = scan.continuous_exp_weights(seq_lp, alpha=1.0, c=8.0)
    u = torch.tensor([1.0, 2.0])
    assert torch.allclose(actual, torch.exp(-8.0 * u.square()))
    assert actual.requires_grad is False


def test_config_rejects_positive_only_seed_or_metric_drift() -> None:
    config = _config()
    config["sweep"]["parameter_points"][0] = {
        "alpha": 0.0,
        "c": 0.0,
        "role": "unauthorized",
    }
    with pytest.raises(ValueError):
        scan.validate_grid_config(config)

    config = _config()
    config["sweep"]["seed_offsets"][0] = 8000
    with pytest.raises(ValueError, match="development seed offsets changed"):
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


def test_one_click_uses_highc_config_and_unregistered_acknowledgement() -> None:
    source = ONE_CLICK.read_text()
    assert "countdown_e8_oracle_offline_v2_alpha1_highc_scan_0p5b.yaml" in source
    assert "--allow-dev-unregistered" in source
    assert "E8_ALPHA1_HIGHC_MAX_DEVICES:-8" in source
