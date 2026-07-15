from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from drpo import countdown_e8_alpha1_c_scan_common as root_base
from drpo import countdown_e8_alpha1_highc_scan_common as predecessor
from drpo import countdown_e8_alpha1_logc_boundary_scan_common as scan


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "countdown_e8_oracle_offline_v2_alpha1_logc_boundary_scan_0p5b.yaml"
)
RUNTIME = (
    ROOT / "src" / "drpo" / "countdown_e8_alpha1_logc_boundary_scan_runtime.py"
)
AUTO = (
    ROOT
    / "scripts"
    / "run_countdown_e8_oracle_offline_v2_alpha1_logc_boundary_scan_auto.py"
)
ONE_CLICK = (
    ROOT
    / "scripts"
    / "run_countdown_e8_oracle_offline_v2_alpha1_logc_boundary_scan_auto_one_click.sh"
)


def _config() -> dict:
    value = yaml.safe_load(CONFIG.read_text())
    assert isinstance(value, dict)
    return value


def test_logc_scan_has_8_points_and_32_cells_without_positive_only() -> None:
    config = _config()
    points = scan.parameter_points(config)
    cells = scan.build_cells(config)
    assert points == (
        (0.5, 1.0),
        (1.0, 8.0),
        (1.0, 12.0),
        (1.0, 16.0),
        (1.0, 24.0),
        (1.0, 32.0),
        (1.0, 64.0),
        (1.0, 128.0),
    )
    assert len(cells) == 32
    assert len({cell.name for cell in cells}) == 32
    assert {cell.seed_offset for cell in cells} == {13000, 14000, 15000, 16000}
    assert all(cell.method == "continuous_exp" for cell in cells)
    assert all(cell.alpha > 0.0 for cell in cells)
    assert config["sweep"]["positive_only_same_seed_control"] is False


def test_same_seed_control_and_previous_anchors_are_frozen() -> None:
    points = set(scan.parameter_points(_config()))
    assert (0.5, 1.0) in points
    assert (1.0, 8.0) in points
    assert (1.0, 12.0) in points
    assert {(1.0, c) for c in (16.0, 24.0, 32.0, 64.0, 128.0)} <= points


def test_predecessor_modules_are_restored_after_scoped_validation() -> None:
    predecessor_before = (
        predecessor.EXPERIMENT_ID,
        predecessor.PARAMETER_POINTS,
        predecessor.SEED_OFFSETS,
    )
    root_before = (
        root_base.EXPERIMENT_ID,
        root_base.PARAMETER_POINTS,
        root_base.SEED_OFFSETS,
    )
    scan.validate_grid_config(_config())
    assert (
        predecessor.EXPERIMENT_ID,
        predecessor.PARAMETER_POINTS,
        predecessor.SEED_OFFSETS,
    ) == predecessor_before
    assert (
        root_base.EXPERIMENT_ID,
        root_base.PARAMETER_POINTS,
        root_base.SEED_OFFSETS,
    ) == root_before


def test_activation_exposes_new_identity_only_inside_context() -> None:
    predecessor_id = predecessor.EXPERIMENT_ID
    root_id = root_base.EXPERIMENT_ID
    with scan.activated():
        assert predecessor.EXPERIMENT_ID == scan.EXPERIMENT_ID
        assert root_base.EXPERIMENT_ID == scan.EXPERIMENT_ID
        assert root_base.PARAMETER_POINTS == scan.PARAMETER_POINTS
        assert root_base.SEED_OFFSETS == scan.SEED_OFFSETS
    assert predecessor.EXPERIMENT_ID == predecessor_id
    assert root_base.EXPERIMENT_ID == root_id


def test_weight_formula_is_inherited_unchanged() -> None:
    seq_lp = torch.tensor([-2.0, -4.0])
    actual = scan.continuous_exp_weights(seq_lp, alpha=1.0, c=32.0)
    u = torch.tensor([1.0, 2.0])
    assert torch.allclose(actual, torch.exp(-32.0 * u.square()))
    assert actual.requires_grad is False


def test_config_rejects_grid_seed_or_gate_drift() -> None:
    config = _config()
    config["sweep"]["parameter_points"][3]["c"] = 20.0
    with pytest.raises(ValueError):
        scan.validate_grid_config(config)

    config = _config()
    config["sweep"]["seed_offsets"][0] = 12000
    with pytest.raises(ValueError, match="development seed offsets changed"):
        scan.validate_grid_config(config)

    config = _config()
    config["sweep"]["automatic_further_extension_forbidden"] = False
    with pytest.raises(ValueError, match="Automatic further c extension"):
        scan.validate_grid_config(config)

    config = _config()
    config["evaluation"]["paired_direction_gate_for_followup"] = "2_of_4"
    with pytest.raises(ValueError, match="Paired direction gate"):
        scan.validate_grid_config(config)

    config = _config()
    config["evaluation"]["material_lift_gate_absolute_pass_at_8"] = 0.0
    with pytest.raises(ValueError, match="Material late-window"):
        scan.validate_grid_config(config)


def test_runtime_and_auto_are_thin_base_adapters() -> None:
    runtime_source = RUNTIME.read_text()
    auto_source = AUTO.read_text()
    assert "logc.activate()" in runtime_source
    assert "countdown_e8_alpha1_c_scan_runtime" in runtime_source
    assert "countdown_e8_alpha1_logc_boundary_scan_runtime.py" in auto_source
    assert "e8_alpha1_logc_boundary_scan_cuda_dev_v1" in auto_source


def test_one_click_uses_logc_config_and_unregistered_acknowledgement() -> None:
    source = ONE_CLICK.read_text()
    assert "countdown_e8_oracle_offline_v2_alpha1_logc_boundary_scan_0p5b.yaml" in source
    assert "--allow-dev-unregistered" in source
    assert "E8_ALPHA1_LOGC_MAX_DEVICES:-8" in source
    assert "e8_v2_alpha1_logc_boundary_scan_dev" in source
