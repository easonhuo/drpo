from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml

from drpo import countdown_e8_alpha1_c_scan_common as scan
from drpo import countdown_e8_alpha1_c_scan_runtime as runtime
from drpo import countdown_e8_alpha1_c_scan_trainer as trainer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs" / "countdown_e8_oracle_offline_v2_alpha1_c_scan_0p5b.yaml"
)
ONE_CLICK = (
    ROOT
    / "scripts"
    / "run_countdown_e8_oracle_offline_v2_alpha1_c_scan_auto_one_click.sh"
)


def _config() -> dict:
    value = yaml.safe_load(CONFIG.read_text())
    assert isinstance(value, dict)
    return value


def test_explicit_scan_has_8_points_and_32_cells() -> None:
    config = _config()
    points = scan.parameter_points(config)
    cells = scan.build_cells(config)
    assert points == (
        (0.0, 0.0),
        (0.5, 1.0),
        (1.0, 1.5),
        (1.0, 2.0),
        (1.0, 2.25),
        (1.0, 2.5),
        (1.0, 3.0),
        (1.0, 4.0),
    )
    assert len(cells) == 32
    assert len({cell.name for cell in cells}) == 32
    assert {cell.seed_offset for cell in cells} == {5000, 6000, 7000, 8000}
    assert sum(cell.method == "positive_only" for cell in cells) == 4
    assert sum(cell.method == "global" for cell in cells) == 0
    assert sum(cell.method == "continuous_exp" for cell in cells) == 28


def test_alpha1_scan_and_same_seed_controls_are_present() -> None:
    points = set(scan.parameter_points(_config()))
    assert (0.0, 0.0) in points
    assert (0.5, 1.0) in points
    assert {(1.0, c) for c in (1.5, 2.0, 2.25, 2.5, 3.0, 4.0)} <= points


def test_weight_formula_is_unchanged() -> None:
    seq_lp = torch.tensor([-2.0, -4.0])
    weights = scan.continuous_exp_weights(seq_lp, alpha=1.0, c=2.25)
    u = torch.tensor([1.0, 2.0])
    assert torch.allclose(weights, torch.exp(-2.25 * u.square()))
    assert weights.requires_grad is False


def test_previous_best_control_uses_exact_original_formula() -> None:
    seq_lp = torch.tensor([-1.0, -2.0, -3.0])
    weights = scan.continuous_exp_weights(seq_lp, alpha=0.5, c=1.0)
    u = -seq_lp / 2.0
    assert torch.allclose(weights, 0.5 * torch.exp(-u.square()))


def test_negative_term_uses_unique_count_not_weight_sum() -> None:
    seq_lp = torch.tensor([-1.0, -3.0, -2.0])
    weights = torch.tensor([1.0, 0.5, 0.25])
    result = scan.mean_unique_negative_term(
        seq_lp,
        weights,
        torch.tensor([0, 0, 1]),
        torch.tensor([2, 1]),
    )
    expected = (((-1.0 + 0.5 * -3.0) / 2.0) + 0.25 * -2.0) / 2.0
    assert result.item() == pytest.approx(expected)


def test_duplicate_bank_expressions_are_removed_before_encoding(monkeypatch) -> None:
    monkeypatch.setattr(scan.arena, "clean_expression", lambda value: value.strip())
    row = {
        "id": "row",
        "negative_bank": [
            {"expression": "(1+2)"},
            {"expression": " (1+2) "},
            {"expression": "(1*2)"},
            "(1*2)",
        ],
    }
    assert scan.unique_negative_expressions(row) == ["(1+2)", "(1*2)"]


def test_registration_transition_does_not_require_code_change() -> None:
    config = _config()
    assert config["registration_state"] == "dev_code_first_unregistered"
    scan.validate_grid_config(config)
    config["registration_state"] = "registered_pilot"
    scan.validate_grid_config(config)


def test_two_runtime_slots_per_gpu_are_frozen() -> None:
    config = _config()
    assert config["execution"]["parallel_cells_per_gpu"] == 2
    parser = runtime.parser()
    args = parser.parse_args(
        [
            "run",
            "--model_path",
            "/m",
            "--bank",
            "/b",
            "--val",
            "/v",
            "--base_config",
            "/base",
            "--grid_config",
            "/grid",
            "--work_dir",
            "/work",
            "--gpus",
            "0,1",
            "--runtime-slots-per-gpu",
            "2",
        ]
    )
    assert args.runtime_slots_per_gpu == 2


def test_config_rejects_grid_or_scientific_drift() -> None:
    config = _config()
    config["sweep"]["parameter_points"][2]["c"] = 1.75
    with pytest.raises(ValueError, match="parameter points changed"):
        scan.validate_grid_config(config)

    config = _config()
    config["training"]["steps"] = 1000
    with pytest.raises(ValueError, match="1200"):
        scan.validate_grid_config(config)


def test_training_path_contains_no_extreme_selection_or_test_cli() -> None:
    source = inspect.getsource(scan) + inspect.getsource(trainer) + inspect.getsource(runtime)
    for forbidden in (
        "current_bank_extreme_indices(",
        "select_current_bank_extremes(",
        "current_bank_training_batches(",
    ):
        assert forbidden not in source
    parser = runtime.parser()
    option_strings = {
        option
        for action in parser._subparsers._group_actions[0].choices["run"]._actions
        for option in action.option_strings
    }
    assert "--test" not in option_strings


def test_one_click_uses_new_config_and_unregistered_acknowledgement() -> None:
    source = ONE_CLICK.read_text()
    assert "countdown_e8_oracle_offline_v2_alpha1_c_scan_0p5b.yaml" in source
    assert "--allow-dev-unregistered" in source
    assert "E8_ALPHA1_C_SCAN_MAX_DEVICES:-8" in source
