from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml

from drpo import countdown_e8_continuous_exp_common as continuous
from drpo import countdown_e8_continuous_exp_runtime as runtime
from drpo import countdown_e8_continuous_exp_trainer as trainer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "countdown_e8_oracle_offline_v2_continuous_exp_grid_0p5b.yaml"
)


def _config() -> dict:
    value = yaml.safe_load(CONFIG.read_text())
    assert isinstance(value, dict)
    return value


def test_frozen_joint_grid_has_31_points_and_62_cells() -> None:
    config = _config()
    assert config["registration_state"] == "registered_pilot"
    points = continuous.parameter_points(config)
    cells = continuous.build_cells(config)
    assert len(points) == 31
    assert len(set(points)) == 31
    assert points.count((0.0, 0.0)) == 1
    assert len(cells) == 62
    assert len({cell.name for cell in cells}) == 62
    assert {cell.seed_offset for cell in cells} == {3000, 4000}
    assert sum(cell.method == "positive_only" for cell in cells) == 2
    assert sum(cell.method == "global" for cell in cells) == 12
    assert sum(cell.method == "continuous_exp" for cell in cells) == 48


def test_weight_is_exact_alpha_exp_minus_c_u_squared() -> None:
    seq_lp = torch.tensor([-2.0, -4.0])
    weights = continuous.continuous_exp_weights(seq_lp, alpha=0.5, c=1.0)
    u = torch.tensor([1.0, 2.0])
    assert torch.allclose(weights, 0.5 * torch.exp(-u.square()))
    assert weights.requires_grad is False


def test_c_zero_is_global_and_alpha_zero_is_positive_only() -> None:
    seq_lp = torch.tensor([-0.5, -2.0, -8.0])
    global_weights = continuous.continuous_exp_weights(seq_lp, alpha=0.11, c=0.0)
    positive_only = continuous.continuous_exp_weights(seq_lp, alpha=0.0, c=1.5)
    assert torch.allclose(global_weights, torch.full_like(seq_lp, 0.11))
    assert torch.equal(positive_only, torch.zeros_like(seq_lp))


def test_negative_term_uses_unique_count_not_weight_sum() -> None:
    seq_lp = torch.tensor([-1.0, -3.0, -2.0])
    weights = torch.tensor([1.0, 0.5, 0.25])
    row_index = torch.tensor([0, 0, 1])
    counts = torch.tensor([2, 1])
    result = continuous.mean_unique_negative_term(seq_lp, weights, row_index, counts)
    expected_row0 = (-1.0 + 0.5 * -3.0) / 2.0
    expected_row1 = 0.25 * -2.0
    assert result.item() == pytest.approx((expected_row0 + expected_row1) / 2.0)
    weight_normalized_row0 = (-1.0 + 0.5 * -3.0) / 1.5
    assert result.item() != pytest.approx((weight_normalized_row0 + -2.0) / 2.0)


def test_duplicate_bank_expressions_are_removed_before_encoding(monkeypatch) -> None:
    monkeypatch.setattr(continuous.arena, "clean_expression", lambda value: value.strip())
    row = {
        "id": "row",
        "negative_bank": [
            {"expression": "(1+2)"},
            {"expression": " (1+2) "},
            {"expression": "(1*2)"},
            "(1*2)",
        ],
    }
    assert continuous.unique_negative_expressions(row) == ["(1+2)", "(1*2)"]


def test_config_rejects_legacy_or_hidden_scaling_coordinates() -> None:
    config = _config()
    config["sweep"]["rho_values"] = [0.5]
    with pytest.raises(ValueError, match="Forbidden"):
        continuous.validate_grid_config(config)


def test_training_source_contains_no_extreme_selection_calls() -> None:
    source = inspect.getsource(continuous)
    for forbidden in (
        "current_bank_extreme_indices(",
        "select_current_bank_extremes(",
        "current_bank_training_batches(",
    ):
        assert forbidden not in source
    assert "torch.exp(-float(c) * u.square())" in source


def test_every_unique_negative_contributes_gradient() -> None:
    seq_lp = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    weights = continuous.continuous_exp_weights(seq_lp, alpha=0.5, c=0.25)
    value = continuous.mean_unique_negative_term(
        seq_lp, weights, torch.tensor([0, 0, 0]), torch.tensor([3])
    )
    value.backward()
    assert seq_lp.grad is not None
    assert torch.all(seq_lp.grad != 0)


def test_trainer_and_runtime_do_not_call_extreme_selection() -> None:
    source = inspect.getsource(trainer) + inspect.getsource(runtime)
    for forbidden in (
        "current_bank_extreme_indices(",
        "select_current_bank_extremes(",
        "current_bank_training_batches(",
    ):
        assert forbidden not in source
    assert "training_diagnostics.jsonl" in source


def test_runtime_has_no_test_split_cli_argument() -> None:
    parser = runtime.parser()
    option_strings = {
        option
        for action in parser._subparsers._group_actions[0].choices["run"]._actions
        for option in action.option_strings
    }
    assert "--test" not in option_strings


def test_registered_launcher_has_no_unregistered_acknowledgement() -> None:
    source = (
        ROOT / "scripts" / "run_countdown_e8_oracle_offline_v2_continuous_exp_auto.py"
    ).read_text()
    one_click = (
        ROOT
        / "scripts"
        / "run_countdown_e8_oracle_offline_v2_continuous_exp_auto_one_click.sh"
    ).read_text()
    assert "--allow-dev-unregistered" not in source
    assert "--allow-dev-unregistered" not in one_click
    assert '"registration_state": config["registration_state"]' in source
