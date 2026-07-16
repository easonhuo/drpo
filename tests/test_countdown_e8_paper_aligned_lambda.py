from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import pytest
import torch

from drpo import countdown_e8_paper_aligned_lambda_common as common
from drpo import countdown_e8_paper_aligned_lambda_runtime as runtime


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / common.DEFAULT_GRID_CONFIG


def _config() -> dict:
    return common.load_yaml(CONFIG)


def test_round1_grid_is_registered_bounded_and_lambda_only() -> None:
    config = _config()
    common.validate_grid_config(config)
    points = common.parameter_points(config)
    cells = common.build_cells(config)
    assert points == ((0.0, 0.0),) + tuple((1.0, value) for value in common.LAMBDA_VALUES)
    assert len(cells) == 18
    assert sum(cell.method == "positive_only" for cell in cells) == 3
    assert sum(cell.method == "paper_aligned_exp" for cell in cells) == 15
    assert config["sweep"]["global_rerun"] is False
    assert config["execution"]["default_gpu_slots"] == 2


def test_paper_formula_golden_cases_have_no_extra_square() -> None:
    tau = 2.0
    scale_c = 3.0
    lambda_value = math.log(2.0)
    # seq_lp is negative mean-token log probability, so D=-seq_lp.
    seq_lp = torch.tensor([-1.0, -2.0, -5.0, -8.0], requires_grad=True)
    weights = common.paper_aligned_lambda_weights(
        seq_lp,
        alpha=1.0,
        lambda_value=lambda_value,
        tau=tau,
        scale_c=scale_c,
    )
    expected = torch.tensor([1.0, 1.0, 0.5, 0.25])
    assert torch.allclose(weights, expected, atol=1e-6)
    assert not math.isclose(float(weights[-1]), math.exp(-4.0 * lambda_value))
    assert weights.requires_grad is False


def test_large_lambda_keeps_near_field_and_removes_only_far_tail() -> None:
    weights = common.paper_aligned_lambda_weights(
        torch.tensor([-1.0, -3.0]),
        alpha=1.0,
        lambda_value=100.0,
        tau=2.0,
        scale_c=1.0,
    )
    assert weights[0].item() == pytest.approx(1.0)
    assert weights[1].item() < 1e-20


def test_calibration_is_deterministic_and_non_degenerate() -> None:
    calibration = common.calibration_from_surprisals(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        minimum_scale=1e-6,
        minimum_active_fraction=0.25,
    )
    assert calibration["tau"] == pytest.approx(3.5)
    assert calibration["scale_c"] == pytest.approx(3.0)
    assert calibration["active_fraction"] == pytest.approx(0.5)


def test_negative_term_uses_unique_count_not_weight_sum() -> None:
    seq_lp = torch.tensor([-1.0, -3.0, -2.0])
    weights = torch.tensor([1.0, 0.0, 0.5])
    row_index = torch.tensor([0, 0, 1])
    counts = torch.tensor([2, 1])
    value = common.mean_unique_negative_term(seq_lp, weights, row_index, counts)
    # Row 0: (-1 + 0)/2=-0.5. Row 1: (-1)/1=-1. Mean=-0.75.
    assert value.item() == pytest.approx(-0.75)


def test_duplicate_negatives_are_removed_by_cleaned_expression() -> None:
    row = {
        "id": "row",
        "negative_bank": [
            {"expression": "1 + 2"},
            {"expression": "1 + 2"},
            {"expression": "2 + 1"},
        ],
    }
    assert common.unique_negative_expressions(row) == ["1 + 2", "2 + 1"]


def test_formula_source_contains_linear_z_not_z_square() -> None:
    source = inspect.getsource(common.paper_aligned_lambda_weights)
    assert "torch.exp(-float(lambda_value) * z)" in source
    assert "z.square" not in source
    assert "z ** 2" not in source


def test_runtime_contract_has_calibration_and_no_test_argument() -> None:
    parser = runtime.parser()
    calibrate = parser.parse_args(
        [
            "calibrate",
            "--model_path",
            "m",
            "--bank",
            "b",
            "--val",
            "v",
            "--base_config",
            "base",
            "--grid_config",
            "grid",
            "--work_dir",
            "work",
        ]
    )
    assert calibrate.command == "calibrate"
    assert "--test" not in parser.format_help()


def test_target_driven_protocol_is_checked_in() -> None:
    general = (
        REPO / "docs/experiment_governance/TARGET_DRIVEN_EXPERIMENT_MECHANISM.md"
    ).read_text()
    e8 = (REPO / "docs/experiments/E8_PAPER_ALIGNED_LAMBDA_TUNING_PROTOCOL.md").read_text()
    assert "non_transfer_under_successor_protocol" in general
    assert "not Positive-only" in e8
    assert "At no point" in e8


def test_auto_launcher_defaults_to_two_gpu_slots() -> None:
    source = (REPO / "scripts/run_countdown_e8_paper_aligned_lambda_auto.py").read_text()
    assert 'default_gpu_slots = int(execution["default_gpu_slots"])' in source
    assert "requested_slots = default_gpu_slots if args.max_devices is None" in source
    assert '"configured_default_gpu_slots": default_gpu_slots' in source


def test_trainer_calls_unique_negative_term_with_exact_signature() -> None:
    path = REPO / "src/drpo/countdown_e8_paper_aligned_lambda_trainer.py"
    tree = ast.parse(path.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "mean_unique_negative_term"
    ]
    assert len(calls) == 1
    assert len(calls[0].args) == 4
    assert not calls[0].keywords
