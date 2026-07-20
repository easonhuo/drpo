from __future__ import annotations

import copy
import csv
import math
from pathlib import Path

import pytest
import torch

from drpo import drpo_cu1_e1_e4_oneclick as legacy
from drpo_reference.continuous.cu1 import CU1Protocol, make_actor, make_environment
from drpo_reference.continuous.cu1_control import (
    CU1ControlProtocol,
    control_gradients,
    run_far_pressure_control,
)
from drpo_reference.continuous.cu1_phase import (
    CU1PhaseProtocol,
    analytic_local_solution,
    analytic_mean_critical_alpha,
    analytic_positive_sigma,
    analytic_variance_boundary_alpha,
    run_phase_scan,
)
from drpo_reference.continuous.cu1_training import CU1PositiveProtocol


def _configure_legacy(tmp_path: Path) -> legacy.Protocol:
    protocol = legacy.Protocol(
        n_train_states=32,
        n_test_states=24,
        hidden_dim=16,
        positive_batch_states=8,
        eval_every=1,
        probe_states=6,
        positive_steps=3,
        positive_continuation_steps=2,
        lbfgs_max_iter=1,
        positive_polish_min_steps=1,
        positive_polish_max_steps=1,
        positive_polish_check_every=1,
        e4_local_warm_steps=2,
        e4_local_continuation_steps=1,
        e4_runaway_steps=4,
        e4_control_steps=4,
        e1_e2_seeds=(10,),
        e3_seeds=(30,),
        e4_seeds=(50,),
        variance_robustness_seeds=(5,),
    )
    legacy.P = protocol
    legacy.DEVICE = torch.device("cpu")
    legacy.DTYPE = torch.float32
    legacy.ROOT = tmp_path
    return protocol


def _reference_protocol(old: legacy.Protocol) -> CU1Protocol:
    return CU1Protocol(
        state_dim=old.state_dim,
        action_dim=old.action_dim,
        n_train_states=old.n_train_states,
        n_test_states=old.n_test_states,
        positive_samples_per_state=old.positive_samples_per_state,
        negative_samples_per_state=old.negative_samples_per_state,
        gap_to_unseen_optimum=old.gap_to_unseen_optimum,
        negative_offset_from_positive=old.negative_offset_from_positive,
        positive_contour_radius=old.positive_contour_radius,
        negative_contour_radius=old.negative_contour_radius,
        reward_width=old.reward_width,
        baseline=old.baseline,
        positive_angle_1=old.positive_angle_1,
        hidden_dim=old.hidden_dim,
        hidden_layers=old.hidden_layers,
        initial_sigma=old.initial_sigma,
        near_far_standardized_threshold=old.near_far_standardized_threshold,
        task_failure_retention=old.task_failure_retention,
        task_failure_consecutive_evals=old.task_failure_consecutive_evals,
        log_sigma_event_boundary=old.log_sigma_event_boundary,
    )


def _positive_training(old: legacy.Protocol) -> CU1PositiveProtocol:
    return CU1PositiveProtocol(
        positive_batch_states=old.positive_batch_states,
        eval_every=old.eval_every,
        adam_beta1=old.adam_beta1,
        adam_beta2=old.adam_beta2,
        adam_eps=old.adam_eps,
        absolute_residual_threshold_alpha_zero=(old.absolute_residual_threshold_alpha_zero),
        formal_seeds=(10,),
    )


def _phase_protocol(old: legacy.Protocol) -> CU1PhaseProtocol:
    return CU1PhaseProtocol(
        fixed_alphas=old.e4_fixed_alphas,
        learnable_alphas=old.e4_learn_alphas,
        learning_rate=old.e4_local_lr,
        warm_steps=old.e4_local_warm_steps,
        continuation_steps=old.e4_local_continuation_steps,
        runaway_steps=old.e4_runaway_steps,
        evaluation_interval=old.eval_every,
        normalized_residual_threshold=old.normalized_residual_threshold,
        absolute_residual_threshold_alpha_zero=(old.absolute_residual_threshold_alpha_zero),
        formal_seeds=(50,),
    )


def _control_protocol(old: legacy.Protocol) -> CU1ControlProtocol:
    return CU1ControlProtocol(
        alpha_local=old.e4_control_alpha_local,
        lambda_far=old.e4_control_lambda_far,
        far_cap_ratio=old.e4_control_far_cap_ratio,
        learning_rate=old.e4_control_lr,
        steps=old.e4_control_steps,
        evaluation_interval=old.eval_every,
        formal_seeds=(50,),
    )


def _matching_state(protocol: CU1Protocol, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    old_actor = legacy.GaussianActor().to("cpu")
    torch.manual_seed(seed)
    new_actor = make_actor(protocol).to("cpu")
    for name, old_value in old_actor.state_dict().items():
        torch.testing.assert_close(new_actor.state_dict()[name], old_value, rtol=0.0, atol=0.0)
    return copy.deepcopy(old_actor.state_dict())


def _assert_value(actual: object, expected: object) -> None:
    if expected is None or isinstance(expected, (bool, str)):
        assert actual == expected
        return
    if isinstance(expected, (int, float)):
        expected_float = float(expected)
        actual_float = float(actual)
        if math.isnan(expected_float):
            assert math.isnan(actual_float)
        else:
            assert actual_float == pytest.approx(expected_float, rel=1e-6, abs=1e-7)
        return
    assert actual == expected


def _assert_mapping(actual: dict[str, object], expected: dict[str, object]) -> None:
    for key, expected_value in expected.items():
        _assert_value(actual[key], expected_value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _assert_trajectory(actual: list[dict[str, object]], expected: list[dict[str, str]]) -> None:
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        for key, text in expected_row.items():
            if text == "":
                assert actual_row[key] is None
            elif text in {"True", "False"}:
                assert actual_row[key] is (text == "True")
            elif key in {
                "stage",
                "method",
                "optimizer",
                "support_event_type",
            }:
                assert str(actual_row[key]) == text
            else:
                _assert_value(actual_row[key], float(text))


def test_phase_analytic_helpers_match_authoritative_runner(
    tmp_path: Path,
) -> None:
    old = _configure_legacy(tmp_path)
    protocol = _reference_protocol(old)
    assert analytic_positive_sigma(protocol) == pytest.approx(
        legacy.analytic_positive_sigma(), abs=1e-15
    )
    assert analytic_mean_critical_alpha(protocol) == pytest.approx(
        legacy.analytic_mean_critical_alpha(), abs=1e-15
    )
    assert analytic_variance_boundary_alpha(protocol) == pytest.approx(
        legacy.analytic_variance_boundary_alpha(), abs=1e-15
    )
    for alpha in (0.0, 0.25, 0.5, 1.5, 1.75):
        _assert_mapping(
            analytic_local_solution(protocol, alpha),
            legacy.analytic_local_solution(alpha),
        )


@pytest.mark.parametrize(
    ("alpha", "fixed_sigma", "branch"),
    (
        (0.25, "analytic", "fixed_variance"),
        (0.50, None, "learnable_variance"),
    ),
)
def test_phase_scan_matches_authoritative_short_trajectory(
    tmp_path: Path,
    alpha: float,
    fixed_sigma: str | None,
    branch: str,
) -> None:
    old = _configure_legacy(tmp_path)
    protocol = _reference_protocol(old)
    phase = _phase_protocol(old)
    positive = _positive_training(old)
    old_environment = legacy.make_environment(50)
    new_environment = make_environment(50, protocol)
    initialization = _matching_state(protocol, seed=711)
    sigma = legacy.analytic_positive_sigma() if fixed_sigma else None

    expected_summary = legacy.run_local_scan_seed(
        50,
        initialization,
        old_environment,
        alpha,
        sigma,
        branch,
    )
    actual = run_phase_scan(
        seed=50,
        initialization_state=initialization,
        environment=new_environment,
        protocol=protocol,
        positive_training=positive,
        phase=phase,
        alpha=alpha,
        fixed_sigma=sigma,
        branch=branch,
    )
    expected_rows = _read_csv(
        tmp_path / "e4" / branch / f"alpha_{alpha:.2f}" / "seed_50_trajectory.csv"
    )
    _assert_trajectory(actual.trajectory, expected_rows)
    _assert_mapping(actual.summary, expected_summary)


@pytest.mark.parametrize("method", ("uncontrolled_all", "far_cap", "budget_matched_global"))
def test_phase_control_gradients_match_authoritative_runner(
    tmp_path: Path,
    method: str,
) -> None:
    old = _configure_legacy(tmp_path)
    protocol = _reference_protocol(old)
    control = _control_protocol(old)
    old_environment = legacy.make_environment(50)
    new_environment = make_environment(50, protocol)
    initialization = _matching_state(protocol, seed=912)
    old_actor = legacy.GaussianActor().to("cpu")
    old_actor.load_state_dict(copy.deepcopy(initialization))
    new_actor = make_actor(protocol).to("cpu")
    new_actor.load_state_dict(copy.deepcopy(initialization))
    ids = torch.tensor([0, 3, 7, 11, 15, 19, 23, 27])
    fixed_sigma = legacy.analytic_positive_sigma()

    expected_gradients, expected_diagnostics = legacy.e4_control_gradients(
        old_actor, old_environment.train, ids, method, fixed_sigma
    )
    actual_gradients, actual_diagnostics = control_gradients(
        new_actor,
        new_environment.train,
        protocol,
        control,
        ids,
        method=method,
        fixed_sigma=fixed_sigma,
    )
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        if expected_gradient is None:
            assert actual_gradient is None
        else:
            assert actual_gradient is not None
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)
    _assert_mapping(actual_diagnostics, expected_diagnostics)


def test_phase_control_short_trajectory_matches_authoritative_runner(
    tmp_path: Path,
) -> None:
    old = _configure_legacy(tmp_path)
    protocol = _reference_protocol(old)
    control = _control_protocol(old)
    positive = _positive_training(old)
    old_environment = legacy.make_environment(50)
    new_environment = make_environment(50, protocol)
    initialization = _matching_state(protocol, seed=313)

    expected_summary = legacy.run_control_seed(50, initialization, old_environment, "far_cap")
    actual = run_far_pressure_control(
        seed=50,
        initialization_state=initialization,
        environment=new_environment,
        protocol=protocol,
        positive_training=positive,
        control=control,
        method="far_cap",
    )
    expected_rows = _read_csv(tmp_path / "e4" / "control" / "far_cap" / "seed_50_trajectory.csv")
    _assert_trajectory(actual.trajectory, expected_rows)
    _assert_mapping(actual.summary, expected_summary)
