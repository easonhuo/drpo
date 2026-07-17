from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch

from drpo import cu1_distance_taper_formal as legacy_taper
from drpo import drpo_cu1_e1_e4_oneclick as legacy
from drpo_reference.continuous.cu1 import CU1Protocol, make_actor, make_environment
from drpo_reference.continuous.cu1_taper import (
    CU1TaperProtocol,
    retention_weight,
    run_taper_method,
    weighted_negative_loss,
)
from drpo_reference.continuous.cu1_training import CU1PositiveProtocol


def _configure_legacy(tmp_path: Path) -> legacy.Protocol:
    protocol = legacy.Protocol(
        n_train_states=32,
        n_test_states=24,
        hidden_dim=16,
        positive_batch_states=8,
        eval_every=1,
        probe_states=4,
        positive_steps=3,
        positive_continuation_steps=2,
        lbfgs_max_iter=1,
        positive_polish_min_steps=1,
        positive_polish_max_steps=1,
        positive_polish_check_every=1,
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


def _legacy_taper() -> legacy_taper.TaperProtocol:
    return legacy_taper.TaperProtocol(
        formal_seeds=(70,),
        reference_distance=5.0,
        primary_rho=0.25,
        sensitivity_rhos=(),
        negative_alpha=1.0,
        learning_rate=5e-4,
        batch_states=8,
        evaluation_interval=1,
        minimum_steps=2,
        maximum_steps=4,
        stable_windows=2,
        normalized_slope_threshold=0.0,
        normalized_field_residual_threshold=2e-3,
        positive_absolute_gradient_threshold=1e-3,
        task_failure_retention=0.45,
        log_sigma_event_boundary=12.0,
        probe_states=4,
        bootstrap_samples=20,
    )


def _reference_taper(old: legacy_taper.TaperProtocol) -> CU1TaperProtocol:
    return CU1TaperProtocol(
        formal_seeds=old.formal_seeds,
        reference_distance=old.reference_distance,
        primary_retention=old.primary_rho,
        sensitivity_retentions=old.sensitivity_rhos,
        negative_alpha=old.negative_alpha,
        learning_rate=old.learning_rate,
        batch_states=old.batch_states,
        evaluation_interval=old.evaluation_interval,
        minimum_steps=old.minimum_steps,
        maximum_steps=old.maximum_steps,
        stable_windows=old.stable_windows,
        normalized_slope_threshold=old.normalized_slope_threshold,
        normalized_field_residual_threshold=(
            old.normalized_field_residual_threshold
        ),
        positive_absolute_gradient_threshold=(
            old.positive_absolute_gradient_threshold
        ),
        task_failure_retention=old.task_failure_retention,
        probe_states=old.probe_states,
    )


def _positive_training(old: legacy.Protocol) -> CU1PositiveProtocol:
    return CU1PositiveProtocol(
        positive_batch_states=old.positive_batch_states,
        eval_every=old.eval_every,
        adam_beta1=old.adam_beta1,
        adam_beta2=old.adam_beta2,
        adam_eps=old.adam_eps,
        formal_seeds=(10,),
    )


def _matching_state(protocol: CU1Protocol, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    old_actor = legacy.GaussianActor().to("cpu")
    torch.manual_seed(seed)
    new_actor = make_actor(protocol).to("cpu")
    for name, old_value in old_actor.state_dict().items():
        torch.testing.assert_close(
            new_actor.state_dict()[name],
            old_value,
            rtol=0.0,
            atol=0.0,
        )
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
            assert actual_float == pytest.approx(
                expected_float,
                rel=1e-6,
                abs=1e-7,
            )
        return
    assert actual == expected


def _assert_mapping(
    actual: dict[str, object],
    expected: dict[str, object],
    *,
    skip: set[str] | None = None,
) -> None:
    ignored = skip or set()
    for key, expected_value in expected.items():
        if key not in ignored:
            _assert_value(actual[key], expected_value)


@pytest.mark.parametrize(
    "family",
    ("unweighted", "reciprocal_linear", "reciprocal_quadratic", "exponential"),
)
def test_retention_weight_matches_authoritative_helper(family: str) -> None:
    old_taper = _legacy_taper()
    taper = _reference_taper(old_taper)
    distance = torch.tensor([0.0, 1.5, 5.0, 9.0])
    expected = legacy_taper.taper_weight(
        distance,
        family,
        0.25,
        old_taper,
    )
    actual = retention_weight(
        distance,
        family=family,
        retention=0.25,
        protocol=taper,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize(
    "family",
    ("unweighted", "reciprocal_linear", "reciprocal_quadratic", "exponential"),
)
def test_weighted_loss_and_gradients_match_authoritative_runner(
    tmp_path: Path,
    family: str,
) -> None:
    old = _configure_legacy(tmp_path)
    protocol = _reference_protocol(old)
    old_taper = _legacy_taper()
    taper = _reference_taper(old_taper)
    old_environment = legacy.make_environment(70)
    new_environment = make_environment(70, protocol)
    initialization = _matching_state(protocol, seed=411)
    old_actor = legacy.GaussianActor().to("cpu")
    old_actor.load_state_dict(copy.deepcopy(initialization))
    new_actor = make_actor(protocol).to("cpu")
    new_actor.load_state_dict(copy.deepcopy(initialization))
    ids = torch.tensor([0, 3, 7, 11, 15, 19, 23, 27])

    expected_loss, expected_diagnostics = legacy_taper.weighted_negative_loss(
        old_actor,
        old_environment.train,
        ids,
        family,
        0.25,
        old_taper,
    )
    actual_loss, actual_diagnostics = weighted_negative_loss(
        new_actor,
        new_environment.train,
        protocol,
        taper,
        ids,
        family=family,
        retention=0.25,
    )
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-7)
    _assert_mapping(actual_diagnostics, expected_diagnostics)

    expected_gradients = torch.autograd.grad(
        expected_loss,
        old_actor.all_parameters(),
        allow_unused=True,
    )
    actual_gradients = torch.autograd.grad(
        actual_loss,
        new_actor.all_parameters(),
        allow_unused=True,
    )
    for actual, expected in zip(actual_gradients, expected_gradients):
        if expected is None:
            assert actual is None
        else:
            assert actual is not None
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_short_taper_trajectory_matches_authoritative_runner(
    tmp_path: Path,
) -> None:
    old = _configure_legacy(tmp_path)
    protocol = _reference_protocol(old)
    old_taper = _legacy_taper()
    taper = _reference_taper(old_taper)
    positive = _positive_training(old)
    old_environment = legacy.make_environment(70)
    new_environment = make_environment(70, protocol)
    initialization = _matching_state(protocol, seed=922)

    expected_summary, expected_trajectory, expected_diagnostics = (
        legacy_taper.train_method(
            70,
            initialization,
            old_environment,
            "reciprocal_quadratic",
            0.25,
            tmp_path / "legacy_taper",
            old_taper,
        )
    )
    actual = run_taper_method(
        seed=70,
        initialization_state=initialization,
        environment=new_environment,
        protocol=protocol,
        positive_training=positive,
        taper=taper,
        family="reciprocal_quadratic",
        retention=0.25,
    )

    assert len(actual.trajectory) == len(expected_trajectory)
    for actual_row, expected_row in zip(actual.trajectory, expected_trajectory):
        _assert_mapping(actual_row, expected_row)
    _assert_mapping(
        actual.summary,
        expected_summary,
        skip={"elapsed_seconds", "milliseconds_per_update", "terminal_checkpoint"},
    )
    assert len(actual.diagnostics) == len(expected_diagnostics)
    for actual_row, expected_row in zip(actual.diagnostics, expected_diagnostics):
        _assert_mapping(actual_row, expected_row)
