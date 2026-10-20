from __future__ import annotations

import math

import pytest
import torch

from drpo_reference.controls import (
    TaperFamily,
    far_mask,
    gradient_l2_norm,
    near_mask,
    normalized_excess_surprisal,
    point_retention_coefficient,
    scale_to_match_norm,
    taper_weight,
)
from drpo_reference.experiments.d4rl import (
    DRPO_EXPONENTIAL_MULTIPLIER,
    EXPONENTIAL_COEFFICIENT,
    REFERENCE_DISTANCE,
    canonical_drpo_negative_factors,
    canonical_standardized_action_remoteness,
)


def test_cu1_point_retention_formulas_match_reference_definitions() -> None:
    distance = torch.tensor([0.0, 2.5, 5.0, 7.5], dtype=torch.float64)
    rho = 0.25
    reference = 5.0
    normalized = distance / reference
    expected = {
        TaperFamily.RECIPROCAL_LINEAR: 1.0 / (1.0 + (1.0 / rho - 1.0) * normalized),
        TaperFamily.RECIPROCAL_QUADRATIC: 1.0 / (1.0 + (1.0 / rho - 1.0) * normalized.square()),
        TaperFamily.EXPONENTIAL_LINEAR: torch.exp(math.log(rho) * normalized),
    }
    for family, reference_value in expected.items():
        coefficient = point_retention_coefficient(
            family, retention=rho, reference_distance=reference
        )
        actual = taper_weight(distance, family=family, coefficient=coefficient)
        torch.testing.assert_close(actual, reference_value, rtol=0.0, atol=1.0e-12)
        assert actual[2].item() == pytest.approx(rho, abs=1.0e-12)


def test_du1_v4_distance_coordinate_matches_reference_formulas() -> None:
    normalized_excess = torch.tensor([0.0, 0.25, 1.0, 4.0], dtype=torch.float64)
    distance = torch.sqrt(normalized_excess)
    rho = 0.25
    reciprocal = 1.0 / rho - 1.0
    exponential = -math.log(rho)
    expected = {
        TaperFamily.RECIPROCAL_LINEAR: 1.0 / (1.0 + reciprocal * torch.sqrt(normalized_excess)),
        TaperFamily.RECIPROCAL_QUADRATIC: 1.0 / (1.0 + reciprocal * normalized_excess),
        TaperFamily.EXPONENTIAL_QUADRATIC: torch.exp(-exponential * normalized_excess),
    }
    for family, reference_value in expected.items():
        coefficient = point_retention_coefficient(family, retention=rho)
        actual = taper_weight(distance, family=family, coefficient=coefficient)
        torch.testing.assert_close(actual, reference_value, rtol=0.0, atol=1.0e-12)
        assert actual[2].item() == pytest.approx(rho, abs=1.0e-12)


def test_exponential_quadratic_weight_is_linear_in_normalized_excess() -> None:
    log_probability = torch.tensor([-1.0, -3.0, -5.0], dtype=torch.float64)
    normalized = normalized_excess_surprisal(log_probability, threshold=1.0, scale=2.0)
    distance = torch.sqrt(normalized)
    coefficient = 0.7
    actual = taper_weight(
        distance,
        family=TaperFamily.EXPONENTIAL_QUADRATIC,
        coefficient=coefficient,
    )
    torch.testing.assert_close(actual, torch.exp(-coefficient * normalized))


def test_remoteness_weights_are_detached_by_default() -> None:
    log_probability = torch.tensor([-1.0, -3.0], requires_grad=True)
    normalized = normalized_excess_surprisal(log_probability, threshold=0.5, scale=2.0)
    distance = torch.sqrt(normalized)
    weight = taper_weight(
        distance,
        family=TaperFamily.EXPONENTIAL_QUADRATIC,
        coefficient=1.5,
    )
    assert not distance.requires_grad
    assert not weight.requires_grad


def test_hard_masks_are_complementary_and_boundary_is_near() -> None:
    distance = torch.tensor([0.0, 4.999, 5.0, 5.001, 10.0])
    near = near_mask(distance, threshold=5.0)
    far = far_mask(distance, threshold=5.0)
    assert near.tolist() == [True, True, True, False, False]
    assert far.tolist() == [False, False, False, True, True]
    assert torch.equal(~near, far)


def test_raw_gradient_norm_and_budget_scale() -> None:
    target = [torch.tensor([3.0, 4.0]), None]
    source = [torch.tensor([1.5, 2.0])]
    assert gradient_l2_norm(target).item() == pytest.approx(5.0)
    assert gradient_l2_norm(source).item() == pytest.approx(2.5)
    scale = scale_to_match_norm(target, source)
    assert scale == pytest.approx(2.0)
    scaled = [source[0] * scale]
    assert gradient_l2_norm(scaled).item() == pytest.approx(5.0)


def test_budget_scale_rejects_nonzero_target_with_zero_source() -> None:
    with pytest.raises(ZeroDivisionError):
        scale_to_match_norm([torch.tensor([1.0])], [torch.tensor([0.0])])


def test_invalid_coordinates_raise() -> None:
    with pytest.raises(ValueError):
        taper_weight(torch.tensor([-1.0]), family="reciprocal_linear", coefficient=1.0)
    with pytest.raises(ValueError):
        normalized_excess_surprisal(torch.tensor([-1.0]), threshold=0.0, scale=0.0)
    with pytest.raises(ValueError):
        near_mask(torch.tensor([float("nan")]), threshold=1.0)


def test_d4rl_drpo_factor_matches_reference_formula() -> None:
    remoteness = torch.tensor(
        [0.0, 4.0, 16.0, 36.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    normalized_excess = remoteness.detach() / (REFERENCE_DISTANCE**2)
    expected = DRPO_EXPONENTIAL_MULTIPLIER * torch.exp(-EXPONENTIAL_COEFFICIENT * normalized_excess)
    actual = canonical_drpo_negative_factors(remoteness)
    torch.testing.assert_close(actual, expected)
    assert actual.requires_grad is False


def test_d4rl_standardized_remoteness_is_detached() -> None:
    mean = torch.tensor([[0.0, 0.0], [1.0, -1.0]], requires_grad=True)
    log_std = torch.zeros_like(mean, requires_grad=True)
    actions = torch.tensor([[3.0, 4.0], [1.0, 1.0]], requires_grad=True)
    remoteness = canonical_standardized_action_remoteness(mean, log_std, actions)
    assert remoteness.tolist() == pytest.approx([12.5, 2.0])
    assert remoteness.requires_grad is False
