from __future__ import annotations

import math

import torch

from drpo import e7_canonical_injection as canonical
from drpo.e7_squared_exp_kernel import (
    install_squared_exponential_kernel,
    squared_exponential_factor,
)


def test_squared_exponential_factor_matches_registered_formula() -> None:
    distance = torch.tensor([0.0, 1.0, 2.0, 4.0])
    actual = squared_exponential_factor(
        distance,
        coefficient=1.0,
        reference_distance=2.0,
    )
    expected = torch.exp(-torch.tensor([0.0, 0.25, 1.0, 4.0]))
    assert torch.allclose(actual, expected)


def test_squared_kernel_preserves_near_and_suppresses_far_relative_to_linear() -> None:
    near = 0.5
    far = 2.0
    squared_near = math.exp(-(near**2))
    linear_near = math.exp(-near)
    squared_far = math.exp(-(far**2))
    linear_far = math.exp(-far)
    assert squared_near > linear_near
    assert squared_far < linear_far


def test_install_squared_kernel_changes_only_exponential_and_restores() -> None:
    distance = torch.tensor([1.0, 2.0])
    exponential = canonical.NegativeControl(
        method="exponential",
        negative_scale=1.0,
        canonical_alpha=0.11,
        reference_distance=2.0,
        exponential_coefficient=1.0,
    )
    reciprocal = canonical.NegativeControl(
        method="reciprocal_quadratic",
        negative_scale=1.0,
        canonical_alpha=0.11,
        reference_distance=2.0,
        reciprocal_quadratic_coefficient=1.0,
    )
    original = canonical.taper_factor
    linear_value = original(distance, exponential)
    reciprocal_value = original(distance, reciprocal)
    with install_squared_exponential_kernel():
        assert canonical.taper_factor is not original
        squared_value = canonical.taper_factor(distance, exponential)
        assert torch.allclose(
            squared_value,
            torch.exp(-torch.tensor([0.25, 1.0])),
        )
        assert torch.allclose(
            canonical.taper_factor(distance, reciprocal),
            reciprocal_value,
        )
    assert canonical.taper_factor is original
    assert torch.allclose(canonical.taper_factor(distance, exponential), linear_value)


def test_squared_kernel_rejects_invalid_inputs() -> None:
    distance = torch.tensor([1.0])
    try:
        squared_exponential_factor(
            distance,
            coefficient=-1.0,
            reference_distance=2.0,
        )
    except ValueError as exc:
        assert "coefficient" in str(exc)
    else:
        raise AssertionError("negative coefficient was accepted")

    try:
        squared_exponential_factor(
            torch.tensor([float("nan")]),
            coefficient=1.0,
            reference_distance=2.0,
        )
    except FloatingPointError as exc:
        assert "distance" in str(exc)
    else:
        raise AssertionError("non-finite distance was accepted")
