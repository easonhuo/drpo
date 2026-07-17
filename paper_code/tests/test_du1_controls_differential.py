from __future__ import annotations

import pytest
import torch

from drpo import du1_e6_cartesian_taper_v4 as legacy
from drpo_reference.categorical.du1_controls import (
    active_cell_loss,
    coordinate_calibration,
    normalized_excess_surprisal,
    taper_coefficients,
    taper_weight,
)
from drpo_reference.categorical.du1_policy import (
    batch_indices,
    cell_log_probs,
    trainable_parameters,
)
from drpo_reference.categorical.du1_protocol import (
    CELL_NAMES,
    method_specs,
)

from du1_helpers import matching_models, small_protocol


@pytest.mark.parametrize(
    "family",
    (
        "reciprocal_linear_distance",
        "reciprocal_quadratic_distance",
        "exponential_quadratic_distance",
    ),
)
def test_shared_taper_kernel_matches_revision_4(
    family: str,
) -> None:
    protocol = small_protocol()
    coefficients = taper_coefficients(
        protocol.reference_rare_retention
    )
    assert coefficients == legacy.taper_coefficients(
        protocol.reference_rare_retention
    )
    log_probability = torch.tensor(
        [-2.0, -4.0, -8.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    calibration = {
        "threshold": 2.0,
        "scale": 3.0,
    }
    old_coordinate = legacy.normalized_excess_surprisal(
        log_probability,
        calibration,
    )
    new_coordinate = normalized_excess_surprisal(
        log_probability,
        calibration,
    )
    torch.testing.assert_close(
        new_coordinate,
        old_coordinate,
        rtol=0.0,
        atol=0.0,
    )
    old_weight = legacy.taper_weight(
        old_coordinate,
        family,
        coefficients[family],
    )
    new_weight = taper_weight(
        new_coordinate,
        family,
        coefficients[family],
    )
    torch.testing.assert_close(
        new_weight,
        old_weight,
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    assert not new_weight.requires_grad


@pytest.mark.parametrize(
    "method",
    (
        "positive_only",
        "all_negative",
        "reciprocal_linear_distance",
        "reciprocal_quadratic_distance",
        "exponential_quadratic_distance",
    ),
)
def test_active_cell_loss_and_gradients_match(
    method: str,
) -> None:
    protocol = small_protocol()
    old_env, new_env, old_model, new_model = matching_models(
        protocol,
        202,
    )
    index = batch_indices(
        202,
        1,
        protocol.train_states,
        protocol.batch_size,
    )
    _, old_cells, _ = legacy.cell_log_probs(
        old_model,
        old_env,
        old_env.train,
        index,
    )
    _, new_cells, _ = cell_log_probs(
        new_model,
        new_env,
        new_env.train,
        index,
    )
    old_calibration = legacy.coordinate_calibration(
        old_model,
        old_env,
        protocol.legacy_config(),
    )
    new_calibration = coordinate_calibration(
        new_model,
        new_env,
        protocol,
    )
    old_spec = legacy.method_specs([method])[0]
    new_spec = method_specs((method,))[0]
    old_coefficients = legacy.taper_coefficients(
        protocol.reference_rare_retention
    )
    new_coefficients = taper_coefficients(
        protocol.reference_rare_retention
    )
    old_loss, old_diagnostics = legacy.active_cell_loss(
        old_cells,
        old_spec,
        old_calibration,
        old_coefficients,
        1.0,
    )
    new_loss, new_diagnostics = active_cell_loss(
        new_cells,
        new_spec,
        new_calibration,
        new_coefficients,
        1.0,
    )
    torch.testing.assert_close(
        new_loss,
        old_loss,
        rtol=1.0e-7,
        atol=1.0e-8,
    )
    assert new_diagnostics == pytest.approx(
        old_diagnostics,
        rel=1.0e-7,
        abs=1.0e-8,
    )
    old_gradients = torch.autograd.grad(
        old_loss,
        legacy.trainable_parameters(old_model),
        allow_unused=True,
    )
    new_gradients = torch.autograd.grad(
        new_loss,
        trainable_parameters(new_model),
        allow_unused=True,
    )
    for actual, expected in zip(
        new_gradients,
        old_gradients,
    ):
        if expected is None:
            assert actual is None
        else:
            assert actual is not None
            torch.testing.assert_close(
                actual,
                expected,
                rtol=1.0e-6,
                atol=1.0e-7,
            )
