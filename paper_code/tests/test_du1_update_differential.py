from __future__ import annotations

import pytest
import torch

from drpo import du1_e6_cartesian_taper_v4 as legacy
from drpo_reference.categorical.du1_controls import (
    coordinate_calibration,
    negative_loss_and_diagnostics,
    rarity_logit_anchor_loss,
)
from drpo_reference.categorical.du1_policy import (
    batch_indices,
    cell_log_probs,
    trainable_parameters,
)
from drpo_reference.categorical.du1_protocol import method_specs

from du1_helpers import matching_models, small_protocol


def test_global_match_and_first_adam_update_match() -> None:
    protocol = small_protocol()
    old_env, new_env, old_model, new_model = matching_models(
        protocol,
        203,
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
    index = batch_indices(
        203,
        1,
        protocol.train_states,
        protocol.batch_size,
    )
    old_positive, old_cells, _ = legacy.cell_log_probs(
        old_model,
        old_env,
        old_env.train,
        index,
    )
    new_positive, new_cells, _ = cell_log_probs(
        new_model,
        new_env,
        new_env.train,
        index,
    )
    old_spec = legacy.method_specs(["global_matched"])[0]
    new_spec = method_specs(("global_matched",))[0]
    coefficients = legacy.taper_coefficients(protocol.reference_rare_retention)
    raw_spec = legacy.MethodSpec(
        "all_negative",
        legacy.CELL_NAMES,
    )
    target_spec = legacy.MethodSpec(
        "exponential_quadratic_distance",
        legacy.CELL_NAMES,
        "exponential_quadratic_distance",
    )
    old_parameters = legacy.trainable_parameters(old_model)
    old_raw, _ = legacy.active_cell_loss(
        old_cells,
        raw_spec,
        old_calibration,
        coefficients,
        1.0,
    )
    old_target, _ = legacy.active_cell_loss(
        old_cells,
        target_spec,
        old_calibration,
        coefficients,
        1.0,
    )
    old_raw_norm = legacy.flat_grad_norm(
        old_raw,
        old_parameters,
        retain_graph=True,
    )
    old_target_norm = legacy.flat_grad_norm(
        old_target,
        old_parameters,
        retain_graph=True,
    )
    old_scale = old_target_norm / max(
        old_raw_norm,
        legacy.EPS,
    )
    old_negative, old_diagnostics = legacy.active_cell_loss(
        old_cells,
        old_spec,
        old_calibration,
        coefficients,
        old_scale,
    )
    old_diagnostics = {
        **old_diagnostics,
        "negative_raw_gradient_norm": old_raw_norm,
        "negative_target_gradient_norm": old_target_norm,
        "negative_applied_gradient_norm": (old_scale * old_raw_norm),
        "stepwise_budget_match_error": abs(old_scale * old_raw_norm - old_target_norm),
        "stepwise_global_scale": old_scale,
    }

    new_negative, new_diagnostics = negative_loss_and_diagnostics(
        cells=new_cells,
        spec=new_spec,
        calibration=new_calibration,
        protocol=protocol,
        model=new_model,
    )
    torch.testing.assert_close(
        new_negative,
        old_negative,
        rtol=1.0e-7,
        atol=1.0e-8,
    )
    assert new_diagnostics == pytest.approx(
        old_diagnostics,
        rel=1.0e-7,
        abs=1.0e-8,
    )

    old_states = old_env.train["states"][index]
    new_states = new_env.train["states"][index]
    old_anchor = legacy.rarity_logit_anchor_loss(
        old_model,
        old_states,
    )
    new_anchor = rarity_logit_anchor_loss(
        new_model,
        new_states,
    )
    old_loss = (
        -old_positive.mean()
        + protocol.negative_alpha * old_negative
        + protocol.rarity_logit_anchor_coefficient * old_anchor
    )
    new_loss = (
        -new_positive.mean()
        + protocol.negative_alpha * new_negative
        + protocol.rarity_logit_anchor_coefficient * new_anchor
    )
    torch.testing.assert_close(
        new_loss,
        old_loss,
        rtol=1.0e-7,
        atol=1.0e-8,
    )

    old_optimizer = torch.optim.Adam(
        old_parameters,
        lr=protocol.learning_rate,
        betas=(protocol.adam_beta1, protocol.adam_beta2),
        eps=protocol.adam_eps,
    )
    new_optimizer = torch.optim.Adam(
        trainable_parameters(new_model),
        lr=protocol.learning_rate,
        betas=(protocol.adam_beta1, protocol.adam_beta2),
        eps=protocol.adam_eps,
    )
    old_optimizer.zero_grad(set_to_none=True)
    new_optimizer.zero_grad(set_to_none=True)
    old_loss.backward()
    new_loss.backward()
    old_optimizer.step()
    new_optimizer.step()
    for name, expected in old_model.state_dict().items():
        torch.testing.assert_close(
            new_model.state_dict()[name],
            expected,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
