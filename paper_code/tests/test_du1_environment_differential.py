from __future__ import annotations

import pytest
import torch

from drpo import du1_e6_cartesian_taper_v4 as legacy
from drpo_reference.categorical.du1_controls import (
    coordinate_calibration,
)
from drpo_reference.categorical.du1_environment import (
    CartesianSemanticEnvironment,
)
from drpo_reference.categorical.du1_metrics import (
    evaluate,
    policy_geometry_audit,
)
from drpo_reference.categorical.du1_policy import cell_log_probs
from drpo_reference.categorical.du1_protocol import CELL_NAMES

from du1_helpers import (
    assert_split_equal,
    matching_models,
    small_protocol,
)


def test_environment_exactly_matches_revision_4() -> None:
    protocol = small_protocol()
    config = protocol.legacy_config()
    old = legacy.CartesianSemanticEnvironment(config, 200)
    new = CartesianSemanticEnvironment(protocol, 200)
    for name in (
        "prototype_embeddings",
        "action_prototype",
        "action_rarity",
        "action_rarity_sign",
        "action_embeddings",
        "w_plus",
        "w_direction",
    ):
        torch.testing.assert_close(
            getattr(new, name),
            getattr(old, name),
            rtol=0.0,
            atol=0.0,
        )
    assert_split_equal(new.train, old.train)
    assert_split_equal(new.test, old.test)
    assert new.audit() == old.audit()


def test_policy_cells_calibration_and_metrics_match() -> None:
    protocol = small_protocol()
    old_env, new_env, old_model, new_model = matching_models(
        protocol,
        201,
    )
    index = torch.tensor([0, 3, 7, 11, 15, 19])
    old_positive, old_cells, old_residual = (
        legacy.cell_log_probs(
            old_model,
            old_env,
            old_env.train,
            index,
        )
    )
    new_positive, new_cells, new_residual = cell_log_probs(
        new_model,
        new_env,
        new_env.train,
        index,
    )
    torch.testing.assert_close(
        new_positive,
        old_positive,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        new_residual,
        old_residual,
        rtol=0.0,
        atol=0.0,
    )
    for cell in CELL_NAMES:
        torch.testing.assert_close(
            new_cells[cell],
            old_cells[cell],
            rtol=0.0,
            atol=0.0,
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
    assert new_calibration == old_calibration

    old_metrics = legacy.evaluate(
        old_model,
        old_env,
        old_env.test,
        old_calibration,
    )
    new_metrics = evaluate(
        new_model,
        new_env,
        new_env.test,
        new_calibration,
    )
    assert new_metrics == pytest.approx(
        old_metrics,
        rel=1.0e-7,
        abs=1.0e-8,
    )
    old_geometry = legacy.policy_geometry_audit(
        old_model,
        old_env,
        protocol.legacy_config(),
    )
    new_geometry = policy_geometry_audit(
        new_model,
        new_env,
        protocol,
    )
    assert new_geometry["passed"] == old_geometry["passed"]
    assert new_geometry[
        "positive_rarity_gradient_norm"
    ] == pytest.approx(
        old_geometry["positive_rarity_gradient_norm"],
        rel=1.0e-7,
        abs=1.0e-8,
    )
    assert new_geometry[
        "cell_shared_rarity_gradient_norms"
    ] == pytest.approx(
        old_geometry["cell_shared_rarity_gradient_norms"],
        rel=1.0e-7,
        abs=1.0e-8,
    )
    for key in (
        "useful_rare_to_common_shared_rarity_gradient_ratio",
        "unhelpful_rare_to_common_shared_rarity_gradient_ratio",
        "utility_oracle_sign_valid_fraction",
        "rarity_shift_reward_drop",
        "rarity_shift_hidden_probability_drop",
    ):
        assert new_geometry[key] == pytest.approx(
            old_geometry[key],
            rel=1.0e-7,
            abs=1.0e-8,
        )
