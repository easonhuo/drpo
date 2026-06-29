from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from drpo.e7_hopper_q2 import (
    METHODS,
    SquashedGaussianPolicy,
    actor_batch_loss,
    classify_actor_terminal,
    load_config,
    normalized_window_drift,
    parameter_update_statistics,
    spearman,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "e7_hopper_q2_medium_replay_v2.yaml"


def test_protocol_v42_loads_and_uses_dynamic_global_control() -> None:
    config = load_config(CONFIG)
    assert "dynamic_budget_matched_global" in METHODS
    assert "budget_matched_global" not in METHODS
    assert config.critic_validation_r2_min == 0.50
    assert config.critic_update_tolerance == 0.0002
    assert config.actor_update_tolerance == 0.0002
    assert config.actor_state_drift_tolerance == 0.01


def test_parameter_update_statistics_are_relative_to_model_scale() -> None:
    layer = torch.nn.Linear(3, 2, bias=False)
    before = [parameter.detach().clone() for parameter in layer.parameters()]
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.add_(0.01)
    stats = parameter_update_statistics(before, layer.parameters(), elapsed_steps=10)
    assert stats["raw_per_step"] > 0.0
    assert stats["rms_per_step"] > 0.0
    assert stats["relative_per_step"] > 0.0
    assert stats["relative_per_step"] < stats["raw_per_step"] * 10.0


def test_spearman_handles_monotone_values_and_ties() -> None:
    first = np.asarray([1.0, 2.0, 2.0, 4.0])
    second = np.asarray([10.0, 20.0, 20.0, 40.0])
    assert abs(spearman(first, second) - 1.0) < 1e-12


def test_dynamic_global_matches_far_cap_output_score_proxy_per_batch() -> None:
    torch.manual_seed(3)
    policy = SquashedGaussianPolicy(3, 2, (8,), -5.0, 2.0, 1e-6)
    obs = torch.randn(12, 3)
    actions = torch.tanh(torch.randn(12, 2))
    advantages = torch.tensor(
        [1.0, 0.5, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -0.3, -0.7, 0.2, -0.9]
    )
    _, diagnostics = actor_batch_loss(
        policy,
        obs,
        actions,
        advantages,
        "dynamic_budget_matched_global",
        far_threshold=1.0,
        global_scale=0.123,
        far_cap_score=1.5,
    )
    assert 0.0 <= diagnostics["dynamic_global_scale"] <= 1.0
    assert diagnostics["negative_influence_proxy_before"] > 0.0
    assert abs(
        diagnostics["negative_influence_proxy_after"]
        - diagnostics["negative_influence_proxy_target"]
    ) <= 1e-5 * max(diagnostics["negative_influence_proxy_target"], 1.0)


def test_actor_terminal_uses_relative_update_not_raw_gradient_gate() -> None:
    config = load_config(CONFIG)
    config = replace(
        config,
        audit_windows=3,
        actor_relative_slope_tolerance=1e-4,
        actor_state_drift_tolerance=1e-4,
        actor_update_tolerance=1e-3,
        actor_gradient_tolerance=1e-12,
    )
    rows = []
    for step in (100, 200, 300):
        rows.append(
            {
                "step": step,
                "loss": 1.0,
                "positive_nll": 1.0,
                "gradient_norm": 100.0,
                "update_norm": 0.1,
                "relative_update_norm": 1e-5,
                "sigma_mean": 1.0,
                "mean_abs": 0.2,
                "phantom_distance_mean": 2.0,
                "mean_boundary_fraction": 0.0,
                "log_std_min_fraction": 0.0,
                "log_std_max_fraction": 0.0,
                "normalized_return": 30.0,
                "rollout_status": "available",
            }
        )
    audit = classify_actor_terminal(
        rows, config, candidate_step=100, extension_complete=True
    )
    assert audit["state"] == "finite_terminal"
    assert not audit["numerical_nonfinite"]


def test_normalized_window_drift_uses_training_step_and_state_scale() -> None:
    rows = [
        {"step": 100, "value": 2.00},
        {"step": 200, "value": 2.01},
        {"step": 300, "value": 2.02},
    ]
    drift = normalized_window_drift(rows, "value", 3)
    assert 0.009 < drift < 0.011
