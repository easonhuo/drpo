"""D-U1 shared initialization and six-method training loop."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from drpo_reference.common import seed_all

from .du1_controls import (
    coordinate_calibration,
    negative_loss,
    rarity_logit_anchor_loss,
)
from .du1_environment import CartesianSemanticEnvironment
from .du1_metrics import evaluate
from .du1_policy import (
    CartesianPolicy,
    batch_indices,
    cache_reference_directions,
    cell_log_probs,
    trainable_parameters,
)
from .du1_protocol import DU1Protocol, MethodSpec


def move_environment(
    environment: CartesianSemanticEnvironment,
    device: torch.device,
) -> None:
    environment.action_embeddings = environment.action_embeddings.to(device)
    for split in (environment.train, environment.test):
        for key, value in list(split.items()):
            if isinstance(value, torch.Tensor):
                split[key] = value.to(device)


def build_shared_start(
    protocol: DU1Protocol,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Build the common model shared by every method."""

    seed_all(seed)
    environment = CartesianSemanticEnvironment(protocol, seed)
    model = CartesianPolicy(protocol, environment).to(device)
    move_environment(environment, device)
    cache_reference_directions(model, environment)
    return (
        {name: value.detach().clone() for name, value in model.state_dict().items()},
        coordinate_calibration(model, environment, protocol),
    )


def run_method(
    *,
    protocol: DU1Protocol,
    seed: int,
    spec: MethodSpec,
    base_state: Mapping[str, torch.Tensor],
    calibration: Mapping[str, float],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train one method from the shared initialization."""

    seed_all(seed)
    environment = CartesianSemanticEnvironment(protocol, seed)
    model = CartesianPolicy(protocol, environment).to(device)
    model.load_state_dict(base_state)
    move_environment(environment, device)
    cache_reference_directions(model, environment)
    optimizer = torch.optim.Adam(
        trainable_parameters(model),
        lr=protocol.learning_rate,
        betas=(protocol.adam_beta1, protocol.adam_beta2),
        eps=protocol.adam_eps,
    )

    trajectory: list[dict[str, Any]] = []

    def record(step: int) -> None:
        metrics = evaluate(
            model,
            environment,
            environment.test,
            calibration,
        )
        trajectory.append(
            {
                "seed": seed,
                "method": spec.method,
                "step": step,
                **metrics,
            }
        )

    record(0)
    for step in range(1, protocol.maximum_steps + 1):
        index = batch_indices(
            seed,
            step,
            environment.train_count,
            protocol.batch_size,
        ).to(device)
        states = environment.train["states"][index]
        positive_log_probability, cells, _ = cell_log_probs(
            model,
            environment,
            environment.train,
            index,
        )
        positive_loss = -positive_log_probability.mean()
        negative = negative_loss(
            cells=cells,
            spec=spec,
            calibration=calibration,
            protocol=protocol,
            model=model,
        )
        anchor = rarity_logit_anchor_loss(model, states)
        loss = (
            positive_loss
            + protocol.negative_alpha * negative
            + protocol.rarity_logit_anchor_coefficient * anchor
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % protocol.evaluation_interval_steps == 0 or step == protocol.maximum_steps:
            record(step)

    final = trajectory[-1]
    summary = {
        "seed": seed,
        "method": spec.method,
        "steps_completed": int(final["step"]),
        "final_expected_semantic_reward": float(final["expected_semantic_reward"]),
        "final_hidden_optimal_family_probability": float(
            final["hidden_optimal_family_probability"]
        ),
        "final_action_effective_support": float(final["action_effective_support"]),
        "final_prototype_effective_support": float(final["prototype_effective_support"]),
        "final_rare_total_probability": float(final["rare_total_probability"]),
        "final_rarity_logit_gap_mean": float(final["rarity_logit_gap_mean"]),
    }
    return trajectory, summary
