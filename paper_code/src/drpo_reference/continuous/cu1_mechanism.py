"""C-U1 source diagnostics and causal near/far interventions.

The source diagnostic measures gradient amplification while holding negative
advantage fixed. The causal runner then intervenes on dynamically recomputed
near/far components and keeps task, support-boundary, and numerical events
separate.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .cu1 import (
    CU1Protocol,
    Environment,
    Split,
    actor_log_prob,
    evaluation,
    local_negative_loss,
    near_far_losses,
    negative_loss,
    positive_loss,
)
from .cu1_training import (
    EPS,
    CU1PositiveProtocol,
    add_gradients,
    finite_model,
    gradient_norm,
    gradients,
    initialized_actor,
    make_adam,
    sample_ids,
    scale_gradients,
    set_parameter_gradients,
)
from .gaussian import GaussianActor

GradientTuple = tuple[torch.Tensor | None, ...]


@dataclass(frozen=True)
class CU1SourceProtocol:
    """C-U1 source-isolation probe settings."""

    probe_states: int = 128
    seeds: tuple[int, ...] = tuple(range(10, 30))


@dataclass(frozen=True)
class CU1CausalProtocol:
    """C-U1 causal-intervention settings."""

    fixed_alpha: float = 1.40
    fixed_learning_rate: float = 1e-4
    fixed_steps: int = 2000
    learnable_alpha: float = 0.15
    learnable_learning_rate: float = 5e-4
    learnable_steps: int = 2000
    far_cap_ratio: float = 0.05
    evaluation_interval: int = 100
    seeds: tuple[int, ...] = tuple(range(30, 50))
    primary_methods: tuple[str, ...] = (
        "baseline",
        "near_zero",
        "far_zero",
        "far_cap",
    )
    appendix_methods: tuple[str, ...] = ("global_scale", "far_to_near")


def _flatten_present(gradients: Sequence[torch.Tensor | None]) -> torch.Tensor:
    present = [gradient.reshape(-1) for gradient in gradients if gradient is not None]
    if not present:
        return torch.empty(0)
    return torch.cat(present)


def per_sample_negative_gradient(
    actor,
    state: torch.Tensor,
    action: torch.Tensor,
    advantage: torch.Tensor,
    protocol: CU1Protocol,
) -> torch.Tensor:
    """Return the full-parameter gradient of ``A log pi(a|s)`` for one sample."""

    log_probability, _, _ = actor_log_prob(
        actor,
        state[None, :],
        action[None, None, :],
        protocol,
    )
    objective = advantage * log_probability.squeeze()
    return _flatten_present(gradients(objective, actor.all_parameters()))


def source_diagnostic(
    *,
    seed: int,
    actor: GaussianActor,
    environment: Environment,
    protocol: CU1Protocol,
    source: CU1SourceProtocol | None = None,
) -> dict[str, float | int]:
    """Measure equal-advantage near/far gradient amplification."""

    source = CU1SourceProtocol() if source is None else source
    split = environment.train
    count = min(source.probe_states, len(split.s))

    def sample_gradients(action_index: int) -> torch.Tensor:
        return torch.stack(
            [
                per_sample_negative_gradient(
                    actor,
                    split.s[index],
                    split.negative_actions[index, action_index],
                    split.negative_advantages[index, 0],
                    protocol,
                )
                for index in range(count)
            ]
        )

    with torch.enable_grad():
        near, far = (sample_gradients(index) for index in (0, 4))
    per_sample_ratio = far.norm(dim=1) / (near.norm(dim=1) + EPS)

    ids = torch.arange(count, device=split.s.device)
    actions = split.negative_actions[ids][:, [0, 4]]
    advantages = split.negative_advantages[ids][:, [0, 4]]
    log_probability, mu, log_std = actor_log_prob(actor, split.s[ids], actions, protocol)
    parameters = actor.all_parameters()
    aggregate_near = gradients(
        (advantages[:, 0] * log_probability[:, 0]).mean(),
        parameters,
        retain_graph=True,
    )
    aggregate_far = gradients(
        (advantages[:, 1] * log_probability[:, 1]).mean(),
        parameters,
    )

    with torch.no_grad():
        sigma = torch.exp(log_std)[:, None]
        distance = torch.linalg.vector_norm(actions - mu[:, None, :], dim=-1)
        score = torch.sqrt(
            (distance / sigma.square()).square()
            + ((distance / sigma).square() - protocol.action_dim).square()
        )
        advantage_ratio = (advantages[:, 1].abs().mean() / advantages[:, 0].abs().mean()).item()

    return {
        "seed": seed,
        "advantage_far_near_ratio": advantage_ratio,
        "output_score_far_near_ratio": (score[:, 1] / score[:, 0]).mean().item(),
        "full_parameter_single_sample_far_near_ratio": per_sample_ratio.mean().item(),
        "full_parameter_single_sample_far_near_median_ratio": per_sample_ratio.median().item(),
        "aggregate_far_near_ratio": (
            gradient_norm(aggregate_far) / (gradient_norm(aggregate_near) + EPS)
        ).item(),
    }


def intervention_gradients(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    ids: torch.Tensor,
    *,
    fixed_sigma: float | None,
    alpha: float,
    method: str,
    cap_ratio: float,
    partition: str = "dynamic",
    component_scales: tuple[float, float] | None = None,
) -> GradientTuple:
    """Return the controlled positive + near/far gradient."""

    parameters = actor.mean_parameters() if fixed_sigma is not None else actor.all_parameters()
    positive = positive_loss(actor, split, protocol, ids, fixed_sigma)
    if partition == "dynamic":
        near, far = near_far_losses(actor, split, protocol, ids, fixed_sigma)
    elif partition == "contour":
        near = local_negative_loss(actor, split, protocol, ids, fixed_sigma)
        far = negative_loss(actor, split, protocol, ids, fixed_sigma, slice(1, None))
    else:
        raise ValueError(f"unknown negative partition: {partition}")

    positive_gradient = gradients(positive, parameters, retain_graph=True)
    near_gradient = gradients(near, parameters, retain_graph=True)
    far_gradient = gradients(far, parameters)
    near_scale, far_scale = component_scales or (alpha, alpha)
    weighted_near = scale_gradients(near_gradient, near_scale)
    weighted_far = scale_gradients(far_gradient, far_scale)
    raw = add_gradients(weighted_near, weighted_far)
    near_norm = gradient_norm(weighted_near).item()
    far_norm = gradient_norm(weighted_far).item()
    raw_norm = gradient_norm(raw).item()
    far_cap = min(1.0, cap_ratio * near_norm / (far_norm + EPS))
    capped_far = scale_gradients(weighted_far, far_cap)
    capped = add_gradients(weighted_near, capped_far)

    if method in {"baseline", "uncontrolled_all"}:
        controlled = raw
    elif method == "near_zero":
        controlled = weighted_far
    elif method == "far_zero":
        controlled = weighted_near
    elif method == "far_cap":
        controlled = capped
    elif method in {"global_scale", "budget_matched_global"}:
        controlled = scale_gradients(raw, gradient_norm(capped).item() / (raw_norm + EPS))
    elif method == "far_to_near":
        near_flat = _flatten_present(weighted_near)
        far_flat = _flatten_present(capped_far)
        a = torch.dot(near_flat, near_flat).item()
        b = 2.0 * torch.dot(near_flat, far_flat).item()
        c = torch.dot(far_flat, far_flat).item() - raw_norm**2
        if a < EPS:
            multiplier = 1.0
        else:
            root = math.sqrt(max(0.0, b**2 - 4.0 * a * c))
            candidates = ((-b + root) / (2.0 * a), (-b - root) / (2.0 * a))
            multiplier = max((value for value in candidates if value >= 0.0), default=0.0)
        controlled = add_gradients(
            scale_gradients(weighted_near, multiplier),
            capped_far,
        )
    else:
        raise ValueError(f"unknown negative-control method: {method}")
    return add_gradients(positive_gradient, controlled)


def run_causal_intervention(
    *,
    seed: int,
    initialization_state: dict[str, torch.Tensor],
    environment: Environment,
    protocol: CU1Protocol,
    positive_training: CU1PositiveProtocol | None = None,
    method: str,
    fixed_sigma: float | None,
    alpha: float,
    learning_rate: float,
    steps: int,
    branch: str,
    causal: CU1CausalProtocol | None = None,
    partition: str = "dynamic",
    component_scales: tuple[float, float] | None = None,
    cap_ratio: float | None = None,
    generator_offset: int = 300007,
) -> dict[str, Any]:
    """Run one C-U1 causal branch from the shared positive-only initialization."""

    positive_training = CU1PositiveProtocol() if positive_training is None else positive_training
    causal = CU1CausalProtocol() if causal is None else causal
    actor = initialized_actor(protocol, environment, initialization_state)
    parameters = actor.mean_parameters() if fixed_sigma is not None else actor.all_parameters()
    optimizer = make_adam(
        parameters,
        learning_rate=learning_rate,
        training=positive_training,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + generator_offset)
    positive_reference = float(evaluation(actor, environment.test, protocol, fixed_sigma)["reward"])
    task_threshold = protocol.task_failure_retention * positive_reference
    below_threshold: deque[int] = deque(maxlen=protocol.task_failure_consecutive_evals)
    task_onset: int | None = None
    support_onset: int | None = None
    first_support_event_type: str | None = None
    stop_reason = "max_steps"

    for step in range(1, steps + 1):
        ids = sample_ids(
            generator,
            environment.train,
            positive_training.positive_batch_states,
        )
        gradients = intervention_gradients(
            actor,
            environment.train,
            protocol,
            ids,
            fixed_sigma=fixed_sigma,
            alpha=alpha,
            method=method,
            cap_ratio=causal.far_cap_ratio if cap_ratio is None else cap_ratio,
            partition=partition,
            component_scales=component_scales,
        )
        optimizer.zero_grad(set_to_none=True)
        set_parameter_gradients(parameters, gradients)
        optimizer.step()

        finite = finite_model(actor)
        post_support = evaluation(actor, environment.train, protocol) if fixed_sigma is None else {}
        support_type = post_support["support_event_type"] if fixed_sigma is None else None
        if support_type is not None and support_onset is None:
            support_onset = step
            first_support_event_type = support_type
        if not finite:
            stop_reason = "non_finite_parameter"
        elif support_type is not None:
            stop_reason = f"{support_type}_boundary_event"

        should_record = (
            step % causal.evaluation_interval == 0
            or step == 1
            or step == steps
            or support_type is not None
            or not finite
        )
        if should_record:
            metrics = evaluation(
                actor,
                environment.test,
                protocol,
                fixed_sigma,
            )
            reward = float(metrics["reward"])
            if reward < task_threshold:
                below_threshold.append(step)
            else:
                below_threshold.clear()
            if (
                len(below_threshold) == protocol.task_failure_consecutive_evals
                and task_onset is None
            ):
                task_onset = below_threshold[0]
        if not finite or support_type is not None:
            break

    final = evaluation(actor, environment.test, protocol, fixed_sigma)
    final_support = evaluation(actor, environment.train, protocol) if fixed_sigma is None else None
    finite_parameters = finite_model(actor)
    summary: dict[str, Any] = {
        "seed": seed,
        "method": method,
        "branch": branch,
        **final,
        "task_failure_onset": task_onset,
        "support_boundary_onset": support_onset,
        "support_event_type": first_support_event_type,
        "stop_reason": stop_reason,
        "finite_parameters": finite_parameters,
        "steps_completed": step,
        "task_performance_collapse": task_onset is not None,
        "support_or_variance_boundary_event": bool(
            final_support and final_support["support_or_variance_boundary_event"]
        ),
        "nan_inf_numerical_event": bool(
            not finite_parameters or (final_support and final_support["nan_inf_numerical_event"])
        ),
    }
    return summary
