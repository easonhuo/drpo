"""C-U1 source diagnostics and causal near/far interventions.

The source diagnostic measures gradient amplification while holding negative
advantage fixed. The causal runner then intervenes on dynamically recomputed
near/far components and keeps task, support-boundary, and numerical events
separate.
"""

from __future__ import annotations

import copy
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
    make_actor,
    near_far_losses,
    positive_loss,
    support_diagnostics,
)
from .cu1_training import (
    EPS,
    CU1PositiveProtocol,
    add_gradients,
    finite_model,
    gradient_norm,
    make_adam,
    scale_gradients,
    set_parameter_gradients,
)

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
    gradients = torch.autograd.grad(
        objective,
        actor.all_parameters(),
        allow_unused=True,
    )
    return _flatten_present(gradients)


def source_diagnostic(
    *,
    seed: int,
    actor: GaussianActor,
    environment: Environment,
    protocol: CU1Protocol,
    source: CU1SourceProtocol = CU1SourceProtocol(),
) -> dict[str, float | int]:
    """Measure the equal-advantage near/far amplification ratios."""

    count = min(source.probe_states, len(environment.train.s))
    near_gradients: list[torch.Tensor] = []
    far_gradients: list[torch.Tensor] = []
    with torch.enable_grad():
        for index in range(count):
            advantage = environment.train.negative_advantages[index, 0]
            near_gradients.append(
                per_sample_negative_gradient(
                    actor,
                    environment.train.s[index],
                    environment.train.negative_actions[index, 0],
                    advantage,
                    protocol,
                )
            )
            far_gradients.append(
                per_sample_negative_gradient(
                    actor,
                    environment.train.s[index],
                    environment.train.negative_actions[index, 4],
                    advantage,
                    protocol,
                )
            )
    near = torch.stack(near_gradients)
    far = torch.stack(far_gradients)
    per_sample_ratio = far.norm(dim=1) / (near.norm(dim=1) + EPS)

    ids = torch.arange(count, device=environment.train.s.device)
    parameters = actor.all_parameters()
    near_log_probability, mu, log_std = actor_log_prob(
        actor,
        environment.train.s[ids],
        environment.train.negative_actions[ids, 0:1],
        protocol,
    )
    far_log_probability, _, _ = actor_log_prob(
        actor,
        environment.train.s[ids],
        environment.train.negative_actions[ids, 4:5],
        protocol,
    )
    near_advantage = environment.train.negative_advantages[ids, 0:1]
    far_advantage = environment.train.negative_advantages[ids, 4:5]
    aggregate_near = torch.autograd.grad(
        (near_advantage * near_log_probability).mean(),
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    aggregate_far = torch.autograd.grad(
        (far_advantage * far_log_probability).mean(),
        parameters,
        allow_unused=True,
    )

    with torch.no_grad():
        sigma = torch.exp(log_std)
        near_distance = torch.linalg.vector_norm(
            environment.train.negative_actions[ids, 0] - mu,
            dim=-1,
        )
        far_distance = torch.linalg.vector_norm(
            environment.train.negative_actions[ids, 4] - mu,
            dim=-1,
        )
        near_score = torch.sqrt(
            (near_distance / sigma.square()).square()
            + ((near_distance / sigma).square() - protocol.action_dim).square()
        )
        far_score = torch.sqrt(
            (far_distance / sigma.square()).square()
            + ((far_distance / sigma).square() - protocol.action_dim).square()
        )
        advantage_ratio = (far_advantage.abs().mean() / near_advantage.abs().mean()).item()

    return {
        "seed": seed,
        "advantage_far_near_ratio": advantage_ratio,
        "output_score_far_near_ratio": (far_score / near_score).mean().item(),
        "full_parameter_single_sample_far_near_ratio": per_sample_ratio.mean().item(),
        "full_parameter_single_sample_far_near_median_ratio": (per_sample_ratio.median().item()),
        "aggregate_far_near_ratio": (
            gradient_norm(aggregate_far) / (gradient_norm(aggregate_near) + EPS)
        ).item(),
    }


def solve_near_scale_for_budget(
    near: Sequence[torch.Tensor | None],
    far_capped: Sequence[torch.Tensor | None],
    target_norm: float,
) -> float:
    """Solve ``||c * near + far_capped||_2 = target_norm`` exactly."""

    near_flat = _flatten_present(near)
    far_flat = _flatten_present(far_capped)
    coefficient_a = torch.dot(near_flat, near_flat).item()
    coefficient_b = 2.0 * torch.dot(near_flat, far_flat).item()
    coefficient_c = torch.dot(far_flat, far_flat).item() - target_norm**2
    if coefficient_a < EPS:
        return 1.0
    discriminant = max(
        0.0,
        coefficient_b**2 - 4.0 * coefficient_a * coefficient_c,
    )
    root = math.sqrt(discriminant)
    candidates = (
        (-coefficient_b + root) / (2.0 * coefficient_a),
        (-coefficient_b - root) / (2.0 * coefficient_a),
    )
    non_negative = [value for value in candidates if value >= 0.0]
    return max(non_negative) if non_negative else 0.0


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
) -> GradientTuple:
    """Return the E3 controlled gradient."""

    parameters = actor.mean_parameters() if fixed_sigma is not None else actor.all_parameters()
    positive = positive_loss(actor, split, protocol, ids, fixed_sigma)
    near, far = near_far_losses(
        actor,
        split,
        protocol,
        ids,
        fixed_sigma,
    )
    positive_gradient = torch.autograd.grad(
        positive,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    near_gradient = torch.autograd.grad(
        near,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    far_gradient = torch.autograd.grad(
        far,
        parameters,
        allow_unused=True,
    )
    weighted_near = scale_gradients(near_gradient, alpha)
    weighted_far = scale_gradients(far_gradient, alpha)
    raw_negative = add_gradients(weighted_near, weighted_far)

    near_norm = gradient_norm(weighted_near).item()
    far_norm = gradient_norm(weighted_far).item()
    raw_norm = gradient_norm(raw_negative).item()
    far_scale = min(1.0, cap_ratio * near_norm / (far_norm + EPS))
    capped_far = scale_gradients(weighted_far, far_scale)
    capped_negative = add_gradients(weighted_near, capped_far)
    capped_norm = gradient_norm(capped_negative).item()

    if method == "baseline":
        controlled_negative = raw_negative
    elif method == "near_zero":
        controlled_negative = weighted_far
    elif method == "far_zero":
        controlled_negative = weighted_near
    elif method == "far_cap":
        controlled_negative = capped_negative
    elif method == "global_scale":
        controlled_negative = scale_gradients(
            raw_negative,
            capped_norm / (raw_norm + EPS),
        )
    elif method == "far_to_near":
        near_scale = solve_near_scale_for_budget(
            weighted_near,
            capped_far,
            raw_norm,
        )
        controlled_negative = add_gradients(
            scale_gradients(weighted_near, near_scale),
            capped_far,
        )
    else:
        raise ValueError(f"unknown E3 intervention method: {method}")

    return add_gradients(positive_gradient, controlled_negative)


def _support_event_type(support: dict[str, Any]) -> str | None:
    if not support["log_sigma_output_finite_all_states"]:
        return "nonfinite_log_sigma_output"
    if not support["sigma_output_finite_all_states"]:
        return "nonfinite_sigma_output"
    if support["support_contraction_boundary"]:
        return "support_contraction"
    if support["unexpected_support_expansion_boundary"]:
        return "unexpected_support_expansion"
    return None


def run_causal_intervention(
    *,
    seed: int,
    initialization_state: dict[str, torch.Tensor],
    environment: Environment,
    protocol: CU1Protocol,
    positive_training: CU1PositiveProtocol = CU1PositiveProtocol(),
    method: str,
    fixed_sigma: float | None,
    alpha: float,
    learning_rate: float,
    steps: int,
    branch: str,
    causal: CU1CausalProtocol = CU1CausalProtocol(),
) -> dict[str, Any]:
    """Run one C-U1 causal branch from the shared positive-only initialization."""

    actor = make_actor(protocol).to(
        environment.train.s.device,
        dtype=environment.train.s.dtype,
    )
    actor.load_state_dict(copy.deepcopy(initialization_state))
    parameters = actor.mean_parameters() if fixed_sigma is not None else actor.all_parameters()
    optimizer = make_adam(
        parameters,
        learning_rate=learning_rate,
        training=positive_training,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 300007)
    last_recorded_step = 0
    positive_reference = float(evaluation(actor, environment.test, protocol, fixed_sigma)["reward"])
    task_threshold = protocol.task_failure_retention * positive_reference
    below_threshold: deque[int] = deque(maxlen=protocol.task_failure_consecutive_evals)
    task_onset: int | None = None
    support_onset: int | None = None
    first_support_event_type: str | None = None
    stop_reason = "max_steps"

    for step in range(1, steps + 1):
        ids = torch.randint(
            0,
            protocol.n_train_states,
            (positive_training.positive_batch_states,),
            generator=generator,
        ).to(environment.train.s.device)
        gradients = intervention_gradients(
            actor,
            environment.train,
            protocol,
            ids,
            fixed_sigma=fixed_sigma,
            alpha=alpha,
            method=method,
            cap_ratio=causal.far_cap_ratio,
        )
        optimizer.zero_grad(set_to_none=True)
        set_parameter_gradients(parameters, gradients)
        optimizer.step()

        finite = finite_model(actor)
        post_support = (
            support_diagnostics(actor, environment.train, protocol) if fixed_sigma is None else {}
        )
        support_type = _support_event_type(post_support) if fixed_sigma is None else None
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
            last_recorded_step = step
        if not finite or support_type is not None:
            break

    final = evaluation(actor, environment.test, protocol, fixed_sigma)
    final_support = (
        support_diagnostics(actor, environment.train, protocol)
        if fixed_sigma is None
        else {
            "log_sigma_output_finite_all_states": True,
            "sigma_output_finite_all_states": True,
            "support_contraction_boundary": False,
            "unexpected_support_expansion_boundary": False,
        }
    )
    summary: dict[str, Any] = {
        "seed": seed,
        "method": method,
        "branch": branch,
        **final,
        "task_failure_threshold": task_threshold,
        "task_failure_onset": task_onset,
        "support_boundary_onset": support_onset,
        "support_event_type": first_support_event_type,
        "unexpected_support_expansion": (
            first_support_event_type == "unexpected_support_expansion"
        ),
        "stop_reason": stop_reason,
        "finite_parameters": finite_model(actor),
        "steps_completed": last_recorded_step,
        "task_performance_collapse": task_onset is not None,
        "support_or_probability_boundary": bool(
            final_support["support_contraction_boundary"]
            or final_support["unexpected_support_expansion_boundary"]
        ),
        "nan_inf_numerical_failure": bool(
            not finite_model(actor)
            or not final_support["log_sigma_output_finite_all_states"]
            or not final_support["sigma_output_finite_all_states"]
        ),
    }
    return summary
