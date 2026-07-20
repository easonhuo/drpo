"""C-U1 distance-taper comparison and terminal audit.

The implementation keeps the frozen C-U1 environment, initialization,
advantages, optimizer, and minibatch stream fixed. Methods differ only in the
detached negative-sample weight applied to standardized action distance.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from drpo_reference.common import seed_all
from drpo_reference.controls import (
    TaperFamily,
    point_retention_coefficient,
    taper_weight,
)

from .cu1 import (
    CU1Protocol,
    Environment,
    Split,
    actor_log_prob,
    evaluation,
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
)
from .gaussian import (
    GaussianActor,
    gaussian_output_components,
    standardized_distance,
)

GradientTuple = tuple[torch.Tensor | None, ...]


@dataclass(frozen=True)
class CU1TaperProtocol:
    """Frozen C-U1-E4-TAPER-01 settings."""

    formal_seeds: tuple[int, ...] = tuple(range(70, 90))
    reference_distance: float = 5.0
    primary_retention: float = 0.25
    sensitivity_retentions: tuple[float, ...] = (0.50, 0.75)
    negative_alpha: float = 1.0
    learning_rate: float = 5e-4
    batch_states: int = 256
    evaluation_interval: int = 100
    minimum_steps: int = 1000
    maximum_steps: int = 8000
    stable_windows: int = 10
    normalized_slope_threshold: float = 1e-4
    normalized_field_residual_threshold: float = 2e-3
    positive_absolute_gradient_threshold: float = 1e-3
    task_failure_retention: float = 0.45
    probe_states: int = 64

    def __post_init__(self) -> None:
        if self.reference_distance <= 0.0:
            raise ValueError("reference_distance must be positive")
        for value in (self.primary_retention, *self.sensitivity_retentions):
            if not 0.0 < value <= 1.0:
                raise ValueError("retention values must lie in (0, 1]")
        if self.negative_alpha < 0.0:
            raise ValueError("negative_alpha must be non-negative")
        for name in (
            "batch_states",
            "evaluation_interval",
            "minimum_steps",
            "maximum_steps",
            "stable_windows",
            "probe_states",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.minimum_steps > self.maximum_steps:
            raise ValueError("minimum_steps must not exceed maximum_steps")


@dataclass
class TaperRun:
    """One in-memory taper run and its scientific outputs."""

    actor: GaussianActor
    trajectory: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    summary: dict[str, Any]


def method_configs(protocol: CU1TaperProtocol) -> list[tuple[str, float]]:
    configs = [
        ("positive_only", 1.0),
        ("unweighted", 1.0),
        ("reciprocal_linear", protocol.primary_retention),
        ("reciprocal_quadratic", protocol.primary_retention),
        ("exponential", protocol.primary_retention),
    ]
    for retention in protocol.sensitivity_retentions:
        for family in (
            "reciprocal_linear",
            "reciprocal_quadratic",
            "exponential",
        ):
            configs.append((family, retention))
    return configs


def config_name(family: str, retention: float) -> str:
    if family in {"positive_only", "unweighted"}:
        return family
    return f"{family}_rho{retention:.2f}"


def _canonical_family(family: str) -> TaperFamily:
    mapping = {
        "positive_only": TaperFamily.POSITIVE_ONLY,
        "unweighted": TaperFamily.UNCONTROLLED,
        "reciprocal_linear": TaperFamily.RECIPROCAL_LINEAR,
        "reciprocal_quadratic": TaperFamily.RECIPROCAL_QUADRATIC,
        "exponential": TaperFamily.EXPONENTIAL_LINEAR,
    }
    try:
        return mapping[family]
    except KeyError as exc:
        raise ValueError(f"unknown C-U1 taper family: {family}") from exc


def retention_weight(
    distance: torch.Tensor,
    *,
    family: str,
    retention: float,
    protocol: CU1TaperProtocol,
) -> torch.Tensor:
    """Return the detached weight with ``w(reference_distance)=retention``."""

    canonical = _canonical_family(family)
    if canonical in {TaperFamily.POSITIVE_ONLY, TaperFamily.UNCONTROLLED}:
        return taper_weight(distance, family=canonical, detach_distance=True)
    coefficient = point_retention_coefficient(
        canonical,
        retention=retention,
        reference_distance=protocol.reference_distance,
    )
    return taper_weight(
        distance,
        family=canonical,
        coefficient=coefficient,
        detach_distance=True,
    )


def weighted_negative_loss(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    taper: CU1TaperProtocol,
    ids: torch.Tensor | None,
    *,
    family: str,
    retention: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    states = split.s if ids is None else split.s[ids]
    actions = split.negative_actions if ids is None else split.negative_actions[ids]
    advantages = split.negative_advantages if ids is None else split.negative_advantages[ids]
    log_probability, mu, log_std = actor_log_prob(
        actor,
        states,
        actions,
        protocol,
    )
    distance = standardized_distance(mu, log_std, actions)
    weight = retention_weight(
        distance,
        family=family,
        retention=retention,
        protocol=taper,
    )
    loss = -(advantages * weight * log_probability).mean()
    with torch.no_grad():
        near = distance <= taper.reference_distance
        far = ~near
        diagnostics = {
            "weight_mean": float(weight.mean().item()),
            "near_weight_mean": (
                float(weight[near].mean().item()) if bool(near.any()) else float("nan")
            ),
            "far_weight_mean": (
                float(weight[far].mean().item()) if bool(far.any()) else float("nan")
            ),
            "standardized_distance_mean": float(distance.mean().item()),
            "near_occupancy": float(near.float().mean().item()),
            "far_occupancy": float(far.float().mean().item()),
        }
    return loss, diagnostics


def _gradient_tuple(
    loss: torch.Tensor,
    actor: GaussianActor,
    *,
    retain_graph: bool,
) -> GradientTuple:
    return tuple(
        torch.autograd.grad(
            loss,
            actor.all_parameters(),
            retain_graph=retain_graph,
            allow_unused=True,
        )
    )


def full_field_diagnostics(
    actor: GaussianActor,
    split: Split,
    protocol: CU1Protocol,
    taper: CU1TaperProtocol,
    *,
    family: str,
    retention: float,
) -> dict[str, Any]:
    positive = positive_loss(actor, split, protocol)
    positive_gradient = _gradient_tuple(
        positive,
        actor,
        retain_graph=family != "positive_only",
    )
    positive_norm = float(gradient_norm(positive_gradient).item())
    if family == "positive_only":
        return {
            "positive_gradient_norm": positive_norm,
            "negative_gradient_norm": 0.0,
            "total_gradient_norm": positive_norm,
            "normalized_field_residual": float("nan"),
            "stationarity_residual": positive_norm,
            "stationarity_residual_kind": "absolute_positive_gradient_norm",
        }
    negative, _ = weighted_negative_loss(
        actor,
        split,
        protocol,
        taper,
        None,
        family=family,
        retention=retention,
    )
    negative_gradient = _gradient_tuple(negative, actor, retain_graph=False)
    total_gradient = add_gradients(
        positive_gradient,
        negative_gradient,
        scales=(1.0, taper.negative_alpha),
    )
    negative_norm = float(gradient_norm(negative_gradient).item())
    total_norm = float(gradient_norm(total_gradient).item())
    residual = total_norm / (positive_norm + taper.negative_alpha * negative_norm + EPS)
    return {
        "positive_gradient_norm": positive_norm,
        "negative_gradient_norm": negative_norm,
        "total_gradient_norm": total_norm,
        "normalized_field_residual": residual,
        "stationarity_residual": residual,
        "stationarity_residual_kind": "normalized_signed_field_residual",
    }


def two_times_audit_target(candidate_step: int, maximum_steps: int) -> int | None:
    target = 2 * candidate_step
    return target if target <= maximum_steps else None


def normalized_slope(rows: list[dict[str, Any]], field: str) -> float:
    steps = np.asarray([float(row["step"]) for row in rows], dtype=float)
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return float("inf")
    slope = float(np.polyfit(steps, values, 1)[0])
    scale = max(float(np.mean(np.abs(values))), 1e-8)
    return abs(slope) / scale


def evaluate_taper_state(
    actor: GaussianActor,
    environment: Environment,
    protocol: CU1Protocol,
    taper: CU1TaperProtocol,
    *,
    family: str,
    retention: float,
    initial_reward: float,
) -> dict[str, Any]:
    task = evaluation(actor, environment.test, protocol)
    field = full_field_diagnostics(
        actor,
        environment.train,
        protocol,
        taper,
        family=family,
        retention=retention,
    )
    finite_parameters = finite_model(actor)
    support = support_diagnostics(actor, environment.train, protocol)
    numerical = bool(
        not finite_parameters
        or not support["log_sigma_output_finite_all_states"]
        or not support["sigma_output_finite_all_states"]
    )
    boundary = bool(
        support["support_contraction_boundary"] or support["unexpected_support_expansion_boundary"]
    )
    task_failure = bool(float(task["reward"]) < taper.task_failure_retention * initial_reward)
    return {
        **task,
        **field,
        "task_performance_collapse_event": task_failure,
        "support_or_variance_boundary_event": boundary,
        "nan_inf_numerical_event": numerical,
        "environment_invalid_event": False,
    }


def per_sample_gradient_diagnostic(
    *,
    seed: int,
    stage: str,
    actor: GaussianActor,
    environment: Environment,
    protocol: CU1Protocol,
    taper: CU1TaperProtocol,
    family: str,
    retention: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    actor.eval()
    count = min(taper.probe_states, len(environment.train.s))
    for state_index in range(count):
        state = environment.train.s[state_index : state_index + 1]
        actions = environment.train.negative_actions[state_index]
        advantages = environment.train.negative_advantages[state_index]
        mu, log_std = actor(state)
        for contour_index in range(actions.shape[0]):
            action = actions[contour_index : contour_index + 1]
            log_probability, _, _ = actor_log_prob(
                actor,
                state,
                action[None, :, :],
                protocol,
            )
            objective = advantages[contour_index] * log_probability.squeeze()
            gradients = torch.autograd.grad(
                objective,
                actor.all_parameters(),
                allow_unused=True,
            )
            raw_norm = float(gradient_norm(gradients).item())
            components = gaussian_output_components(
                mu,
                log_std,
                action[None, :, :],
                protocol.action_dim,
            )
            distance = float(components["standardized_distance"].item())
            advantage_abs = abs(float(advantages[contour_index].item()))
            raw_output_mean = advantage_abs * float(components["mean_score"].item())
            raw_output_log_scale = advantage_abs * abs(float(components["log_scale_score"].item()))
            raw_output_joint = advantage_abs * float(components["joint_score"].item())
            weight = float(
                retention_weight(
                    torch.tensor(distance),
                    family=family,
                    retention=retention,
                    protocol=taper,
                ).item()
            )
            rows.append(
                {
                    "seed": seed,
                    "stage": stage,
                    "family": family,
                    "rho": retention,
                    "state_index": state_index,
                    "contour_index": contour_index,
                    "advantage": float(advantages[contour_index].item()),
                    "standardized_distance": distance,
                    "weight": weight,
                    "raw_output_mean_gradient_norm": raw_output_mean,
                    "raw_output_log_scale_gradient_abs": raw_output_log_scale,
                    "raw_output_joint_gradient_norm": raw_output_joint,
                    "weighted_output_mean_gradient_norm": weight * raw_output_mean,
                    "weighted_output_log_scale_gradient_abs": weight * raw_output_log_scale,
                    "weighted_output_joint_gradient_norm": weight * raw_output_joint,
                    "raw_full_parameter_gradient_norm": raw_norm,
                    "weighted_full_parameter_gradient_norm": weight * raw_norm,
                }
            )

    def ratio_and_slope(field: str) -> tuple[float, float]:
        near = [row[field] for row in rows if row["contour_index"] == 0]
        far = [row[field] for row in rows if row["contour_index"] == 4]
        far_region = [
            row
            for row in rows
            if row["standardized_distance"] >= taper.reference_distance and row[field] > 0.0
        ]
        ratio = float(np.mean(far) / (np.mean(near) + EPS)) if near and far else float("nan")
        if len(far_region) < 2:
            return ratio, float("nan")
        x = np.log(
            np.asarray(
                [row["standardized_distance"] for row in far_region],
                dtype=float,
            )
        )
        y = np.log(np.asarray([row[field] for row in far_region], dtype=float))
        return ratio, float(np.polyfit(x, y, 1)[0])

    full_ratio, full_slope = ratio_and_slope("weighted_full_parameter_gradient_norm")
    output_ratio, output_slope = ratio_and_slope("weighted_output_joint_gradient_norm")
    return rows, {
        "far_near_weighted_gradient_ratio": full_ratio,
        "far_loglog_slope": full_slope,
        "far_near_weighted_output_joint_ratio": output_ratio,
        "far_loglog_weighted_output_joint_slope": output_slope,
        "probe_rows": len(rows),
    }


def run_taper_method(
    *,
    seed: int,
    initialization_state: dict[str, torch.Tensor],
    environment: Environment,
    protocol: CU1Protocol,
    positive_training: CU1PositiveProtocol = CU1PositiveProtocol(),
    taper: CU1TaperProtocol = CU1TaperProtocol(),
    family: str,
    retention: float,
) -> TaperRun:
    """Run one taper branch from the exact positive-only Adam checkpoint."""

    seed_all(seed + 900_000)
    actor = GaussianActor(
        state_dim=protocol.state_dim,
        action_dim=protocol.action_dim,
        hidden_dim=protocol.hidden_dim,
        initial_sigma=protocol.initial_sigma,
    ).to(environment.train.s.device, dtype=environment.train.s.dtype)
    actor.load_state_dict(copy.deepcopy(initialization_state))
    optimizer = make_adam(
        actor.all_parameters(),
        learning_rate=taper.learning_rate,
        training=positive_training,
    )
    index_generator = torch.Generator(device="cpu").manual_seed(seed + 700_003)
    initial_reward = float(evaluation(actor, environment.test, protocol)["reward"])
    diagnostic_rows, initial_diagnostic = per_sample_gradient_diagnostic(
        seed=seed,
        stage="initial",
        actor=actor,
        environment=environment,
        protocol=protocol,
        taper=taper,
        family=family,
        retention=retention,
    )
    trajectory: list[dict[str, Any]] = []
    stable_candidate_step: int | None = None
    audit_target_step: int | None = None
    candidate_classification: tuple[bool, bool, bool] | None = None
    stop_reason = "maximum_steps"

    for step in range(1, taper.maximum_steps + 1):
        ids = torch.randint(
            0,
            protocol.n_train_states,
            (taper.batch_states,),
            generator=index_generator,
        ).to(environment.train.s.device)
        positive = positive_loss(actor, environment.train, protocol, ids)
        if family == "positive_only":
            loss = positive
        else:
            negative, _ = weighted_negative_loss(
                actor,
                environment.train,
                protocol,
                taper,
                ids,
                family=family,
                retention=retention,
            )
            loss = positive + taper.negative_alpha * negative
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        should_evaluate = (
            step == 1
            or step % taper.evaluation_interval == 0
            or step == taper.maximum_steps
            or (audit_target_step is not None and step >= audit_target_step)
        )
        if not should_evaluate:
            continue
        state = evaluate_taper_state(
            actor,
            environment,
            protocol,
            taper,
            family=family,
            retention=retention,
            initial_reward=initial_reward,
        )
        trajectory.append(
            {
                "seed": seed,
                "step": step,
                "family": family,
                "rho": retention,
                "loss": float(loss.detach().item()),
                **state,
            }
        )
        if state["nan_inf_numerical_event"]:
            stop_reason = "nan_inf_numerical_event"
            break
        if state["support_or_variance_boundary_event"]:
            stop_reason = "support_or_variance_boundary_event"
            break
        if (
            stable_candidate_step is None
            and step >= taper.minimum_steps
            and len(trajectory) >= taper.stable_windows
        ):
            window = trajectory[-taper.stable_windows :]
            max_slope = max(
                normalized_slope(window, field)
                for field in (
                    "reward",
                    "normalized_extrapolation_displacement",
                    "sigma_mean",
                )
            )
            residual = float(state["stationarity_residual"])
            residual_threshold = (
                taper.positive_absolute_gradient_threshold
                if family == "positive_only"
                else taper.normalized_field_residual_threshold
            )
            if max_slope < taper.normalized_slope_threshold and residual < residual_threshold:
                stable_candidate_step = step
                audit_target_step = two_times_audit_target(
                    step,
                    taper.maximum_steps,
                )
                if audit_target_step is not None:
                    candidate_classification = (
                        bool(state["task_performance_collapse_event"]),
                        bool(state["support_or_variance_boundary_event"]),
                        bool(state["nan_inf_numerical_event"]),
                    )
        if audit_target_step is not None and step >= audit_target_step:
            terminal_classification = (
                bool(state["task_performance_collapse_event"]),
                bool(state["support_or_variance_boundary_event"]),
                bool(state["nan_inf_numerical_event"]),
            )
            stop_reason = (
                "stable_plateau_2x_confirmed"
                if terminal_classification == candidate_classification
                else "terminal_classification_reversed"
            )
            break

    final_state = evaluate_taper_state(
        actor,
        environment,
        protocol,
        taper,
        family=family,
        retention=retention,
        initial_reward=initial_reward,
    )
    terminal_rows, terminal_diagnostic = per_sample_gradient_diagnostic(
        seed=seed,
        stage="terminal",
        actor=actor,
        environment=environment,
        protocol=protocol,
        taper=taper,
        family=family,
        retention=retention,
    )
    diagnostic_rows.extend(terminal_rows)
    completed_steps = int(trajectory[-1]["step"]) if trajectory else 0
    summary: dict[str, Any] = {
        "seed": seed,
        "family": family,
        "rho": retention,
        "steps_completed": completed_steps,
        "stop_reason": stop_reason,
        "stable_candidate_step": stable_candidate_step,
        "audit_target_step": audit_target_step,
        "initial_reward": initial_reward,
        **final_state,
        "initial_far_near_weighted_gradient_ratio": initial_diagnostic[
            "far_near_weighted_gradient_ratio"
        ],
        "terminal_far_near_weighted_gradient_ratio": terminal_diagnostic[
            "far_near_weighted_gradient_ratio"
        ],
        "initial_far_loglog_slope": initial_diagnostic["far_loglog_slope"],
        "terminal_far_loglog_slope": terminal_diagnostic["far_loglog_slope"],
        "initial_far_near_weighted_output_joint_ratio": initial_diagnostic[
            "far_near_weighted_output_joint_ratio"
        ],
        "terminal_far_near_weighted_output_joint_ratio": terminal_diagnostic[
            "far_near_weighted_output_joint_ratio"
        ],
        "initial_far_loglog_weighted_output_joint_slope": initial_diagnostic[
            "far_loglog_weighted_output_joint_slope"
        ],
        "terminal_far_loglog_weighted_output_joint_slope": terminal_diagnostic[
            "far_loglog_weighted_output_joint_slope"
        ],
    }
    return TaperRun(
        actor=actor,
        trajectory=trajectory,
        diagnostics=diagnostic_rows,
        summary=summary,
    )
