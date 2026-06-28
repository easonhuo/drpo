#!/usr/bin/env python3
"""Formal C-U1 near-retention-matched taper-family comparison.

Experiment ID: C-U1-E4-TAPER-NEAR-RETENTION-01.

This is the first v60 E4-TAPER fairness follow-up.  Every taper coefficient is
calibrated once, using only development seeds and the frozen 2000-step
positive-only Adam checkpoint, so that the average weight over the preregistered
near region is the same.  The calibrated coefficients are then frozen for all
formal seeds and all training steps.

The experiment does not extend or rewrite C-U1-E4-TAPER-01.  It uses the shared
C-U1 environment and actor, reports same-distribution held-out-context metrics,
and keeps task-performance collapse, support/variance-boundary events, and
NaN/Inf numerical failure separate.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

try:
    from . import cu1_core
    from . import cu1_distance_taper_formal as taper_base
    from . import drpo_cu1_e1_e4_oneclick as experiment
except ImportError:  # direct script execution
    import cu1_core
    import cu1_distance_taper_formal as taper_base
    import drpo_cu1_e1_e4_oneclick as experiment


EXPERIMENT_ID = "C-U1-E4-TAPER-NEAR-RETENTION-01"
SCRIPT_VERSION = "2026.06.28-near-retention-formal-v1"
EPS = 1e-12


@dataclass(frozen=True)
class NearRetentionProtocol:
    development_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    formal_seeds: tuple[int, ...] = tuple(range(90, 110))
    reference_distance: float = 5.0
    near_region_boundary: float = 5.0
    primary_retention: float = 0.75
    sensitivity_retentions: tuple[float, ...] = (0.50, 0.25)
    families: tuple[str, ...] = (
        "reciprocal_linear",
        "reciprocal_quadratic",
        "exponential",
        "squared_distance_exponential",
    )
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
    log_sigma_event_boundary: float = 12.0
    calibration_tolerance: float = 1e-6
    calibration_bisection_steps: int = 100
    calibration_max_coefficient: float = 1e8
    probe_states: int = 16
    bootstrap_samples: int = 4000
    checkpoint_every_formal_seeds: int = 5
    utility_bin_edges: tuple[float, ...] = (0.0, 2.5, 5.0, 7.5, 10.0, float("inf"))
    normalized_utility_kappa: float = 1.0

    @property
    def retention_levels(self) -> tuple[float, ...]:
        return (self.primary_retention, *self.sensitivity_retentions)


PROTOCOL = NearRetentionProtocol()


def atomic_json(path: Path, value: Any) -> None:
    taper_base.atomic_json(path, value)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    taper_base.write_csv(path, rows)


def log_message(output_root: Path, message: str) -> None:
    taper_base.log_message(output_root, message)


def config_name(family: str, target_retention: float | None) -> str:
    if family in {"positive_only", "unweighted"}:
        return family
    assert target_retention is not None
    return f"{family}_near{target_retention:.2f}"


def method_configs(
    protocol: NearRetentionProtocol,
) -> list[tuple[str, float | None]]:
    configs: list[tuple[str, float | None]] = [
        ("positive_only", None),
        ("unweighted", None),
    ]
    for retention in protocol.retention_levels:
        for family in protocol.families:
            configs.append((family, retention))
    return configs


def taper_weight_from_coefficient(
    standardized_distance: torch.Tensor,
    family: str,
    coefficient: float,
    reference_distance: float,
) -> torch.Tensor:
    """Evaluate one frozen taper family with an explicit coefficient."""
    if coefficient < 0.0 or not math.isfinite(coefficient):
        raise ValueError("coefficient must be finite and non-negative")
    u = standardized_distance / reference_distance
    if family == "reciprocal_linear":
        return 1.0 / (1.0 + coefficient * u)
    if family == "reciprocal_quadratic":
        return 1.0 / (1.0 + coefficient * u.square())
    if family == "exponential":
        return torch.exp(-coefficient * u)
    if family == "squared_distance_exponential":
        return torch.exp(-coefficient * u.square())
    if family == "unweighted":
        return torch.ones_like(standardized_distance)
    if family == "positive_only":
        return torch.zeros_like(standardized_distance)
    raise ValueError(f"unknown taper family: {family}")


def solve_matching_coefficient(
    near_distances: torch.Tensor,
    family: str,
    target_retention: float,
    protocol: NearRetentionProtocol,
) -> tuple[float, float]:
    """Deterministically solve E[w(d)|near] == target by bisection."""
    if not (0.0 < target_retention < 1.0):
        raise ValueError("target_retention must lie strictly between zero and one")
    values = near_distances.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    if values.numel() == 0:
        raise RuntimeError("calibration near region is empty")
    if not torch.isfinite(values).all() or torch.any(values < 0):
        raise RuntimeError("calibration distances must be finite and non-negative")

    def mean_weight(coefficient: float) -> float:
        weight = taper_weight_from_coefficient(
            values,
            family=family,
            coefficient=coefficient,
            reference_distance=protocol.reference_distance,
        )
        return float(weight.mean().item())

    low = 0.0
    high = 1.0
    while mean_weight(high) > target_retention:
        high *= 2.0
        if high > protocol.calibration_max_coefficient:
            raise RuntimeError(
                f"could not bracket coefficient for {family} at {target_retention}"
            )
    for _ in range(protocol.calibration_bisection_steps):
        middle = 0.5 * (low + high)
        if mean_weight(middle) > target_retention:
            low = middle
        else:
            high = middle
    coefficient = 0.5 * (low + high)
    achieved = mean_weight(coefficient)
    if abs(achieved - target_retention) > protocol.calibration_tolerance:
        raise RuntimeError(
            f"calibration mismatch for {family}: target={target_retention} "
            f"achieved={achieved}"
        )
    return coefficient, achieved


def collect_calibration_distances(
    output_root: Path,
    protocol: NearRetentionProtocol,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    pooled: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for seed in protocol.development_seeds:
        log_message(output_root, f"calibration development seed={seed}")
        _, environment, _, positive_summary = experiment.train_positive(seed)
        initial_state = copy.deepcopy(experiment.load_initialization_state(seed))
        actor = experiment.GaussianActor().to(experiment.DEVICE)
        actor.load_state_dict(initial_state)
        actor.eval()
        with torch.no_grad():
            mu, log_std = actor(environment.train.s)
            distance = cu1_core.standardized_distance(
                mu, log_std, environment.train.negative_actions
            )
            near = distance <= protocol.near_region_boundary
            near_values = distance[near].detach().cpu()
        if near_values.numel() == 0:
            raise RuntimeError(f"development seed {seed} has no near negatives")
        pooled.append(near_values)
        rows.append(
            {
                "seed": seed,
                "positive_checkpoint_reward": float(positive_summary["reward"]),
                "negative_samples": int(distance.numel()),
                "near_samples": int(near.sum().item()),
                "near_fraction": float(near.float().mean().item()),
                "near_distance_mean": float(near_values.mean().item()),
                "near_distance_min": float(near_values.min().item()),
                "near_distance_max": float(near_values.max().item()),
            }
        )
    return torch.cat(pooled), rows


def calibrate_families(
    output_root: Path,
    protocol: NearRetentionProtocol,
) -> tuple[dict[tuple[str, float], float], dict[str, Any]]:
    distances, seed_rows = collect_calibration_distances(output_root, protocol)
    coefficients: dict[tuple[str, float], float] = {}
    calibration_rows: list[dict[str, Any]] = []
    for retention in protocol.retention_levels:
        for family in protocol.families:
            coefficient, achieved = solve_matching_coefficient(
                distances, family, retention, protocol
            )
            coefficients[(family, retention)] = coefficient
            calibration_rows.append(
                {
                    "family": family,
                    "target_retention": retention,
                    "coefficient": coefficient,
                    "achieved_near_retention": achieved,
                    "absolute_error": abs(achieved - retention),
                }
            )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "calibration_scope": "development_seeds_only",
        "development_seeds": list(protocol.development_seeds),
        "formal_or_confirmatory_seed_accessed": False,
        "checkpoint": "positive_only_adam_2000_step_checkpoint",
        "near_region": {
            "distance": "standardized Gaussian distance",
            "boundary": protocol.near_region_boundary,
            "predicate": "d <= 5.0",
        },
        "matching_target": "pooled E[w(d) | d <= 5.0]",
        "near_distance_count": int(distances.numel()),
        "near_distance_mean": float(distances.mean().item()),
        "near_distance_min": float(distances.min().item()),
        "near_distance_max": float(distances.max().item()),
        "seed_rows": seed_rows,
        "calibrations": calibration_rows,
        "coefficient_freeze": "fixed_for_all_formal_seeds_and_all_training_steps",
    }
    atomic_json(output_root / "calibration.json", payload)
    write_csv(output_root / "calibration.csv", calibration_rows)
    return coefficients, payload


def coefficient_for(
    family: str,
    target_retention: float | None,
    coefficients: dict[tuple[str, float], float],
) -> float:
    if family == "positive_only":
        return 0.0
    if family == "unweighted":
        return 0.0
    if target_retention is None:
        raise ValueError(f"missing retention target for {family}")
    return coefficients[(family, target_retention)]


def weighted_negative_loss(
    actor: experiment.GaussianActor,
    split: experiment.Split,
    ids: torch.Tensor | None,
    family: str,
    coefficient: float,
    protocol: NearRetentionProtocol,
) -> tuple[torch.Tensor, dict[str, float]]:
    if ids is None:
        states = split.s
        actions = split.negative_actions
        advantages = split.negative_advantages
    else:
        states = split.s[ids]
        actions = split.negative_actions[ids]
        advantages = split.negative_advantages[ids]
    mu, log_std = actor(states)
    log_prob = cu1_core.gaussian_log_prob(
        mu, log_std, actions, experiment.P.action_dim
    )
    distance = cu1_core.standardized_distance(mu, log_std, actions)
    weight = taper_weight_from_coefficient(
        distance.detach(), family, coefficient, protocol.reference_distance
    )
    loss = -(advantages * weight * log_prob).mean()
    with torch.no_grad():
        near = distance <= protocol.near_region_boundary
        far = ~near
        diagnostics = {
            "weight_mean": float(weight.mean().item()),
            "near_weight_mean": (
                float(weight[near].mean().item()) if near.any() else float("nan")
            ),
            "far_weight_mean": (
                float(weight[far].mean().item()) if far.any() else float("nan")
            ),
            "near_occupancy": float(near.float().mean().item()),
            "far_occupancy": float(far.float().mean().item()),
        }
    return loss, diagnostics


def full_field_diagnostics(
    actor: experiment.GaussianActor,
    split: experiment.Split,
    family: str,
    coefficient: float,
    protocol: NearRetentionProtocol,
) -> dict[str, Any]:
    positive = experiment.positive_loss(actor, split)
    positive_grad = taper_base.gradient_tuple(
        positive, actor, retain_graph=family != "positive_only"
    )
    positive_norm = float(experiment.norm_tuple(positive_grad).item())
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
        actor, split, None, family, coefficient, protocol
    )
    negative_grad = taper_base.gradient_tuple(negative, actor, retain_graph=True)
    total_grad = experiment.add_tuples(
        positive_grad, negative_grad, scales=(1.0, protocol.negative_alpha)
    )
    negative_norm = float(experiment.norm_tuple(negative_grad).item())
    total_norm = float(experiment.norm_tuple(total_grad).item())
    residual = total_norm / (
        positive_norm + protocol.negative_alpha * negative_norm + EPS
    )
    return {
        "positive_gradient_norm": positive_norm,
        "negative_gradient_norm": negative_norm,
        "total_gradient_norm": total_norm,
        "normalized_field_residual": residual,
        "stationarity_residual": residual,
        "stationarity_residual_kind": "normalized_signed_field_residual",
    }


def retention_and_harm_diagnostics(
    actor: experiment.GaussianActor,
    split: experiment.Split,
    family: str,
    coefficient: float,
    protocol: NearRetentionProtocol,
) -> dict[str, float]:
    actor.eval()
    with torch.no_grad():
        mu, log_std = actor(split.s)
        actions = split.negative_actions
        distance = cu1_core.standardized_distance(mu, log_std, actions)
        weight = taper_weight_from_coefficient(
            distance, family, coefficient, protocol.reference_distance
        )
        inverse_variance = torch.exp(-2.0 * log_std)[:, None]
        advantage_abs = split.negative_advantages.abs()
        negative_mean_update = (
            advantage_abs[..., None]
            * (mu[:, None, :] - actions)
            * inverse_variance[..., None]
        )
        oracle_direction = split.a_star - mu
        oracle_unit = oracle_direction / (
            torch.linalg.vector_norm(oracle_direction, dim=-1, keepdim=True) + EPS
        )
        projection = (negative_mean_update * oracle_unit[:, None, :]).sum(-1)
        update_norm = torch.linalg.vector_norm(negative_mean_update, dim=-1)
        near = distance <= protocol.near_region_boundary
        far = ~near
        useful = projection > 0.0
        harmful = projection < 0.0
        near_useful = near & useful
        far_harmful = far & harmful

        def safe_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
            return float(values[mask].mean().item()) if mask.any() else float("nan")

        def retention_ratio(mass: torch.Tensor, mask: torch.Tensor) -> float:
            denominator = mass[mask].sum()
            if not mask.any() or float(denominator.item()) <= EPS:
                return float("nan")
            return float((weight[mask] * mass[mask]).sum().item() / denominator.item())

        near_useful_mass = torch.clamp(projection, min=0.0)
        far_harmful_mass = torch.clamp(-projection, min=0.0)
        return {
            "near_region_weight_mean": safe_mean(weight, near),
            "far_region_weight_mean": safe_mean(weight, far),
            "near_useful_weight_mean": safe_mean(weight, near_useful),
            "near_useful_gradient_retention": retention_ratio(
                near_useful_mass, near_useful
            ),
            "far_harmful_influence_retention": retention_ratio(
                far_harmful_mass, far_harmful
            ),
            "far_harmful_weighted_projection_mean": safe_mean(
                weight * far_harmful_mass, far_harmful
            ),
            "far_harmful_weighted_update_norm_mean": safe_mean(
                weight * update_norm, far_harmful
            ),
            "near_occupancy": float(near.float().mean().item()),
            "far_occupancy": float(far.float().mean().item()),
            "near_useful_fraction": float(near_useful.float().mean().item()),
            "far_harmful_fraction": float(far_harmful.float().mean().item()),
        }


def evaluate_state(
    actor: experiment.GaussianActor,
    environment: experiment.Environment,
    family: str,
    coefficient: float,
    protocol: NearRetentionProtocol,
    initial_reward: float,
) -> dict[str, Any]:
    task = experiment.evaluation(actor, environment.test)
    field = full_field_diagnostics(
        actor, environment.train, family, coefficient, protocol
    )
    retention = retention_and_harm_diagnostics(
        actor, environment.train, family, coefficient, protocol
    )
    finite_parameters = experiment.finite_model(actor)
    log_sigma_min = float(task["log_sigma_min"])
    log_sigma_max = float(task["log_sigma_max"])
    support_boundary = bool(
        not math.isfinite(log_sigma_min)
        or not math.isfinite(log_sigma_max)
        or abs(log_sigma_min) > protocol.log_sigma_event_boundary
        or abs(log_sigma_max) > protocol.log_sigma_event_boundary
    )
    task_failure = bool(
        float(task["reward"]) < protocol.task_failure_retention * initial_reward
    )
    return {
        **task,
        **field,
        **retention,
        "task_performance_collapse_event": task_failure,
        "support_or_variance_boundary_event": support_boundary,
        "nan_inf_numerical_event": not finite_parameters,
    }


def _flatten_gradients(
    gradients: tuple[torch.Tensor | None, ...],
    parameters: list[torch.nn.Parameter],
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for gradient, parameter in zip(gradients, parameters):
        values.append(
            gradient.reshape(-1)
            if gradient is not None
            else torch.zeros_like(parameter).reshape(-1)
        )
    return torch.cat(values)


def _utility_bin(distance: float, edges: tuple[float, ...]) -> str:
    for low, high in zip(edges[:-1], edges[1:]):
        if low <= distance < high:
            high_text = "inf" if math.isinf(high) else f"{high:g}"
            return f"[{low:g},{high_text})"
    return "unassigned"


def per_sample_utility_diagnostic(
    seed: int,
    stage: str,
    actor: experiment.GaussianActor,
    environment: experiment.Environment,
    family: str,
    target_retention: float | None,
    coefficient: float,
    protocol: NearRetentionProtocol,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    actor.eval()
    parameters = list(actor.parameters())
    count = min(protocol.probe_states, len(environment.train.s))
    for state_index in range(count):
        state = environment.train.s[state_index : state_index + 1]
        actions = environment.train.negative_actions[state_index]
        advantages = environment.train.negative_advantages[state_index]
        star = environment.train.a_star[state_index : state_index + 1]
        mu, log_std = actor(state)
        oracle_objective = -0.5 * (mu - star).square().sum()
        oracle_gradients = torch.autograd.grad(
            oracle_objective,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        oracle_vector = _flatten_gradients(oracle_gradients, parameters)
        oracle_norm = torch.linalg.vector_norm(oracle_vector)
        for contour_index in range(actions.shape[0]):
            action = actions[contour_index : contour_index + 1]
            log_prob = cu1_core.gaussian_log_prob(
                mu,
                log_std,
                action[None, :, :],
                experiment.P.action_dim,
            ).squeeze()
            objective = advantages[contour_index] * log_prob
            negative_gradients = torch.autograd.grad(
                objective,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            negative_vector = _flatten_gradients(negative_gradients, parameters)
            raw_norm = torch.linalg.vector_norm(negative_vector)
            cosine = float(
                torch.dot(negative_vector, oracle_vector).item()
                / (float(raw_norm.item() * oracle_norm.item()) + EPS)
            )
            cosine = max(-1.0, min(1.0, cosine))
            orthogonal_fraction_sq = max(0.0, 1.0 - cosine * cosine)
            normalized_net_utility = (
                cosine
                - protocol.normalized_utility_kappa * orthogonal_fraction_sq
            )
            distance = float(
                cu1_core.standardized_distance(
                    mu, log_std, action[None, :, :]
                ).item()
            )
            weight = float(
                taper_weight_from_coefficient(
                    torch.tensor(distance),
                    family,
                    coefficient,
                    protocol.reference_distance,
                ).item()
            )
            raw_norm_value = float(raw_norm.item())
            rows.append(
                {
                    "seed": seed,
                    "stage": stage,
                    "family": family,
                    "target_retention": target_retention,
                    "coefficient": coefficient,
                    "state_index": state_index,
                    "contour_index": contour_index,
                    "standardized_distance": distance,
                    "distance_bin": _utility_bin(distance, protocol.utility_bin_edges),
                    "weight": weight,
                    "raw_full_parameter_gradient_norm": raw_norm_value,
                    "weighted_full_parameter_gradient_norm": weight * raw_norm_value,
                    "alignment_cosine": cosine,
                    "orthogonal_fraction_sq": orthogonal_fraction_sq,
                    "normalized_directional_net_utility": normalized_net_utility,
                    "weighted_directional_utility": (
                        weight * raw_norm_value * normalized_net_utility
                    ),
                    "weighted_aligned_projection": (
                        weight * raw_norm_value * cosine
                    ),
                    "weighted_orthogonal_norm": (
                        weight * raw_norm_value * math.sqrt(orthogonal_fraction_sq)
                    ),
                }
            )

    def ratio(field: str) -> float:
        near = [row[field] for row in rows if row["contour_index"] == 0]
        far = [row[field] for row in rows if row["contour_index"] == 4]
        if not near or not far:
            return float("nan")
        return float(np.mean(far) / (np.mean(near) + EPS))

    summary = {
        "far_near_weighted_gradient_ratio": ratio(
            "weighted_full_parameter_gradient_norm"
        ),
        "far_near_raw_gradient_ratio": ratio("raw_full_parameter_gradient_norm"),
        "probe_rows": len(rows),
    }
    return rows, summary


def aggregate_utility_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["seed"],
            row["stage"],
            row["family"],
            row["target_retention"],
            row["distance_bin"],
        )
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    fields = (
        "standardized_distance",
        "weight",
        "raw_full_parameter_gradient_norm",
        "weighted_full_parameter_gradient_norm",
        "alignment_cosine",
        "orthogonal_fraction_sq",
        "normalized_directional_net_utility",
        "weighted_directional_utility",
        "weighted_aligned_projection",
        "weighted_orthogonal_norm",
    )
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        seed, stage, family, target, distance_bin = key
        row: dict[str, Any] = {
            "seed": seed,
            "stage": stage,
            "family": family,
            "target_retention": target,
            "distance_bin": distance_bin,
            "sample_count": len(values),
        }
        for field in fields:
            row[f"{field}_mean"] = float(
                np.mean([float(value[field]) for value in values])
            )
        output.append(row)
    return output


def train_method(
    seed: int,
    initial_state: dict[str, torch.Tensor],
    environment: experiment.Environment,
    family: str,
    target_retention: float | None,
    coefficient: float,
    output_root: Path,
    protocol: NearRetentionProtocol,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    name = config_name(family, target_retention)
    summary_path = output_root / "runs" / name / f"seed_{seed}.json"
    trajectory_path = output_root / "runs" / name / f"seed_{seed}_trajectory.csv"
    diagnostic_path = output_root / "runs" / name / f"seed_{seed}_utility.csv"
    if summary_path.exists() and trajectory_path.exists() and diagnostic_path.exists():
        return (
            json.loads(summary_path.read_text(encoding="utf-8")),
            experiment.read_csv(trajectory_path),
            experiment.read_csv(diagnostic_path),
        )

    experiment.seed_all(seed + 1_900_000)
    actor = experiment.GaussianActor().to(experiment.DEVICE)
    actor.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.Adam(actor.parameters(), lr=protocol.learning_rate)
    index_generator = torch.Generator(device="cpu").manual_seed(seed + 1_700_003)

    initial_task = experiment.evaluation(actor, environment.test)
    initial_reward = float(initial_task["reward"])
    diagnostic_rows: list[dict[str, Any]] = []
    initial_raw, initial_diag = per_sample_utility_diagnostic(
        seed,
        "initial",
        actor,
        environment,
        family,
        target_retention,
        coefficient,
        protocol,
    )
    diagnostic_rows.extend(initial_raw)

    trajectory: list[dict[str, Any]] = []
    stable_candidate_step: int | None = None
    audit_target_step: int | None = None
    candidate_classification: tuple[bool, bool, bool] | None = None
    stop_reason = "maximum_steps"
    started = time.perf_counter()

    for step in range(1, protocol.maximum_steps + 1):
        ids = torch.randint(
            0,
            experiment.P.n_train_states,
            (protocol.batch_states,),
            generator=index_generator,
        ).to(experiment.DEVICE)
        positive = experiment.positive_loss(actor, environment.train, ids)
        if family == "positive_only":
            loss = positive
        else:
            negative, _ = weighted_negative_loss(
                actor,
                environment.train,
                ids,
                family,
                coefficient,
                protocol,
            )
            loss = positive + protocol.negative_alpha * negative

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        should_evaluate = (
            step == 1
            or step % protocol.evaluation_interval == 0
            or step == protocol.maximum_steps
            or (audit_target_step is not None and step >= audit_target_step)
        )
        if not should_evaluate:
            continue

        state = evaluate_state(
            actor,
            environment,
            family,
            coefficient,
            protocol,
            initial_reward,
        )
        row = {
            "seed": seed,
            "step": step,
            "family": family,
            "target_retention": target_retention,
            "coefficient": coefficient,
            "loss": float(loss.detach().item()),
            **state,
        }
        trajectory.append(row)
        write_csv(trajectory_path, trajectory)

        if state["nan_inf_numerical_event"]:
            stop_reason = "nan_inf_numerical_event"
            break
        if state["support_or_variance_boundary_event"]:
            stop_reason = "support_or_variance_boundary_event"
            break

        if (
            stable_candidate_step is None
            and step >= protocol.minimum_steps
            and len(trajectory) >= protocol.stable_windows
        ):
            window = trajectory[-protocol.stable_windows :]
            max_slope = max(
                taper_base.normalized_slope(window, field)
                for field in (
                    "reward",
                    "normalized_extrapolation_displacement",
                    "sigma_mean",
                )
            )
            residual = float(state["stationarity_residual"])
            threshold = (
                protocol.positive_absolute_gradient_threshold
                if family == "positive_only"
                else protocol.normalized_field_residual_threshold
            )
            if max_slope < protocol.normalized_slope_threshold and residual < threshold:
                stable_candidate_step = step
                audit_target_step = taper_base.two_times_audit_target(
                    step, protocol.maximum_steps
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

    elapsed = time.perf_counter() - started
    final_state = evaluate_state(
        actor,
        environment,
        family,
        coefficient,
        protocol,
        initial_reward,
    )
    terminal_raw, terminal_diag = per_sample_utility_diagnostic(
        seed,
        "terminal",
        actor,
        environment,
        family,
        target_retention,
        coefficient,
        protocol,
    )
    diagnostic_rows.extend(terminal_raw)
    write_csv(diagnostic_path, diagnostic_rows)

    completed_steps = int(trajectory[-1]["step"])
    checkpoint_path = output_root / "checkpoints" / name / f"seed_{seed}_terminal.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), checkpoint_path)

    summary = {
        "seed": seed,
        "family": family,
        "target_retention": target_retention,
        "coefficient": coefficient,
        "steps_completed": completed_steps,
        "stop_reason": stop_reason,
        "stable_candidate_step": stable_candidate_step,
        "audit_target_step": audit_target_step,
        "elapsed_seconds": elapsed,
        "milliseconds_per_update": 1000.0 * elapsed / max(completed_steps, 1),
        "initial_reward": initial_reward,
        **final_state,
        "initial_far_near_weighted_gradient_ratio": initial_diag[
            "far_near_weighted_gradient_ratio"
        ],
        "terminal_far_near_weighted_gradient_ratio": terminal_diag[
            "far_near_weighted_gradient_ratio"
        ],
        "initial_far_near_raw_gradient_ratio": initial_diag[
            "far_near_raw_gradient_ratio"
        ],
        "terminal_far_near_raw_gradient_ratio": terminal_diag[
            "far_near_raw_gradient_ratio"
        ],
        "terminal_checkpoint": str(checkpoint_path),
    }
    atomic_json(summary_path, summary)
    return summary, trajectory, diagnostic_rows


def bootstrap_mean_ci(
    values: list[float], samples: int, seed: int
) -> tuple[float, float, float]:
    return taper_base.bootstrap_mean_ci(values, samples=samples, seed=seed)


def aggregate_results(
    summaries: list[dict[str, Any]],
    output_root: Path,
    protocol: NearRetentionProtocol,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
    for row in summaries:
        grouped.setdefault((row["family"], row["target_retention"]), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    metric_fields = (
        "reward",
        "near_region_weight_mean",
        "near_useful_gradient_retention",
        "far_harmful_influence_retention",
        "far_harmful_weighted_projection_mean",
        "terminal_far_near_weighted_gradient_ratio",
        "sigma_mean",
        "distance_to_a_star",
    )
    for (family, target), rows in sorted(
        grouped.items(), key=lambda item: (str(item[0][1]), item[0][0])
    ):
        aggregate: dict[str, Any] = {
            "family": family,
            "target_retention": target,
            "seeds": len(rows),
            "coefficient": float(rows[0]["coefficient"]),
            "task_performance_collapse_events": sum(
                bool(row["task_performance_collapse_event"]) for row in rows
            ),
            "support_or_variance_boundary_events": sum(
                bool(row["support_or_variance_boundary_event"]) for row in rows
            ),
            "nan_inf_numerical_events": sum(
                bool(row["nan_inf_numerical_event"]) for row in rows
            ),
            "stable_plateau_2x_confirmed_runs": sum(
                row["stop_reason"] == "stable_plateau_2x_confirmed" for row in rows
            ),
        }
        for field in metric_fields:
            values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
            aggregate[f"{field}_mean"] = (
                float(np.mean(values)) if values else float("nan")
            )
        aggregate_rows.append(aggregate)
    write_csv(output_root / "aggregate.csv", aggregate_rows)

    by_key = {
        (int(row["seed"]), row["family"], row["target_retention"]): row
        for row in summaries
    }
    paired_rows: list[dict[str, Any]] = []
    paired_summary: dict[str, Any] = {
        "primary_target_retention": protocol.primary_retention,
        "comparisons": [],
        "reward_is_secondary_and_no_winner_is_assumed": True,
    }
    for target in protocol.retention_levels:
        for family in protocol.families:
            if family == "reciprocal_linear":
                continue
            differences: dict[str, list[float]] = {
                "reward": [],
                "near_useful_gradient_retention": [],
                "far_harmful_influence_retention": [],
                "terminal_far_near_weighted_gradient_ratio": [],
            }
            for seed in protocol.formal_seeds:
                linear = by_key.get((seed, "reciprocal_linear", target))
                candidate = by_key.get((seed, family, target))
                if linear is None or candidate is None:
                    continue
                paired = {
                    "seed": seed,
                    "target_retention": target,
                    "candidate_family": family,
                }
                for field in differences:
                    difference = float(candidate[field]) - float(linear[field])
                    paired[f"candidate_minus_linear_{field}"] = difference
                    differences[field].append(difference)
                paired_rows.append(paired)
            comparison: dict[str, Any] = {
                "target_retention": target,
                "candidate_family": family,
                "paired_seeds": len(differences["reward"]),
            }
            for field, values in differences.items():
                if values:
                    comparison[f"{field}_difference_mean_ci95"] = bootstrap_mean_ci(
                        values,
                        samples=protocol.bootstrap_samples,
                        seed=20260628
                        + int(round(target * 100))
                        + sum(ord(char) for char in family),
                    )
            paired_summary["comparisons"].append(comparison)
    write_csv(output_root / "paired_comparisons.csv", paired_rows)
    atomic_json(output_root / "paired_summary.json", paired_summary)
    return aggregate_rows, paired_rows, paired_summary


def build_terminal_audit(
    summaries: list[dict[str, Any]],
    calibration: dict[str, Any],
    paired_summary: dict[str, Any],
    protocol: NearRetentionProtocol,
    *,
    base_commit: str,
    smoke: bool,
) -> dict[str, Any]:
    expected_configs = method_configs(protocol)
    expected_runs = len(protocol.formal_seeds) * len(expected_configs)
    calibration_errors = [
        float(row["absolute_error"]) for row in calibration["calibrations"]
    ]
    expected_pair_count = len(protocol.formal_seeds)
    pair_counts = [
        int(row["paired_seeds"]) for row in paired_summary["comparisons"]
    ]
    resolved_reasons = {
        "stable_plateau_2x_confirmed",
        "support_or_variance_boundary_event",
        "nan_inf_numerical_event",
    }
    checks = [
        {
            "name": "all_registered_runs_present",
            "passed": len(summaries) == expected_runs,
            "value": len(summaries),
            "expected": expected_runs,
        },
        {
            "name": "development_only_calibration",
            "passed": calibration["formal_or_confirmatory_seed_accessed"] is False,
            "value": calibration["development_seeds"],
            "expected": list(protocol.development_seeds),
        },
        {
            "name": "near_retention_calibration_within_tolerance",
            "passed": bool(calibration_errors)
            and max(calibration_errors) <= protocol.calibration_tolerance,
            "value": max(calibration_errors) if calibration_errors else None,
            "expected": protocol.calibration_tolerance,
        },
        {
            "name": "all_candidate_linear_pairs_complete",
            "passed": bool(pair_counts)
            and all(count == expected_pair_count for count in pair_counts),
            "value": pair_counts,
            "expected": expected_pair_count,
        },
        {
            "name": "failure_types_reported_separately",
            "passed": all(
                all(
                    key in row
                    for key in (
                        "task_performance_collapse_event",
                        "support_or_variance_boundary_event",
                        "nan_inf_numerical_event",
                    )
                )
                for row in summaries
            ),
            "value": True,
            "expected": True,
        },
        {
            "name": "all_runs_terminally_resolved",
            "passed": all(str(row.get("stop_reason")) in resolved_reasons for row in summaries),
            "value": {
                reason: sum(str(row.get("stop_reason")) == reason for row in summaries)
                for reason in sorted({str(row.get("stop_reason")) for row in summaries})
            },
            "expected": sorted(resolved_reasons),
            "gating_for_scientific_status": False,
        },
    ]
    coverage_checks = [check for check in checks if check["name"] != "all_runs_terminally_resolved"]
    coverage_passed = all(bool(check["passed"]) for check in coverage_checks)
    terminal_resolution_complete = bool(checks[-1]["passed"])
    scientific_status = (
        "not run / 尚未运行"
        if smoke
        else (
            "finite-step validated / 有限训练步数验证"
            if coverage_passed
            else "not run / 尚未运行"
        )
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "base_commit": base_commit,
        "execution_status": "engineering_smoke" if smoke else "formal_run",
        "scientific_status": scientific_status,
        "coverage_checks_passed": coverage_passed,
        "terminal_resolution_complete": terminal_resolution_complete,
        "all_checks_passed": coverage_passed and terminal_resolution_complete,
        "checks": checks,
        "interpretation_boundary": (
            "This experiment tests taper shape at matched average near-region "
            "retention. It does not match total negative-update budget, does not "
            "resolve the final long-run shortlist, and does not establish a "
            "universal family winner. Held-out states are sampled from the same "
            "state distribution and are not an OOD protocol."
        ),
    }


def write_protocol_freeze(
    output_root: Path,
    protocol: NearRetentionProtocol,
    coefficients: dict[tuple[str, float], float],
) -> None:
    atomic_json(
        output_root / "formal_protocol_freeze.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "script_version": SCRIPT_VERSION,
            "protocol": asdict(protocol),
            "near_region": "standardized_distance <= 5.0 at development calibration checkpoint",
            "matching_estimator": "pooled development-seed sample mean",
            "coefficient_solver": "deterministic monotone bisection",
            "coefficient_application": "fixed across all formal seeds and training steps",
            "candidate_formulas": {
                "reciprocal_linear": "1/(1+c*u)",
                "reciprocal_quadratic": "1/(1+c*u^2)",
                "exponential": "exp(-c*u)",
                "squared_distance_exponential": "exp(-c*u^2)",
                "u": "d/5.0",
            },
            "calibrated_coefficients": [
                {
                    "family": family,
                    "target_retention": retention,
                    "coefficient": coefficient,
                }
                for (family, retention), coefficient in sorted(
                    coefficients.items(), key=lambda item: (item[0][1], item[0][0])
                )
            ],
            "primary_target_retention": protocol.primary_retention,
            "secondary_targets": list(protocol.sensitivity_retentions),
            "primary_scientific_question": (
                "At matched average near retention, how do taper shapes allocate "
                "remaining influence to useful near and harmful far negatives?"
            ),
            "reward_directional_winner_preregistered": False,
            "long_run_method_ranking_allowed": False,
            "budget_matching_performed": False,
        },
    )


def checkpoint_manifest(
    output_root: Path,
    completed_seeds: list[int],
    protocol: NearRetentionProtocol,
) -> None:
    atomic_json(
        output_root / "checkpoints" / f"seed_block_{completed_seeds[-1]}.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "completed_formal_seeds": completed_seeds,
            "remaining_formal_seeds": [
                seed for seed in protocol.formal_seeds if seed not in completed_seeds
            ],
            "checkpoint_kind": "runner_progress_index_not_final_scientific_result",
            "partial_summary": "per_seed_runs_partial.csv",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_dir.resolve()
    supervisor_owned = {
        "heartbeat.json",
        "logs",
        "provenance_launch",
        "run_manifest.json",
        "scientific_run_manifest.json",
    }
    if output_root.exists():
        unexpected = sorted(
            path.name
            for path in output_root.iterdir()
            if path.name not in supervisor_owned
        )
        if unexpected:
            raise SystemExit(
                "output directory contains non-supervisor files: "
                + ", ".join(unexpected)
            )
    output_root.mkdir(parents=True, exist_ok=True)

    global PROTOCOL
    if args.smoke:
        experiment.P = replace(
            experiment.P,
            n_train_states=64,
            n_test_states=64,
            hidden_dim=16,
            positive_batch_states=32,
            positive_steps=4,
            positive_continuation_steps=2,
            lbfgs_max_iter=1,
            positive_polish_min_steps=1,
            positive_polish_max_steps=1,
            positive_polish_check_every=1,
            probe_states=2,
        )
        PROTOCOL = NearRetentionProtocol(
            development_seeds=(0,),
            formal_seeds=(90,),
            sensitivity_retentions=(),
            batch_states=32,
            evaluation_interval=1,
            minimum_steps=1,
            maximum_steps=3,
            stable_windows=2,
            probe_states=2,
            bootstrap_samples=20,
            checkpoint_every_formal_seeds=1,
        )

    experiment.ROOT = output_root
    started = time.time()
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "base_commit": args.base_commit,
        "result_status": "engineering_smoke" if args.smoke else "running",
        "cu1_protocol": asdict(experiment.P),
        "near_retention_protocol": asdict(PROTOCOL),
        "device": str(experiment.DEVICE),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "start_unix": started,
        "scope": (
            "same-distribution held-out-context generalization; not OOD; "
            "development-only coefficient calibration; no total-budget matching"
        ),
    }
    atomic_json(output_root / "experiment_manifest.json", manifest)

    source_snapshot = output_root / "source_snapshot"
    source_snapshot.mkdir(parents=True, exist_ok=True)
    for source in (
        Path(__file__),
        Path(experiment.__file__),
        Path(cu1_core.__file__),
        Path(taper_base.__file__),
    ):
        shutil.copy2(source, source_snapshot / source.name)

    preflight_environment = experiment.audit_environment(
        experiment.make_environment(PROTOCOL.formal_seeds[0])
    )
    atomic_json(output_root / "environment_audit.json", preflight_environment)
    if not preflight_environment["passed"]:
        raise SystemExit("shared C-U1 environment audit failed")

    coefficients, calibration = calibrate_families(output_root, PROTOCOL)
    write_protocol_freeze(output_root, PROTOCOL, coefficients)

    summaries: list[dict[str, Any]] = []
    all_trajectories: list[dict[str, Any]] = []
    all_utility_rows: list[dict[str, Any]] = []
    positive_rows: list[dict[str, Any]] = []
    completed_seeds: list[int] = []

    for seed_index, seed in enumerate(PROTOCOL.formal_seeds, start=1):
        log_message(output_root, f"formal seed={seed} positive-only initialization")
        _, environment, _, positive_summary = experiment.train_positive(seed)
        positive_rows.append(
            {
                **positive_summary,
                "method_initialization_source": "positive_only_adam_2000_step_checkpoint",
            }
        )
        initial_state = copy.deepcopy(experiment.load_initialization_state(seed))

        for family, target_retention in method_configs(PROTOCOL):
            coefficient = coefficient_for(family, target_retention, coefficients)
            log_message(
                output_root,
                f"seed={seed} family={family} target={target_retention} c={coefficient:.8g}",
            )
            summary, trajectory, utility_rows = train_method(
                seed,
                initial_state,
                environment,
                family,
                target_retention,
                coefficient,
                output_root,
                PROTOCOL,
            )
            summaries.append(summary)
            all_trajectories.extend(trajectory)
            all_utility_rows.extend(utility_rows)
            write_csv(output_root / "per_seed_runs_partial.csv", summaries)

        completed_seeds.append(seed)
        if (
            seed_index % PROTOCOL.checkpoint_every_formal_seeds == 0
            or seed_index == len(PROTOCOL.formal_seeds)
        ):
            checkpoint_manifest(output_root, completed_seeds, PROTOCOL)

    write_csv(output_root / "positive_summary.csv", positive_rows)
    write_csv(output_root / "per_seed_runs.csv", summaries)
    write_csv(output_root / "all_trajectories.csv", all_trajectories)
    write_csv(output_root / "all_utility_diagnostics.csv", all_utility_rows)
    write_csv(
        output_root / "utility_bins.csv", aggregate_utility_bins(all_utility_rows)
    )
    aggregate_rows, paired_rows, paired_summary = aggregate_results(
        summaries, output_root, PROTOCOL
    )
    terminal_audit = build_terminal_audit(
        summaries,
        calibration,
        paired_summary,
        PROTOCOL,
        base_commit=args.base_commit,
        smoke=args.smoke,
    )
    atomic_json(output_root / "terminal_audit.json", terminal_audit)

    manifest.update(
        {
            "result_status": "engineering_smoke"
            if args.smoke
            else terminal_audit["scientific_status"],
            "end_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "runs_completed": len(summaries),
            "paired_rows": len(paired_rows),
        }
    )
    atomic_json(output_root / "experiment_manifest.json", manifest)
    atomic_json(
        output_root / "RUN_COMPLETE.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "base_commit": args.base_commit,
            "result_status": manifest["result_status"],
            "coverage_checks_passed": terminal_audit["coverage_checks_passed"],
            "terminal_resolution_complete": terminal_audit[
                "terminal_resolution_complete"
            ],
            "aggregate_rows": len(aggregate_rows),
            "formal_run_started": not args.smoke,
        },
    )
    log_message(output_root, "run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
