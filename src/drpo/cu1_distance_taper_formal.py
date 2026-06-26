#!/usr/bin/env python3
"""Formal C-U1 reciprocal-linear versus reciprocal-quadratic taper experiment.

Experiment ID: C-U1-E4-TAPER-01.

The runner imports the frozen C-U1 environment, Gaussian actor, positive-only
trainer, and evaluation routines from the shared implementation. It changes
only the detached negative-sample taper. All methods use the same standardized
distance and paired minibatch stream.
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
    from . import drpo_cu1_e1_e4_oneclick as experiment
except ImportError:  # direct script execution
    import cu1_core
    import drpo_cu1_e1_e4_oneclick as experiment

EXPERIMENT_ID = "C-U1-E4-TAPER-01"
SCRIPT_VERSION = "2026.06.25-formal-v1-shared-cu1-core"
EPS = 1e-12


@dataclass(frozen=True)
class TaperProtocol:
    formal_seeds: tuple[int, ...] = tuple(range(70, 90))
    reference_distance: float = 5.0
    primary_rho: float = 0.25
    sensitivity_rhos: tuple[float, ...] = (0.50, 0.75)
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
    probe_states: int = 64
    bootstrap_samples: int = 4000


TAPER = TaperProtocol()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    experiment.write_csv(path, list(rows))


def log_message(output_root: Path, message: str) -> None:
    stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(stamped, flush=True)
    with (output_root / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(stamped + "\n")


def method_configs(protocol: TaperProtocol) -> list[tuple[str, float]]:
    configs: list[tuple[str, float]] = [
        ("positive_only", 1.0),
        ("unweighted", 1.0),
        ("reciprocal_linear", protocol.primary_rho),
        ("reciprocal_quadratic", protocol.primary_rho),
        ("exponential", protocol.primary_rho),
    ]
    for rho in protocol.sensitivity_rhos:
        for family in (
            "reciprocal_linear",
            "reciprocal_quadratic",
            "exponential",
        ):
            configs.append((family, rho))
    return configs


def config_name(family: str, rho: float) -> str:
    if family in {"positive_only", "unweighted"}:
        return family
    return f"{family}_rho{rho:.2f}"


def taper_weight(
    standardized_distance: torch.Tensor,
    family: str,
    rho: float,
    protocol: TaperProtocol,
) -> torch.Tensor:
    return cu1_core.distance_taper_weight(
        standardized_distance,
        family=family,
        rho=rho,
        reference_distance=protocol.reference_distance,
    )


def weighted_negative_loss(
    actor: experiment.GaussianActor,
    split: experiment.Split,
    ids: torch.Tensor | None,
    family: str,
    rho: float,
    protocol: TaperProtocol,
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
    # The theorem and registered method require pure sample reweighting.
    weight = taper_weight(distance.detach(), family, rho, protocol)
    loss = -(advantages * weight * log_prob).mean()

    with torch.no_grad():
        near = distance <= protocol.reference_distance
        far = ~near
        diagnostics = {
            "weight_mean": float(weight.mean().item()),
            "near_weight_mean": (
                float(weight[near].mean().item()) if near.any() else float("nan")
            ),
            "far_weight_mean": (
                float(weight[far].mean().item()) if far.any() else float("nan")
            ),
            "standardized_distance_mean": float(distance.mean().item()),
            "near_occupancy": float(near.float().mean().item()),
            "far_occupancy": float(far.float().mean().item()),
        }
    return loss, diagnostics


def gradient_tuple(
    loss: torch.Tensor,
    actor: experiment.GaussianActor,
    retain_graph: bool,
) -> tuple[torch.Tensor | None, ...]:
    return tuple(
        torch.autograd.grad(
            loss,
            list(actor.parameters()),
            retain_graph=retain_graph,
            allow_unused=True,
        )
    )


def full_field_diagnostics(
    actor: experiment.GaussianActor,
    split: experiment.Split,
    family: str,
    rho: float,
    protocol: TaperProtocol,
) -> dict[str, Any]:
    positive = experiment.positive_loss(actor, split)
    positive_grad = gradient_tuple(positive, actor, retain_graph=family != "positive_only")
    positive_norm = float(experiment.norm_tuple(positive_grad).item())

    if family == "positive_only":
        return {
            "positive_gradient_norm": positive_norm,
            "negative_gradient_norm": 0.0,
            "total_gradient_norm": positive_norm,
            # Cancellation normalization is not meaningful without a negative field.
            "normalized_field_residual": float("nan"),
            "stationarity_residual": positive_norm,
            "stationarity_residual_kind": "absolute_positive_gradient_norm",
        }

    negative, _ = weighted_negative_loss(
        actor, split, None, family, rho, protocol
    )
    negative_grad = gradient_tuple(negative, actor, retain_graph=True)
    total_grad = experiment.add_tuples(
        positive_grad,
        negative_grad,
        scales=(1.0, protocol.negative_alpha),
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


def two_times_audit_target(candidate_step: int, maximum_steps: int) -> int | None:
    """Return the exact 2x audit step, or None when the horizon cannot fit it."""
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


def evaluate_state(
    actor: experiment.GaussianActor,
    environment: experiment.Environment,
    family: str,
    rho: float,
    protocol: TaperProtocol,
    initial_reward: float,
) -> dict[str, Any]:
    task = experiment.evaluation(actor, environment.test)
    field = full_field_diagnostics(
        actor, environment.train, family, rho, protocol
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
        float(task["reward"])
        < protocol.task_failure_retention * initial_reward
    )
    return {
        **task,
        **field,
        "task_performance_collapse_event": task_failure,
        "support_or_variance_boundary_event": support_boundary,
        "nan_inf_numerical_event": not finite_parameters,
    }


def per_sample_gradient_diagnostic(
    seed: int,
    stage: str,
    actor: experiment.GaussianActor,
    environment: experiment.Environment,
    family: str,
    rho: float,
    protocol: TaperProtocol,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    actor.eval()
    count = min(protocol.probe_states, len(environment.train.s))
    for state_index in range(count):
        state = environment.train.s[state_index : state_index + 1]
        actions = environment.train.negative_actions[state_index]
        advantages = environment.train.negative_advantages[state_index]
        mu, log_std = actor(state)
        for contour_index in range(actions.shape[0]):
            action = actions[contour_index : contour_index + 1]
            log_prob = cu1_core.gaussian_log_prob(
                mu,
                log_std,
                action[None, :, :],
                experiment.P.action_dim,
            ).squeeze()
            objective = advantages[contour_index] * log_prob
            gradients = torch.autograd.grad(
                objective,
                list(actor.parameters()),
                retain_graph=True,
                allow_unused=True,
            )
            raw_norm = float(experiment.norm_tuple(gradients).item())
            components = cu1_core.gaussian_output_components(
                mu,
                log_std,
                action[None, :, :],
                experiment.P.action_dim,
            )
            distance = float(components["standardized_distance"].item())
            advantage_abs = abs(float(advantages[contour_index].item()))
            raw_output_mean = advantage_abs * float(components["mean_score"].item())
            raw_output_log_scale = advantage_abs * abs(
                float(components["log_scale_score"].item())
            )
            raw_output_joint = advantage_abs * float(components["joint_score"].item())
            if family == "positive_only":
                weight = 0.0
            else:
                weight = float(
                    taper_weight(
                        torch.tensor(distance), family, rho, protocol
                    ).item()
                )
            rows.append(
                {
                    "seed": seed,
                    "stage": stage,
                    "family": family,
                    "rho": rho,
                    "state_index": state_index,
                    "contour_index": contour_index,
                    "advantage": float(advantages[contour_index].item()),
                    "standardized_distance": distance,
                    "weight": weight,
                    "raw_output_mean_gradient_norm": raw_output_mean,
                    "raw_output_log_scale_gradient_abs": raw_output_log_scale,
                    "raw_output_joint_gradient_norm": raw_output_joint,
                    "weighted_output_mean_gradient_norm": weight * raw_output_mean,
                    "weighted_output_log_scale_gradient_abs": (
                        weight * raw_output_log_scale
                    ),
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
            if row["standardized_distance"] >= protocol.reference_distance
            and row[field] > 0.0
        ]
        ratio = (
            float(np.mean(far) / (np.mean(near) + EPS))
            if near and far
            else float("nan")
        )
        if len(far_region) < 2:
            return ratio, float("nan")
        x = np.log(
            np.asarray(
                [row["standardized_distance"] for row in far_region], dtype=float
            )
        )
        y = np.log(np.asarray([row[field] for row in far_region], dtype=float))
        return ratio, float(np.polyfit(x, y, 1)[0])

    full_ratio, full_slope = ratio_and_slope(
        "weighted_full_parameter_gradient_norm"
    )
    output_ratio, output_slope = ratio_and_slope(
        "weighted_output_joint_gradient_norm"
    )
    summary = {
        "far_near_weighted_gradient_ratio": full_ratio,
        "far_loglog_slope": full_slope,
        "far_near_weighted_output_joint_ratio": output_ratio,
        "far_loglog_weighted_output_joint_slope": output_slope,
        "probe_rows": len(rows),
    }
    return rows, summary


def train_method(
    seed: int,
    initial_state: dict[str, torch.Tensor],
    environment: experiment.Environment,
    family: str,
    rho: float,
    output_root: Path,
    protocol: TaperProtocol,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    name = config_name(family, rho)
    summary_path = output_root / "runs" / name / f"seed_{seed}.json"
    trajectory_path = output_root / "runs" / name / f"seed_{seed}_trajectory.csv"
    diagnostic_path = output_root / "runs" / name / f"seed_{seed}_diagnostics.csv"
    if summary_path.exists() and trajectory_path.exists() and diagnostic_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        trajectory = experiment.read_csv(trajectory_path)
        diagnostics = experiment.read_csv(diagnostic_path)
        return summary, trajectory, diagnostics

    experiment.seed_all(seed + 900_000)
    actor = experiment.GaussianActor().to(experiment.DEVICE)
    actor.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.Adam(actor.parameters(), lr=protocol.learning_rate)
    index_generator = torch.Generator(device="cpu").manual_seed(seed + 700_003)

    initial_task = experiment.evaluation(actor, environment.test)
    initial_reward = float(initial_task["reward"])
    diagnostic_rows: list[dict[str, Any]] = []
    initial_raw, initial_diag = per_sample_gradient_diagnostic(
        seed, "initial", actor, environment, family, rho, protocol
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
                rho,
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
            rho,
            protocol,
            initial_reward,
        )
        row = {
            "seed": seed,
            "step": step,
            "family": family,
            "rho": rho,
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
                normalized_slope(window, field)
                for field in (
                    "reward",
                    "normalized_extrapolation_displacement",
                    "sigma_mean",
                )
            )
            residual = float(state["stationarity_residual"])
            residual_threshold = (
                protocol.positive_absolute_gradient_threshold
                if family == "positive_only"
                else protocol.normalized_field_residual_threshold
            )
            if (
                max_slope < protocol.normalized_slope_threshold
                and residual < residual_threshold
            ):
                stable_candidate_step = step
                # A 2x terminal audit is valid only when the full equal-length
                # continuation fits inside the pre-registered horizon.  A
                # candidate discovered after maximum_steps/2 remains
                # unresolved instead of being truncated and mislabeled 2x.
                audit_target_step = two_times_audit_target(
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
            if terminal_classification == candidate_classification:
                stop_reason = "stable_plateau_2x_confirmed"
            else:
                stop_reason = "terminal_classification_reversed"
            break

    elapsed = time.perf_counter() - started
    final_state = evaluate_state(
        actor,
        environment,
        family,
        rho,
        protocol,
        initial_reward,
    )
    terminal_raw, terminal_diag = per_sample_gradient_diagnostic(
        seed, "terminal", actor, environment, family, rho, protocol
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
        "rho": rho,
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
        "initial_far_loglog_slope": initial_diag["far_loglog_slope"],
        "terminal_far_loglog_slope": terminal_diag["far_loglog_slope"],
        "initial_far_near_weighted_output_joint_ratio": initial_diag[
            "far_near_weighted_output_joint_ratio"
        ],
        "terminal_far_near_weighted_output_joint_ratio": terminal_diag[
            "far_near_weighted_output_joint_ratio"
        ],
        "initial_far_loglog_weighted_output_joint_slope": initial_diag[
            "far_loglog_weighted_output_joint_slope"
        ],
        "terminal_far_loglog_weighted_output_joint_slope": terminal_diag[
            "far_loglog_weighted_output_joint_slope"
        ],
        "terminal_checkpoint": str(checkpoint_path),
    }
    atomic_json(summary_path, summary)
    return summary, trajectory, diagnostic_rows


def bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        value = float(array[0])
        return value, value, value
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def aggregate_results(
    summaries: list[dict[str, Any]],
    output_root: Path,
    protocol: TaperProtocol,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in summaries:
        grouped.setdefault((str(row["family"]), float(row["rho"])), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (family, rho), group in sorted(grouped.items()):
        reward = [float(row["reward"]) for row in group]
        ratio = [
            float(row["terminal_far_near_weighted_gradient_ratio"])
            for row in group
        ]
        reward_ci = bootstrap_mean_ci(
            reward,
            samples=protocol.bootstrap_samples,
            seed=20260625,
        )
        ratio_ci = bootstrap_mean_ci(
            ratio,
            samples=protocol.bootstrap_samples,
            seed=20260626,
        )
        aggregate_rows.append(
            {
                "family": family,
                "rho": rho,
                "n_seeds": len(group),
                "reward_mean": reward_ci[0],
                "reward_ci_low": reward_ci[1],
                "reward_ci_high": reward_ci[2],
                "terminal_far_near_gradient_ratio_mean": ratio_ci[0],
                "terminal_far_near_gradient_ratio_ci_low": ratio_ci[1],
                "terminal_far_near_gradient_ratio_ci_high": ratio_ci[2],
                "task_performance_collapse_events": sum(
                    bool(row["task_performance_collapse_event"]) for row in group
                ),
                "support_or_variance_boundary_events": sum(
                    bool(row["support_or_variance_boundary_event"]) for row in group
                ),
                "nan_inf_numerical_events": sum(
                    bool(row["nan_inf_numerical_event"]) for row in group
                ),
                "stable_plateau_2x_confirmed": sum(
                    row["stop_reason"] == "stable_plateau_2x_confirmed"
                    for row in group
                ),
            }
        )
    write_csv(output_root / "aggregate.csv", aggregate_rows)

    primary = {
        (str(row["family"]), int(row["seed"])): row
        for row in summaries
        if abs(float(row["rho"]) - protocol.primary_rho) < 1e-12
    }
    paired_rows: list[dict[str, Any]] = []
    for seed in protocol.formal_seeds:
        linear = primary.get(("reciprocal_linear", seed))
        quadratic = primary.get(("reciprocal_quadratic", seed))
        if linear is None or quadratic is None:
            continue
        paired_rows.append(
            {
                "seed": seed,
                "quadratic_minus_linear_reward": (
                    float(quadratic["reward"]) - float(linear["reward"])
                ),
                "quadratic_minus_linear_far_near_ratio": (
                    float(quadratic["terminal_far_near_weighted_gradient_ratio"])
                    - float(linear["terminal_far_near_weighted_gradient_ratio"])
                ),
                "quadratic_reward_higher": (
                    float(quadratic["reward"]) > float(linear["reward"])
                ),
                "quadratic_far_near_ratio_lower": (
                    float(quadratic["terminal_far_near_weighted_gradient_ratio"])
                    < float(linear["terminal_far_near_weighted_gradient_ratio"])
                ),
            }
        )
    write_csv(output_root / "paired_primary_comparison.csv", paired_rows)

    reward_differences = [
        float(row["quadratic_minus_linear_reward"]) for row in paired_rows
    ]
    ratio_differences = [
        float(row["quadratic_minus_linear_far_near_ratio"])
        for row in paired_rows
    ]
    paired_summary: dict[str, Any] = {
        "rho": protocol.primary_rho,
        "paired_seeds": len(paired_rows),
        "quadratic_reward_wins": sum(
            bool(row["quadratic_reward_higher"]) for row in paired_rows
        ),
        "quadratic_suppression_wins": sum(
            bool(row["quadratic_far_near_ratio_lower"]) for row in paired_rows
        ),
    }
    if paired_rows:
        paired_summary["reward_difference_mean_ci95"] = bootstrap_mean_ci(
            reward_differences,
            samples=protocol.bootstrap_samples,
            seed=20260627,
        )
        paired_summary["far_near_ratio_difference_mean_ci95"] = bootstrap_mean_ci(
            ratio_differences,
            samples=protocol.bootstrap_samples,
            seed=20260628,
        )
    atomic_json(output_root / "paired_primary_summary.json", paired_summary)
    return aggregate_rows, paired_summary


def build_terminal_audit(
    summaries: list[dict[str, Any]],
    paired_summary: dict[str, Any],
    protocol: TaperProtocol,
    *,
    base_commit: str,
    smoke: bool = False,
) -> dict[str, Any]:
    expected_configs = method_configs(protocol)
    resolved_terminal_reasons = {
        "stable_plateau_2x_confirmed",
        "support_or_variance_boundary_event",
        "nan_inf_numerical_event",
    }
    checks = [
        {
            "name": "all_registered_runs_present",
            "passed": len(summaries)
            == len(protocol.formal_seeds) * len(expected_configs),
            "value": len(summaries),
            "expected": len(protocol.formal_seeds) * len(expected_configs),
        },
        {
            "name": "primary_linear_quadratic_pairs_complete",
            "passed": paired_summary.get("paired_seeds")
            == len(protocol.formal_seeds),
            "value": paired_summary.get("paired_seeds"),
            "expected": len(protocol.formal_seeds),
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
            "name": "terminal_classification_available",
            "passed": all(bool(row.get("stop_reason")) for row in summaries),
            "value": True,
            "expected": True,
        },
        {
            "name": "all_runs_terminally_resolved",
            "passed": all(
                str(row.get("stop_reason")) in resolved_terminal_reasons
                for row in summaries
            ),
            "value": {
                reason: sum(str(row.get("stop_reason")) == reason for row in summaries)
                for reason in sorted(
                    {str(row.get("stop_reason")) for row in summaries}
                )
            },
            "expected": sorted(resolved_terminal_reasons),
        },
    ]
    all_checks_passed = all(
        check["passed"]
        for check in checks
        if not (smoke and check["name"] == "all_runs_terminally_resolved")
    )
    if smoke:
        scientific_status = "not run / 尚未运行"
    elif all_checks_passed:
        scientific_status = "long-run validated / 已长期验证"
    else:
        scientific_status = "finite-step validated / 有限训练步数验证"
    return {
        "experiment_id": EXPERIMENT_ID,
        "base_commit": base_commit,
        "execution_status": "engineering_smoke" if smoke else "formal_run",
        "scientific_status": scientific_status,
        "all_checks_passed": all_checks_passed,
        "checks": checks,
        "interpretation_boundary": (
            "The registered primary claim is stronger far-field suppression by "
            "reciprocal-quadratic versus reciprocal-linear at matched reference "
            "attenuation. A task-reward ranking is empirical and is not assumed."
        ),
    }


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

    global TAPER
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
        TAPER = TaperProtocol(
            formal_seeds=(70,),
            sensitivity_rhos=(),
            batch_states=32,
            evaluation_interval=1,
            minimum_steps=1,
            maximum_steps=3,
            stable_windows=2,
            probe_states=2,
            bootstrap_samples=20,
        )

    experiment.ROOT = output_root
    started = time.time()
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "base_commit": args.base_commit,
        "result_status": "engineering_smoke" if args.smoke else "running",
        "cu1_protocol": asdict(experiment.P),
        "taper_protocol": asdict(TAPER),
        "device": str(experiment.DEVICE),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "start_unix": started,
        "scope": (
            "same-distribution held-out-context generalization; not OOD; "
            "same standardized distance for every taper"
        ),
    }
    atomic_json(output_root / "experiment_manifest.json", manifest)
    source_snapshot = output_root / "source_snapshot"
    source_snapshot.mkdir(parents=True, exist_ok=True)
    for source in (
        Path(__file__),
        Path(experiment.__file__),
        Path(cu1_core.__file__),
    ):
        shutil.copy2(source, source_snapshot / source.name)

    preflight_environment = experiment.audit_environment(
        experiment.make_environment(TAPER.formal_seeds[0])
    )
    atomic_json(output_root / "environment_audit.json", preflight_environment)
    if not preflight_environment["passed"]:
        raise SystemExit("shared C-U1 environment audit failed")

    summaries: list[dict[str, Any]] = []
    all_trajectories: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    positive_rows: list[dict[str, Any]] = []

    for seed in TAPER.formal_seeds:
        log_message(output_root, f"seed={seed} positive-only initialization")
        _, environment, _, positive_summary = experiment.train_positive(seed)
        positive_rows.append(
            {
                **positive_summary,
                "method_initialization_source": "positive_only_adam_2000_step_checkpoint",
            }
        )
        # Match the frozen E3/E4 Adam protocol exactly.  train_positive also
        # performs later E2-only terminal audits, so the returned actor is not
        # a valid downstream initialization.
        initial_state = copy.deepcopy(experiment.load_initialization_state(seed))

        for family, rho in method_configs(TAPER):
            log_message(output_root, f"seed={seed} family={family} rho={rho:.2f}")
            summary, trajectory, diagnostics = train_method(
                seed,
                initial_state,
                environment,
                family,
                rho,
                output_root,
                TAPER,
            )
            summaries.append(summary)
            all_trajectories.extend(trajectory)
            all_diagnostics.extend(diagnostics)
            write_csv(output_root / "per_seed_runs_partial.csv", summaries)

    write_csv(output_root / "positive_summary.csv", positive_rows)
    write_csv(output_root / "per_seed_runs.csv", summaries)
    write_csv(output_root / "all_trajectories.csv", all_trajectories)
    write_csv(output_root / "all_gradient_diagnostics.csv", all_diagnostics)
    aggregate_rows, paired_summary = aggregate_results(
        summaries, output_root, TAPER
    )
    terminal_audit = build_terminal_audit(
        summaries,
        paired_summary,
        TAPER,
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
        }
    )
    atomic_json(output_root / "experiment_manifest.json", manifest)
    atomic_json(
        output_root / "RUN_COMPLETE.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "base_commit": args.base_commit,
            "result_status": manifest["result_status"],
            "terminal_audit_passed": terminal_audit["all_checks_passed"],
            "aggregate_rows": len(aggregate_rows),
        },
    )
    log_message(output_root, "run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
