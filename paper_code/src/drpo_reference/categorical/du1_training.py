"""D-U1 revision-4 shared start, method training, and terminal audit."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from drpo_reference.common import seed_all

from .du1_controls import (
    coordinate_calibration,
    negative_loss_and_diagnostics,
    rarity_logit_anchor_loss,
)
from .du1_environment import CartesianSemanticEnvironment
from .du1_metrics import evaluate, policy_geometry_audit
from .du1_policy import (
    CartesianPolicy,
    batch_indices,
    cache_reference_directions,
    cell_log_probs,
    trainable_parameters,
)
from .du1_protocol import DU1Protocol, MethodSpec


@dataclass(frozen=True)
class DU1TerminalProtocol:
    """Frozen two-window terminal audit for the revision-4 formal horizon."""

    window_1_steps: tuple[int, int] = (4000, 6000)
    window_2_steps: tuple[int, int] = (6000, 8000)
    metric_tolerances: tuple[tuple[str, float], ...] = (
        ("expected_semantic_reward", 0.01),
        ("hidden_optimal_family_probability", 0.02),
        ("prototype_entropy_mean", 0.08),
        ("rarity_logit_gap_mean", 0.20),
    )

    def __post_init__(self) -> None:
        if self.window_1_steps[0] > self.window_1_steps[1]:
            raise ValueError("window_1_steps are reversed")
        if self.window_2_steps[0] > self.window_2_steps[1]:
            raise ValueError("window_2_steps are reversed")
        if self.window_1_steps[1] > self.window_2_steps[0]:
            raise ValueError("terminal windows must not overlap out of order")
        if any(tolerance < 0.0 for _, tolerance in self.metric_tolerances):
            raise ValueError("terminal tolerances must be non-negative")

    def legacy_mapping(self) -> dict[str, Any]:
        return {
            "mode": "formal_two_x_windows",
            "formal_horizon_steps": self.window_2_steps[1],
            "window_1_steps": list(self.window_1_steps),
            "window_2_steps": list(self.window_2_steps),
            "metric_window_mean_abs_tolerances": dict(self.metric_tolerances),
        }


@dataclass
class SharedStart:
    environment: CartesianSemanticEnvironment
    state_dict: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    calibration: dict[str, float]
    audit: dict[str, Any]


@dataclass
class MethodRun:
    model: CartesianPolicy
    trajectory: list[dict[str, Any]]
    summary: dict[str, Any]


def legacy_run_config(
    protocol: DU1Protocol,
    terminal: DU1TerminalProtocol,
) -> dict[str, Any]:
    config = protocol.legacy_config()
    config["terminal_audit"] = terminal.legacy_mapping()
    return config


def move_environment(
    environment: CartesianSemanticEnvironment,
    device: torch.device,
) -> None:
    environment.action_embeddings = environment.action_embeddings.to(device)
    for split in (environment.train, environment.test):
        for key, value in list(split.items()):
            if isinstance(value, torch.Tensor):
                split[key] = value.to(device)


def parameter_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            parameter.detach().reshape(-1).cpu()
            for parameter in trainable_parameters(model)
        ]
    )


def make_optimizer(
    model: CartesianPolicy,
    protocol: DU1Protocol,
) -> torch.optim.Adam:
    return torch.optim.Adam(
        trainable_parameters(model),
        lr=protocol.learning_rate,
        betas=(protocol.adam_beta1, protocol.adam_beta2),
        eps=protocol.adam_eps,
    )


def build_shared_start(
    protocol: DU1Protocol,
    seed: int,
    device: torch.device,
) -> SharedStart:
    """Build the exact common model/Adam start shared by all six methods."""

    seed_all(seed)
    environment = CartesianSemanticEnvironment(protocol, seed)
    environment_audit = environment.audit()
    model = CartesianPolicy(protocol, environment).to(device)
    move_environment(environment, device)
    cache_reference_directions(model, environment)
    initial_geometry = policy_geometry_audit(model, environment, protocol)
    environment_audit["policy_geometry_initial"] = initial_geometry
    environment_audit["passed"] = bool(
        environment_audit["passed"] and initial_geometry["passed"]
    )
    if not environment_audit["passed"]:
        raise RuntimeError(
            f"initial D-U1 environment audit failed for seed {seed}: "
            f"{environment_audit}"
        )

    optimizer = make_optimizer(model, protocol)
    initial_gap = float(
        (
            2.0
            * model.rarity_coordinate(environment.train["states"]).abs()
        )
        .mean()
        .detach()
    )
    initial_vector = parameter_vector(model)
    final_loss = 0.0
    for step in range(1, protocol.positive_warm_start_steps + 1):
        index = batch_indices(
            seed + 50_000,
            step,
            environment.train_count,
            protocol.batch_size,
        ).to(device)
        states = environment.train["states"][index]
        positive_log_probability, _, _ = cell_log_probs(
            model,
            environment,
            environment.train,
            index,
        )
        loss = (
            -positive_log_probability.mean()
            + protocol.rarity_logit_anchor_coefficient
            * rarity_logit_anchor_loss(model, states)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable_parameters(model)
        )
        if not finite:
            raise RuntimeError(
                f"positive warm-start numerical failure for seed {seed}"
            )
        optimizer.step()
        final_loss = float(loss.detach())

    final_gap = float(
        (
            2.0
            * model.rarity_coordinate(environment.train["states"]).abs()
        )
        .mean()
        .detach()
    )
    gap_drift = abs(final_gap - initial_gap)
    if gap_drift > protocol.maximum_positive_only_rarity_gap_drift:
        raise RuntimeError(
            f"positive warm-start changed rarity gap for seed {seed}: "
            f"{gap_drift}"
        )
    warm_summary = {
        "steps": protocol.positive_warm_start_steps,
        "final_loss": final_loss,
        "parameter_delta_norm": float(
            (parameter_vector(model) - initial_vector).norm()
        ),
        "initial_rarity_logit_gap_mean": initial_gap,
        "final_rarity_logit_gap_mean": final_gap,
        "rarity_logit_gap_abs_drift": gap_drift,
        "numerical_failure": False,
    }
    warm_geometry = policy_geometry_audit(model, environment, protocol)
    environment_audit["positive_warm_start"] = warm_summary
    environment_audit["policy_geometry_after_warm_start"] = warm_geometry
    environment_audit["passed"] = bool(
        environment_audit["passed"] and warm_geometry["passed"]
    )
    if not environment_audit["passed"]:
        raise RuntimeError(
            f"post-warm-start D-U1 audit failed for seed {seed}: "
            f"{environment_audit}"
        )
    calibration = coordinate_calibration(model, environment, protocol)
    return SharedStart(
        environment=environment,
        state_dict=copy.deepcopy(model.state_dict()),
        optimizer_state=copy.deepcopy(optimizer.state_dict()),
        calibration=calibration,
        audit=environment_audit,
    )


def terminal_classification(
    trajectory: Sequence[Mapping[str, Any]],
    terminal: DU1TerminalProtocol,
) -> dict[str, Any]:
    if any(
        not bool(row.get("environment_valid", True))
        for row in trajectory
    ):
        return {
            "class": "environment_invalid",
            "formal_acceptance": False,
        }
    if any(
        bool(row["nan_inf_numerical_failure"])
        for row in trajectory
    ):
        return {
            "class": "nan_inf_numerical_failure",
            "formal_acceptance": True,
        }
    if any(bool(row["support_boundary_event"]) for row in trajectory):
        return {
            "class": "support_boundary",
            "formal_acceptance": True,
        }

    first_rows = [
        row
        for row in trajectory
        if terminal.window_1_steps[0]
        <= int(row["step"])
        <= terminal.window_1_steps[1]
    ]
    second_rows = [
        row
        for row in trajectory
        if terminal.window_2_steps[0]
        <= int(row["step"])
        <= terminal.window_2_steps[1]
    ]
    if not first_rows or not second_rows:
        return {
            "class": "incomplete_terminal_windows",
            "formal_acceptance": False,
        }

    deltas: dict[str, float] = {}
    passed = True
    for metric, tolerance in terminal.metric_tolerances:
        first = float(
            np.mean([float(row[metric]) for row in first_rows])
        )
        second = float(
            np.mean([float(row[metric]) for row in second_rows])
        )
        delta = abs(second - first)
        deltas[metric] = delta
        passed = passed and delta <= tolerance
    return {
        "class": (
            "terminal_plateau"
            if passed
            else "persistent_drift_or_inconclusive"
        ),
        "formal_acceptance": passed,
        "window_mean_abs_deltas": deltas,
    }


def run_method(
    *,
    protocol: DU1Protocol,
    terminal: DU1TerminalProtocol,
    seed: int,
    spec: MethodSpec,
    base_state: Mapping[str, torch.Tensor],
    base_optimizer_state: Mapping[str, Any],
    calibration: Mapping[str, float],
    device: torch.device,
) -> MethodRun:
    """Run one frozen revision-4 method from the shared model/Adam state."""

    seed_all(seed)
    environment = CartesianSemanticEnvironment(protocol, seed)
    model = CartesianPolicy(protocol, environment).to(device)
    model.load_state_dict(base_state)
    move_environment(environment, device)
    cache_reference_directions(model, environment)
    parameters = trainable_parameters(model)
    optimizer = make_optimizer(model, protocol)
    optimizer.load_state_dict(copy.deepcopy(base_optimizer_state))

    trajectory: list[dict[str, Any]] = []
    last_diagnostics = {
        "weight_useful_common": 0.0,
        "weight_useful_rare": 0.0,
        "weight_unhelpful_common": 0.0,
        "weight_unhelpful_rare": 0.0,
        "negative_raw_gradient_norm": 0.0,
        "negative_target_gradient_norm": 0.0,
        "negative_applied_gradient_norm": 0.0,
        "stepwise_budget_match_error": 0.0,
        "stepwise_global_scale": 0.0,
        "rarity_logit_anchor_loss": 0.0,
    }
    environment_failure = False
    last_update_norm = 0.0

    def record(step: int, numerical_failure: bool = False) -> None:
        nonlocal environment_failure
        metrics = evaluate(
            model,
            environment,
            environment.test,
            calibration,
        )
        utility_valid = (
            metrics["utility_oracle_sign_valid_fraction"]
            >= protocol.utility_sign_fraction_min
        )
        rarity_valid = (
            metrics["rarity_coordinate_positive_fraction"]
            >= protocol.utility_sign_fraction_min
        )
        environment_valid = bool(utility_valid and rarity_valid)
        environment_failure = (
            environment_failure or not environment_valid
        )
        prototype_event = (
            metrics["prototype_effective_support"]
            < protocol.prototype_effective_support_boundary
        )
        rarity_event = (
            min(
                metrics["common_total_probability"],
                metrics["rare_total_probability"],
            )
            < protocol.rarity_mass_boundary
        )
        trajectory.append(
            {
                "seed": seed,
                "method": spec.method,
                "step": step,
                **metrics,
                **last_diagnostics,
                "adam_parameter_update_norm": last_update_norm,
                "environment_utility_sign_valid": bool(
                    utility_valid
                ),
                "environment_rarity_role_valid": bool(
                    rarity_valid
                ),
                "environment_valid": environment_valid,
                "prototype_support_boundary_event": bool(
                    prototype_event
                ),
                "rarity_mass_boundary_event": bool(rarity_event),
                "support_boundary_event": bool(
                    prototype_event or rarity_event
                ),
                "nan_inf_numerical_failure": (
                    numerical_failure
                ),
            }
        )

    record(0)
    numerical_failure = False
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
        negative_loss, last_diagnostics = (
            negative_loss_and_diagnostics(
                cells=cells,
                spec=spec,
                calibration=calibration,
                protocol=protocol,
                model=model,
            )
        )
        anchor = rarity_logit_anchor_loss(model, states)
        last_diagnostics["rarity_logit_anchor_loss"] = float(
            anchor.detach()
        )
        loss = (
            positive_loss
            + protocol.negative_alpha * negative_loss
            + protocol.rarity_logit_anchor_coefficient * anchor
        )
        measure = (
            step % protocol.evaluation_interval_steps == 0
            or step == protocol.maximum_steps
        )
        before = parameter_vector(model) if measure else None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )
        if not finite:
            numerical_failure = True
            record(step, True)
            break
        optimizer.step()
        if measure:
            assert before is not None
            last_update_norm = float(
                (parameter_vector(model) - before).norm()
            )
            record(step)

    terminal_result = terminal_classification(
        trajectory,
        terminal,
    )
    final = trajectory[-1]
    summary = {
        "seed": seed,
        "method": spec.method,
        "active_cells": list(spec.active_cells),
        "taper_family": spec.taper_family,
        "steps_completed": int(final["step"]),
        "terminal_class": terminal_result["class"],
        "terminal_formal_acceptance": bool(
            terminal_result["formal_acceptance"]
        ),
        "terminal_audit": terminal_result,
        "task_performance_collapse": False,
        "prototype_support_boundary_event": any(
            bool(row["prototype_support_boundary_event"])
            for row in trajectory
        ),
        "rarity_mass_boundary_event": any(
            bool(row["rarity_mass_boundary_event"])
            for row in trajectory
        ),
        "support_boundary_event": any(
            bool(row["support_boundary_event"])
            for row in trajectory
        ),
        "nan_inf_numerical_failure": numerical_failure,
        "environment_validity_failure": environment_failure,
        "minimum_utility_oracle_sign_valid_fraction": min(
            float(row["utility_oracle_sign_valid_fraction"])
            for row in trajectory
        ),
        "final_expected_semantic_reward": float(
            final["expected_semantic_reward"]
        ),
        "final_hidden_optimal_family_probability": float(
            final["hidden_optimal_family_probability"]
        ),
        "final_action_effective_support": float(
            final["action_effective_support"]
        ),
        "final_prototype_effective_support": float(
            final["prototype_effective_support"]
        ),
        "final_rare_total_probability": float(
            final["rare_total_probability"]
        ),
        "final_rarity_logit_gap_mean": float(
            final["rarity_logit_gap_mean"]
        ),
        "max_stepwise_budget_match_error": max(
            float(row["stepwise_budget_match_error"])
            for row in trajectory
        ),
        "coordinate_calibration": dict(calibration),
    }
    return MethodRun(
        model=model,
        trajectory=trajectory,
        summary=summary,
    )
