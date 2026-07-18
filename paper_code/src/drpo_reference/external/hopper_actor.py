"""Hopper E7-Q2 actor objectives, training, and terminal artifacts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drpo_reference.common.io import atomic_json, write_csv

from .hopper_metrics import classify_actor_terminal, normalized_window_drift
from .hopper_models import SquashedGaussianPolicy
from .hopper_optim import (
    full_gradient_statistics,
    parameter_update_statistics,
    sample_indices,
    tensor,
)
from .hopper_protocol import METHODS, HopperProtocol

EPS = 1.0e-6
RolloutEvaluator = Callable[
    [SquashedGaussianPolicy, int, str],
    dict[str, Any],
]
Heartbeat = Callable[[str, int], None]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def actor_eval_metrics(
    *,
    policy: SquashedGaussianPolicy,
    observations: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    audit_indices: np.ndarray,
    fixed_negative_indices: np.ndarray,
    device: torch.device | str,
    loss_value: float,
    gradient_norm: float,
    gradient_rms: float,
    relative_gradient_norm: float,
    update_norm: float,
    relative_update_norm: float,
    step: int,
    boundary_threshold: float,
    rollout_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the fixed actor audit set without changing training mode."""

    resolved_device = torch.device(device)
    policy.eval()
    with torch.no_grad():
        obs_t = tensor(observations[audit_indices], resolved_device)
        action_t = tensor(actions[audit_indices], resolved_device)
        log_prob = policy.log_prob(obs_t, action_t)
        advantage_t = tensor(advantages[audit_indices], resolved_device)
        positive = advantage_t > 0
        positive_nll = (
            float((-log_prob[positive]).mean().cpu())
            if bool(positive.any())
            else float("nan")
        )
        mean_latent, log_std = policy.latent_parameters(obs_t)
        action_mean = torch.tanh(mean_latent)
        boundary = (
            action_mean.abs() >= boundary_threshold
        ).any(dim=-1)
        sigma = torch.exp(log_std)
        log_std_vector = policy.log_std.detach().clamp(
            policy.log_std_min,
            policy.log_std_max,
        )
        minimum_fraction = float(
            (
                log_std_vector
                <= policy.log_std_min + 1.0e-7
            )
            .float()
            .mean()
            .cpu()
        )
        maximum_fraction = float(
            (
                log_std_vector
                >= policy.log_std_max - 1.0e-7
            )
            .float()
            .mean()
            .cpu()
        )
        negative_obs = tensor(
            observations[fixed_negative_indices],
            resolved_device,
        )
        negative_actions = tensor(
            actions[fixed_negative_indices],
            resolved_device,
        )
        components = policy.score_components(
            negative_obs,
            negative_actions,
        )
        metrics: dict[str, Any] = {
            "step": step,
            "loss": loss_value,
            "positive_nll": positive_nll,
            "gradient_norm": gradient_norm,
            "gradient_rms": gradient_rms,
            "relative_gradient_norm": relative_gradient_norm,
            "update_norm": update_norm,
            "relative_update_norm": relative_update_norm,
            "mean_abs": float(action_mean.abs().mean().cpu()),
            "mean_boundary_fraction": float(
                boundary.float().mean().cpu()
            ),
            "sigma_mean": float(sigma.mean().cpu()),
            "sigma_min": float(sigma.min().cpu()),
            "sigma_max": float(sigma.max().cpu()),
            "log_std_min_fraction": minimum_fraction,
            "log_std_max_fraction": maximum_fraction,
            "phantom_distance_mean": float(
                components["radius"].mean().cpu()
            ),
            "phantom_mean_score_norm": float(
                components["mean_score_norm"].mean().cpu()
            ),
            "phantom_raw_log_scale_score_norm": float(
                components["raw_log_scale_score_norm"].mean().cpu()
            ),
            "phantom_corrected_q_xi_mean": float(
                components["corrected_q_xi"].mean().cpu()
            ),
            "phantom_joint_output_score_mean": float(
                components["joint_output_score_norm"].mean().cpu()
            ),
            "phantom_log_scale_to_mean_ratio": float(
                components["log_scale_to_mean_ratio"].mean().cpu()
            ),
        }
        metrics["phantom_score_mean"] = metrics[
            "phantom_joint_output_score_mean"
        ]
    if rollout_metrics:
        metrics.update(rollout_metrics)
    else:
        metrics.update(
            {
                "rollout_status": "not_evaluated",
                "rollout_return_mean": float("nan"),
                "rollout_return_std": float("nan"),
                "normalized_return": float("nan"),
                "normalized_return_available": False,
                "rollout_episodes": 0,
            }
        )
    policy.train()
    return metrics


def actor_batch_loss(
    policy: SquashedGaussianPolicy,
    observations: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    method: str,
    far_threshold: float,
    global_scale: float,
    far_cap_score: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply one of the six frozen E7-Q2 negative-weight controls."""

    if method not in METHODS:
        raise ValueError(f"unknown Hopper actor method: {method}")
    log_prob = policy.log_prob(observations, actions)
    weights = advantages.clone()
    components = policy.score_components(observations, actions)
    distance = components["radius"].detach()
    joint_score = components["joint_output_score_norm"].detach()
    negative = weights < 0
    far = negative & (distance > far_threshold)
    near = negative & ~far
    cap_factor = torch.ones_like(weights)
    dynamic_scale = 1.0
    proxy_before = 0.0
    proxy_target = 0.0
    proxy_after = 0.0

    if method == "positive_only":
        weights = torch.where(
            weights > 0,
            weights,
            torch.zeros_like(weights),
        )
    elif method == "near_zero":
        weights = torch.where(
            near,
            torch.zeros_like(weights),
            weights,
        )
    elif method == "far_zero":
        weights = torch.where(
            far,
            torch.zeros_like(weights),
            weights,
        )
    elif method == "far_cap":
        cap_factor = torch.minimum(
            torch.ones_like(weights),
            torch.full_like(weights, far_cap_score)
            / joint_score.clamp_min(EPS),
        )
        weights = torch.where(
            far,
            weights * cap_factor,
            weights,
        )
    elif method == "dynamic_budget_matched_global":
        if bool(negative.any()):
            magnitude = (-weights[negative]).detach()
            score = joint_score[negative]
            negative_far = far[negative]
            target_factor = torch.ones_like(magnitude)
            target_factor = torch.where(
                negative_far,
                torch.minimum(
                    torch.ones_like(magnitude),
                    torch.full_like(magnitude, far_cap_score)
                    / score.clamp_min(EPS),
                ),
                target_factor,
            )
            proxy_before_t = torch.sum(magnitude * score)
            proxy_target_t = torch.sum(
                magnitude * score * target_factor
            )
            dynamic_scale = float(
                torch.clamp(
                    proxy_target_t
                    / proxy_before_t.clamp_min(EPS),
                    0.0,
                    1.0,
                )
                .detach()
                .cpu()
            )
            weights = torch.where(
                negative,
                weights * dynamic_scale,
                weights,
            )
            proxy_before = float(proxy_before_t.detach().cpu())
            proxy_target = float(proxy_target_t.detach().cpu())
            proxy_after = proxy_before * dynamic_scale
    elif method == "signed":
        pass

    active = weights.ne(0)
    if not bool(active.any()):
        raise RuntimeError(
            f"method {method} produced an empty active batch"
        )
    loss = -(weights[active] * log_prob[active]).mean()
    diagnostics = {
        "active_fraction": float(
            active.float().mean().detach().cpu()
        ),
        "negative_fraction": float(
            negative.float().mean().detach().cpu()
        ),
        "far_negative_fraction": float(
            far.float().mean().detach().cpu()
        ),
        "far_cap_factor_mean": (
            float(cap_factor[far].mean().detach().cpu())
            if bool(far.any())
            else 1.0
        ),
        "dynamic_global_scale": dynamic_scale,
        "negative_influence_proxy_before": proxy_before,
        "negative_influence_proxy_target": proxy_target,
        "negative_influence_proxy_after": proxy_after,
    }
    return loss, diagnostics


def train_actor_stage(
    *,
    policy: SquashedGaussianPolicy,
    method: str,
    observations: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    train_indices: np.ndarray,
    audit_indices: np.ndarray,
    fixed_negative_indices: np.ndarray,
    protocol: HopperProtocol,
    min_steps: int,
    max_steps: int,
    eval_interval: int,
    seed: int,
    device: torch.device | str,
    output_dir: Path,
    far_threshold: float = float("inf"),
    global_scale: float = 1.0,
    far_cap_score: float = float("inf"),
    rollout_evaluator: RolloutEvaluator | None = None,
    rollout_eval_interval: int = 0,
    heartbeat: Heartbeat | None = None,
) -> tuple[SquashedGaussianPolicy, dict[str, Any]]:
    """Run the exact fixed-step E7-Q2 actor stage and terminal audit."""

    if method not in METHODS:
        raise ValueError(f"unknown Hopper actor method: {method}")
    if min_steps <= 0 or max_steps <= 0 or eval_interval <= 0:
        raise ValueError("actor step budgets and interval must be positive")
    if min_steps > max_steps:
        raise ValueError("min_steps exceeds max_steps")

    resolved_device = torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=protocol.actor_learning_rate,
        weight_decay=protocol.weight_decay,
    )
    generator = np.random.default_rng(seed + 2000)
    rows: list[dict[str, Any]] = []
    candidate_step: int | None = None
    extension_target: int | None = None
    train_batch_loss = float("nan")
    train_batch_gradient = float("inf")
    early_stop_reason: str | None = None
    last_diagnostics: dict[str, float] = {}
    evaluation_snapshot = [
        parameter.detach().clone()
        for parameter in policy.parameters()
    ]
    last_evaluation_step = 0

    def evaluate(
        step: int,
        update_statistics: dict[str, float],
    ) -> dict[str, Any]:
        audit_obs = tensor(
            observations[audit_indices],
            resolved_device,
        )
        audit_actions = tensor(
            actions[audit_indices],
            resolved_device,
        )
        audit_advantages = tensor(
            advantages[audit_indices],
            resolved_device,
        )
        audit_loss, audit_diagnostics = actor_batch_loss(
            policy,
            audit_obs,
            audit_actions,
            audit_advantages,
            method,
            far_threshold,
            global_scale,
            far_cap_score,
        )
        audit_gradient = full_gradient_statistics(
            audit_loss,
            policy.parameters(),
        )
        should_rollout = bool(
            rollout_evaluator
            and (
                step == 0
                or rollout_eval_interval <= 0
                or step % rollout_eval_interval == 0
            )
        )
        row = actor_eval_metrics(
            policy=policy,
            observations=observations,
            actions=actions,
            advantages=advantages,
            audit_indices=audit_indices,
            fixed_negative_indices=fixed_negative_indices,
            device=resolved_device,
            loss_value=float(audit_loss.detach().cpu()),
            gradient_norm=audit_gradient["raw"],
            gradient_rms=audit_gradient["rms"],
            relative_gradient_norm=(
                audit_gradient["relative_to_parameter_norm"]
            ),
            update_norm=update_statistics["raw_per_step"],
            relative_update_norm=(
                update_statistics["relative_per_step"]
            ),
            step=step,
            boundary_threshold=protocol.support_boundary_threshold,
            rollout_metrics=(
                rollout_evaluator(policy, step, method)
                if should_rollout and rollout_evaluator is not None
                else None
            ),
        )
        row.update(
            {
                f"audit_{key}": value
                for key, value in audit_diagnostics.items()
            }
        )
        row.update(
            {
                "train_batch_loss": train_batch_loss,
                "train_batch_gradient_norm": train_batch_gradient,
            }
        )
        return row

    policy.train()
    rows.append(
        evaluate(
            0,
            {
                "raw_per_step": 0.0,
                "rms_per_step": 0.0,
                "relative_per_step": 0.0,
            },
        )
    )
    if heartbeat is not None:
        heartbeat(f"actor:{method}", 0)

    for step in range(1, max_steps + 1):
        indices = sample_indices(
            generator,
            train_indices,
            protocol.actor_batch_size,
        )
        observation_t = tensor(
            observations[indices],
            resolved_device,
        )
        action_t = tensor(actions[indices], resolved_device)
        advantage_t = tensor(
            advantages[indices],
            resolved_device,
        )
        loss, last_diagnostics = actor_batch_loss(
            policy,
            observation_t,
            action_t,
            advantage_t,
            method,
            far_threshold,
            global_scale,
            far_cap_score,
        )
        optimizer.zero_grad(set_to_none=True)
        train_batch_loss = float(loss.detach().cpu())
        if not math.isfinite(train_batch_loss):
            early_stop_reason = "nonfinite_train_loss"
            train_batch_gradient = float("nan")
        else:
            loss.backward()
            train_batch_gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    protocol.max_gradient_norm,
                ).cpu()
            )
            if not math.isfinite(train_batch_gradient):
                early_stop_reason = "nonfinite_train_gradient"
            else:
                optimizer.step()

        if early_stop_reason is not None:
            optimizer.zero_grad(set_to_none=True)
            row = actor_eval_metrics(
                policy=policy,
                observations=observations,
                actions=actions,
                advantages=advantages,
                audit_indices=audit_indices,
                fixed_negative_indices=fixed_negative_indices,
                device=resolved_device,
                loss_value=train_batch_loss,
                gradient_norm=train_batch_gradient,
                gradient_rms=float("nan"),
                relative_gradient_norm=float("nan"),
                update_norm=float("nan"),
                relative_update_norm=float("nan"),
                step=step,
                boundary_threshold=(
                    protocol.support_boundary_threshold
                ),
            )
            row.update(last_diagnostics)
            row["numerical_failure_reason"] = (
                early_stop_reason
            )
            rows.append(row)
            if heartbeat is not None:
                heartbeat(f"actor:{method}", step)
            break

        if step % eval_interval == 0 or step == max_steps:
            update_statistics = parameter_update_statistics(
                evaluation_snapshot,
                policy.parameters(),
                step - last_evaluation_step,
            )
            evaluation_snapshot = [
                parameter.detach().clone()
                for parameter in policy.parameters()
            ]
            last_evaluation_step = step
            row = evaluate(step, update_statistics)
            rows.append(row)
            if heartbeat is not None:
                heartbeat(f"actor:{method}", step)
            state_drifts = [
                normalized_window_drift(
                    rows,
                    key,
                    protocol.audit_windows,
                )
                for key in (
                    "mean_abs",
                    "sigma_mean",
                    "phantom_distance_mean",
                )
            ]
            if (
                candidate_step is None
                and step >= min_steps
                and 2 * step <= max_steps
                and all(
                    value
                    <= protocol.actor_state_drift_tolerance
                    for value in state_drifts
                )
                and float(row["relative_update_norm"])
                <= protocol.actor_update_tolerance
            ):
                candidate_step = step
                extension_target = 2 * step

    parameters_finite = all(
        bool(torch.isfinite(parameter).all())
        for parameter in policy.parameters()
    )
    if (
        rollout_evaluator is not None
        and parameters_finite
        and not math.isfinite(
            float(
                rows[-1].get(
                    "normalized_return",
                    float("nan"),
                )
            )
        )
    ):
        rows[-1].update(
            rollout_evaluator(
                policy,
                int(rows[-1]["step"]),
                method,
            )
        )

    final_step = int(rows[-1]["step"])
    fixed_budget_completed = bool(
        final_step == max_steps
        and early_stop_reason is None
        and parameters_finite
        and math.isfinite(
            float(rows[-1].get("loss", float("nan")))
        )
    )
    if not fixed_budget_completed and early_stop_reason is None:
        early_stop_reason = (
            "incomplete_fixed_budget_unknown_reason"
        )

    checkpoint_path = output_dir / "terminal_actor.pt"
    torch.save(
        {
            "model": policy.state_dict(),
            "method": method,
            "step": final_step,
            "checkpoint_role": (
                "fixed_budget_final_checkpoint"
            ),
            "fixed_budget_steps": max_steps,
            "fixed_budget_completed": fixed_budget_completed,
            "far_threshold": far_threshold,
            "global_scale": global_scale,
            "global_scale_semantics": (
                "dynamic_per_batch_detached_output_score_proxy"
                if method == "dynamic_budget_matched_global"
                else "fixed_compatibility_or_unused"
            ),
            "far_cap_score": far_cap_score,
        },
        checkpoint_path,
    )
    extension_complete = bool(
        candidate_step is not None
        and rows[-1]["step"] >= 2 * candidate_step
    )
    audit = classify_actor_terminal(
        rows,
        protocol,
        candidate_step,
        extension_complete,
        fixed_budget_completed=fixed_budget_completed,
    )
    audit.update(
        {
            "method": method,
            "final_step": final_step,
            "max_steps": max_steps,
            "fixed_budget_steps": max_steps,
            "fixed_budget_completed": fixed_budget_completed,
            "reached_max_steps": fixed_budget_completed,
            "stopping_rule": "fixed_optimizer_steps",
            "early_stop_reason": early_stop_reason,
            "terminal_audit_controls_stopping": False,
            "extension_target": extension_target,
            "far_threshold": far_threshold,
            "global_scale": global_scale,
            "global_scale_semantics": (
                "dynamic_per_batch_detached_output_score_proxy"
                if method == "dynamic_budget_matched_global"
                else "fixed_compatibility_or_unused"
            ),
            "far_cap_score": far_cap_score,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256_file(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
            },
            "final_metrics": rows[-1],
        }
    )
    audit["terminal_audit_complete"] = bool(
        audit["fixed_budget_completed"]
        or audit["numerical_nonfinite"]
    )
    write_csv(output_dir / "curves.csv", rows)
    atomic_json(output_dir / "terminal_audit.json", audit)
    return policy, audit
