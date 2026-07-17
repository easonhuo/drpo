"""Canonical Hopper E7-Q2 critic training and terminal audit."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from drpo_reference.common import atomic_json, write_csv

from .hopper_advantages import critic_advantage_arrays
from .hopper_data import Normalizer, OfflineData
from .hopper_metrics import pearson, r2_score, relative_slope
from .hopper_models import ValueNetwork
from .hopper_optim import (
    EPS,
    full_gradient_statistics,
    parameter_update_statistics,
    sample_indices,
    spearman,
    tensor,
)
from .hopper_protocol import HopperProtocol


@dataclass
class CriticRun:
    critic: ValueNetwork
    target_normalizer: Normalizer
    metrics: list[dict[str, Any]]
    audit: dict[str, Any]
    advantages: dict[str, Any]


def _predict(
    model: ValueNetwork,
    observations: np.ndarray,
    indices: np.ndarray,
    target_normalizer: Normalizer,
    device: torch.device,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for offset in range(0, len(indices), 65_536):
            selected = indices[offset : offset + 65_536]
            chunks.append(
                model(tensor(observations[selected], device))
                .cpu()
                .numpy()
            )
    normalized = np.concatenate(chunks)
    return (
        normalized * float(target_normalizer.std[0])
        + float(target_normalizer.mean[0])
    )


def train_critic(
    *,
    data: OfflineData,
    split: dict[str, np.ndarray],
    observation_normalizer: Normalizer,
    returns: np.ndarray,
    protocol: HopperProtocol,
    seed: int,
    device: torch.device | str = "cpu",
    output_dir: Path | None = None,
) -> CriticRun:
    """Train the frozen-budget critic and select best validation MSE."""

    target_device = torch.device(device)
    observations = observation_normalizer.transform(data.observations)
    target_normalizer = Normalizer.fit(
        returns[split["train"]].reshape(-1, 1)
    )
    normalized_targets = target_normalizer.transform(
        returns.reshape(-1, 1)
    ).reshape(-1)
    model = ValueNetwork(
        observations.shape[1],
        protocol.hidden_sizes,
        protocol.activation,
        protocol.init_scheme,
        protocol.init_gain,
    ).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol.critic_learning_rate,
        weight_decay=protocol.weight_decay,
    )
    rng = np.random.default_rng(seed + 1000)
    train_audit = rng.choice(
        split["train"],
        size=min(protocol.audit_sample_size, len(split["train"])),
        replace=False,
    )
    validation_audit = rng.choice(
        split["validation"],
        size=min(
            protocol.audit_sample_size,
            len(split["validation"]),
        ),
        replace=False,
    )
    rows: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    candidate_step: int | None = None
    extension_target: int | None = None
    early_stop_reason: str | None = None
    snapshot = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]
    last_eval_step = 0

    def evaluate(
        step: int,
        update: dict[str, float],
    ) -> dict[str, Any]:
        model.eval()
        result: dict[str, Any] = {
            "step": step,
            "update_norm_per_step": update["raw_per_step"],
            "update_rms_per_step": update["rms_per_step"],
            "relative_update_norm_per_step": update[
                "relative_per_step"
            ],
        }
        for name in ("train", "validation", "test"):
            indices = split[name]
            prediction = _predict(
                model,
                observations,
                indices,
                target_normalizer,
                target_device,
            )
            truth = returns[indices]
            result[f"{name}_mse"] = float(
                np.mean((truth - prediction) ** 2)
            )
            result[f"{name}_r2"] = r2_score(truth, prediction)
            result[f"{name}_pearson"] = pearson(
                truth,
                prediction,
            )
        for name, indices in (
            ("train", train_audit),
            ("validation", validation_audit),
        ):
            prediction = model(
                tensor(observations[indices], target_device)
            )
            target = tensor(
                normalized_targets[indices],
                target_device,
            )
            audit_loss = F.mse_loss(prediction, target)
            gradient = full_gradient_statistics(
                audit_loss,
                model.parameters(),
            )
            result[f"{name}_audit_loss_normalized"] = float(
                audit_loss.detach().cpu()
            )
            result[f"{name}_gradient_norm"] = gradient["raw"]
            result[f"{name}_gradient_rms"] = gradient["rms"]
            result[f"{name}_relative_gradient_norm"] = gradient[
                "relative_to_parameter_norm"
            ]
        model.train()
        return result

    model.train()
    for step in range(1, protocol.critic_steps + 1):
        indices = sample_indices(
            rng,
            split["train"],
            protocol.critic_batch_size,
        )
        prediction = model(
            tensor(observations[indices], target_device)
        )
        loss = F.mse_loss(
            prediction,
            tensor(normalized_targets[indices], target_device),
        )
        optimizer.zero_grad(set_to_none=True)
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            early_stop_reason = "nonfinite_train_loss"
            break
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                10.0,
            )
            .detach()
            .cpu()
        )
        if not math.isfinite(gradient_norm):
            early_stop_reason = "nonfinite_train_gradient"
            break
        optimizer.step()
        if (
            step % protocol.critic_eval_interval == 0
            or step == protocol.critic_steps
        ):
            update = parameter_update_statistics(
                snapshot,
                model.parameters(),
                step - last_eval_step,
            )
            snapshot = [
                parameter.detach().clone()
                for parameter in model.parameters()
            ]
            last_eval_step = step
            row = evaluate(step, update)
            row["train_batch_loss_normalized"] = loss_value
            row["train_batch_gradient_norm"] = gradient_norm
            rows.append(row)
            validation_loss = float(row["validation_mse"])
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
            validation_slope = relative_slope(
                rows,
                "validation_mse",
                protocol.audit_windows,
            )
            train_slope = relative_slope(
                rows,
                "train_audit_loss_normalized",
                protocol.audit_windows,
            )
            if (
                candidate_step is None
                and step >= protocol.critic_min_steps
                and 2 * step <= protocol.critic_steps
                and validation_slope
                <= protocol.critic_relative_slope_tolerance
                and train_slope
                <= protocol.critic_relative_slope_tolerance
                and float(row["relative_update_norm_per_step"])
                <= protocol.critic_update_tolerance
            ):
                candidate_step = step
                extension_target = 2 * step

    if not rows or best_state is None:
        raise RuntimeError("critic produced no auditable checkpoint")
    final_step = int(rows[-1]["step"])
    fixed_budget_completed = bool(
        final_step == protocol.critic_steps
        and early_stop_reason is None
    )
    final_metrics = dict(rows[-1])
    final_state = copy.deepcopy(model.state_dict())
    final_advantages = critic_advantage_arrays(
        critic=model,
        data=data,
        observation_normalizer=observation_normalizer,
        target_normalizer=target_normalizer,
        gamma=protocol.gamma,
        standardize=protocol.advantage_standardize_once,
        standardization_indices=split["train"],
        device=target_device,
    )
    model.load_state_dict(best_state)
    selected_metrics = next(
        dict(row)
        for row in rows
        if int(row["step"]) == best_step
    )
    best_advantages = critic_advantage_arrays(
        critic=model,
        data=data,
        observation_normalizer=observation_normalizer,
        target_normalizer=target_normalizer,
        gamma=protocol.gamma,
        standardize=protocol.advantage_standardize_once,
        standardization_indices=split["train"],
        device=target_device,
    )
    indices = split["train"]
    best_stability = best_advantages["advantage"][indices]
    final_stability = final_advantages["advantage"][indices]
    sign_agreement = float(
        np.mean(
            np.sign(best_stability) == np.sign(final_stability)
        )
    )
    advantage_pearson = pearson(best_stability, final_stability)
    advantage_spearman = spearman(best_stability, final_stability)
    best_negative = best_stability < 0
    final_negative = final_stability < 0
    union = int(np.sum(best_negative | final_negative))
    negative_jaccard = (
        float(np.sum(best_negative & final_negative)) / union
        if union
        else 1.0
    )
    final_to_best = float(final_metrics["validation_mse"]) / max(
        float(selected_metrics["validation_mse"]),
        EPS,
    )
    extension_complete = bool(
        candidate_step is not None
        and final_step >= 2 * candidate_step
    )
    final_validation_slope = relative_slope(
        rows,
        "validation_mse",
        protocol.audit_windows,
    )
    final_train_slope = relative_slope(
        rows,
        "train_audit_loss_normalized",
        protocol.audit_windows,
    )
    stationarity_reconfirmed = bool(
        final_validation_slope
        <= protocol.critic_relative_slope_tolerance
        and final_train_slope
        <= protocol.critic_relative_slope_tolerance
        and float(rows[-1]["relative_update_norm_per_step"])
        <= protocol.critic_update_tolerance
    )
    optimization_terminal = bool(
        candidate_step is not None
        and extension_complete
        and stationarity_reconfirmed
    )
    operational_checks = {
        "fixed_budget_completed": fixed_budget_completed,
        "finite_selected_metrics": all(
            math.isfinite(float(selected_metrics[key]))
            for key in (
                "validation_mse",
                "validation_r2",
                "validation_pearson",
            )
        ),
    }
    quality_checks = {
        "validation_r2": (
            float(selected_metrics["validation_r2"])
            >= protocol.critic_validation_r2_min
        ),
        "validation_pearson": (
            float(selected_metrics["validation_pearson"])
            >= protocol.critic_validation_pearson_min
        ),
        "final_to_best_validation_mse_ratio": (
            final_to_best
            <= protocol.critic_max_final_to_best_validation_mse_ratio
        ),
        "advantage_sign_agreement": (
            sign_agreement
            >= protocol.critic_advantage_sign_agreement_min
        ),
        "advantage_pearson": (
            advantage_pearson
            >= protocol.critic_advantage_pearson_min
        ),
        "advantage_spearman": (
            advantage_spearman
            >= protocol.critic_advantage_spearman_min
        ),
        "negative_set_jaccard": (
            negative_jaccard
            >= protocol.critic_negative_set_jaccard_min
        ),
    }
    critic_accepted = all(operational_checks.values())
    audit = {
        "best_step": best_step,
        "best_validation_mse": best_loss,
        "stopping_rule": "fixed_optimizer_steps",
        "fixed_budget_steps": protocol.critic_steps,
        "fixed_budget_completed": fixed_budget_completed,
        "early_stop_reason": early_stop_reason,
        "terminal_audit_controls_stopping": False,
        "candidate_step": candidate_step,
        "extension_target": extension_target,
        "extension_complete": extension_complete,
        "final_stationarity_reconfirmed": stationarity_reconfirmed,
        "validation_mse_relative_slope": final_validation_slope,
        "train_audit_loss_relative_slope": final_train_slope,
        "final_train_gradient_norm_diagnostic": rows[-1][
            "train_gradient_norm"
        ],
        "final_validation_gradient_norm_diagnostic": rows[-1][
            "validation_gradient_norm"
        ],
        "final_update_norm_per_step_raw": rows[-1][
            "update_norm_per_step"
        ],
        "final_relative_update_norm_per_step": rows[-1][
            "relative_update_norm_per_step"
        ],
        "optimization_terminal": optimization_terminal,
        "critic_accepted_for_frozen_advantage": critic_accepted,
        "operational_acceptance_checks": operational_checks,
        "critic_quality_audit_passed": all(quality_checks.values()),
        "quality_audit_checks": quality_checks,
        "acceptance_metrics": {
            "final_to_best_validation_mse_ratio": final_to_best,
            "advantage_sign_agreement": sign_agreement,
            "advantage_pearson": advantage_pearson,
            "advantage_spearman": advantage_spearman,
            "negative_set_jaccard": negative_jaccard,
            "stability_scope": "actor_training_split",
            "stability_sample_count": int(len(indices)),
            "test_r2_report_only": float(selected_metrics["test_r2"]),
            "test_pearson_report_only": float(
                selected_metrics["test_pearson"]
            ),
        },
        "selected_checkpoint_role": (
            "best_validation_checkpoint"
            if critic_accepted
            else "best_validation_checkpoint_for_pilot_diagnostics"
        ),
        "selected_checkpoint_step": best_step,
        "terminal_checkpoint_eligible": False,
        "selected_checkpoint_metrics": selected_metrics,
        "final_training_metrics": final_metrics,
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": best_state,
                "obs_mean": observation_normalizer.mean,
                "obs_std": observation_normalizer.std,
                "target_mean": target_normalizer.mean,
                "target_std": target_normalizer.std,
                "step": best_step,
                "checkpoint_role": audit[
                    "selected_checkpoint_role"
                ],
            },
            output / "canonical_critic.pt",
        )
        torch.save(
            {
                "model": final_state,
                "step": final_step,
                "checkpoint_role": "final_training_checkpoint",
            },
            output / "final_training_critic.pt",
        )
        write_csv(output / "critic_metrics.csv", rows)
        atomic_json(output / "critic_terminal_audit.json", audit)
        np.savez_compressed(
            output / "frozen_advantages.npz",
            **best_advantages,
        )
    return CriticRun(
        critic=model,
        target_normalizer=target_normalizer,
        metrics=rows,
        audit=audit,
        advantages=best_advantages,
    )
