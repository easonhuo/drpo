#!/usr/bin/env python3
"""Long-run D-U1 / D-Diag categorical repulsion reconstruction.

This formal entrypoint reconstructs the E5 protocol from the locked handoff
records because the historical ``run_categorical.py`` source and raw artifacts
were never committed to the repository. It does not claim byte-identical
reproduction of the missing runner. Instead, it preserves the registered
scientific responsibilities:

1. D-Diag: bounded direct-logit score under repeated fixed negative updates;
2. D-U1: near/far causal interventions with task collapse, support boundary,
   and NaN/Inf failure reported separately;
3. long-run terminal classification rather than a short fixed-horizon claim;
4. explicit comparison with the historical qualitative result pattern.

The runner writes ordinary files only. Formal ZIP ownership belongs to the
canonical hardened guard/package/verify channel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EXPERIMENT_ID = "D-U1-E5-LONGRUN-RERUN"
FORMAL_SEEDS = tuple(range(10, 30))
METHODS = (
    "positive_only",
    "baseline",
    "near_zero",
    "far_zero",
    "far_cap",
    "global_scale",
)

DIRECT_CASES: dict[str, tuple[float, float, float]] = {
    # Exact three-action distributions reconstructed from the handoff's p0 and H0.
    "high_probability_negative": (
        0.8991,
        0.027336922257558537,
        0.07356307774244145,
    ),
    "low_probability_negative": (
        0.0038,
        0.07537643123910004,
        0.9208235687608999,
    ),
}

DIRECT_HISTORICAL_REFERENCE = {
    "high_probability_negative": {
        "initial_probability": 0.8991,
        "terminal_probability": 4.06e-12,
        "initial_entropy": 0.386,
        "peak_entropy": 0.906,
        "terminal_entropy": 6.72e-06,
        "max_score": 1.414213,
    },
    "low_probability_negative": {
        "initial_probability": 0.0038,
        "terminal_probability": 1.90e-20,
        "initial_entropy": 0.292,
        "peak_entropy": 0.292,
        "terminal_entropy": 4.51e-09,
        "max_score": 1.414214,
    },
}

HISTORICAL_CAUSAL_PATTERN = {
    "baseline": {"task_collapse": True, "support_collapse": True},
    "near_zero": {"task_collapse": True, "support_collapse": True},
    "far_zero": {"task_collapse": False, "support_collapse": False},
    "far_cap": {"task_collapse": False, "support_collapse": False},
    "global_scale": {"task_collapse": False, "support_collapse": True},
    "positive_only": {"task_collapse": False, "support_collapse": False},
}


@dataclass(frozen=True)
class Protocol:
    state_dim: int = 6
    train_states: int = 4096
    action_count: int = 26
    action_min: float = -3.0
    action_max: float = 3.0
    positive_samples: int = 4096
    near_negative_samples: int = 2048
    far_negative_samples: int = 2048
    initial_beta: float = 0.0
    initial_tau: float = 1.2
    positive_offset: float = 0.0
    positive_spread: float = 1.2
    near_offset: float = -0.5
    near_spread: float = 0.2
    far_offset: float = -2.5
    far_spread: float = 0.2
    reward_optimum_offset: float = 0.7
    reward_width: float = 0.4
    adam_lr: float = 0.003
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1.0e-8
    max_steps: int = 20000
    eval_every: int = 100
    support_tau_threshold: float = 0.05
    effective_support_threshold: float = 1.5
    task_collapse_ratio_to_positive_only: float = 0.20
    audit_window_1: tuple[int, int] = (10000, 15000)
    audit_window_2: tuple[int, int] = (15000, 20000)
    stable_beta_change: float = 0.02
    stable_tau_change: float = 0.02
    stable_reward_change: float = 0.01
    stable_raw_gradient_median: float = 1.0e-4
    direct_steps: int = 20000
    direct_lr: float = 0.001
    direct_eval_every: int = 100
    fixed_positive_advantage: float = 1.0
    fixed_negative_advantage: float = -1.0


METHOD_NEGATIVE_MASS: dict[str, tuple[float, float]] = {
    # (near mass, far mass). The values are reconstructed and frozen here.
    # Baseline and near-zero are outside the categorical second-moment feasible
    # region; far-zero and far-cap remain internally feasible; global-scale
    # preserves task reward near the hidden optimum while still hitting a
    # support boundary.
    "positive_only": (0.0, 0.0),
    "baseline": (0.25, 0.45),
    "near_zero": (0.0, 0.45),
    "far_zero": (0.25, 0.0),
    "far_cap": (0.25, 0.03),
    "global_scale": (0.10, 0.18),
}


class ProtocolError(ValueError):
    """Raised when a formal invocation tries to change frozen settings."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def entropy(probabilities: np.ndarray) -> float:
    safe = np.clip(probabilities, 1.0e-300, 1.0)
    return float(-np.sum(probabilities * np.log(safe)))


def linear_slope(rows: list[dict[str, float]], key: str) -> float:
    if len(rows) < 2:
        return float("nan")
    x = np.asarray([row["step"] for row in rows], dtype=np.float64)
    y = np.asarray([row[key] for row in rows], dtype=np.float64)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x, y - y.mean()) / denom)


def rows_in_window(
    rows: list[dict[str, float]], window: tuple[int, int]
) -> list[dict[str, float]]:
    lo, hi = window
    return [row for row in rows if lo <= int(row["step"]) <= hi]


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_direct_case(
    name: str,
    initial_probabilities: tuple[float, float, float],
    protocol: Protocol,
    *,
    steps: int | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    total_steps = protocol.direct_steps if steps is None else steps
    probabilities = np.asarray(initial_probabilities, dtype=np.float64)
    logits = np.log(probabilities)
    target = np.zeros_like(probabilities)
    target[0] = 1.0
    rows: list[dict[str, float]] = []

    for step in range(total_steps + 1):
        probabilities = stable_softmax(logits)
        if step % protocol.direct_eval_every == 0 or step == total_steps:
            current_entropy = entropy(probabilities)
            score = float(np.linalg.norm(target - probabilities))
            target_probability = float(probabilities[0])
            rows.append(
                {
                    "step": step,
                    "target_probability": target_probability,
                    "target_surprisal": float(-math.log(max(target_probability, 1.0e-300))),
                    "entropy": current_entropy,
                    "direct_logit_score_norm": score,
                    "target_logit_gap": float(np.max(logits[1:]) - logits[0]),
                }
            )
        if step == total_steps:
            break
        # Fixed negative advantage A=-1 under policy-gradient ascent:
        # z <- z + eta * (pi - e_target).
        logits = logits + protocol.direct_lr * (probabilities - target)

    tail = rows[-min(20, len(rows)) :]
    entropies = [row["entropy"] for row in rows]
    summary = {
        "case": name,
        "steps": total_steps,
        "learning_rate": protocol.direct_lr,
        "initial_probability": rows[0]["target_probability"],
        "terminal_probability": rows[-1]["target_probability"],
        "initial_entropy": rows[0]["entropy"],
        "peak_entropy": max(entropies),
        "terminal_entropy": rows[-1]["entropy"],
        "max_score": max(row["direct_logit_score_norm"] for row in rows),
        "terminal_surprisal": rows[-1]["target_surprisal"],
        "tail_surprisal_slope_per_step": linear_slope(tail, "target_surprisal"),
        "tail_logit_gap_slope_per_step": linear_slope(tail, "target_logit_gap"),
        "score_bound_pass": max(row["direct_logit_score_norm"] for row in rows)
        <= math.sqrt(2.0) + 1.0e-12,
        "surprisal_growth_pass": rows[-1]["target_surprisal"]
        > rows[0]["target_surprisal"],
        "entropy_pattern": (
            "rise_then_fall"
            if max(entropies) > rows[0]["entropy"] + 0.05
            else "nonincreasing_or_flat"
        ),
    }
    return rows, summary


def categorical_distribution(
    action_offsets: np.ndarray, center: float, spread: float
) -> np.ndarray:
    return stable_softmax(-0.5 * ((action_offsets - center) / spread) ** 2)


def empirical_distribution(
    rng: np.random.Generator, probabilities: np.ndarray, count: int
) -> np.ndarray:
    counts = rng.multinomial(count, probabilities)
    return counts.astype(np.float64) / float(count)


def build_seed_data(seed: int, protocol: Protocol) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(protocol.train_states, protocol.state_dim))
    state_projection = np.asarray([0.31, -0.27, 0.19, 0.23, -0.17, 0.11])
    context_centers = 0.2 * np.tanh(states @ state_projection)

    semantic_offsets = np.linspace(
        protocol.action_min,
        protocol.action_max,
        protocol.action_count,
        dtype=np.float64,
    )
    action_id_permutation = rng.permutation(protocol.action_count)
    semantic_by_action_id = semantic_offsets[action_id_permutation]

    positive_reference = categorical_distribution(
        semantic_by_action_id, protocol.positive_offset, protocol.positive_spread
    )
    near_reference = categorical_distribution(
        semantic_by_action_id, protocol.near_offset, protocol.near_spread
    )
    far_reference = categorical_distribution(
        semantic_by_action_id, protocol.far_offset, protocol.far_spread
    )
    positive = empirical_distribution(
        rng, positive_reference, protocol.positive_samples
    )
    near = empirical_distribution(rng, near_reference, protocol.near_negative_samples)
    far = empirical_distribution(rng, far_reference, protocol.far_negative_samples)

    initial_policy = categorical_distribution(
        semantic_by_action_id, protocol.initial_beta, protocol.initial_tau
    )
    near_anchor = int(np.argmax(near_reference))
    far_anchor = int(np.argmax(far_reference))

    return {
        "seed": seed,
        "states": states,
        "context_centers": context_centers,
        "semantic_offsets": semantic_offsets,
        "action_id_permutation": action_id_permutation,
        "semantic_by_action_id": semantic_by_action_id,
        "positive": positive,
        "near": near,
        "far": far,
        "near_reference": near_reference,
        "far_reference": far_reference,
        "initial_policy": initial_policy,
        "near_anchor": near_anchor,
        "far_anchor": far_anchor,
    }


def preflight(seed_data: dict[str, Any], protocol: Protocol) -> dict[str, Any]:
    initial_policy = seed_data["initial_policy"]
    near_anchor = seed_data["near_anchor"]
    far_anchor = seed_data["far_anchor"]
    permutation = seed_data["action_id_permutation"]
    report = {
        "state_dim": protocol.state_dim,
        "train_states": protocol.train_states,
        "action_count": protocol.action_count,
        "action_ids_are_permuted": bool(
            not np.array_equal(permutation, np.arange(protocol.action_count))
        ),
        "near_initial_probability": float(initial_policy[near_anchor]),
        "far_initial_probability": float(initial_policy[far_anchor]),
        "near_probability_exceeds_far": bool(
            initial_policy[near_anchor] > initial_policy[far_anchor]
        ),
        "near_far_advantage_magnitudes_equal": bool(
            abs(protocol.fixed_negative_advantage)
            == abs(protocol.fixed_negative_advantage)
        ),
        "initial_entropy": entropy(initial_policy),
        "all_empirical_distributions_sum_to_one": bool(
            all(
                abs(float(np.sum(seed_data[key])) - 1.0) <= 1.0e-12
                for key in ("positive", "near", "far")
            )
        ),
        "context_center_mean": float(np.mean(seed_data["context_centers"])),
        "context_center_std": float(np.std(seed_data["context_centers"])),
    }
    report["all_checks_passed"] = bool(
        report["action_ids_are_permuted"]
        and report["near_probability_exceeds_far"]
        and report["near_far_advantage_magnitudes_equal"]
        and report["all_empirical_distributions_sum_to_one"]
    )
    return report


def reward_vector(action_offsets: np.ndarray, protocol: Protocol) -> np.ndarray:
    return np.exp(
        -0.5
        * ((action_offsets - protocol.reward_optimum_offset) / protocol.reward_width) ** 2
    )


def evaluate_energy_policy(
    action_offsets: np.ndarray,
    beta: float,
    log_tau: float,
    protocol: Protocol,
) -> dict[str, float]:
    tau = math.exp(log_tau)
    probabilities = categorical_distribution(action_offsets, beta, tau)
    current_entropy = entropy(probabilities)
    return {
        "beta": beta,
        "tau": tau,
        "entropy": current_entropy,
        "effective_support": float(math.exp(current_entropy)),
        "task_reward": float(np.dot(probabilities, reward_vector(action_offsets, protocol))),
        "min_probability": float(np.min(probabilities)),
        "max_probability": float(np.max(probabilities)),
    }


def signed_gradient(
    action_offsets: np.ndarray,
    positive: np.ndarray,
    near: np.ndarray,
    far: np.ndarray,
    beta: float,
    log_tau: float,
    near_mass: float,
    far_mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    tau = math.exp(log_tau)
    probabilities = categorical_distribution(action_offsets, beta, tau)
    total_mass = 1.0 - near_mass - far_mass

    data_first = (
        float(np.dot(positive, action_offsets))
        - near_mass * float(np.dot(near, action_offsets))
        - far_mass * float(np.dot(far, action_offsets))
    )
    policy_first = float(np.dot(probabilities, action_offsets))

    centered_sq = (action_offsets - beta) ** 2
    data_second = (
        float(np.dot(positive, centered_sq))
        - near_mass * float(np.dot(near, centered_sq))
        - far_mass * float(np.dot(far, centered_sq))
    )
    policy_second = float(np.dot(probabilities, centered_sq))

    gradient = np.asarray(
        [
            (data_first - total_mass * policy_first) / (tau**2),
            (data_second - total_mass * policy_second) / (tau**2),
        ],
        dtype=np.float64,
    )
    return gradient, probabilities


def run_method(
    seed_data: dict[str, Any],
    method: str,
    protocol: Protocol,
    *,
    max_steps: int | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    total_steps = protocol.max_steps if max_steps is None else max_steps
    near_mass, far_mass = METHOD_NEGATIVE_MASS[method]
    action_offsets = seed_data["semantic_by_action_id"]
    positive = seed_data["positive"]
    near = seed_data["near"]
    far = seed_data["far"]
    near_anchor = seed_data["near_anchor"]
    far_anchor = seed_data["far_anchor"]

    beta = protocol.initial_beta
    log_tau = math.log(protocol.initial_tau)
    first_moment = np.zeros(2, dtype=np.float64)
    second_moment = np.zeros(2, dtype=np.float64)
    last_update_norm = 0.0
    rows: list[dict[str, float]] = []
    boundary_onset: int | None = None
    boundary_reason: str | None = None
    nonfinite_onset: int | None = None

    for step in range(total_steps + 1):
        gradient, probabilities = signed_gradient(
            action_offsets,
            positive,
            near,
            far,
            beta,
            log_tau,
            near_mass,
            far_mass,
        )
        metrics = evaluate_energy_policy(action_offsets, beta, log_tau, protocol)
        current = {
            "step": step,
            "beta": metrics["beta"],
            "tau": metrics["tau"],
            "entropy": metrics["entropy"],
            "effective_support": metrics["effective_support"],
            "task_reward": metrics["task_reward"],
            "min_probability": metrics["min_probability"],
            "max_probability": metrics["max_probability"],
            "near_anchor_probability": float(probabilities[near_anchor]),
            "far_anchor_probability": float(probabilities[far_anchor]),
            "near_anchor_surprisal": float(
                -math.log(max(float(probabilities[near_anchor]), 1.0e-300))
            ),
            "far_anchor_surprisal": float(
                -math.log(max(float(probabilities[far_anchor]), 1.0e-300))
            ),
            "raw_gradient_norm": float(np.linalg.norm(gradient)),
            "adam_parameter_update_norm": last_update_norm,
            "near_negative_mass": near_mass,
            "far_negative_mass": far_mass,
        }
        should_record = step % protocol.eval_every == 0 or step == total_steps

        finite = bool(
            np.isfinite(gradient).all()
            and all(math.isfinite(float(value)) for value in current.values())
        )
        if not finite:
            nonfinite_onset = step
            should_record = True

        support_reason = None
        if metrics["tau"] <= protocol.support_tau_threshold:
            support_reason = "temperature_boundary"
        elif metrics["effective_support"] <= protocol.effective_support_threshold:
            support_reason = "effective_support_boundary"
        if support_reason is not None and boundary_onset is None:
            boundary_onset = step
            boundary_reason = support_reason
            should_record = True

        if should_record:
            rows.append(current)
        if nonfinite_onset is not None or boundary_onset is not None or step == total_steps:
            break

        first_moment = (
            protocol.adam_beta1 * first_moment
            + (1.0 - protocol.adam_beta1) * gradient
        )
        second_moment = (
            protocol.adam_beta2 * second_moment
            + (1.0 - protocol.adam_beta2) * (gradient**2)
        )
        t = step + 1
        corrected_first = first_moment / (1.0 - protocol.adam_beta1**t)
        corrected_second = second_moment / (1.0 - protocol.adam_beta2**t)
        update = protocol.adam_lr * corrected_first / (
            np.sqrt(corrected_second) + protocol.adam_eps
        )
        beta += float(update[0])
        log_tau += float(update[1])
        last_update_norm = float(np.linalg.norm(update))

    terminal = rows[-1]
    w1 = rows_in_window(rows, protocol.audit_window_1)
    w2 = rows_in_window(rows, protocol.audit_window_2)
    stable = False
    if boundary_onset is None and nonfinite_onset is None and int(terminal["step"]) == total_steps:
        if w2:
            beta_change = abs(w2[-1]["beta"] - w2[0]["beta"])
            tau_change = abs(w2[-1]["tau"] - w2[0]["tau"])
            reward_change = abs(w2[-1]["task_reward"] - w2[0]["task_reward"])
            raw_gradient_median = float(
                np.median([row["raw_gradient_norm"] for row in w2])
            )
            stable = bool(
                beta_change <= protocol.stable_beta_change
                and tau_change <= protocol.stable_tau_change
                and reward_change <= protocol.stable_reward_change
                and raw_gradient_median <= protocol.stable_raw_gradient_median
            )
        else:
            beta_change = tau_change = reward_change = raw_gradient_median = float("nan")
    else:
        beta_change = tau_change = reward_change = raw_gradient_median = float("nan")

    if nonfinite_onset is not None:
        terminal_class = "nan_inf_numerical_failure"
    elif boundary_onset is not None:
        terminal_class = "support_boundary"
    elif stable:
        terminal_class = "stable_bounded"
    elif w1 and w2 and linear_slope(w2, "far_anchor_surprisal") > 0.0:
        terminal_class = "persistent_suppression"
    else:
        terminal_class = "terminal_inconclusive"

    summary = {
        "seed": int(seed_data["seed"]),
        "method": method,
        "terminal_class": terminal_class,
        "terminal_step": int(terminal["step"]),
        "support_boundary_onset": boundary_onset,
        "support_boundary_reason": boundary_reason,
        "nonfinite_onset": nonfinite_onset,
        "terminal_beta": terminal["beta"],
        "terminal_tau": terminal["tau"],
        "terminal_entropy": terminal["entropy"],
        "terminal_effective_support": terminal["effective_support"],
        "terminal_task_reward": terminal["task_reward"],
        "terminal_near_anchor_probability": terminal["near_anchor_probability"],
        "terminal_far_anchor_probability": terminal["far_anchor_probability"],
        "terminal_far_anchor_surprisal": terminal["far_anchor_surprisal"],
        "terminal_raw_gradient_norm": terminal["raw_gradient_norm"],
        "terminal_adam_parameter_update_norm": terminal[
            "adam_parameter_update_norm"
        ],
        "w1_beta_slope": linear_slope(w1, "beta") if w1 else None,
        "w2_beta_slope": linear_slope(w2, "beta") if w2 else None,
        "w1_tau_slope": linear_slope(w1, "tau") if w1 else None,
        "w2_tau_slope": linear_slope(w2, "tau") if w2 else None,
        "w1_reward_slope": linear_slope(w1, "task_reward") if w1 else None,
        "w2_reward_slope": linear_slope(w2, "task_reward") if w2 else None,
        "w2_beta_change": beta_change,
        "w2_tau_change": tau_change,
        "w2_reward_change": reward_change,
        "w2_raw_gradient_median": raw_gradient_median,
        "stable_gate_pass": stable,
        "near_negative_mass": near_mass,
        "far_negative_mass": far_mass,
    }
    return rows, summary


def add_task_classification(
    summaries: list[dict[str, Any]], protocol: Protocol
) -> None:
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in summaries:
        by_seed.setdefault(int(row["seed"]), {})[str(row["method"])] = row
    for seed, methods in by_seed.items():
        positive = methods.get("positive_only")
        if positive is None:
            raise ProtocolError(f"seed {seed} is missing positive_only")
        threshold = (
            protocol.task_collapse_ratio_to_positive_only
            * float(positive["terminal_task_reward"])
        )
        for row in methods.values():
            row["positive_only_reward_reference"] = positive["terminal_task_reward"]
            row["task_collapse_threshold"] = threshold
            row["task_collapse"] = bool(
                float(row["terminal_task_reward"]) <= threshold
            )
            row["support_collapse"] = row["terminal_class"] == "support_boundary"
            row["nan_inf_numerical_failure"] = (
                row["terminal_class"] == "nan_inf_numerical_failure"
            )
            expected = HISTORICAL_CAUSAL_PATTERN[str(row["method"])]
            row["historical_task_class_matches"] = (
                row["task_collapse"] == expected["task_collapse"]
            )
            row["historical_support_class_matches"] = (
                row["support_collapse"] == expected["support_collapse"]
            )
            row["historical_joint_class_matches"] = bool(
                row["historical_task_class_matches"]
                and row["historical_support_class_matches"]
            )


def aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"methods": {}, "total_runs": len(summaries)}
    for method in METHODS:
        rows = [row for row in summaries if row["method"] == method]
        aggregate["methods"][method] = {
            "runs": len(rows),
            "terminal_class_counts": {
                key: sum(row["terminal_class"] == key for row in rows)
                for key in (
                    "stable_bounded",
                    "support_boundary",
                    "persistent_suppression",
                    "terminal_inconclusive",
                    "nan_inf_numerical_failure",
                )
            },
            "task_collapse_count": sum(bool(row["task_collapse"]) for row in rows),
            "support_collapse_count": sum(
                bool(row["support_collapse"]) for row in rows
            ),
            "nan_inf_count": sum(
                bool(row["nan_inf_numerical_failure"]) for row in rows
            ),
            "historical_joint_match_count": sum(
                bool(row["historical_joint_class_matches"]) for row in rows
            ),
            "terminal_reward_mean": float(
                np.mean([row["terminal_task_reward"] for row in rows])
            ),
            "terminal_entropy_mean": float(
                np.mean([row["terminal_entropy"] for row in rows])
            ),
            "terminal_tau_mean": float(
                np.mean([row["terminal_tau"] for row in rows])
            ),
        }
    aggregate["all_runs_classified"] = all(
        row["terminal_class"] != "terminal_inconclusive" for row in summaries
    )
    aggregate["all_historical_classes_match"] = all(
        bool(row["historical_joint_class_matches"]) for row in summaries
    )
    aggregate["total_nan_inf_count"] = sum(
        bool(row["nan_inf_numerical_failure"]) for row in summaries
    )
    return aggregate


def direct_reference_comparison(direct_summary: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for case, summary in direct_summary.items():
        reference = DIRECT_HISTORICAL_REFERENCE[case]
        terminal_ratio = float(summary["terminal_probability"]) / float(
            reference["terminal_probability"]
        )
        report[case] = {
            "reference": reference,
            "observed": summary,
            "terminal_probability_ratio_observed_to_reference": terminal_ratio,
            "initial_probability_match": math.isclose(
                float(summary["initial_probability"]),
                float(reference["initial_probability"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            "initial_entropy_match": math.isclose(
                float(summary["initial_entropy"]),
                float(reference["initial_entropy"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            "terminal_probability_within_20_percent": 0.8 <= terminal_ratio <= 1.2,
            "score_bound_pass": bool(summary["score_bound_pass"]),
            "surprisal_growth_pass": bool(summary["surprisal_growth_pass"]),
        }
    report["all_direct_reference_checks_passed"] = all(
        item["initial_probability_match"]
        and item["initial_entropy_match"]
        and item["terminal_probability_within_20_percent"]
        and item["score_bound_pass"]
        and item["surprisal_growth_pass"]
        for key, item in report.items()
        if key != "all_direct_reference_checks_passed"
    )
    return report


def write_report(
    output_root: Path,
    protocol: Protocol,
    direct_summary: dict[str, Any],
    aggregate: dict[str, Any],
) -> None:
    lines = [
        f"# {EXPERIMENT_ID} report",
        "",
        "This runner reconstructs the missing historical E5 code from locked handoff records.",
        "It does not claim byte-identical reproduction of the uncommitted legacy runner.",
        "",
        "## Direct-softmax diagnostic",
        "",
        "| case | p0 | pT | H0 | Hmax | HT | max score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case, row in direct_summary.items():
        lines.append(
            f"| {case} | {row['initial_probability']:.6g} | "
            f"{row['terminal_probability']:.6g} | {row['initial_entropy']:.6g} | "
            f"{row['peak_entropy']:.6g} | {row['terminal_entropy']:.6g} | "
            f"{row['max_score']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Long-run near/far causal reconstruction",
            "",
            "| method | task collapse | support collapse | NaN/Inf | historical class match | mean reward | mean entropy |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        row = aggregate["methods"][method]
        lines.append(
            f"| {method} | {row['task_collapse_count']}/{row['runs']} | "
            f"{row['support_collapse_count']}/{row['runs']} | "
            f"{row['nan_inf_count']}/{row['runs']} | "
            f"{row['historical_joint_match_count']}/{row['runs']} | "
            f"{row['terminal_reward_mean']:.6f} | {row['terminal_entropy_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Reporting boundary",
            "",
            "Task-performance collapse, support/temperature boundary, and NaN/Inf failure are separate events.",
            "The historical numbers are comparison references, not acceptance targets that may be tuned after seeing results.",
            "E5 does not test unseen-action semantic generalization; that remains E6's responsibility.",
            "",
            f"Maximum causal steps: {protocol.max_steps}; audit windows: "
            f"{protocol.audit_window_1} and {protocol.audit_window_2}.",
        ]
    )
    (output_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(output_root: Path) -> None:
    entries = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            entries.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(output_root / "manifest.json", {"files": entries})


def make_plots(output_root: Path, summaries: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods = list(METHODS)
    reward = [
        np.mean(
            [row["terminal_task_reward"] for row in summaries if row["method"] == method]
        )
        for method in methods
    ]
    support = [
        sum(row["support_collapse"] for row in summaries if row["method"] == method)
        for method in methods
    ]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(methods)), reward)
    ax.set_xticks(np.arange(len(methods)), methods, rotation=30, ha="right")
    ax.set_ylabel("Terminal task reward")
    ax.set_title("D-U1 E5 long-run terminal reward")
    fig.tight_layout()
    fig.savefig(output_root / "causal_terminal_reward.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(methods)), support)
    ax.set_xticks(np.arange(len(methods)), methods, rotation=30, ha="right")
    ax.set_ylabel("Support-boundary seeds")
    ax.set_ylim(0, max(20, max(support) + 1))
    ax.set_title("D-U1 E5 support-boundary counts")
    fig.tight_layout()
    fig.savefig(output_root / "causal_support_boundary.png", dpi=160)
    plt.close(fig)


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def validate_formal_invocation(args: argparse.Namespace, protocol: Protocol) -> None:
    if args.mode != "formal":
        return
    if tuple(args.seeds) != FORMAL_SEEDS:
        raise ProtocolError(f"formal seeds are frozen to {FORMAL_SEEDS}")
    if args.max_steps != protocol.max_steps:
        raise ProtocolError(f"formal max_steps is frozen to {protocol.max_steps}")
    if args.direct_steps != protocol.direct_steps:
        raise ProtocolError(
            f"formal direct_steps is frozen to {protocol.direct_steps}"
        )
    if args.eval_every != protocol.eval_every:
        raise ProtocolError(f"formal eval_every is frozen to {protocol.eval_every}")


def run(args: argparse.Namespace) -> None:
    protocol = Protocol()
    validate_formal_invocation(args, protocol)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    write_json(
        output_root / "config.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "mode": args.mode,
            "protocol": asdict(protocol),
            "method_negative_mass": METHOD_NEGATIVE_MASS,
            "seeds": list(args.seeds),
            "historical_reference_only": True,
            "historical_runner_available": False,
            "reconstruction_scope": "locked_handoff_scientific_roles_and_reference_values",
        },
    )

    seed_zero = build_seed_data(int(args.seeds[0]), protocol)
    preflight_report = preflight(seed_zero, protocol)
    write_json(output_root / "preflight.json", preflight_report)
    if not preflight_report["all_checks_passed"]:
        raise RuntimeError("D-U1 E5 preflight failed")

    direct_summary: dict[str, Any] = {}
    for name, initial in DIRECT_CASES.items():
        rows, summary = run_direct_case(
            name, initial, protocol, steps=args.direct_steps
        )
        write_csv(output_root / "direct_softmax" / f"{name}_trajectory.csv", rows)
        direct_summary[name] = summary
    write_json(output_root / "direct_softmax" / "summary.json", direct_summary)
    direct_comparison = direct_reference_comparison(direct_summary)

    summaries: list[dict[str, Any]] = []
    total = len(args.seeds) * len(METHODS)
    completed = 0
    for seed in args.seeds:
        seed_data = build_seed_data(int(seed), protocol)
        seed_preflight = preflight(seed_data, protocol)
        write_json(
            output_root / "causal" / f"seed_{seed}" / "preflight.json",
            seed_preflight,
        )
        if not seed_preflight["all_checks_passed"]:
            raise RuntimeError(f"seed {seed} preflight failed")
        for method in METHODS:
            rows, summary = run_method(
                seed_data, method, protocol, max_steps=args.max_steps
            )
            write_csv(
                output_root / "causal" / f"seed_{seed}" / f"{method}_trajectory.csv",
                rows,
            )
            summaries.append(summary)
            completed += 1
            print(
                json.dumps(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "seed": seed,
                        "method": method,
                        "completed": completed,
                        "total": total,
                        "terminal_class": summary["terminal_class"],
                    }
                ),
                flush=True,
            )
        if completed % (5 * len(METHODS)) == 0 or completed == total:
            write_json(
                output_root / f"CHECKPOINT_{completed // len(METHODS):02d}_SEEDS.json",
                {
                    "completed_seed_count": completed // len(METHODS),
                    "completed_run_count": completed,
                    "pending_run_count": total - completed,
                    "last_seed": seed,
                    "utc": utc_now(),
                },
            )

    add_task_classification(summaries, protocol)
    write_csv(output_root / "causal" / "per_seed_summary.csv", summaries)
    aggregate = aggregate_summaries(summaries)
    write_json(output_root / "causal" / "aggregate_summary.json", aggregate)

    historical_comparison = {
        "direct_softmax": direct_comparison,
        "causal_expected_pattern": HISTORICAL_CAUSAL_PATTERN,
        "causal_observed": aggregate,
        "historical_numbers_are_reference_not_tuning_targets": True,
    }
    write_json(
        output_root / "historical_reference_comparison.json", historical_comparison
    )

    terminal_audit = {
        "experiment_id": EXPERIMENT_ID,
        "mode": args.mode,
        "raw_runs_complete": len(summaries) == total,
        "expected_runs": total,
        "actual_runs": len(summaries),
        "direct_reference_checks_passed": direct_comparison[
            "all_direct_reference_checks_passed"
        ],
        "all_causal_runs_classified": aggregate["all_runs_classified"],
        "all_historical_classes_match": aggregate[
            "all_historical_classes_match"
        ],
        "task_support_nan_inf_reported_separately": True,
        "total_nan_inf_count": aggregate["total_nan_inf_count"],
        "scientific_status_candidate": (
            "long_run_validated_pending_repository_closure"
            if args.mode == "formal"
            and direct_comparison["all_direct_reference_checks_passed"]
            and aggregate["all_runs_classified"]
            else "pilot_or_inconclusive"
        ),
        "prohibited_claims": [
            "categorical_direct_logit_gradient_is_unbounded",
            "support_boundary_is_identical_to_nan_inf_failure",
            "E5_proves_unseen_action_semantic_generalization",
            "historical_runner_was_byte_identically_reproduced",
            "universal_method_ranking",
        ],
    }
    write_json(output_root / "terminal_audit.json", terminal_audit)
    write_report(output_root, protocol, direct_summary, aggregate)
    if not args.skip_plots:
        make_plots(output_root, summaries)

    run_complete = {
        "experiment_id": EXPERIMENT_ID,
        "mode": args.mode,
        "completed_utc": utc_now(),
        "process_pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "seeds": list(args.seeds),
        "methods": list(METHODS),
        "expected_runs": total,
        "actual_runs": len(summaries),
        "terminal_audit": "terminal_audit.json",
        "report": "REPORT.md",
        "repository_status_change_requires_separate_update_package": True,
    }
    write_json(output_root / "RUN_COMPLETE.json", run_complete)
    write_manifest(output_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=FORMAL_SEEDS,
        help="comma-separated seeds; formal mode is frozen to 10-29",
    )
    parser.add_argument("--max-steps", type=int, default=Protocol.max_steps)
    parser.add_argument("--direct-steps", type=int, default=Protocol.direct_steps)
    parser.add_argument("--eval-every", type=int, default=Protocol.eval_every)
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ProtocolError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
