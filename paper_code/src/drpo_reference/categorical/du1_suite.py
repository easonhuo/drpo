"""D-U1 revision-4 seed bundles, paired collapse, and aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .du1_protocol import (
    FORMAL_METHODS,
    DU1Protocol,
    MethodSpec,
    method_specs,
)
from .du1_training import (
    DU1TerminalProtocol,
    MethodRun,
    SharedStart,
    build_shared_start,
    run_method,
)


@dataclass
class SeedBundle:
    seed: int
    shared_start: SharedStart
    runs: list[MethodRun]


def run_seed_bundle(
    *,
    protocol: DU1Protocol,
    terminal: DU1TerminalProtocol,
    seed: int,
    specs: Sequence[MethodSpec] | None = None,
    device: torch.device | str = "cpu",
) -> SeedBundle:
    target = torch.device(device)
    selected = method_specs() if specs is None else list(specs)
    shared = build_shared_start(protocol, seed, target)
    runs = [
        run_method(
            protocol=protocol,
            terminal=terminal,
            seed=seed,
            spec=spec,
            base_state=shared.state_dict,
            base_optimizer_state=shared.optimizer_state,
            calibration=shared.calibration,
            device=target,
        )
        for spec in selected
    ]
    return SeedBundle(seed=seed, shared_start=shared, runs=runs)


def assign_task_collapse(
    summaries: list[dict[str, Any]],
    protocol: DU1Protocol,
) -> None:
    """Assign task collapse only after paired Positive-only is available."""

    reference = {
        int(row["seed"]): float(row["final_expected_semantic_reward"])
        for row in summaries
        if row["method"] == "positive_only"
    }
    observed_seeds = {int(row["seed"]) for row in summaries}
    if set(reference) != observed_seeds:
        raise RuntimeError("paired Positive-only reference missing for one or more seeds")
    for row in summaries:
        paired = reference[int(row["seed"])]
        row["paired_positive_only_reward"] = paired
        row["task_performance_collapse"] = bool(
            float(row["final_expected_semantic_reward"])
            < protocol.task_collapse_ratio_to_paired_positive_only * paired
        )


def paired_effect(
    values: Sequence[float],
    *,
    seed: int = 12345,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    if array.size == 0:
        return {
            "mean": None,
            "ci95": [None, None],
            "wins": 0,
            "ties": 0,
            "losses": 0,
        }
    draws = generator.choice(
        array,
        size=(5000, array.size),
        replace=True,
    ).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "wins": int((array > 0).sum()),
        "ties": int((array == 0).sum()),
        "losses": int((array < 0).sum()),
    }


def aggregate(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in summaries:
        grouped.setdefault(str(row["method"]), []).append(row)
    positive = {int(row["seed"]): row for row in grouped.get("positive_only", [])}
    output: dict[str, Any] = {}
    for method in FORMAL_METHODS:
        rows = grouped.get(method, [])
        if not rows:
            continue
        paired = [
            float(row["final_expected_semantic_reward"])
            - float(positive[int(row["seed"])]["final_expected_semantic_reward"])
            for row in rows
        ]
        output[method] = {
            "runs": len(rows),
            "reward_mean": float(
                np.mean([float(row["final_expected_semantic_reward"]) for row in rows])
            ),
            "reward_delta_vs_positive_only": paired_effect(paired),
            "hidden_probability_mean": float(
                np.mean([float(row["final_hidden_optimal_family_probability"]) for row in rows])
            ),
            "action_effective_support_mean": float(
                np.mean([float(row["final_action_effective_support"]) for row in rows])
            ),
            "prototype_effective_support_mean": float(
                np.mean([float(row["final_prototype_effective_support"]) for row in rows])
            ),
            "rare_total_probability_mean": float(
                np.mean([float(row["final_rare_total_probability"]) for row in rows])
            ),
            "task_performance_collapse_events": sum(
                bool(row["task_performance_collapse"]) for row in rows
            ),
            "support_boundary_events": sum(bool(row["support_boundary_event"]) for row in rows),
            "nan_inf_numerical_failures": sum(
                bool(row["nan_inf_numerical_failure"]) for row in rows
            ),
            "environment_validity_failures": sum(
                bool(row["environment_validity_failure"]) for row in rows
            ),
            "terminal_plateaus": sum(row["terminal_class"] == "terminal_plateau" for row in rows),
        }
    return output


def build_terminal_audit(
    *,
    protocol: DU1Protocol,
    summaries: Sequence[Mapping[str, Any]],
    selected_seeds: Sequence[int],
    smoke: bool,
) -> dict[str, Any]:
    expected = {(int(seed), method) for seed in selected_seeds for method in FORMAL_METHODS}
    observed = {(int(row["seed"]), str(row["method"])) for row in summaries}
    complete_formal_seed_set = tuple(int(seed) for seed in selected_seeds) == protocol.formal_seeds
    environment_failures = sum(bool(row["environment_validity_failure"]) for row in summaries)
    task_events = sum(bool(row["task_performance_collapse"]) for row in summaries)
    support_events = sum(bool(row["support_boundary_event"]) for row in summaries)
    numerical_events = sum(bool(row["nan_inf_numerical_failure"]) for row in summaries)
    all_terminal_accepted = all(bool(row["terminal_formal_acceptance"]) for row in summaries)
    matrix_complete = observed == expected and len(summaries) == len(expected)
    formal_acceptance = bool(
        not smoke
        and complete_formal_seed_set
        and matrix_complete
        and environment_failures == 0
        and all_terminal_accepted
    )
    return {
        "experiment_id": "D-U1-E6-CARTESIAN-TAPER-01",
        "protocol_revision": 4,
        "selected_seeds": list(selected_seeds),
        "registered_formal_seeds": list(protocol.formal_seeds),
        "expected_runs": len(expected),
        "actual_runs": len(summaries),
        "missing_run_identities": sorted(expected - observed),
        "unexpected_run_identities": sorted(observed - expected),
        "all_registered_runs_present": matrix_complete,
        "complete_formal_seed_set": complete_formal_seed_set,
        "environment_validity_failures": environment_failures,
        "task_performance_collapse_events": task_events,
        "support_boundary_events": support_events,
        "nan_inf_numerical_failures": numerical_events,
        "all_terminal_classifications_accepted": (all_terminal_accepted),
        "formal_scientific_acceptance": formal_acceptance,
        "method_ranking_allowed": formal_acceptance,
        "formal_evidence_allowed": formal_acceptance,
        "smoke": smoke,
    }
