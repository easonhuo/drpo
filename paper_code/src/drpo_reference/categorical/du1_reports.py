"""D-U1 mechanism and taper reports derived from complete paired runs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .du1_protocol import FORMAL_METHODS
from .du1_suite import paired_effect


def _summary_index(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, dict[int, Mapping[str, Any]]]:
    output: dict[str, dict[int, Mapping[str, Any]]] = {}
    for row in summaries:
        output.setdefault(str(row["method"]), {})[int(row["seed"])] = row
    return output


def paired_metric_effect(
    index: Mapping[str, Mapping[int, Mapping[str, Any]]],
    lhs: str,
    rhs: str,
    metric: str,
) -> dict[str, Any]:
    common_seeds = sorted(set(index.get(lhs, {})) & set(index.get(rhs, {})))
    values = [
        float(index[lhs][seed][metric]) - float(index[rhs][seed][metric]) for seed in common_seeds
    ]
    return {
        **paired_effect(values),
        "lhs": lhs,
        "rhs": rhs,
        "metric": metric,
        "seeds": common_seeds,
    }


def mechanism_report(
    audits: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    initial = [dict(audit["policy_geometry_initial"]) for audit in audits]
    after = [dict(audit["policy_geometry_after_warm_start"]) for audit in audits]
    return {
        "experiment_id": "D-U1-E6-CARTESIAN-TAPER-01",
        "block": "E6_REV4_ENVIRONMENT_IDENTIFICATION_AUDIT",
        "protocol_revision": 4,
        "seeds": [int(audit["seed"]) for audit in audits],
        "all_environment_audits_passed": all(bool(audit["passed"]) for audit in audits),
        "minimum_initial_utility_oracle_sign_valid_fraction": min(
            float(row["utility_oracle_sign_valid_fraction"]) for row in initial
        ),
        "minimum_runtime_utility_oracle_sign_valid_fraction": min(
            float(row["minimum_utility_oracle_sign_valid_fraction"]) for row in summaries
        ),
        "minimum_initial_rarity_shift_reward_drop": min(
            float(row["rarity_shift_reward_drop"]) for row in initial
        ),
        "minimum_initial_rarity_shift_hidden_probability_drop": min(
            float(row["rarity_shift_hidden_probability_drop"]) for row in initial
        ),
        "minimum_rare_common_shared_rarity_gradient_ratio": min(
            min(
                float(row["useful_rare_to_common_shared_rarity_gradient_ratio"]),
                float(row["unhelpful_rare_to_common_shared_rarity_gradient_ratio"]),
            )
            for row in initial + after
        ),
        "hidden_rare_channel_is_evaluation_only": True,
        "interpretation": (
            "environment validity and causal support-cost audit; not a separate method ranking"
        ),
    }


def taper_report(
    summaries: Sequence[Mapping[str, Any]],
    aggregate_summary: Mapping[str, Any],
) -> dict[str, Any]:
    index = _summary_index(summaries)
    metrics = (
        "final_expected_semantic_reward",
        "final_hidden_optimal_family_probability",
        "final_action_effective_support",
        "final_prototype_effective_support",
        "final_rare_total_probability",
    )
    contrasts: dict[str, Any] = {}
    for candidate in (
        "reciprocal_linear_distance",
        "reciprocal_quadratic_distance",
        "exponential_quadratic_distance",
    ):
        for control in (
            "all_negative",
            "global_matched",
            "positive_only",
        ):
            label = f"{candidate}_minus_{control}"
            contrasts[label] = {
                metric: paired_metric_effect(
                    index,
                    candidate,
                    control,
                    metric,
                )
                for metric in metrics
            }
    label = "exponential_quadratic_distance_minus_reciprocal_quadratic_distance"
    contrasts[label] = {
        metric: paired_metric_effect(
            index,
            "exponential_quadratic_distance",
            "reciprocal_quadratic_distance",
            metric,
        )
        for metric in metrics
    }
    return {
        "experiment_id": "D-U1-E6-CARTESIAN-TAPER-01",
        "block": "E6_REV4_FORMAL_TAPER_COMPARISON",
        "methods": {
            method: aggregate_summary[method]
            for method in FORMAL_METHODS
            if method in aggregate_summary
        },
        "paired_contrasts": contrasts,
        "quartic_active": False,
        "no_method_winner_assumed": True,
        "interpretation_gate": (
            "no ranking unless all formal runs and terminal audits are complete"
        ),
    }
