"""D-U1 revision-4 categorical environment and policy primitives.

Train and held-out contexts are independent draws from the same distribution.
This environment therefore supports same-distribution held-out-context
generalization claims, not OOD generalization claims.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

CELL_NAMES = (
    "useful_common",
    "useful_rare",
    "unhelpful_common",
    "unhelpful_rare",
)
FORMAL_METHODS = (
    "positive_only",
    "all_negative",
    "global_matched",
    "reciprocal_linear_distance",
    "reciprocal_quadratic_distance",
    "exponential_quadratic_distance",
)
HISTORICAL_EXCLUDED_METHODS = ("reciprocal_quartic_distance",)


@dataclass(frozen=True)
class DU1Protocol:
    """Frozen scientific constants for D-U1 E6 protocol revision 4."""

    protocol_revision: int = 4
    state_dim: int = 6
    semantic_dim: int = 4
    semantic_prototypes: int = 32
    hidden_semantic_prototypes: int = 16
    rarity_replicas: int = 2
    observed_action_count: int = 64
    hidden_action_count: int = 16
    action_count: int = 80
    hidden_optimal_actions_per_state: int = 4
    train_states: int = 2048
    test_states: int = 2048
    positive_prototypes_per_state: int = 4

    target_offset: float = 0.45
    positive_advantage: float = 1.0
    negative_advantage: float = -1.0
    neutral_observed_reward: float = 0.4
    positive_observed_reward: float = 0.7
    useful_negative_reward: float = -1.0
    unhelpful_negative_reward: float = 2.2
    hidden_reward_min: float = 1.5
    hidden_reward_max: float = 2.2

    hidden_dim: int = 64
    fixed_concentration: float = 8.0
    initial_rarity_logit_gap: float = 4.0

    learning_rate: float = 1.0e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1.0e-8
    batch_size: int = 128
    maximum_steps: int = 8000
    evaluation_interval_steps: int = 100
    audit_states: int = 512
    negative_alpha: float = 0.5
    rarity_logit_anchor_coefficient: float = 0.25
    positive_warm_start_steps: int = 0

    minimum_calibration_gap: float = 1.0
    reference_rare_retention: float = 0.25

    task_collapse_ratio_to_paired_positive_only: float = 0.2
    prototype_effective_support_boundary: float = 1.5
    rarity_mass_boundary: float = 1.0e-4
    utility_sign_fraction_min: float = 0.995
    rarity_shift_probe: float = 1.0
    minimum_rarity_shift_reward_drop: float = 0.005
    minimum_rarity_shift_hidden_probability_drop: float = 0.001
    maximum_positive_only_rarity_gap_drift: float = 2.0e-6

    formal_seeds: tuple[int, ...] = tuple(range(200, 220))

    def __post_init__(self) -> None:
        if self.protocol_revision != 4:
            raise ValueError("the active D-U1 protocol revision is 4")
        if self.rarity_replicas != 2:
            raise ValueError("D-U1 requires one common and one rare observed replica")
        if self.observed_action_count != self.semantic_prototypes * 2:
            raise ValueError("observed_action_count must equal semantic_prototypes * 2")
        if self.hidden_action_count != self.hidden_semantic_prototypes:
            raise ValueError("hidden action and hidden prototype counts must match")
        if self.action_count != self.observed_action_count + self.hidden_action_count:
            raise ValueError("action_count must equal observed plus hidden actions")
        if not 0.0 < self.reference_rare_retention < 1.0:
            raise ValueError("reference_rare_retention must lie in (0, 1)")
        if self.minimum_calibration_gap <= 0.0:
            raise ValueError("minimum_calibration_gap must be positive")

    def legacy_config(self) -> dict[str, Any]:
        """Return the exact nested fields consumed by the authoritative v4 runner."""

        return {
            "experiment_id": "D-U1-E6-CARTESIAN-TAPER-01",
            "protocol_revision": self.protocol_revision,
            "data": {
                "state_dim": self.state_dim,
                "semantic_dim": self.semantic_dim,
                "semantic_prototypes": self.semantic_prototypes,
                "hidden_semantic_prototypes": self.hidden_semantic_prototypes,
                "rarity_replicas": self.rarity_replicas,
                "observed_action_count": self.observed_action_count,
                "hidden_action_count": self.hidden_action_count,
                "action_count": self.action_count,
                "hidden_optimal_actions_per_state": (self.hidden_optimal_actions_per_state),
                "train_states": self.train_states,
                "test_states": self.test_states,
                "positive_prototypes_per_state": (self.positive_prototypes_per_state),
            },
            "geometry": {
                "target_offset": self.target_offset,
                "positive_advantage": self.positive_advantage,
                "negative_advantage": self.negative_advantage,
                "neutral_observed_reward": self.neutral_observed_reward,
                "positive_observed_reward": self.positive_observed_reward,
                "useful_negative_reward": self.useful_negative_reward,
                "unhelpful_negative_reward": self.unhelpful_negative_reward,
                "hidden_reward_min": self.hidden_reward_min,
                "hidden_reward_max": self.hidden_reward_max,
            },
            "policy": {
                "hidden_dim": self.hidden_dim,
                "fixed_concentration": self.fixed_concentration,
                "initial_rarity_logit_gap": self.initial_rarity_logit_gap,
            },
            "optimization": {
                "learning_rate": self.learning_rate,
                "betas": [self.adam_beta1, self.adam_beta2],
                "eps": self.adam_eps,
                "batch_size": self.batch_size,
                "maximum_steps": self.maximum_steps,
                "evaluation_interval_steps": self.evaluation_interval_steps,
                "audit_states": self.audit_states,
                "negative_alpha": self.negative_alpha,
                "rarity_logit_anchor_coefficient": (self.rarity_logit_anchor_coefficient),
                "positive_warm_start_steps": self.positive_warm_start_steps,
                "cpu_threads_per_run": 1,
            },
            "taper": {
                "minimum_calibration_gap": self.minimum_calibration_gap,
                "reference_rare_retention": self.reference_rare_retention,
            },
            "events": {
                "task_collapse_ratio_to_paired_positive_only": (
                    self.task_collapse_ratio_to_paired_positive_only
                ),
                "prototype_effective_support_boundary": (self.prototype_effective_support_boundary),
                "rarity_mass_boundary": self.rarity_mass_boundary,
            },
            "diagnostics": {
                "utility_sign_fraction_min": self.utility_sign_fraction_min,
                "rarity_shift_probe": self.rarity_shift_probe,
                "minimum_rarity_shift_reward_drop": (self.minimum_rarity_shift_reward_drop),
                "minimum_rarity_shift_hidden_probability_drop": (
                    self.minimum_rarity_shift_hidden_probability_drop
                ),
                "maximum_positive_only_rarity_gap_drift": (
                    self.maximum_positive_only_rarity_gap_drift
                ),
            },
        }


def smoke_protocol() -> DU1Protocol:
    return replace(
        DU1Protocol(),
        train_states=64,
        test_states=64,
        hidden_dim=16,
        batch_size=16,
        maximum_steps=4,
        evaluation_interval_steps=2,
        audit_states=32,
        formal_seeds=(0,),
    )


@dataclass(frozen=True)
class MethodSpec:
    method: str
    active_cells: tuple[str, ...]
    taper_family: str | None = None


def method_specs(method_names: tuple[str, ...] | None = None) -> list[MethodSpec]:
    specs = {
        "positive_only": MethodSpec("positive_only", ()),
        "all_negative": MethodSpec("all_negative", CELL_NAMES),
        "global_matched": MethodSpec("global_matched", CELL_NAMES, "global"),
        "reciprocal_linear_distance": MethodSpec(
            "reciprocal_linear_distance",
            CELL_NAMES,
            "reciprocal_linear_distance",
        ),
        "reciprocal_quadratic_distance": MethodSpec(
            "reciprocal_quadratic_distance",
            CELL_NAMES,
            "reciprocal_quadratic_distance",
        ),
        "exponential_quadratic_distance": MethodSpec(
            "exponential_quadratic_distance",
            CELL_NAMES,
            "exponential_quadratic_distance",
        ),
    }
    names = FORMAL_METHODS if method_names is None else method_names
    unknown = [name for name in names if name not in specs]
    if unknown:
        raise ValueError(f"unknown active methods: {unknown}")
    return [specs[name] for name in names]
