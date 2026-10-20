"""D-U1 revision-4 categorical environment and policy primitives.

Train and held-out contexts are independent draws from the same distribution.
This environment therefore supports same-distribution held-out-context
generalization claims, not OOD generalization claims.
"""

from __future__ import annotations

from dataclasses import dataclass

CELL_NAMES = (
    "useful_common",
    "useful_rare",
    "unhelpful_common",
    "unhelpful_rare",
)
METHODS = (
    "positive_only",
    "all_negative",
    "global_matched",
    "reciprocal_linear_distance",
    "reciprocal_quadratic_distance",
    "exponential_quadratic_distance",
)


@dataclass(frozen=True)
class DU1Protocol:
    """D-U1 revision-4 experiment settings."""

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
    calibration_states: int = 512
    negative_alpha: float = 0.5
    rarity_logit_anchor_coefficient: float = 0.25

    reference_rare_retention: float = 0.25

    task_collapse_ratio_to_paired_positive_only: float = 0.2

    seeds: tuple[int, ...] = tuple(range(200, 220))


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
    names = METHODS if method_names is None else method_names
    return [specs[name] for name in names]
