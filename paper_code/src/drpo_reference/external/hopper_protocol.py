"""Frozen paper-facing protocol for Hopper E7-Q2 external validation."""

from __future__ import annotations

from dataclasses import dataclass, replace


METHODS = (
    "positive_only",
    "signed",
    "near_zero",
    "far_zero",
    "far_cap",
    "dynamic_budget_matched_global",
)


@dataclass(frozen=True)
class HopperProtocol:
    """Scientific constants from the validated E7-Q2 formal run."""

    experiment_id: str = "EXT-H-E7-Q2"
    role: str = "external_mechanism_validation"
    dataset_basename: str = "hopper_medium_replay-v2.hdf5"
    dataset_sha256: str = (
        "e121c5f7c9857a307baa9edc6a2c3b48"
        "e85fedb9ac316ecddd0f48ca7ef4e39b"
    )
    rollout_dataset_id: str = "hopper-medium-replay-v2"
    env_id: str = "Hopper-v4"
    normalized_score_percent: bool = True
    normalized_score_reference_min: float = -20.272305
    normalized_score_reference_max: float = 3234.3

    gamma: float = 0.99
    train_fraction: float = 0.80
    validation_fraction: float = 0.10
    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "tanh"
    init_scheme: str = "default"
    init_gain: float = 1.0
    critic_learning_rate: float = 3.0e-4
    actor_learning_rate: float = 3.0e-4
    critic_batch_size: int = 1024
    actor_batch_size: int = 1024
    weight_decay: float = 1.0e-4
    max_gradient_norm: float = 100.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    action_clip_epsilon: float = 1.0e-6

    advantage_standardize_once: bool = True
    near_quantile: float = 0.25
    far_quantile: float = 0.75
    advantage_bins: int = 20
    advantage_match_relative_tolerance: float = 0.05
    gradient_probe_pairs: int = 64
    distance_bins: int = 8
    far_cap_reference_quantile: float = 0.95
    global_budget_audit_size: int = 256

    audit_windows: int = 6
    actor_state_drift_tolerance: float = 0.01
    actor_update_tolerance: float = 2.0e-4
    support_boundary_threshold: float = 0.99
    support_boundary_fraction: float = 0.10
    task_return_drop_threshold: float = 20.0

    formal_seeds: tuple[int, ...] = tuple(range(100, 110))
    canonical_critic_seed: int = 100
    critic_steps: int = 100_000
    critic_eval_interval: int = 2_000
    positive_steps: int = 100_000
    actor_eval_interval: int = 5_000
    branch_steps: int = 200_000
    matched_pairs: int = 256
    audit_sample_size: int = 16_384
    rollout_episodes: int = 5
    final_rollout_episodes: int = 20
    rollout_eval_interval: int = 25_000

    def __post_init__(self) -> None:
        if self.experiment_id != "EXT-H-E7-Q2":
            raise ValueError("Hopper protocol identity is frozen")
        if self.role != "external_mechanism_validation":
            raise ValueError("Hopper E7-Q2 is an external mechanism validation")
        if len(self.formal_seeds) != 10:
            raise ValueError("Hopper E7-Q2 requires ten formal seeds")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must lie in (0, 1)")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0, 1)")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train and validation fractions leave no test split")
        if self.critic_steps <= 0 or self.positive_steps <= 0:
            raise ValueError("formal training budgets must be positive")
        if self.branch_steps <= 0:
            raise ValueError("formal branch budget must be positive")


def smoke_protocol() -> HopperProtocol:
    """Small non-formal protocol for integration tests only."""

    return replace(
        HopperProtocol(),
        hidden_sizes=(16, 16),
        critic_batch_size=8,
        actor_batch_size=8,
        formal_seeds=(42,),
        canonical_critic_seed=42,
        critic_steps=4,
        critic_eval_interval=2,
        positive_steps=4,
        actor_eval_interval=2,
        branch_steps=4,
        matched_pairs=4,
        audit_sample_size=16,
        rollout_episodes=1,
        final_rollout_episodes=1,
        rollout_eval_interval=4,
    )
