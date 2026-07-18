"""External-validity experiment components."""

from .hopper_actor import actor_batch_loss, actor_eval_metrics, train_actor_stage
from .hopper_advantages import critic_advantage_arrays
from .hopper_critic import CriticRun, train_critic
from .hopper_data import (
    Normalizer,
    OfflineData,
    build_episode_ids,
    discounted_returns,
    load_hopper_hdf5,
    split_episode_indices,
)
from .hopper_metrics import (
    aggregate_negative_gradient_norm,
    analytic_output_autograd_relative_error,
    classify_actor_terminal,
    create_gradient_probe,
    loglog_slope,
    match_near_far_indices,
    normalized_window_drift,
    pearson,
    per_sample_gradient_norm,
    r2_score,
    relative_slope,
    resolve_global_scale,
)
from .hopper_models import SquashedGaussianPolicy, ValueNetwork, make_mlp
from .hopper_protocol import METHODS, HopperProtocol, smoke_protocol
from .hopper_suite import (
    PreparedActor,
    clone_policy,
    make_policy,
    prepare_positive_only_actor,
    run_hopper_six_branch_suite,
)

__all__ = [
    "METHODS",
    "CriticRun",
    "HopperProtocol",
    "Normalizer",
    "OfflineData",
    "PreparedActor",
    "SquashedGaussianPolicy",
    "ValueNetwork",
    "actor_batch_loss",
    "actor_eval_metrics",
    "aggregate_negative_gradient_norm",
    "analytic_output_autograd_relative_error",
    "build_episode_ids",
    "clone_policy",
    "critic_advantage_arrays",
    "classify_actor_terminal",
    "create_gradient_probe",
    "discounted_returns",
    "load_hopper_hdf5",
    "loglog_slope",
    "make_mlp",
    "make_policy",
    "match_near_far_indices",
    "normalized_window_drift",
    "pearson",
    "per_sample_gradient_norm",
    "prepare_positive_only_actor",
    "r2_score",
    "relative_slope",
    "resolve_global_scale",
    "run_hopper_six_branch_suite",
    "smoke_protocol",
    "split_episode_indices",
    "train_actor_stage",
    "train_critic",
]
