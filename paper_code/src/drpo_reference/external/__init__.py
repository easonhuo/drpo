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
    classify_actor_terminal,
    normalized_window_drift,
    pearson,
    r2_score,
    relative_slope,
)
from .hopper_models import SquashedGaussianPolicy, ValueNetwork, make_mlp
from .hopper_protocol import METHODS, HopperProtocol, smoke_protocol

__all__ = [
    "METHODS",
    "CriticRun",
    "HopperProtocol",
    "Normalizer",
    "OfflineData",
    "SquashedGaussianPolicy",
    "ValueNetwork",
    "actor_batch_loss",
    "actor_eval_metrics",
    "build_episode_ids",
    "critic_advantage_arrays",
    "classify_actor_terminal",
    "discounted_returns",
    "load_hopper_hdf5",
    "make_mlp",
    "normalized_window_drift",
    "pearson",
    "r2_score",
    "relative_slope",
    "smoke_protocol",
    "split_episode_indices",
    "train_actor_stage",
    "train_critic",
]
