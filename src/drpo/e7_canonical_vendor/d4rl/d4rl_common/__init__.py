"""d4rl_common: single source of truth for D4RL normalization constants.

This package re-exports the official d4rl/infos.py reference scores
(REF_MIN_SCORE / REF_MAX_SCORE) under a consistent (min, max) tuple
convention, plus a normalize_score() helper used across the repo.

Usage:
    from d4rl_common import normalize_score, get_ref
    ns = normalize_score('hopper-medium-v2', raw_return)  # 0..100
    lo, hi = get_ref('hopper-medium-v2')

Rule (MUST not violate):
    Constants here are the ONLY source. No file outside this package should
    hard-code REF_MIN_SCORE / REF_MAX_SCORE for locomotion datasets.

Locomotion-v2 convention (per official d4rl/infos.py):
    v1/v2 envs inherit REF_MIN/MAX from the corresponding `-random-v0`.
    So hopper-{medium,random,expert,medium-replay,medium-expert,full-replay}-v2
    ALL use min=-20.272305, max=3234.3.
"""
from .constants import (
    REF_MIN_SCORE,
    REF_MAX_SCORE,
    D4RL_REF,
    get_ref,
    normalize_score,
)
from .datasets import (
    dataset_to_key,
    dataset_to_gym_env,
)
from .normalize import (
    reward_norm_locomotion,
    normalize_rewards,
    compute_ep_returns,
)
from .train_loop import (
    EVAL_ENV_LOCOMOTION,
    get_eval_env,
    load_hdf5,
    compute_mc_returns,
    evaluate,
)
from .scheduler import (
    SchedulerResult,
    run_sliding_window,
)

__all__ = [
    "REF_MIN_SCORE",
    "REF_MAX_SCORE",
    "D4RL_REF",
    "get_ref",
    "normalize_score",
    "dataset_to_key",
    "dataset_to_gym_env",
    "reward_norm_locomotion",
    "normalize_rewards",
    "compute_ep_returns",
    "EVAL_ENV_LOCOMOTION",
    "get_eval_env",
    "load_hdf5",
    "compute_mc_returns",
    "evaluate",
    "SchedulerResult",
    "run_sliding_window",
]
