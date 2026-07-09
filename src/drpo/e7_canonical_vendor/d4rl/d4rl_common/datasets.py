"""Single source of truth for D4RL dataset name routing.

Two helpers:
  dataset_to_key(name)      e.g. 'hopper_medium-v2' -> 'hopper-medium-v2'
                                  (canonical D4RL key, used to look up
                                  REF_MIN_SCORE / REF_MAX_SCORE)
  dataset_to_gym_env(name)  e.g. 'hopper-medium-v2' -> 'Hopper-v4'
                                  (Gymnasium env id used for online eval)

Historical state (2026-05-09 audit): four files each had their own copy
of these helpers, with subtly different supported prefixes:
  - train_all_vs_iql.py        : locomotion-4 + adroit-4 (most complete)
  - train_d4rl_mp.py           : locomotion-4
  - eval_d4rl_mp_greedy.py     : locomotion-3 (no ant), and **lacked**
                                  dataset_to_key entirely
  - train_iql_origin.py        : own copy (under third_party/IQL/, do not change)

This file is the new single source of truth (the union of all four). The
three in-house files (`train_all_vs_iql.py`, `train_d4rl_mp.py`,
`eval_d4rl_mp_greedy.py`) should `from d4rl_common.datasets import ...`
instead of redefining locally. The third-party copy in train_iql_origin.py
is left untouched per the official-code-immutability rule.
"""

# Prefix → Gymnasium env id. Locomotion uses v4 (modern mujoco),
# Adroit uses v1 (Gymnasium-Robotics naming).
_DATASET_PREFIX_TO_GYM_ENV = {
    # Locomotion (D4RL Gym MuJoCo)
    'hopper':      'Hopper-v4',
    'halfcheetah': 'HalfCheetah-v4',
    'walker2d':    'Walker2d-v4',
    'ant':         'Ant-v4',
    # Adroit
    'door':        'AdroitHandDoor-v1',
    'hammer':      'AdroitHandHammer-v1',
    'pen':         'AdroitHandPen-v1',
    'relocate':    'AdroitHandRelocate-v1',
}


def dataset_to_key(dataset_name):
    """Convert underscore form to hyphenated D4RL key.

    Examples:
        'hopper_medium-v2'        -> 'hopper-medium-v2'
        'hopper_medium_replay-v2' -> 'hopper-medium-replay-v2'
        'hopper-medium-v2'        -> 'hopper-medium-v2'  (already hyphenated)

    Strategy: only the body (everything before the trailing '-vN') has
    underscores converted to hyphens; the version suffix is preserved.
    """
    if '-v' in dataset_name:
        body, ver = dataset_name.rsplit('-v', 1)
        return body.replace('_', '-') + '-v' + ver
    return dataset_name.replace('_', '-')


def dataset_to_gym_env(dataset_name):
    """Auto-derive Gymnasium env id from dataset name.

    Examples:
        'hopper-medium-v2'           -> 'Hopper-v4'
        'halfcheetah_medium_replay-v2' -> 'HalfCheetah-v4'
        'door-expert-v1'              -> 'AdroitHandDoor-v1'

    Raises a verbose ValueError on unknown prefix (preferred over KeyError
    so callers get a clear error message).
    """
    prefix = dataset_name.replace('_', '-').split('-')[0]
    if prefix in _DATASET_PREFIX_TO_GYM_ENV:
        return _DATASET_PREFIX_TO_GYM_ENV[prefix]
    raise ValueError(
        f"Cannot auto-derive Gymnasium env from dataset '{dataset_name}' "
        f"(prefix='{prefix}'). Known prefixes: "
        f"{sorted(_DATASET_PREFIX_TO_GYM_ENV.keys())}. "
        f"Please specify --env explicitly."
    )


__all__ = ['dataset_to_key', 'dataset_to_gym_env',
           '_DATASET_PREFIX_TO_GYM_ENV']
