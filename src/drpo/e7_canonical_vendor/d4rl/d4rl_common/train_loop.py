"""d4rl_common.train_loop — Shared training-loop utilities for in-house trainers.

Created 2026-05-11 (P1-A refactor). Single source of truth for:
  - load_hdf5(path, dataset_name=None): load D4RL HDF5 + IQL-style reward norm
  - compute_mc_returns(rews, terms, touts, gamma=0.99): MC return per transition
  - evaluate(agent, dataset, n, seed, eval_env=None): online inline eval
  - get_eval_env(dataset, eval_env_table=None): dataset -> Gymnasium env id

These were duplicated across:
  /root/d4rl/train_sna2c_only.py
  /root/d4rl/train_drpoq_only.py
  /root/d4rl/train_sna2c_variant.py
  /root/d4rl/train_drpoq_diag.py

Audited 2026-05-11:
  * compute_mc_returns: 4/4 md5-identical → byte-identical extraction
  * load_hdf5:          4/4 logically equivalent (only whitespace/comment
                        styling differs). Unified signature
                        `load_hdf5(path, dataset_name=None)` preserves call-site
                        behavior of all four trainers.
  * evaluate:           4/4 logically equivalent. Adds optional `eval_env`
                        kwarg so callers can pass their own EVAL_ENV table for
                        backward compatibility.

Scope rule: this module ONLY hosts *byte-identical-equivalent* helpers. Trainer
loop bodies, agent construction, ckpt save format, and history/summary fields
remain in each trainer (they encode trainer-specific exploration logic).

DO NOT modify agents.py or third_party/* (project-level immutability rules).
"""
from __future__ import annotations

import os
import time  # noqa: F401  (re-exported convenience for trainers)
import numpy as np
import torch
import h5py
import gymnasium as gym

# reward_norm_locomotion lives in d4rl_common.normalize (P0-A single source).
from .normalize import reward_norm_locomotion
from .datasets import dataset_to_gym_env


# ── EVAL_ENV table ──────────────────────────────────────────────────────────
# Locomotion (D4RL Gym MuJoCo) — full 15-entry coverage (3 envs × 5 levels).
# This is the union of the per-trainer EVAL_ENV dicts. Trainers that listed
# only 7 entries (sna2c_only / drpoq_only / drpoq_diag) get a strict superset
# here, so behavior on previously-listed datasets is unchanged AND the same
# table now also handles the random/expert/medium-expert variants used by
# train_sna2c_variant.py.
EVAL_ENV_LOCOMOTION = {
    'hopper-random-v2':             'Hopper-v4',
    'hopper-medium-v2':             'Hopper-v4',
    'hopper-medium-replay-v2':      'Hopper-v4',
    'hopper-medium-expert-v2':      'Hopper-v4',
    'hopper-expert-v2':             'Hopper-v4',
    'halfcheetah-random-v2':        'HalfCheetah-v4',
    'halfcheetah-medium-v2':        'HalfCheetah-v4',
    'halfcheetah-medium-replay-v2': 'HalfCheetah-v4',
    'halfcheetah-medium-expert-v2': 'HalfCheetah-v4',
    'halfcheetah-expert-v2':        'HalfCheetah-v4',
    'walker2d-random-v2':           'Walker2d-v4',
    'walker2d-medium-v2':           'Walker2d-v4',
    'walker2d-medium-replay-v2':    'Walker2d-v4',
    'walker2d-medium-expert-v2':    'Walker2d-v4',
    'walker2d-expert-v2':           'Walker2d-v4',
}


def get_eval_env(dataset, eval_env_table=None):
    """Resolve a dataset name to a Gymnasium env id.

    Behavior precedence (preserves all four trainers' original semantics):
      1. If `eval_env_table` is given and `dataset` is in it -> return that.
         (Callers that want strict fallback-to-input-string semantics can
         pass their original local table.)
      2. Else: try `dataset_to_gym_env(dataset)` (prefix-based, e.g.
         'hopper'->'Hopper-v4'). On unknown prefix, fall back to returning
         `dataset` as-is — matching the original `EVAL_ENV.get(dataset, dataset)`
         pattern used in all four trainers.
    """
    if eval_env_table is not None and dataset in eval_env_table:
        return eval_env_table[dataset]
    if eval_env_table is not None and dataset not in eval_env_table:
        # Strict legacy semantics: trainer passed its own table and the
        # dataset is not in it -> return the dataset string verbatim.
        return dataset
    try:
        return dataset_to_gym_env(dataset)
    except ValueError:
        return dataset


# ── HDF5 loading ────────────────────────────────────────────────────────────
def load_hdf5(path, dataset_name=None):
    """Load a D4RL HDF5 file. Returns dict(obs, acts, rews, terms, touts, next_obs).

    If `dataset_name` is given AND its prefix is a locomotion env
    (hopper/halfcheetah/walker2d), apply IQL-style reward normalization
    (reward_norm_locomotion) — exactly matches train_d4rl_mp.py.

    `dataset_name=None` skips the reward normalization (pre-existing
    behavior of train_sna2c_only / train_drpoq_only / train_sna2c_variant
    when they were called without dataset_name; train_drpoq_diag always
    passes it).
    """
    print(f"Loading {path} ...", flush=True)
    with h5py.File(path, 'r') as f:
        obs   = f['observations'][:].astype(np.float32)
        acts  = f['actions'][:].astype(np.float32)
        rews  = f['rewards'][:].astype(np.float32)
        terms = f['terminals'][:].astype(bool)
        touts = f.get('timeouts', np.zeros(len(rews), dtype=bool))[:].astype(bool)
        if 'next_observations' in f:
            next_obs = f['next_observations'][:].astype(np.float32)
        else:
            next_obs = np.zeros_like(obs)
            next_obs[:-1] = obs[1:]; next_obs[-1] = obs[-1]
            for idx in np.where((terms | touts)[:-1])[0]:
                next_obs[idx] = obs[idx]
    lim = 1.0 - 1e-5
    acts = np.clip(acts, -lim, lim)

    # IQL-style reward normalization for D4RL locomotion (hopper/halfcheetah/walker2d).
    # Matches train_d4rl_mp.py exactly — essential for cross-experiment comparability.
    if dataset_name is not None:
        prefix = dataset_name.replace('_', '-').split('-')[0]
        if prefix in ('hopper', 'halfcheetah', 'walker2d'):
            r_before = (float(rews.min()), float(rews.max()))
            rews = reward_norm_locomotion(rews, terms, touts).astype(np.float32)
            print(f"  Reward normalized: {r_before} -> ({rews.min():.4f},{rews.max():.4f})", flush=True)

    print(f"  {len(obs):,} transitions | obs={obs.shape[1]} act={acts.shape[1]}", flush=True)
    return dict(obs=obs, acts=acts, rews=rews, terms=terms, touts=touts, next_obs=next_obs)


# ── MC returns ──────────────────────────────────────────────────────────────
def compute_mc_returns(rews, terms, touts, gamma=0.99):
    """Episode-aware Monte-Carlo return per transition.

    md5-identical to the version in 4 in-house trainers (audited 2026-05-11).
    """
    n = len(rews); mc = np.zeros(n, dtype=np.float32); ret = 0.0
    for i in range(n - 1, -1, -1):
        if terms[i] or touts[i]: ret = 0.0
        ret = rews[i] + gamma * ret
        mc[i] = ret
    return mc


# ── Inline online evaluation ────────────────────────────────────────────────
def evaluate(agent, dataset, n=10, seed=42, eval_env=None):
    """Inline evaluation. Mirrors train_d4rl_mp.py worker: deterministic reset
    seeds (seed+20000+i) + action clip to env bounds.

    Args:
        agent:     must expose `get_action(obs) -> (action, info)`
        dataset:   D4RL dataset key (used to look up the Gym env id)
        n:         number of episodes
        seed:      base seed
        eval_env:  optional dict[str -> str] override; if dataset is a key in
                   it, that mapping wins. Used by trainers that historically
                   passed their own EVAL_ENV table; new code can omit it and
                   rely on `dataset_to_gym_env`.
    """
    env_name = get_eval_env(dataset, eval_env)
    env = gym.make(env_name)
    env.action_space.seed(seed + 10_000)
    returns = []
    for i in range(n):
        obs_out = env.reset(seed=seed + 20_000 + i)
        obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
        done = False; ep_ret = 0.0
        for _ in range(1000):
            a, _ = agent.get_action(obs)
            a = np.clip(a, env.action_space.low, env.action_space.high)
            result = env.step(a)
            if len(result) == 5:
                obs2, r, term, trunc, _ = result; done = term or trunc
            else:
                obs2, r, done, _ = result
            ep_ret += r; obs = obs2
            if done: break
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))


__all__ = [
    'EVAL_ENV_LOCOMOTION',
    'get_eval_env',
    'load_hdf5',
    'compute_mc_returns',
    'evaluate',
]
