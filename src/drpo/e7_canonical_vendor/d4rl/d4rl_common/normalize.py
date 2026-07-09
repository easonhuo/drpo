"""IQL-style reward normalization for D4RL locomotion datasets.

Single source of truth (replaces 6 inline copies that previously lived in
train_d4rl_mp.py / train_drpoq_only.py / train_sna2c_only.py /
train_sna2c_variant.py / train_drpoq_diag.py / train_iql_origin.py).

Aligned with `ikostrikov/implicit_q_learning` train_offline.py
(per-trajectory return span normalization).

  rewards /= (max_traj_return - min_traj_return)
  rewards *= 1000.0

Only call for locomotion datasets (halfcheetah / hopper / walker2d).
For non-locomotion datasets (antmaze / kitchen / adroit) the IQL paper uses
different reward shaping; do NOT call this function for them.
"""
import numpy as np


def reward_norm_locomotion(rewards, terminals, timeouts):
    """ikostrikov/IQL-style reward normalization (locomotion only).

    rewards /= (max_traj_return - min_traj_return) * 1000.

    AST-verified byte-identical to the 6 inline copies it replaces
    (audit hash 905698c26f64 on 2026-05-10).
    """
    n = len(rewards); returns = []; start = 0
    for i in range(n):
        if terminals[i] or timeouts[i] or i == n - 1:
            returns.append(rewards[start:i+1].sum())
            start = i + 1
    if len(returns) < 2:
        return rewards
    span = max(returns) - min(returns)
    if span < 1e-8:
        return rewards
    return rewards / span * 1000.0


# Backward-compat alias for train_iql_origin.py historical name.
# Does NOT change behavior — same function, different name.
normalize_rewards = reward_norm_locomotion


def compute_ep_returns(rewards, terminals, timeouts, dtype=np.float32):
    """Per-step episode-return broadcast: each transition gets the sum of its
    own trajectory's rewards (used by some Agent classes that need ep_ret per
    transition for SNA2C / NegDown shaping).

    Args:
        rewards / terminals / timeouts: 1D arrays of length N (transitions).
        dtype: output dtype. Default float32 matches train_d4rl_mp.py;
               train_iql_origin.py historically used the numpy default
               (float64) — pass dtype=np.float64 for byte-identical output
               with that file.

    AST-verified equivalent to the 2 inline copies (train_d4rl_mp.py +
    train_iql_origin.py); they only differ in the dtype default.
    """
    n = len(rewards); ep_ret = np.zeros(n, dtype=dtype); start = 0
    for i in range(n):
        if terminals[i] or timeouts[i] or i == n - 1:
            ep_ret[start:i+1] = rewards[start:i+1].sum()
            start = i + 1
    return ep_ret


__all__ = ['reward_norm_locomotion', 'normalize_rewards', 'compute_ep_returns']
