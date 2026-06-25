#!/usr/bin/env python3
"""One-click reproduction of the DRPO C-U1 continuous experiments E1-E4.

Run one registered stage without editing source or hyperparameters:

    python drpo_cu1_e1_e4_oneclick.py --stage e3 --output-root outputs/cu1_e3_adam
    python drpo_cu1_e1_e4_oneclick.py --stage e4 --output-root outputs/cu1_e4_adam

The script automatically chooses CUDA when available and freezes all protocol
values in code. E3 and E4 are separate formal delivery boundaries. Smoke mode
may exercise the integrated pipeline with ``--stage all``.

Important scope:
- train and test states are independent draws from the SAME state distribution.
  Therefore this script reports held-out-context generalization, not strict OOD
  generalization.
- fixed advantages are computed once from the environment and never updated.
- E2, E3, and E4 use Adam for paper-facing training. Raw gradient norms and
  actual Adam parameter-update norms are reported separately.
- no variance clamp is used in the main learnable-variance runs. The negative
  log-sigma boundary is an event detector for support contraction, not clipping.
  A positive-boundary crossing is treated as unexpected, not as a scientific
  variance-expansion branch.
- the normalized extrapolation diagnostic is written with the descriptive field
  name ``normalized_extrapolation_displacement``; no retired plotting symbol is reintroduced.

This file reconstructs the frozen v14 protocol from the surviving environment,
Master document, and E2-E4 reports. Because the original transient drivers were
not persisted, exact bitwise identity with those lost runs is impossible to
promise. The script therefore records both empirical outputs and comparisons to
pre-registered reference ranges.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from collections import deque

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# =============================================================================
# 0. Frozen one-click protocol
# =============================================================================

SCRIPT_VERSION = "2026.06.25-reconstruction-v5-unified-adam"
SMOKE = os.environ.get("DRPO_CU1_SMOKE", "0") == "1"  # developer-only shortcut
ROOT = Path(__file__).resolve().parent / ("drpo_cu1_reproduction_results_smoke" if SMOKE else "drpo_cu1_e3_adam_results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
EPS = 1e-12
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class Protocol:
    # Environment
    state_dim: int = 6
    action_dim: int = 2
    n_train_states: int = 4096
    n_test_states: int = 4096
    positive_samples_per_state: int = 4
    negative_samples_per_state: int = 8
    gap_to_unseen_optimum: float = 0.70
    negative_offset_from_positive: float = 0.50
    positive_contour_radius: float = 0.75
    negative_contour_radius: float = 1.20
    reward_width: float = 0.75
    baseline: float = 0.40
    positive_angle_1: float = 0.20
    # Policy
    hidden_dim: int = 64
    hidden_layers: int = 2
    initial_sigma: float = 0.60
    # Positive-only / E1 / E2
    positive_adam_lr: float = 1e-3
    positive_batch_states: int = 256
    positive_steps: int = 2000
    positive_continuation_steps: int = 2000
    lbfgs_lr: float = 0.25
    lbfgs_max_iter: int = 120
    positive_polish_min_steps: int = 100
    positive_polish_max_steps: int = 500
    positive_polish_check_every: int = 25
    positive_polish_lr: float = 1e-4
    eval_every: int = 100
    probe_states: int = 128
    # E3 / E4 optimizer: unified Adam training for the paper-facing C-U1 pipeline.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    # E3
    near_far_standardized_threshold: float = 5.0
    e3_fixed_alpha: float = 1.40
    e3_fixed_lr: float = 1e-4
    e3_fixed_steps: int = 2000
    e3_learn_alpha: float = 0.15
    e3_learn_lr: float = 5e-4
    e3_learn_steps: int = 2000
    e3_cap_ratio: float = 0.05
    # Explicitly re-registered task-failure criterion for the reconstruction.
    task_failure_retention: float = 0.45
    task_failure_consecutive_evals: int = 3
    log_sigma_event_boundary: float = 12.0
    # E4
    e4_fixed_alphas: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75)
    e4_learn_alphas: tuple[float, ...] = (0.0, 0.10, 0.20, 0.30, 0.35, 0.38, 0.40, 0.50)
    e4_local_lr: float = 5e-4
    e4_local_warm_steps: int = 200
    e4_local_continuation_steps: int = 200
    e4_runaway_steps: int = 4000
    e4_control_alpha_local: float = 1.0
    e4_control_lambda_far: float = 1.0
    e4_control_far_cap_ratio: float = 0.05
    e4_control_lr: float = 5e-4
    e4_control_steps: int = 4000
    # Convergence
    normalized_residual_threshold: float = 2e-3
    absolute_residual_threshold_alpha_zero: float = 1e-3
    # Formal seeds
    e1_e2_seeds: tuple[int, ...] = tuple(range(10, 30))
    e3_seeds: tuple[int, ...] = tuple(range(30, 50))
    e4_seeds: tuple[int, ...] = tuple(range(50, 70))
    variance_robustness_seeds: tuple[int, ...] = tuple(range(5, 10))


P = Protocol()
if SMOKE:
    P = Protocol(
        n_train_states=64,
        n_test_states=64,
        hidden_dim=16,
        positive_batch_states=32,
        positive_steps=4,
        positive_continuation_steps=2,
        lbfgs_max_iter=1,
        positive_polish_min_steps=1,
        positive_polish_max_steps=1,
        positive_polish_check_every=1,
        probe_states=4,
        e3_fixed_steps=3,
        e3_learn_steps=3,
        e4_fixed_alphas=(0.0, 1.0, 1.75),
        e4_learn_alphas=(0.0, 0.38, 0.40),
        e4_local_warm_steps=2,
        e4_local_continuation_steps=1,
        e4_runaway_steps=3,
        e4_control_steps=3,
        e1_e2_seeds=(10,),
        e3_seeds=(30,),
        e4_seeds=(50,),
        variance_robustness_seeds=(5,),
    )


# =============================================================================
# 1. General utilities
# =============================================================================


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_ci(values: Sequence[float], seed: int = 20260624, n_boot: int = 4000) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def norm_tuple(grads: Sequence[torch.Tensor | None]) -> torch.Tensor:
    vals = [g.reshape(-1) for g in grads if g is not None]
    if not vals:
        return torch.zeros((), device=DEVICE)
    return torch.linalg.vector_norm(torch.cat(vals))


def add_tuples(*items: Sequence[torch.Tensor | None], scales: Sequence[float] | None = None) -> tuple[torch.Tensor | None, ...]:
    if scales is None:
        scales = [1.0] * len(items)
    out: list[torch.Tensor | None] = []
    for components in zip(*items):
        value = None
        for g, c in zip(components, scales):
            if g is not None:
                value = c * g if value is None else value + c * g
        out.append(value)
    return tuple(out)


def scale_tuple(grads: Sequence[torch.Tensor | None], scale: float | torch.Tensor) -> tuple[torch.Tensor | None, ...]:
    return tuple(None if g is None else g * scale for g in grads)


def set_parameter_grads(params: Sequence[nn.Parameter], grads: Sequence[torch.Tensor | None]) -> None:
    for p, g in zip(params, grads):
        p.grad = None if g is None else g.detach().clone()


def finite_model(model: nn.Module) -> bool:
    return all(torch.isfinite(p).all().item() for p in model.parameters())


def make_adam(params: Sequence[nn.Parameter], lr: float) -> torch.optim.Adam:
    return torch.optim.Adam(
        params,
        lr=lr,
        betas=(P.adam_beta1, P.adam_beta2),
        eps=P.adam_eps,
    )


def gradient_norm_from_parameters(params: Sequence[nn.Parameter]) -> float:
    grads = [p.grad.reshape(-1) for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return torch.linalg.vector_norm(torch.cat(grads)).item()


def optimizer_step_with_norm(
    optimizer: torch.optim.Optimizer, params: Sequence[nn.Parameter]
) -> float:
    before = [p.detach().clone() for p in params]
    optimizer.step()
    deltas = [(p.detach() - old).reshape(-1) for p, old in zip(params, before)]
    return torch.linalg.vector_norm(torch.cat(deltas)).item() if deltas else 0.0


def support_diagnostics(actor: "GaussianActor", split: "Split") -> dict[str, Any]:
    """Audit all registered states; never sample only a prefix.

    The theory-facing failure mode is support contraction. A positive-boundary
    crossing is retained only as an unexpected implementation/numerical event,
    not promoted to a second scientific branch.
    """
    with torch.no_grad():
        _, log_sigma = actor(split.s)
        sigma = torch.exp(log_sigma)
    finite_log_sigma = bool(torch.isfinite(log_sigma).all().item())
    finite_sigma = bool(torch.isfinite(sigma).all().item())
    log_min = float(log_sigma.min().item()) if finite_log_sigma else float("nan")
    log_max = float(log_sigma.max().item()) if finite_log_sigma else float("nan")
    return {
        "log_sigma_min_all_states": log_min,
        "log_sigma_max_all_states": log_max,
        "sigma_output_finite_all_states": finite_sigma,
        "log_sigma_output_finite_all_states": finite_log_sigma,
        "support_contraction_boundary": finite_log_sigma and log_min < -P.log_sigma_event_boundary,
        "unexpected_support_expansion_boundary": finite_log_sigma and log_max > P.log_sigma_event_boundary,
    }


def support_event_type(diag: dict[str, Any]) -> str | None:
    # NaN/Inf output failures are numerical events and take precedence over
    # finite support-boundary labels. The individual diagnostic flags are still
    # retained so a single step can be audited without collapsing categories.
    if not diag["log_sigma_output_finite_all_states"]:
        return "nonfinite_log_sigma_output"
    if not diag["sigma_output_finite_all_states"]:
        return "nonfinite_sigma_output"
    if diag["support_contraction_boundary"]:
        return "support_contraction"
    if diag["unexpected_support_expansion_boundary"]:
        return "unexpected_support_expansion"
    return None


# =============================================================================
# 2. Exact C-U1 environment
# =============================================================================


def base_from_state(s: torch.Tensor) -> torch.Tensor:
    b1 = 0.70 * torch.tanh(0.85 * s[:, 0] - 0.30 * s[:, 1] * s[:, 2] + 0.20 * torch.sin(1.6 * s[:, 3]))
    b2 = 0.65 * torch.tanh(-0.50 * s[:, 1] + 0.35 * torch.cos(1.1 * s[:, 4]) + 0.22 * s[:, 0] * s[:, 5])
    return torch.stack([b1, b2], dim=1)


def task_direction_from_state(s: torch.Tensor) -> torch.Tensor:
    angle = 1.15 * torch.tanh(0.75 * s[:, 0] + 0.50 * s[:, 2] - 0.30 * s[:, 5])
    angle = angle + 0.30 * torch.sin(1.35 * s[:, 1])
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)


def orthogonal(direction: torch.Tensor) -> torch.Tensor:
    return torch.stack([-direction[:, 1], direction[:, 0]], dim=1)


def reward_from_optimum(action: torch.Tensor, optimum: torch.Tensor) -> torch.Tensor:
    d = torch.linalg.vector_norm(action - optimum, dim=-1)
    return torch.exp(-0.5 * (d / P.reward_width).square())


@dataclass
class Split:
    s: torch.Tensor
    a_plus: torch.Tensor
    a_star: torch.Tensor
    a_minus: torch.Tensor
    direction: torch.Tensor
    orthogonal: torch.Tensor
    positive_actions: torch.Tensor
    positive_rewards: torch.Tensor
    positive_advantages: torch.Tensor
    negative_actions: torch.Tensor
    negative_rewards: torch.Tensor
    negative_advantages: torch.Tensor


@dataclass
class Environment:
    train: Split
    test: Split


def positive_angles() -> torch.Tensor:
    # Four equal-reward points with exact centroid a_plus.
    t1 = P.positive_angle_1
    cos_t2 = 2.0 * P.gap_to_unseen_optimum / P.positive_contour_radius - math.cos(t1)
    if not (-1.0 <= cos_t2 <= 1.0):
        raise RuntimeError("Invalid positive contour geometry")
    t2 = math.acos(cos_t2)
    return torch.tensor([math.pi - t1, math.pi + t1, math.pi - t2, math.pi + t2], dtype=DTYPE)


def negative_angles() -> torch.Tensor:
    # Index 0 is a_minus; index 4 is the farthest contour copy.
    return torch.tensor([
        math.pi,
        3.0 * math.pi / 4.0,
        math.pi / 2.0,
        math.pi / 4.0,
        0.0,
        -math.pi / 4.0,
        -math.pi / 2.0,
        -3.0 * math.pi / 4.0,
    ], dtype=DTYPE)


def make_split(s: torch.Tensor) -> Split:
    plus = base_from_state(s)
    u = task_direction_from_state(s)
    v = orthogonal(u)
    star = plus + P.gap_to_unseen_optimum * u
    minus = plus - P.negative_offset_from_positive * u

    pa = positive_angles().to(s.device)
    pdir = torch.cos(pa)[None, :, None] * u[:, None, :] + torch.sin(pa)[None, :, None] * v[:, None, :]
    pos = star[:, None, :] + P.positive_contour_radius * pdir
    pos_r = reward_from_optimum(pos, star[:, None, :])
    pos_a = pos_r - P.baseline

    na = negative_angles().to(s.device)
    ndir = torch.cos(na)[None, :, None] * u[:, None, :] + torch.sin(na)[None, :, None] * v[:, None, :]
    neg = star[:, None, :] + P.negative_contour_radius * ndir
    neg_r = reward_from_optimum(neg, star[:, None, :])
    neg_a = neg_r - P.baseline

    return Split(s, plus, star, minus, u, v, pos, pos_r, pos_a, neg, neg_r, neg_a)


def make_environment(seed: int) -> Environment:
    g = torch.Generator(device="cpu").manual_seed(seed)
    train_s = torch.randn(P.n_train_states, P.state_dim, generator=g, dtype=DTYPE).to(DEVICE)
    test_s = torch.randn(P.n_test_states, P.state_dim, generator=g, dtype=DTYPE).to(DEVICE)
    return Environment(make_split(train_s), make_split(test_s))


def audit_environment(env: Environment) -> dict[str, Any]:
    t = env.train
    pos_centroid_error = torch.max(torch.abs(t.positive_actions.mean(dim=1) - t.a_plus)).item()
    neg_reward_range = (t.negative_rewards.max(1).values - t.negative_rewards.min(1).values).abs().max().item()
    neg_adv_range = (t.negative_advantages.max(1).values - t.negative_advantages.min(1).values).abs().max().item()
    pos_reward_range = (t.positive_rewards.max(1).values - t.positive_rewards.min(1).values).abs().max().item()
    d = torch.linalg.vector_norm(t.negative_actions - t.a_plus[:, None, :], dim=-1)
    nearest = d[:, 0].mean().item()
    farthest = d[:, 4].mean().item()
    out = {
        "state_distribution": "Normal(0,I_6)",
        "n_train_states": P.n_train_states,
        "n_test_states": P.n_test_states,
        "positive_actions_per_state": 4,
        "negative_actions_per_state": 8,
        "positive_centroid_max_error": pos_centroid_error,
        "positive_reward_max_range_per_state": pos_reward_range,
        "negative_reward_max_range_per_state": neg_reward_range,
        "negative_advantage_max_range_per_state": neg_adv_range,
        "positive_advantage_fraction": (t.positive_advantages > 0).float().mean().item(),
        "negative_advantage_fraction": (t.negative_advantages < 0).float().mean().item(),
        "nearest_negative_distance": nearest,
        "farthest_negative_distance": farthest,
        "farthest_nearest_distance_ratio": farthest / nearest,
        "analytic_positive_sigma": analytic_positive_sigma(),
        "analytic_mean_critical_alpha": analytic_mean_critical_alpha(),
        "analytic_variance_boundary_alpha": analytic_variance_boundary_alpha(),
    }
    ok = (
        pos_centroid_error < 2e-6
        and pos_reward_range < 2e-6
        and neg_reward_range < 2e-6
        and neg_adv_range < 2e-6
        and out["positive_advantage_fraction"] == 1.0
        and out["negative_advantage_fraction"] == 1.0
    )
    out["passed"] = ok
    return out


def positive_advantage_value() -> float:
    return math.exp(-0.5 * (P.positive_contour_radius / P.reward_width) ** 2) - P.baseline


def negative_advantage_value() -> float:
    return math.exp(-0.5 * (P.negative_contour_radius / P.reward_width) ** 2) - P.baseline


def analytic_positive_sigma() -> float:
    residual_second_moment = P.positive_contour_radius ** 2 - P.gap_to_unseen_optimum ** 2
    return math.sqrt(residual_second_moment / P.action_dim)


def analytic_mean_critical_alpha() -> float:
    return positive_advantage_value() / abs(negative_advantage_value())


def analytic_variance_boundary_alpha() -> float:
    # Solve p*M_pos(mu*) - q*M_neg(mu*) = 0 by bisection.
    p = positive_advantage_value()
    n = abs(negative_advantage_value())
    residual = P.positive_contour_radius ** 2 - P.gap_to_unseen_optimum ** 2

    def f(alpha: float) -> float:
        q = alpha * n
        disp = q * P.negative_offset_from_positive / (p - q)
        m_pos = residual + disp * disp
        m_neg = (P.negative_offset_from_positive + disp) ** 2
        return p * m_pos - q * m_neg

    lo, hi = 0.0, min(analytic_mean_critical_alpha() - 1e-8, 1.0)
    for _ in range(100):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# =============================================================================
# 3. Policy and objectives
# =============================================================================


class GaussianActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(P.state_dim, P.hidden_dim)
        self.fc2 = nn.Linear(P.hidden_dim, P.hidden_dim)
        self.mu_head = nn.Linear(P.hidden_dim, P.action_dim)
        self.log_std_head = nn.Linear(P.hidden_dim, 1)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, math.log(P.initial_sigma))

    def features(self, s: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc2(torch.relu(self.fc1(s))))

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(s)
        return self.mu_head(h), self.log_std_head(h).squeeze(-1)

    def mean_parameters(self) -> list[nn.Parameter]:
        return list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.mu_head.parameters())

    def all_parameters(self) -> list[nn.Parameter]:
        return list(self.parameters())


def gaussian_log_prob(mu: torch.Tensor, log_std: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    # actions: [B,K,2], mu [B,2], log_std [B]
    inv_std = torch.exp(-log_std)[:, None, None]
    z = (actions - mu[:, None, :]) * inv_std
    return -0.5 * z.square().sum(-1) - P.action_dim * log_std[:, None] - 0.5 * P.action_dim * math.log(2.0 * math.pi)


def actor_log_prob(actor: GaussianActor, s: torch.Tensor, actions: torch.Tensor, fixed_sigma: float | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, log_std_pred = actor(s)
    if fixed_sigma is None:
        log_std = log_std_pred
    else:
        log_std = torch.full_like(log_std_pred, math.log(fixed_sigma))
    return gaussian_log_prob(mu, log_std, actions), mu, log_std


def positive_loss(actor: GaussianActor, split: Split, ids: torch.Tensor | None = None, fixed_sigma: float | None = None) -> torch.Tensor:
    if ids is None:
        s, a, adv = split.s, split.positive_actions, split.positive_advantages
    else:
        s, a, adv = split.s[ids], split.positive_actions[ids], split.positive_advantages[ids]
    lp, _, _ = actor_log_prob(actor, s, a, fixed_sigma)
    return -(adv * lp).mean()


def local_negative_loss(actor: GaussianActor, split: Split, ids: torch.Tensor | None = None, fixed_sigma: float | None = None) -> torch.Tensor:
    if ids is None:
        s, a, adv = split.s, split.negative_actions[:, :1], split.negative_advantages[:, :1]
    else:
        s, a, adv = split.s[ids], split.negative_actions[ids, :1], split.negative_advantages[ids, :1]
    lp, _, _ = actor_log_prob(actor, s, a, fixed_sigma)
    return -(adv * lp).mean()


def all_negative_loss(actor: GaussianActor, split: Split, ids: torch.Tensor | None = None, fixed_sigma: float | None = None) -> torch.Tensor:
    if ids is None:
        s, a, adv = split.s, split.negative_actions, split.negative_advantages
    else:
        s, a, adv = split.s[ids], split.negative_actions[ids], split.negative_advantages[ids]
    lp, _, _ = actor_log_prob(actor, s, a, fixed_sigma)
    return -(adv * lp).mean()


def near_far_losses(actor: GaussianActor, split: Split, ids: torch.Tensor, fixed_sigma: float | None) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    s = split.s[ids]
    a = split.negative_actions[ids]
    adv = split.negative_advantages[ids]
    lp, mu, log_std = actor_log_prob(actor, s, a, fixed_sigma)
    standardized = torch.linalg.vector_norm(a - mu[:, None, :], dim=-1) / torch.exp(log_std)[:, None]
    near = (standardized.detach() <= P.near_far_standardized_threshold).to(lp.dtype)
    far = 1.0 - near
    denom = float(lp.numel())
    l_near = -(adv * lp * near).sum() / denom
    l_far = -(adv * lp * far).sum() / denom
    raw = torch.linalg.vector_norm(a - mu[:, None, :], dim=-1)
    diagnostics = {
        "near_occupancy": near.mean().item(),
        "far_occupancy": far.mean().item(),
        "raw_distance_mean": raw.mean().item(),
        "standardized_distance_mean": standardized.mean().item(),
    }
    return l_near, l_far, diagnostics


def evaluation(actor: GaussianActor, split: Split, fixed_sigma: float | None = None) -> dict[str, float]:
    actor.eval()
    with torch.no_grad():
        mu, pred = actor(split.s)
        log_std = pred if fixed_sigma is None else torch.full_like(pred, math.log(fixed_sigma))
        reward = reward_from_optimum(mu, split.a_star)
        axis = ((mu - split.a_plus) * split.direction).sum(-1)
        normalized = axis / P.gap_to_unseen_optimum
        sigma = torch.exp(log_std)
        return {
            "reward": reward.mean().item(),
            "normalized_extrapolation_displacement": normalized.mean().item(),
            "distance_to_a_plus": torch.linalg.vector_norm(mu - split.a_plus, dim=-1).mean().item(),
            "distance_to_a_star": torch.linalg.vector_norm(mu - split.a_star, dim=-1).mean().item(),
            "sigma_mean": sigma.mean().item(),
            "sigma_min": sigma.min().item(),
            "sigma_max": sigma.max().item(),
            "log_sigma_min": log_std.min().item(),
            "log_sigma_max": log_std.max().item(),
            "log_sigma_output_finite": bool(torch.isfinite(log_std).all().item()),
            "sigma_output_finite": bool(torch.isfinite(sigma).all().item()),
        }


def policy_distance_diagnostics(actor: GaussianActor, split: Split, fixed_sigma: float | None = None) -> dict[str, float]:
    """Population diagnostics for raw/standardized remoteness and dynamic near/far occupancy."""
    actor.eval()
    with torch.no_grad():
        mu, pred = actor(split.s)
        log_std = pred if fixed_sigma is None else torch.full_like(pred, math.log(fixed_sigma))
        raw = torch.linalg.vector_norm(split.negative_actions - mu[:, None, :], dim=-1)
        standardized = raw / torch.exp(log_std)[:, None]
        near = standardized <= P.near_far_standardized_threshold
        local_raw = raw[:, 0]
        local_std = standardized[:, 0]
        farthest_raw = raw[:, 4]
        farthest_std = standardized[:, 4]
        return {
            "negative_raw_distance_mean": raw.mean().item(),
            "negative_standardized_distance_mean": standardized.mean().item(),
            "dynamic_near_occupancy": near.float().mean().item(),
            "dynamic_far_occupancy": (~near).float().mean().item(),
            "local_negative_raw_distance": local_raw.mean().item(),
            "local_negative_standardized_distance": local_std.mean().item(),
            "farthest_negative_raw_distance": farthest_raw.mean().item(),
            "farthest_negative_standardized_distance": farthest_std.mean().item(),
        }


def normalized_field_residual(actor: GaussianActor, split: Split, alpha: float, fixed_sigma: float | None, local_only: bool = True) -> dict[str, float]:
    params = actor.mean_parameters() if fixed_sigma is not None else actor.all_parameters()
    lp = positive_loss(actor, split, None, fixed_sigma)
    ln = local_negative_loss(actor, split, None, fixed_sigma) if local_only else all_negative_loss(actor, split, None, fixed_sigma)
    gp = torch.autograd.grad(lp, params, retain_graph=True, allow_unused=True)
    gn = torch.autograd.grad(ln, params, allow_unused=True)
    gt = add_tuples(gp, gn, scales=[1.0, alpha])
    np_, nn_, nt = norm_tuple(gp).item(), norm_tuple(scale_tuple(gn, alpha)).item(), norm_tuple(gt).item()
    return {
        "positive_gradient_norm": np_,
        "negative_gradient_norm": nn_,
        "total_gradient_norm": nt,
        "normalized_field_residual": nt / (np_ + nn_ + EPS),
    }


# =============================================================================
# 4. Positive-only training, E1 and E2
# =============================================================================


def positive_checkpoint_path(seed: int) -> Path:
    return ROOT / "positive_checkpoints" / f"seed_{seed}.pt"


def positive_initialization_checkpoint_path(seed: int) -> Path:
    return ROOT / "positive_checkpoints" / f"seed_{seed}_adam2000_initialization.pt"


def phantom_metrics(actor: GaussianActor, split: Split, probe_n: int | None = None) -> dict[str, float]:
    if probe_n is None:
        ids = torch.arange(len(split.s), device=DEVICE)
    else:
        ids = torch.arange(min(probe_n, len(split.s)), device=DEVICE)
    params = actor.all_parameters()
    loss = all_negative_loss(actor, split, ids, None)
    g = torch.autograd.grad(loss, params, allow_unused=True)
    with torch.no_grad():
        mu, log_std = actor(split.s[ids])
        a = split.negative_actions[ids]
        sigma = torch.exp(log_std)
        raw = torch.linalg.vector_norm(a - mu[:, None, :], dim=-1)
        z = raw / sigma[:, None]
    return {
        "aggregate_phantom_negative_gradient_norm": norm_tuple(g).item(),
        "negative_raw_distance_mean": raw.mean().item(),
        "negative_standardized_distance_mean": z.mean().item(),
        "near_standardized_distance": z[:, 0].mean().item(),
        "far_standardized_distance": z[:, 4].mean().item(),
    }


def train_positive(seed: int) -> tuple[GaussianActor, Environment, list[dict[str, Any]], dict[str, float]]:
    ckpt = positive_checkpoint_path(seed)
    trajectory_path = ROOT / "e2" / f"seed_{seed}_trajectory.csv"
    summary_path = ROOT / "e2" / f"seed_{seed}.json"
    env = make_environment(seed)
    actor = GaussianActor().to(DEVICE)
    init_ckpt = positive_initialization_checkpoint_path(seed)
    if ckpt.exists() and init_ckpt.exists() and summary_path.exists() and trajectory_path.exists():
        try:
            state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location=DEVICE)
        actor.load_state_dict(state)
        return actor, env, read_csv(trajectory_path) if trajectory_path.exists() else [], json.loads(summary_path.read_text())

    seed_all(seed)
    actor = GaussianActor().to(DEVICE)
    initial_phantom = phantom_metrics(actor, env.train, P.probe_states)
    optimizer = make_adam(list(actor.parameters()), P.positive_adam_lr)
    gen = torch.Generator(device="cpu").manual_seed(seed + 100003)
    trajectory: list[dict[str, Any]] = []

    def record(step: int, stage: str) -> None:
        row = {"step": step, "stage": stage, **evaluation(actor, env.test)}
        if step % max(P.eval_every, 1) == 0 or stage != "adam":
            row.update(phantom_metrics(actor, env.train, P.probe_states))
        trajectory.append(row)

    record(0, "initial")
    actor.train()
    for step in range(1, P.positive_steps + 1):
        ids = torch.randint(0, P.n_train_states, (P.positive_batch_states,), generator=gen).to(DEVICE)
        loss = positive_loss(actor, env.train, ids)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % P.eval_every == 0 or step == P.positive_steps:
            record(step, "adam")

    # E3/E4 must start from the frozen 2000-step Adam checkpoint. The later
    # LBFGS/continuation/polish sequence is an E2 terminal audit only and must
    # not silently alter the initialization of downstream experiments.
    init_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), init_ckpt)

    # Full-data stationary audit. Snapshot and restore if LBFGS fails.
    snapshot = copy.deepcopy(actor.state_dict())
    lbfgs = torch.optim.LBFGS(actor.parameters(), lr=P.lbfgs_lr, max_iter=P.lbfgs_max_iter, history_size=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(set_to_none=True)
        loss = positive_loss(actor, env.train)
        loss.backward()
        return loss

    try:
        lbfgs.step(closure)
        if not finite_model(actor):
            raise FloatingPointError("non-finite LBFGS state")
    except Exception:
        actor.load_state_dict(snapshot)
    record(P.positive_steps, "stationary_audit")

    # Equal-length 2x continuation.
    optimizer = make_adam(list(actor.parameters()), P.positive_adam_lr * 0.25)
    for extra in range(1, P.positive_continuation_steps + 1):
        ids = torch.randint(0, P.n_train_states, (P.positive_batch_states,), generator=gen).to(DEVICE)
        loss = positive_loss(actor, env.train, ids)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step = P.positive_steps + extra
        if extra % P.eval_every == 0 or extra == P.positive_continuation_steps:
            record(step, "continuation")

    # A second full-data audit is required after the continuation. Without this,
    # stochastic continuation noise can leave a small but non-negligible residual
    # even though the observable trajectory has plateaued.
    post_continuation_snapshot = copy.deepcopy(actor.state_dict())
    final_lbfgs = torch.optim.LBFGS(
        actor.parameters(), lr=P.lbfgs_lr, max_iter=P.lbfgs_max_iter,
        history_size=50, line_search_fn="strong_wolfe"
    )

    def final_closure() -> torch.Tensor:
        final_lbfgs.zero_grad(set_to_none=True)
        loss = positive_loss(actor, env.train)
        loss.backward()
        return loss

    final_audit_succeeded = False
    try:
        final_lbfgs.step(final_closure)
        if not finite_model(actor):
            raise FloatingPointError("non-finite final LBFGS state")
        final_audit_succeeded = True
    except Exception:
        actor.load_state_dict(post_continuation_snapshot)

    # LBFGS can stop with a small float32 residual. A short full-data Adam
    # polish is deterministic and enforces the pre-registered alpha=0
    # absolute-gradient criterion instead of silently relaxing it.
    polish = make_adam(list(actor.parameters()), P.positive_polish_lr)
    polish_steps_used = 0
    for polish_step in range(1, P.positive_polish_max_steps + 1):
        loss = positive_loss(actor, env.train)
        polish.zero_grad(set_to_none=True)
        loss.backward()
        polish.step()
        polish_steps_used = polish_step
        should_check = (
            polish_step >= P.positive_polish_min_steps
            and (polish_step % P.positive_polish_check_every == 0
                 or polish_step == P.positive_polish_max_steps)
        )
        if should_check:
            polish_field = normalized_field_residual(actor, env.train, 0.0, None)
            if polish_field["total_gradient_norm"] < P.absolute_residual_threshold_alpha_zero:
                break
    record(P.positive_steps + P.positive_continuation_steps, "final_stationary_audit_and_adaptive_polish")

    final_phantom = phantom_metrics(actor, env.train, None)
    final_eval = evaluation(actor, env.test)
    field = normalized_field_residual(actor, env.train, 0.0, None)
    summary = {
        "seed": seed,
        **final_eval,
        **final_phantom,
        **field,
        "initial_probe_phantom_gradient_norm": initial_phantom["aggregate_phantom_negative_gradient_norm"],
        "probe_phantom_growth": trajectory[-1].get("aggregate_phantom_negative_gradient_norm", float("nan")) / (initial_phantom["aggregate_phantom_negative_gradient_norm"] + EPS),
        "final_stationary_audit_succeeded": final_audit_succeeded,
        "full_data_polish_steps": polish_steps_used,
        "full_data_polish_max_steps": P.positive_polish_max_steps,
        "status": "stable_plateau_2x_confirmed" if final_audit_succeeded and field["total_gradient_norm"] < P.absolute_residual_threshold_alpha_zero else "finite_but_residual_above_strict_threshold",
    }
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), ckpt)
    write_csv(trajectory_path, trajectory)
    atomic_json(summary_path, summary)
    return actor, env, trajectory, summary


def e1_per_sample_gradient(actor: GaussianActor, s: torch.Tensor, a: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
    params = actor.all_parameters()
    lp, _, _ = actor_log_prob(actor, s[None, :], a[None, None, :], None)
    objective = advantage * lp.squeeze()
    grads = torch.autograd.grad(objective, params, allow_unused=True)
    return torch.cat([g.reshape(-1) for g in grads if g is not None])


def run_e1_seed(seed: int, actor: GaussianActor, env: Environment) -> dict[str, float]:
    out_path = ROOT / "e1" / f"seed_{seed}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    n = min(P.probe_states, len(env.train.s))
    near_grads, far_grads = [], []
    with torch.enable_grad():
        for i in range(n):
            adv = env.train.negative_advantages[i, 0]
            near_grads.append(e1_per_sample_gradient(actor, env.train.s[i], env.train.negative_actions[i, 0], adv))
            far_grads.append(e1_per_sample_gradient(actor, env.train.s[i], env.train.negative_actions[i, 4], adv))
    near = torch.stack(near_grads)
    far = torch.stack(far_grads)
    per_ratio = far.norm(dim=1) / (near.norm(dim=1) + EPS)

    ids = torch.arange(n, device=DEVICE)
    params = actor.all_parameters()
    lp_near, mu, log_std = actor_log_prob(actor, env.train.s[ids], env.train.negative_actions[ids, 0:1], None)
    lp_far, _, _ = actor_log_prob(actor, env.train.s[ids], env.train.negative_actions[ids, 4:5], None)
    adv_near = env.train.negative_advantages[ids, 0:1]
    adv_far = env.train.negative_advantages[ids, 4:5]
    g_near = torch.autograd.grad((adv_near * lp_near).mean(), params, retain_graph=True, allow_unused=True)
    g_far = torch.autograd.grad((adv_far * lp_far).mean(), params, allow_unused=True)

    with torch.no_grad():
        sigma = torch.exp(log_std)
        dn = torch.linalg.vector_norm(env.train.negative_actions[ids, 0] - mu, dim=-1)
        df = torch.linalg.vector_norm(env.train.negative_actions[ids, 4] - mu, dim=-1)
        sn = torch.sqrt((dn / sigma.square()).square() + ((dn / sigma).square() - P.action_dim).square())
        sf = torch.sqrt((df / sigma.square()).square() + ((df / sigma).square() - P.action_dim).square())
        advantage_ratio = (adv_far.abs().mean() / adv_near.abs().mean()).item()
    row = {
        "seed": seed,
        "advantage_far_near_ratio": advantage_ratio,
        "output_score_far_near_ratio": (sf / sn).mean().item(),
        "full_parameter_single_sample_far_near_ratio": per_ratio.mean().item(),
        "full_parameter_single_sample_far_near_median_ratio": per_ratio.median().item(),
        "aggregate_far_near_ratio": (norm_tuple(g_far) / (norm_tuple(g_near) + EPS)).item(),
    }
    atomic_json(out_path, row)
    return row


# =============================================================================
# 5. E3 dynamic Near/Far interventions
# =============================================================================


def solve_near_scale_for_budget(near: Sequence[torch.Tensor | None], far_capped: Sequence[torch.Tensor | None], target_norm: float) -> float:
    n = torch.cat([g.reshape(-1) for g in near if g is not None])
    f = torch.cat([g.reshape(-1) for g in far_capped if g is not None])
    a = torch.dot(n, n).item()
    b = 2.0 * torch.dot(n, f).item()
    c = torch.dot(f, f).item() - target_norm * target_norm
    if a < EPS:
        return 1.0
    disc = max(0.0, b * b - 4.0 * a * c)
    roots = [(-b + math.sqrt(disc)) / (2.0 * a), (-b - math.sqrt(disc)) / (2.0 * a)]
    positive = [r for r in roots if r >= 0]
    return max(positive) if positive else 0.0


def intervention_gradients(actor: GaussianActor, split: Split, ids: torch.Tensor, fixed_sigma: float | None, alpha: float, method: str, cap_ratio: float) -> tuple[tuple[torch.Tensor | None, ...], dict[str, float]]:
    params = actor.mean_parameters() if fixed_sigma is not None else actor.all_parameters()
    lp = positive_loss(actor, split, ids, fixed_sigma)
    ln, lf, diag = near_far_losses(actor, split, ids, fixed_sigma)
    gp = torch.autograd.grad(lp, params, retain_graph=True, allow_unused=True)
    gn = torch.autograd.grad(ln, params, retain_graph=True, allow_unused=True)
    gf = torch.autograd.grad(lf, params, allow_unused=True)
    wn = scale_tuple(gn, alpha)
    wf = scale_tuple(gf, alpha)
    raw_neg = add_tuples(wn, wf)

    near_norm = norm_tuple(wn).item()
    far_norm = norm_tuple(wf).item()
    raw_norm = norm_tuple(raw_neg).item()
    far_scale = min(1.0, cap_ratio * near_norm / (far_norm + EPS))
    wf_cap = scale_tuple(wf, far_scale)
    capped_neg = add_tuples(wn, wf_cap)
    capped_norm = norm_tuple(capped_neg).item()

    if method == "baseline":
        neg = raw_neg
    elif method == "near_zero":
        neg = wf
    elif method == "far_zero":
        neg = wn
    elif method == "far_cap":
        neg = capped_neg
    elif method == "global_scale":
        neg = scale_tuple(raw_neg, capped_norm / (raw_norm + EPS))
    elif method == "far_to_near":
        c = solve_near_scale_for_budget(wn, wf_cap, raw_norm)
        neg = add_tuples(scale_tuple(wn, c), wf_cap)
    else:
        raise ValueError(method)
    total = add_tuples(gp, neg)
    diag.update({
        "positive_gradient_norm": norm_tuple(gp).item(),
        "near_negative_gradient_norm": near_norm,
        "far_negative_gradient_norm": far_norm,
        "raw_negative_gradient_norm": raw_norm,
        "post_control_negative_gradient_norm": norm_tuple(neg).item(),
        "far_scale": far_scale,
        "total_update_norm": norm_tuple(total).item(),
    })
    return total, diag


def run_intervention(seed: int, initial_state: dict[str, torch.Tensor], env: Environment, method: str, fixed_sigma: float | None, alpha: float, lr: float, steps: int, branch: str) -> dict[str, Any]:
    out_dir = ROOT / "e3" / branch / method
    out_path = out_dir / f"seed_{seed}.json"
    traj_path = out_dir / f"seed_{seed}_trajectory.csv"
    if out_path.exists() and traj_path.exists():
        return json.loads(out_path.read_text())
    actor = GaussianActor().to(DEVICE)
    actor.load_state_dict(copy.deepcopy(initial_state))
    params = actor.mean_parameters() if fixed_sigma is not None else actor.all_parameters()
    optimizer = make_adam(params, lr)
    gen = torch.Generator(device="cpu").manual_seed(seed + 300007)
    traj: list[dict[str, Any]] = []
    positive_reference = evaluation(actor, env.test, fixed_sigma)["reward"]
    task_threshold = P.task_failure_retention * positive_reference
    below_threshold_eval_steps: deque[int] = deque(maxlen=P.task_failure_consecutive_evals)
    task_onset = None
    support_onset = None
    first_support_event_type = None
    stop_reason = "max_steps"

    for step in range(1, steps + 1):
        pre_support = support_diagnostics(actor, env.train) if fixed_sigma is None else {}
        ids = torch.randint(0, P.n_train_states, (P.positive_batch_states,), generator=gen).to(DEVICE)
        grads, diag = intervention_gradients(actor, env.train, ids, fixed_sigma, alpha, method, P.e3_cap_ratio)
        optimizer.zero_grad(set_to_none=True)
        set_parameter_grads(params, grads)
        raw_gradient_norm = norm_tuple(grads).item()
        parameter_update_norm = optimizer_step_with_norm(optimizer, params)

        finite = finite_model(actor)
        post_support = support_diagnostics(actor, env.train) if fixed_sigma is None else {}
        event_type = support_event_type(post_support) if fixed_sigma is None else None
        if event_type is not None and support_onset is None:
            support_onset = step
            first_support_event_type = event_type
        if not finite:
            stop_reason = "non_finite_parameter"
        elif event_type is not None:
            stop_reason = event_type + "_boundary_event"

        should_record = (
            step % P.eval_every == 0
            or step == 1
            or step == steps
            or event_type is not None
            or not finite
        )
        if should_record:
            ev = evaluation(actor, env.test, fixed_sigma)
            reward = ev["reward"]
            if reward < task_threshold:
                below_threshold_eval_steps.append(step)
            else:
                below_threshold_eval_steps.clear()
            if len(below_threshold_eval_steps) == P.task_failure_consecutive_evals and task_onset is None:
                task_onset = below_threshold_eval_steps[0]
            row = {
                "step": step,
                "method": method,
                "optimizer": "adam",
                **ev,
                **diag,
                "raw_total_gradient_norm": raw_gradient_norm,
                "parameter_update_norm": parameter_update_norm,
                "task_threshold": task_threshold,
                "support_boundary_event": event_type is not None,
                "support_event_type": event_type,
            }
            if fixed_sigma is None:
                row.update({f"pre_{k}": v for k, v in pre_support.items()})
                row.update({f"post_{k}": v for k, v in post_support.items()})
            traj.append(row)
        if not finite or event_type is not None:
            break

    final = evaluation(actor, env.test, fixed_sigma)
    summary = {
        "seed": seed,
        "method": method,
        "branch": branch,
        "optimizer": "adam",
        **final,
        "task_failure_threshold": task_threshold,
        "task_failure_onset": task_onset,
        "support_boundary_onset": support_onset,
        "support_event_type": first_support_event_type,
        "unexpected_support_expansion": first_support_event_type == "unexpected_support_expansion",
        "stop_reason": stop_reason,
        "finite_parameters": finite_model(actor),
        "steps_completed": traj[-1]["step"] if traj else 0,
    }
    write_csv(traj_path, traj)
    atomic_json(out_path, summary)
    return summary


# =============================================================================
# 6. E4 local extrapolation, learnable variance, and far-pressure controls
# =============================================================================


def analytic_local_solution(alpha: float) -> dict[str, float | bool]:
    p = positive_advantage_value()
    q = alpha * abs(negative_advantage_value())
    if q >= p:
        return {"finite_mean_fixed_point": False}
    displacement = q * P.negative_offset_from_positive / (p - q)
    normalized = displacement / P.gap_to_unseen_optimum
    reward = math.exp(-0.5 * ((P.gap_to_unseen_optimum - displacement) / P.reward_width) ** 2)
    residual = P.positive_contour_radius ** 2 - P.gap_to_unseen_optimum ** 2
    m_pos = residual + displacement ** 2
    m_neg = (P.negative_offset_from_positive + displacement) ** 2
    sigma2 = (p * m_pos - q * m_neg) / (P.action_dim * (p - q))
    return {
        "finite_mean_fixed_point": True,
        "analytic_normalized_extrapolation_displacement": normalized,
        "analytic_reward": reward,
        "analytic_sigma": math.sqrt(sigma2) if sigma2 > 0 else float("nan"),
        "finite_variance_fixed_point": sigma2 > 0,
    }


def local_objective(actor: GaussianActor, split: Split, ids: torch.Tensor | None, alpha: float, fixed_sigma: float | None) -> torch.Tensor:
    return positive_loss(actor, split, ids, fixed_sigma) + alpha * local_negative_loss(actor, split, ids, fixed_sigma)


def run_local_scan_seed(seed: int, initial_state: dict[str, torch.Tensor], env: Environment, alpha: float, fixed_sigma: float | None, branch: str) -> dict[str, Any]:
    label = f"alpha_{alpha:.2f}"
    out_dir = ROOT / "e4" / branch / label
    out_path = out_dir / f"seed_{seed}.json"
    traj_path = out_dir / f"seed_{seed}_trajectory.csv"
    if out_path.exists() and traj_path.exists():
        return json.loads(out_path.read_text())
    actor = GaussianActor().to(DEVICE)
    actor.load_state_dict(copy.deepcopy(initial_state))
    params = actor.mean_parameters() if fixed_sigma is not None else actor.all_parameters()
    optimizer = make_adam(params, P.e4_local_lr)
    # Every alpha within a seed receives the identical minibatch stream.
    gen = torch.Generator(device="cpu").manual_seed(seed + 400009)
    analytic = analytic_local_solution(alpha)
    finite_internal = bool(analytic.get("finite_mean_fixed_point", False)) and (
        fixed_sigma is not None or bool(analytic.get("finite_variance_fixed_point", False))
    )
    first_phase_steps = P.e4_local_warm_steps if finite_internal else P.e4_runaway_steps
    traj: list[dict[str, Any]] = []
    support_onset = None
    first_support_event_type = None
    stop_reason = "completed"

    def record(step: int, stage: str, extra: dict[str, Any] | None = None) -> None:
        ev = evaluation(actor, env.test, fixed_sigma)
        field = normalized_field_residual(actor, env.train, alpha, fixed_sigma, local_only=True)
        distance_diag = policy_distance_diagnostics(actor, env.train, fixed_sigma)
        row = {"step": step, "stage": stage, "optimizer": "adam", **ev, **field, **distance_diag}
        if fixed_sigma is None:
            row.update(support_diagnostics(actor, env.train))
        if extra:
            row.update(extra)
        traj.append(row)

    record(0, "initial")

    def adam_phase(number_of_steps: int, start_step: int, stage: str) -> int:
        nonlocal support_onset, first_support_event_type, stop_reason
        completed = 0
        for offset in range(1, number_of_steps + 1):
            step = start_step + offset
            pre_support = support_diagnostics(actor, env.train) if fixed_sigma is None else {}
            ids = torch.randint(0, P.n_train_states, (P.positive_batch_states,), generator=gen).to(DEVICE)
            loss = local_objective(actor, env.train, ids, alpha, fixed_sigma)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            raw_gradient_norm = gradient_norm_from_parameters(params)
            parameter_update_norm = optimizer_step_with_norm(optimizer, params)
            completed = offset
            finite = finite_model(actor)
            post_support = support_diagnostics(actor, env.train) if fixed_sigma is None else {}
            event_type = support_event_type(post_support) if fixed_sigma is None else None
            if event_type is not None and support_onset is None:
                support_onset = step
                first_support_event_type = event_type
            extra = {
                "raw_total_gradient_norm": raw_gradient_norm,
                "parameter_update_norm": parameter_update_norm,
            }
            if fixed_sigma is None:
                extra.update({f"pre_{k}": v for k, v in pre_support.items()})
                extra.update({f"post_{k}": v for k, v in post_support.items()})
            if not finite:
                stop_reason = "non_finite_parameter"
            elif event_type is not None:
                stop_reason = event_type + "_boundary_event"
            if (
                offset % P.eval_every == 0
                or offset == number_of_steps
                or event_type is not None
                or not finite
            ):
                record(step, stage, extra)
            if not finite or event_type is not None:
                break
        return completed

    completed_first = adam_phase(first_phase_steps, 0, "adam_phase_1")

    # Terminal audits measure the same full-data Adam objective without using a
    # second optimizer. This keeps E3/E4 on one optimizer story while retaining
    # the pre-registered residual and 2x-continuation checks.
    audit_1_ok = False
    audit_2_ok = False
    audit_1_residual = float("nan")
    audit_2_residual = float("nan")
    if finite_internal and finite_model(actor) and support_onset is None:
        field_1 = normalized_field_residual(actor, env.train, alpha, fixed_sigma, local_only=True)
        audit_1_residual = (
            field_1["total_gradient_norm"] if alpha == 0 else field_1["normalized_field_residual"]
        )
        threshold = P.absolute_residual_threshold_alpha_zero if alpha == 0 else P.normalized_residual_threshold
        audit_1_ok = audit_1_residual < threshold
        record(completed_first, "full_data_residual_audit_1", {"audit_residual": audit_1_residual})

        completed_second = adam_phase(
            P.e4_local_continuation_steps,
            completed_first,
            "adam_continuation",
        )
        if finite_model(actor) and support_onset is None:
            field_2 = normalized_field_residual(actor, env.train, alpha, fixed_sigma, local_only=True)
            audit_2_residual = (
                field_2["total_gradient_norm"] if alpha == 0 else field_2["normalized_field_residual"]
            )
            audit_2_ok = audit_2_residual < threshold
            record(
                completed_first + completed_second,
                "full_data_residual_audit_2",
                {"audit_residual": audit_2_residual},
            )

    final = evaluation(actor, env.test, fixed_sigma)
    field = normalized_field_residual(actor, env.train, alpha, fixed_sigma, local_only=True)
    stable = finite_internal and finite_model(actor) and audit_1_ok and audit_2_ok and support_onset is None
    positive_ceiling_reward = evaluation_from_geometry(P.gap_to_unseen_optimum)[0]
    if stable:
        displacement = final["normalized_extrapolation_displacement"]
        reward_gain = final["reward"] - positive_ceiling_reward
        if abs(displacement) <= 0.05:
            state = "stable_imitation_ceiling"
        elif reward_gain > 0.01 and displacement <= 1.25:
            state = "stable_beneficial_extrapolation"
        elif final["reward"] < P.task_failure_retention * positive_ceiling_reward:
            state = "stable_bad_fixed_point"
        else:
            state = "stable_over_extrapolated_fixed_point"
    elif stop_reason == "non_finite_parameter" or stop_reason.endswith("_boundary_event"):
        state = stop_reason
    else:
        state = "finite_continuing_drift_or_runaway"

    displacement_slope = float("nan")
    log_sigma_slope = float("nan")
    dynamic_rows = [r for r in traj if r.get("stage") in {"adam_phase_1", "adam_continuation"}]
    by_step = {int(r["step"]): r for r in dynamic_rows}
    ordered_dynamic = [by_step[k] for k in sorted(by_step)]
    if len(ordered_dynamic) >= 3:
        tail = ordered_dynamic[-min(5, len(ordered_dynamic)):]
        xs = np.asarray([float(r["step"]) for r in tail])
        if np.ptp(xs) > 0:
            displacement_slope = float(np.polyfit(xs, np.asarray([float(r["normalized_extrapolation_displacement"]) for r in tail]), 1)[0])
            log_sigma_slope = float(np.polyfit(xs, np.log(np.maximum(np.asarray([float(r["sigma_mean"]) for r in tail]), 1e-30)), 1)[0])
    summary = {
        "seed": seed,
        "alpha": alpha,
        "branch": branch,
        "optimizer": "adam",
        **analytic,
        **final,
        **field,
        **policy_distance_diagnostics(actor, env.train, fixed_sigma),
        "stationary_audit_attempted": finite_internal,
        "stationary_audit_1_succeeded": audit_1_ok,
        "stationary_audit_2_succeeded": audit_2_ok,
        "stationary_audit_succeeded": audit_1_ok and audit_2_ok,
        "stationary_audit_1_residual": audit_1_residual,
        "stationary_audit_2_residual": audit_2_residual,
        "state_class": state,
        "support_boundary_onset": support_onset,
        "support_event_type": first_support_event_type,
        "unexpected_support_expansion": first_support_event_type == "unexpected_support_expansion",
        "stop_reason": stop_reason,
        "normalized_extrapolation_displacement_window_slope": displacement_slope,
        "log_sigma_window_slope": log_sigma_slope,
    }
    write_csv(traj_path, traj)
    atomic_json(out_path, summary)
    return summary


def evaluation_from_geometry(distance_to_star: float) -> tuple[float]:
    return (math.exp(-0.5 * (distance_to_star / P.reward_width) ** 2),)


def e4_control_gradients(actor: GaussianActor, split: Split, ids: torch.Tensor, method: str, fixed_sigma: float) -> tuple[tuple[torch.Tensor | None, ...], dict[str, float]]:
    params = actor.mean_parameters()
    lp = positive_loss(actor, split, ids, fixed_sigma)
    ll = local_negative_loss(actor, split, ids, fixed_sigma)
    # Far pressure is the remaining seven actions, averaged as one group.
    s = split.s[ids]
    a = split.negative_actions[ids, 1:]
    adv = split.negative_advantages[ids, 1:]
    lpf, _, _ = actor_log_prob(actor, s, a, fixed_sigma)
    lf = -(adv * lpf).mean()
    gp = torch.autograd.grad(lp, params, retain_graph=True, allow_unused=True)
    gl = torch.autograd.grad(ll, params, retain_graph=True, allow_unused=True)
    gf = torch.autograd.grad(lf, params, allow_unused=True)
    wl = scale_tuple(gl, P.e4_control_alpha_local)
    wf = scale_tuple(gf, P.e4_control_lambda_far)
    raw = add_tuples(wl, wf)
    local_norm, far_norm, raw_norm = norm_tuple(wl).item(), norm_tuple(wf).item(), norm_tuple(raw).item()
    far_scale = min(1.0, P.e4_control_far_cap_ratio * local_norm / (far_norm + EPS))
    capped = add_tuples(wl, scale_tuple(wf, far_scale))
    cap_norm = norm_tuple(capped).item()
    if method == "uncontrolled_all":
        neg = raw
    elif method == "far_cap":
        neg = capped
    elif method == "budget_matched_global":
        neg = scale_tuple(raw, cap_norm / (raw_norm + EPS))
    else:
        raise ValueError(method)
    total = add_tuples(gp, neg)
    return total, {
        "positive_gradient_norm": norm_tuple(gp).item(),
        "local_negative_gradient_norm": local_norm,
        "far_negative_gradient_norm": far_norm,
        "raw_negative_gradient_norm": raw_norm,
        "post_control_negative_gradient_norm": norm_tuple(neg).item(),
        "far_scale": far_scale,
        "total_update_norm": norm_tuple(total).item(),
    }


def run_control_seed(seed: int, initial_state: dict[str, torch.Tensor], env: Environment, method: str) -> dict[str, Any]:
    out_dir = ROOT / "e4" / "control" / method
    out_path = out_dir / f"seed_{seed}.json"
    traj_path = out_dir / f"seed_{seed}_trajectory.csv"
    if out_path.exists() and traj_path.exists():
        return json.loads(out_path.read_text())
    actor = GaussianActor().to(DEVICE)
    actor.load_state_dict(copy.deepcopy(initial_state))
    params = actor.mean_parameters()
    optimizer = make_adam(params, P.e4_control_lr)
    gen = torch.Generator(device="cpu").manual_seed(seed + 500009)
    fixed_sigma = analytic_positive_sigma()
    traj: list[dict[str, Any]] = []
    task_threshold = P.task_failure_retention * evaluation(actor, env.test, fixed_sigma)["reward"]
    below_threshold_eval_steps: deque[int] = deque(maxlen=P.task_failure_consecutive_evals)
    task_onset, nonfinite_onset = None, None

    for step in range(1, P.e4_control_steps + 1):
        ids = torch.randint(0, P.n_train_states, (P.positive_batch_states,), generator=gen).to(DEVICE)
        grads, diag = e4_control_gradients(actor, env.train, ids, method, fixed_sigma)
        optimizer.zero_grad(set_to_none=True)
        set_parameter_grads(params, grads)
        raw_gradient_norm = norm_tuple(grads).item()
        parameter_update_norm = optimizer_step_with_norm(optimizer, params)
        diag["raw_total_gradient_norm"] = raw_gradient_norm
        diag["parameter_update_norm"] = parameter_update_norm
        diag["optimizer"] = "adam"
        if not finite_model(actor):
            nonfinite_onset = step
            break
        if step % P.eval_every == 0 or step == 1 or step == P.e4_control_steps:
            ev = evaluation(actor, env.test, fixed_sigma)
            if ev["reward"] < task_threshold:
                below_threshold_eval_steps.append(step)
            else:
                below_threshold_eval_steps.clear()
            if len(below_threshold_eval_steps) == P.task_failure_consecutive_evals and task_onset is None:
                task_onset = below_threshold_eval_steps[0]
            traj.append({"step": step, "method": method, **ev, **diag, "task_threshold": task_threshold})
    final = evaluation(actor, env.test, fixed_sigma) if finite_model(actor) else {k: float("nan") for k in ["reward", "normalized_extrapolation_displacement", "distance_to_a_plus", "distance_to_a_star", "sigma_mean", "sigma_min", "sigma_max", "log_sigma_min", "log_sigma_max"]}
    summary = {"seed": seed, "method": method, "optimizer": "adam", **final, "task_failure_threshold": task_threshold, "task_failure_onset": task_onset, "nonfinite_onset": nonfinite_onset, "finite_parameters": finite_model(actor), "steps_completed": traj[-1]["step"] if traj else 0}
    write_csv(traj_path, traj)
    atomic_json(out_path, summary)
    return summary


# =============================================================================
# 7. Variance-boundary robustness requested after the audit
# =============================================================================


def run_variance_robustness() -> list[dict[str, Any]]:
    """Check that the early variance boundary is not just one event threshold.

    Main training remains unclamped. We rerun alpha in {0.38,0.40,0.50} on the
    development seeds at two learning rates, never stop at |log sigma|=12, and
    record crossings of 8/10/12/14 plus finite-horizon trends. This separates a
    true no-fixed-point trend from an arbitrary reporting cutoff.
    """
    out_path = ROOT / "variance_robustness" / "results.csv"
    if out_path.exists():
        return [dict(r) for r in read_csv(out_path)]
    rows: list[dict[str, Any]] = []
    for seed in P.variance_robustness_seeds:
        _, env, _, _ = train_positive(seed)
        initial = copy.deepcopy(load_initialization_state(seed))
        for alpha in (0.38, 0.40, 0.50):
            for lr in (1.0e-4, 2.5e-4, 5e-4):
                actor = GaussianActor().to(DEVICE)
                actor.load_state_dict(copy.deepcopy(initial))
                optimizer = make_adam(list(actor.parameters()), lr)
                gen = torch.Generator(device="cpu").manual_seed(seed + 600011)
                contraction_crossings = {8: None, 10: None, 12: None, 14: None}
                expansion_crossings = {8: None, 10: None, 12: None, 14: None}
                sigma_trace: list[tuple[int, float]] = []
                steps = P.e4_runaway_steps
                for step in range(1, steps + 1):
                    ids = torch.randint(0, P.n_train_states, (P.positive_batch_states,), generator=gen).to(DEVICE)
                    loss = local_objective(actor, env.train, ids, alpha, None)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    if not finite_model(actor):
                        break
                    if step % P.eval_every == 0 or step == 1:
                        with torch.no_grad():
                            _, ls = actor(env.train.s)
                            min_log_sigma = ls.min().item()
                            max_log_sigma = ls.max().item()
                            sigma_trace.append((step, torch.exp(ls).mean().item()))
                            for threshold in contraction_crossings:
                                if contraction_crossings[threshold] is None and min_log_sigma < -threshold:
                                    contraction_crossings[threshold] = step
                                if expansion_crossings[threshold] is None and max_log_sigma > threshold:
                                    expansion_crossings[threshold] = step
                ev = evaluation(actor, env.test)
                slope = float("nan")
                if len(sigma_trace) >= 5:
                    x = np.array([z[0] for z in sigma_trace[-5:]], dtype=float)
                    y = np.log(np.array([max(z[1], 1e-30) for z in sigma_trace[-5:]], dtype=float))
                    slope = float(np.polyfit(x, y, 1)[0])
                rows.append({
                    "seed": seed, "alpha": alpha, "lr": lr, **ev,
                    "finite_parameters": finite_model(actor),
                    "log_sigma_window_slope": slope,
                    **{f"support_contraction_cross_log_sigma_minus_{k}": v for k, v in contraction_crossings.items()},
                    **{f"unexpected_support_expansion_cross_log_sigma_plus_{k}": v for k, v in expansion_crossings.items()},
                })
    write_csv(out_path, rows)
    return rows


# =============================================================================
# 8. Aggregation, plotting, and reference regression
# =============================================================================


def aggregate_group(rows: list[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in rows:
        k = tuple(r[x] for x in keys)
        groups.setdefault(k, []).append(r)
    out = []
    for k, rs in groups.items():
        row = {name: value for name, value in zip(keys, k)}
        all_names = []
        for item in rs:
            for name in item:
                if name not in keys and name not in all_names:
                    all_names.append(name)
        for name in all_names:
            vals = []
            for item in rs:
                value = item.get(name)
                if isinstance(value, bool):
                    vals.append(float(value))
                elif isinstance(value, (int, float)) and value is not None and math.isfinite(float(value)):
                    vals.append(float(value))
            if vals:
                m, lo, hi = mean_ci(vals)
                row[name] = m
                row[name + "_ci_low"] = lo
                row[name + "_ci_high"] = hi
        # Explicit event rates remain auditable even when the first row has None.
        for event_name in ("task_failure_onset", "support_boundary_onset", "nonfinite_onset"):
            if any(event_name in item for item in rs):
                row[event_name + "_event_rate"] = sum(item.get(event_name) is not None for item in rs) / len(rs)
        row["n"] = len(rs)
        out.append(row)
    return out


def plot_phase(rows: list[dict[str, Any]], branch: str) -> None:
    if plt is None or not rows:
        return
    by_alpha = aggregate_group(rows, ["alpha"])
    by_alpha.sort(key=lambda x: float(x["alpha"]))
    x = [float(r["alpha"]) for r in by_alpha]
    y = [float(r["reward"]) for r in by_alpha]
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("Local negative-gradient strength alpha")
    plt.ylabel("Held-out-context reward")
    plt.title(f"E4 {branch}: performance")
    plt.tight_layout()
    plt.savefig(ROOT / "e4" / f"{branch}_reward_phase.png", dpi=220)
    plt.close()

    # Only finite, stable/analytic points go on the displacement phase plot.
    finite = [r for r in by_alpha if math.isfinite(float(r.get("normalized_extrapolation_displacement", float("nan")))) and float(r["alpha"]) < analytic_mean_critical_alpha()]
    plt.figure(figsize=(7, 5))
    plt.plot([float(r["alpha"]) for r in finite], [float(r["normalized_extrapolation_displacement"]) for r in finite], marker="o")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Local negative-gradient strength alpha")
    plt.ylabel("Normalized extrapolation displacement toward hidden optimum")
    plt.title(f"E4 {branch}: finite fixed-point branch")
    plt.tight_layout()
    plt.savefig(ROOT / "e4" / f"{branch}_normalized_extrapolation_displacement_phase.png", dpi=220)
    plt.close()


def regression_report(e1: list[dict[str, Any]], e2: list[dict[str, Any]], e3f: list[dict[str, Any]], e3l: list[dict[str, Any]], e4f: list[dict[str, Any]], e4l: list[dict[str, Any]], controls: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(rows: list[dict[str, Any]], field: str, predicate: Callable[[dict[str, Any]], bool] = lambda _: True) -> float:
        vals = [float(r[field]) for r in rows if predicate(r) and r.get(field) is not None and math.isfinite(float(r[field]))]
        return float(np.mean(vals)) if vals else float("nan")

    def event_rate(rows: list[dict[str, Any]], method: str) -> float:
        selected = [r for r in rows if r.get("method") == method]
        return sum(r.get("support_boundary_onset") is not None for r in selected) / len(selected) if selected else float("nan")

    checks: list[dict[str, Any]] = []

    def check(name: str, value: float, lo: float | None = None, hi: float | None = None, relation: str | None = None) -> None:
        passed = math.isfinite(value)
        if lo is not None:
            passed &= value >= lo
        if hi is not None:
            passed &= value <= hi
        checks.append({"name": name, "value": value, "expected_low": lo, "expected_high": hi, "passed": bool(passed), "relation": relation})

    if e1:
        check("E1 advantage ratio", avg(e1, "advantage_far_near_ratio"), 0.9999, 1.0001)
        check("E1 output score ratio", avg(e1, "output_score_far_near_ratio"), 7.2, 8.0)
    if e2:
        check("E2 reward", avg(e2, "reward"), 0.62, 0.67)
        check("E2 sigma", avg(e2, "sigma_mean"), 0.17, 0.21)
    if e3f:
        for method, bound, direction in [
            ("baseline", 0.30, "below"),
            ("near_zero", 0.32, "below"),
            ("far_zero", 0.55, "above"),
            ("far_cap", 0.50, "above"),
        ]:
            value = avg(e3f, "reward", lambda r, m=method: r["method"] == m)
            check(f"E3 fixed {method} reward", value, None if direction == "below" else bound, bound if direction == "below" else None)
    if e3l:
        check("E3 learn baseline support contraction", event_rate(e3l, "baseline"), 0.5, 1.0)
        check("E3 learn near-zero support contraction", event_rate(e3l, "near_zero"), 0.5, 1.0)
        check("E3 learn far-zero support events", event_rate(e3l, "far_zero"), 0.0, 0.2)
        unexpected = float(sum(bool(r.get("unexpected_support_expansion")) for r in e3l))
        check("E3 unexpected support-expansion events", unexpected, 0.0, 0.0)
    if e4f:
        required = {float(r["alpha"]) for r in e4f}
        check("E4 fixed alpha grid present", float(len(required)), float(len(P.e4_fixed_alphas)), float(len(P.e4_fixed_alphas)))
    if e4l:
        required = {float(r["alpha"]) for r in e4l}
        check("E4 learnable alpha grid present", float(len(required)), float(len(P.e4_learn_alphas)), float(len(P.e4_learn_alphas)))
        unexpected = float(sum(bool(r.get("unexpected_support_expansion")) for r in e4l))
        check("E4 unexpected support-expansion events", unexpected, 0.0, 0.0)
    if controls:
        methods = {str(r["method"]) for r in controls}
        check("E4 control methods present", float(len(methods)), 3.0, 3.0)

    return {"all_passed": bool(checks) and all(x["passed"] for x in checks), "checks": checks}


def collect_jsons(folder: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(folder.rglob("seed_*.json")) if "trajectory" not in p.name]


def run_preflight_self_tests() -> dict[str, Any]:
    """Fail fast on implementation errors before any formal seed is trained."""
    seed_all(12345)
    env = make_environment(P.e1_e2_seeds[0])
    audit = audit_environment(env)
    tests: dict[str, Any] = {"environment_audit": bool(audit["passed"])}

    # Fixed-sigma baseline gradient must equal autograd on the combined loss.
    actor = GaussianActor().to(DEVICE)
    ids = torch.arange(min(16, P.n_train_states), device=DEVICE)
    params = actor.mean_parameters()
    combined = positive_loss(actor, env.train, ids, analytic_positive_sigma()) + P.e3_fixed_alpha * all_negative_loss(actor, env.train, ids, analytic_positive_sigma())
    direct = torch.autograd.grad(combined, params, allow_unused=True)
    generated, _ = intervention_gradients(actor, env.train, ids, analytic_positive_sigma(), P.e3_fixed_alpha, "baseline", P.e3_cap_ratio)
    rel = (norm_tuple(add_tuples(direct, generated, scales=[1.0, -1.0])) / (norm_tuple(direct) + EPS)).item()
    tests["baseline_gradient_relative_error"] = rel

    # Equal-budget controls must match their declared target norms.
    _, cap_diag = intervention_gradients(actor, env.train, ids, analytic_positive_sigma(), P.e3_fixed_alpha, "far_cap", P.e3_cap_ratio)
    _, global_diag = intervention_gradients(actor, env.train, ids, analytic_positive_sigma(), P.e3_fixed_alpha, "global_scale", P.e3_cap_ratio)
    _, transfer_diag = intervention_gradients(actor, env.train, ids, analytic_positive_sigma(), P.e3_fixed_alpha, "far_to_near", P.e3_cap_ratio)
    tests["far_cap_global_budget_relative_error"] = abs(cap_diag["post_control_negative_gradient_norm"] - global_diag["post_control_negative_gradient_norm"]) / (cap_diag["post_control_negative_gradient_norm"] + EPS)
    tests["far_to_near_raw_budget_relative_error"] = abs(transfer_diag["post_control_negative_gradient_norm"] - transfer_diag["raw_negative_gradient_norm"]) / (transfer_diag["raw_negative_gradient_norm"] + EPS)

    # Analytic landmarks are protocol invariants, not learned outcomes.
    tests["analytic_positive_sigma"] = analytic_positive_sigma()
    tests["analytic_mean_critical_alpha"] = analytic_mean_critical_alpha()
    tests["analytic_variance_boundary_alpha"] = analytic_variance_boundary_alpha()
    g1 = torch.Generator(device="cpu").manual_seed(999)
    g2 = torch.Generator(device="cpu").manual_seed(999)
    tests["paired_minibatch_stream_identical"] = bool(torch.equal(
        torch.randint(0, P.n_train_states, (64,), generator=g1),
        torch.randint(0, P.n_train_states, (64,), generator=g2),
    ))
    adam = make_adam(actor.all_parameters(), P.e3_learn_lr)
    tests["formal_optimizer_is_adam"] = isinstance(adam, torch.optim.Adam)
    support = support_diagnostics(actor, env.train)
    tests["support_audit_covers_all_registered_states"] = bool(
        support["log_sigma_output_finite_all_states"] and len(env.train.s) == P.n_train_states
    )
    tests["all_passed"] = bool(
        tests["environment_audit"]
        and rel < 1e-6
        and tests["far_cap_global_budget_relative_error"] < 1e-6
        and tests["far_to_near_raw_budget_relative_error"] < 2e-5
        and abs(tests["analytic_positive_sigma"] - 0.1903943276465978) < 1e-10
        and abs(tests["analytic_mean_critical_alpha"] - 1.6933920000136828) < 1e-9
        and abs(tests["analytic_variance_boundary_alpha"] - 0.3806850232588901) < 1e-9
        and tests["paired_minibatch_stream_identical"]
        and tests["formal_optimizer_is_adam"]
        and tests["support_audit_covers_all_registered_states"]
    )
    return tests


# =============================================================================
# 9. Main one-click runner
# =============================================================================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the registered C-U1 Adam pipeline by experiment boundary.")
    parser.add_argument(
        "--stage",
        choices=("e1_e2", "e3", "e4", "all"),
        default="e3",
        help="Formal runs must use e3 or e4 separately. 'all' is smoke/integration only.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Fresh output directory. Formal supervisors should always provide one.",
    )
    return parser.parse_args(argv)


def load_initialization_state(seed: int) -> dict[str, torch.Tensor]:
    path = positive_initialization_checkpoint_path(seed)
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def main(argv: Sequence[str] | None = None) -> None:
    global ROOT
    args = parse_args(argv)
    if args.stage == "all" and not SMOKE:
        raise SystemExit("Formal E3 and E4 are separate delivery boundaries; --stage all is smoke-only.")
    if args.output_root is not None:
        ROOT = args.output_root.expanduser().resolve()
    else:
        suffix = "smoke" if SMOKE else args.stage
        ROOT = Path(__file__).resolve().parent / f"drpo_cu1_{suffix}_adam_results"
    ROOT.mkdir(parents=True, exist_ok=True)

    start = time.time()
    current_protocol = asdict(P)
    current_script_hash = sha256(Path(__file__))
    existing_manifest = ROOT / "manifest.json"
    if existing_manifest.exists():
        try:
            old = json.loads(existing_manifest.read_text())
            incompatible = (
                old.get("protocol") != current_protocol
                or old.get("script_version") != SCRIPT_VERSION
                or old.get("script_sha256") != current_script_hash
                or old.get("stage") != args.stage
            )
            if incompatible:
                archived = ROOT.with_name(ROOT.name + "_incompatible_" + time.strftime("%Y%m%d_%H%M%S"))
                ROOT.rename(archived)
                ROOT.mkdir(parents=True, exist_ok=True)
        except Exception:
            archived = ROOT.with_name(ROOT.name + "_unreadable_" + time.strftime("%Y%m%d_%H%M%S"))
            ROOT.rename(archived)
            ROOT.mkdir(parents=True, exist_ok=True)
    print(f"DRPO C-U1 Adam reproduction {SCRIPT_VERSION}")
    print(f"Stage: {args.stage}; device: {DEVICE}; smoke={SMOKE}; output={ROOT}")
    manifest = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": current_script_hash,
        "stage": args.stage,
        "optimizer": "adam",
        "protocol": current_protocol,
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "scope_note": "held-out-context generalization only; train/test states share the same distribution",
        "optimizer_note": "raw gradient norms and Adam parameter-update norms are reported separately",
    }
    atomic_json(ROOT / "manifest.json", manifest)
    source_dir = ROOT / "source_snapshot"
    source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), source_dir / Path(__file__).name)
    atomic_json(source_dir / "protocol.json", current_protocol)

    preflight = run_preflight_self_tests()
    atomic_json(ROOT / "preflight_self_tests.json", preflight)
    if not preflight["all_passed"]:
        raise SystemExit("Preflight self-tests failed; formal training was not started")
    print("[OK] Preflight implementation self-tests")

    audit = audit_environment(make_environment(P.e1_e2_seeds[0]))
    atomic_json(ROOT / "environment_audit.json", audit)
    if not audit["passed"]:
        raise SystemExit("Environment invariant audit failed")
    print("[OK] Environment audit")

    if args.stage == "e1_e2":
        needed_seeds = P.e1_e2_seeds
    elif args.stage == "e3":
        needed_seeds = P.e3_seeds
    elif args.stage == "e4":
        needed_seeds = P.e4_seeds + P.variance_robustness_seeds
    else:
        needed_seeds = tuple(sorted(set(P.e1_e2_seeds + P.e3_seeds + P.e4_seeds + P.variance_robustness_seeds)))

    audited_actors: dict[int, GaussianActor] = {}
    checkpoints: dict[int, dict[str, torch.Tensor]] = {}
    environments: dict[int, Environment] = {}
    e2_prerequisite_rows: list[dict[str, Any]] = []
    for seed in needed_seeds:
        actor, env, _, e2 = train_positive(seed)
        audited_actors[seed] = actor
        checkpoints[seed] = copy.deepcopy(load_initialization_state(seed))
        environments[seed] = env
        e2_prerequisite_rows.append(e2)
        print(f"[OK] Positive prerequisite seed {seed}")
    write_csv(ROOT / "positive_prerequisite_per_seed.csv", e2_prerequisite_rows)

    e1_rows: list[dict[str, Any]] = []
    e2_rows: list[dict[str, Any]] = []
    e3_fixed_rows: list[dict[str, Any]] = []
    e3_learn_rows: list[dict[str, Any]] = []
    e4_fixed_rows: list[dict[str, Any]] = []
    e4_learn_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    robust: list[dict[str, Any]] = []

    if args.stage in {"e1_e2", "all"}:
        for seed in P.e1_e2_seeds:
            e2_rows.append(json.loads((ROOT / "e2" / f"seed_{seed}.json").read_text()))
            e1_rows.append(run_e1_seed(seed, audited_actors[seed], environments[seed]))
        write_csv(ROOT / "e1" / "per_seed.csv", e1_rows)
        write_csv(ROOT / "e1" / "aggregate.csv", aggregate_group(e1_rows, []))
        write_csv(ROOT / "e2" / "per_seed.csv", e2_rows)
        write_csv(ROOT / "e2" / "aggregate.csv", aggregate_group(e2_rows, []))

    if args.stage in {"e3", "all"}:
        fixed_methods = ("baseline", "near_zero", "far_zero", "far_cap", "global_scale", "far_to_near")
        learn_methods = ("baseline", "near_zero", "far_zero", "far_cap", "global_scale")
        for seed in P.e3_seeds:
            for method in fixed_methods:
                print(f"  E3 fixed seed={seed} method={method}", flush=True)
                e3_fixed_rows.append(run_intervention(seed, checkpoints[seed], environments[seed], method, analytic_positive_sigma(), P.e3_fixed_alpha, P.e3_fixed_lr, P.e3_fixed_steps, "fixed_variance"))
            for method in learn_methods:
                print(f"  E3 learn seed={seed} method={method}", flush=True)
                e3_learn_rows.append(run_intervention(seed, checkpoints[seed], environments[seed], method, None, P.e3_learn_alpha, P.e3_learn_lr, P.e3_learn_steps, "learnable_variance"))
            print(f"[OK] E3 seed {seed}")
        write_csv(ROOT / "e3" / "fixed_variance_per_seed.csv", e3_fixed_rows)
        write_csv(ROOT / "e3" / "fixed_variance_aggregate.csv", aggregate_group(e3_fixed_rows, ["method"]))
        write_csv(ROOT / "e3" / "learnable_variance_per_seed.csv", e3_learn_rows)
        write_csv(ROOT / "e3" / "learnable_variance_aggregate.csv", aggregate_group(e3_learn_rows, ["method"]))

    if args.stage in {"e4", "all"}:
        for seed in P.e4_seeds:
            for alpha in P.e4_fixed_alphas:
                print(f"  E4 fixed seed={seed} alpha={alpha}", flush=True)
                e4_fixed_rows.append(run_local_scan_seed(seed, checkpoints[seed], environments[seed], alpha, analytic_positive_sigma(), "fixed_variance"))
            for alpha in P.e4_learn_alphas:
                print(f"  E4 learn seed={seed} alpha={alpha}", flush=True)
                e4_learn_rows.append(run_local_scan_seed(seed, checkpoints[seed], environments[seed], alpha, None, "learnable_variance"))
            for method in ("uncontrolled_all", "far_cap", "budget_matched_global"):
                print(f"  E4 control seed={seed} method={method}", flush=True)
                control_rows.append(run_control_seed(seed, checkpoints[seed], environments[seed], method))
            print(f"[OK] E4 seed {seed}")
        write_csv(ROOT / "e4" / "fixed_variance_per_seed.csv", e4_fixed_rows)
        write_csv(ROOT / "e4" / "fixed_variance_aggregate.csv", aggregate_group(e4_fixed_rows, ["alpha"]))
        write_csv(ROOT / "e4" / "learnable_variance_per_seed.csv", e4_learn_rows)
        write_csv(ROOT / "e4" / "learnable_variance_aggregate.csv", aggregate_group(e4_learn_rows, ["alpha"]))
        write_csv(ROOT / "e4" / "control_per_seed.csv", control_rows)
        write_csv(ROOT / "e4" / "control_aggregate.csv", aggregate_group(control_rows, ["method"]))
        plot_phase(e4_fixed_rows, "fixed_variance")
        plot_phase(e4_learn_rows, "learnable_variance")
        robust = run_variance_robustness()
        print("[OK] Variance-boundary robustness")

    regression = regression_report(e1_rows, e2_rows, e3_fixed_rows, e3_learn_rows, e4_fixed_rows, e4_learn_rows, control_rows)
    atomic_json(ROOT / "reference_regression.json", regression)

    summary = {
        "elapsed_seconds": time.time() - start,
        "stage": args.stage,
        "optimizer": "adam",
        "environment_audit": audit,
        "preflight_self_tests": preflight,
        "reference_regression_all_passed": regression["all_passed"],
        "results_root": str(ROOT),
        "important_scope": "All test-state results are held-out-context generalization under the same state distribution, not strict OOD.",
        "variance_robustness_rows": len(robust),
        "execution_note": "E3 and E4 are separate formal delivery boundaries; --stage all is smoke-only.",
    }
    atomic_json(ROOT / "RUN_COMPLETE.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not regression["all_passed"]:
        print("WARNING: run completed, but one or more registered mechanism checks failed. Inspect reference_regression.json.")
    print("Done. Re-running the same command resumes/skips completed seed files.")


if __name__ == "__main__":
    main()
