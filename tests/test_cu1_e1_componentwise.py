from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "drpo"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cu1_e1_componentwise_rerun as componentwise  # noqa: E402


def test_gaussian_output_components_match_linear_and_quadratic_identities() -> None:
    mu = torch.tensor([[0.2, -0.1], [-0.3, 0.4]], dtype=torch.float64)
    log_std = torch.tensor([math.log(0.4), math.log(0.7)], dtype=torch.float64)
    actions = torch.tensor(
        [
            [[0.7, -0.1], [1.4, 0.8], [-0.4, -0.9]],
            [[-0.3, 0.9], [0.8, -0.6], [-1.5, 0.4]],
        ],
        dtype=torch.float64,
    )

    result = componentwise.gaussian_output_components(mu, log_std, actions, action_dim=2)

    torch.testing.assert_close(result["normalized_mean"], result["raw_distance"])
    torch.testing.assert_close(
        result["normalized_quadratic"], result["raw_distance"].square()
    )
    torch.testing.assert_close(
        result["corrected_log_scale"], result["standardized2"]
    )


def test_gaussian_output_components_match_output_tensor_autograd() -> None:
    mu = torch.tensor([[0.1, -0.2]], dtype=torch.float64)
    log_std = torch.tensor([math.log(0.55)], dtype=torch.float64)
    actions = torch.tensor(
        [[[0.6, -0.2], [1.3, 0.7], [-0.5, -1.0]]], dtype=torch.float64
    )
    analytic = componentwise.gaussian_output_components(mu, log_std, actions, action_dim=2)

    mu_probe = mu[:, None, :].expand_as(actions).clone().requires_grad_(True)
    log_std_probe = log_std[:, None].expand(actions.shape[:2]).clone().requires_grad_(True)
    z = (actions - mu_probe) * torch.exp(-log_std_probe)[..., None]
    log_prob = (
        -0.5 * z.square().sum(-1)
        - 2 * log_std_probe
        - math.log(2.0 * math.pi)
    )
    grad_mu, grad_log_std = torch.autograd.grad(
        log_prob.sum(), (mu_probe, log_std_probe)
    )

    torch.testing.assert_close(
        torch.linalg.vector_norm(grad_mu, dim=-1), analytic["mean_score"]
    )
    torch.testing.assert_close(grad_log_std, analytic["log_scale_score"])
    torch.testing.assert_close(
        torch.sqrt(
            torch.linalg.vector_norm(grad_mu, dim=-1).square()
            + grad_log_std.square()
        ),
        analytic["joint_score"],
    )
