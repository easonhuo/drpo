import torch

from drpo.models import GaussianMLPPolicy


def test_gaussian_policy_log_prob_shape():
    policy = GaussianMLPPolicy(obs_dim=11, action_dim=3, hidden_sizes=(16, 16))
    obs = torch.zeros(5, 11)
    actions = torch.zeros(5, 3)

    log_prob = policy.log_prob(obs, actions)

    assert log_prob.shape == (5,)
