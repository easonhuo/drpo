from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

MODULE_PATH = Path(__file__).parents[1] / "src" / "drpo" / "e7_hopper_q2.py"
spec = importlib.util.spec_from_file_location("e7_hopper_q2", MODULE_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_episode_ids_and_discounted_returns() -> None:
    terminals = np.array([0, 1, 0, 0], dtype=bool)
    timeouts = np.array([0, 0, 0, 1], dtype=bool)
    ids = mod.build_episode_ids(terminals, timeouts)
    np.testing.assert_array_equal(ids, np.array([0, 0, 1, 1]))
    returns = mod.discounted_returns(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        terminals,
        timeouts,
        0.5,
    )
    np.testing.assert_allclose(returns, np.array([2.0, 2.0, 5.0, 4.0]))


def test_gaussian_components_and_autograd_identity() -> None:
    torch.manual_seed(0)
    policy = mod.SquashedGaussianPolicy(3, 2, (8,), -5.0, 2.0, 1e-6)
    obs = torch.randn(6, 3)
    actions = torch.tanh(torch.randn(6, 2))
    components = policy.score_components(obs, actions)
    torch.testing.assert_close(
        components["corrected_q_xi"], components["radius"].square()
    )
    assert torch.isfinite(components["joint_output_score_norm"]).all()
    error = mod.analytic_output_autograd_relative_error(
        policy,
        obs.numpy(),
        actions.numpy(),
        np.arange(6, dtype=np.int64),
        torch.device("cpu"),
    )
    assert error < 1e-6


def test_squashed_gaussian_is_finite_near_action_boundary() -> None:
    policy = mod.SquashedGaussianPolicy(3, 2, (8,), -5.0, 2.0, 1e-6)
    obs = torch.zeros(4, 3)
    actions = torch.tensor([[0.999999, -0.999999]] * 4)
    log_prob = policy.log_prob(obs, actions)
    components = policy.score_components(obs, actions)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(components["radius"]).all()
    assert torch.isfinite(components["raw_log_scale_score_norm"]).all()


def test_advantage_matching_separates_distance_and_matches_magnitude() -> None:
    rng = np.random.default_rng(0)
    count = 400
    advantages = -rng.uniform(0.1, 2.0, size=count).astype(np.float32)
    distances = np.linspace(0.2, 5.0, count).astype(np.float32)
    negative = np.arange(count, dtype=np.int64)
    near, far, summary = mod.match_near_far_indices(
        advantages,
        distances,
        negative,
        0.25,
        0.75,
        20,
        64,
        0.10,
        7,
    )
    assert len(near) > 0
    assert len(near) == len(far)
    assert summary["distance_far_near_ratio"] > 2.0
    assert 0.9 <= summary["advantage_magnitude_far_near_ratio"] <= 1.1


def test_far_cap_and_budget_global_are_supported() -> None:
    torch.manual_seed(1)
    policy = mod.SquashedGaussianPolicy(2, 1, (4,), -5.0, 2.0, 1e-6)
    obs = torch.zeros(4, 2)
    actions = torch.tensor([[0.0], [0.2], [0.95], [-0.95]])
    advantages = torch.tensor([1.0, -1.0, -1.0, -1.0])
    signed_loss, _ = mod.actor_batch_loss(
        policy, obs, actions, advantages, "signed", 1.0, 1.0, 0.5
    )
    capped_loss, diagnostics = mod.actor_batch_loss(
        policy, obs, actions, advantages, "far_cap", 1.0, 1.0, 0.5
    )
    global_loss, _ = mod.actor_batch_loss(
        policy,
        obs,
        actions,
        advantages,
        "budget_matched_global",
        1.0,
        0.5,
        0.5,
    )
    assert torch.isfinite(signed_loss)
    assert torch.isfinite(capped_loss)
    assert torch.isfinite(global_loss)
    assert diagnostics["far_cap_factor_mean"] < 1.0


def test_config_is_bound_to_q2() -> None:
    config = mod.load_config(
        Path(__file__).parents[1] / "configs" / "e7_hopper_q2_medium_replay_v2.yaml"
    )
    assert config.experiment_id == "EXT-H-E7-Q2"
    assert config.dataset_basename == "hopper_medium_replay-v2.hdf5"
    assert config.formal.seeds == tuple(range(100, 110))
    assert config.formal.rollout_episodes == 5


def test_legacy_hdf5_loader(tmp_path: Path) -> None:
    path = tmp_path / "tiny.hdf5"
    with h5py.File(path, "w") as handle:
        handle["observations"] = np.zeros((6, 3), dtype=np.float32)
        handle["actions"] = np.zeros((6, 2), dtype=np.float32)
        handle["rewards"] = np.arange(6, dtype=np.float32)
        handle["next_observations"] = np.ones((6, 3), dtype=np.float32)
        handle["terminals"] = np.array([0, 1, 0, 0, 0, 1], dtype=np.float32)
        handle["timeouts"] = np.zeros(6, dtype=np.float32)
    data = mod.load_hopper_hdf5(path, None)
    assert data.size == 6
    np.testing.assert_array_equal(data.episode_ids, np.array([0, 0, 1, 1, 1, 1]))


def test_registry_keeps_implemented_q2_blocked_without_claiming_results() -> None:
    import yaml

    root = Path(__file__).parents[1]
    registry = yaml.safe_load((root / "experiments" / "registry.yaml").read_text())
    entry = next(item for item in registry["experiments"] if item["id"] == "EXT-H-E7-Q2")
    assert entry["status"] == "not_run"
    assert entry["scientific_status"] == "not_run"
    assert entry["implementation_state"] == "implemented"
    assert entry["execution_gate"]["state"] == "blocked"
    assert entry["execution_gate"]["blocked_by"] == ["D-U1-E6-TAPER-01"]
    assert entry["formal_execution"]["activation_state"] == "blocked"
    assert entry["formal_execution"]["entrypoint"] == "src/drpo/e7_hopper_q2.py"
    assert entry["formal_execution"]["runner_archive_policy"]["mode"] == "forbid"
    assert entry["evidence"]["implementation_tests_passed"] is True
    assert entry["evidence"]["run_started"] is False


def test_handoff_preserves_v41_boundary_under_v42_route() -> None:
    handoff = (Path(__file__).parents[1] / "docs" / "handoff.md").read_text()
    assert "v42 增量登记：状态机一致性、E7 已实现门禁与 E4--E8 路线锁定" in handoff
    assert "v45（E4-TAPER 结果闭环、环境识别与公平性边界版）" in handoff
    assert "v41 增量登记：`EXT-H-E7-Q2` Hopper 实现" in handoff
    assert "implemented + not_run + blocked" in handoff
    assert "CPU 单元测试、静态检查和缩减工程 pilot" in handoff
    assert "尚未运行（not_run）" in handoff
    assert "不能替代 C-U1" in handoff
