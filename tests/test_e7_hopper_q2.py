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
    assert config.pilot.canonical_critic_seed == 42
    assert config.formal.canonical_critic_seed == 100
    assert config.pilot_rollout_required is True
    assert config.formal_rollout_required is True
    assert config.rollout_preflight_max_steps == 2000
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
    assert "v54 增量登记：`EXT-H-E7-Q2` canonical critic" in handoff
    assert "每个 run 只训练或严格复用一个 canonical critic artifact" in handoff
    assert "task_performance_collapse=null" in handoff
    assert "formal gate 也必须为 false" in handoff


def test_task_performance_unavailable_is_not_reported_as_no_collapse() -> None:
    config = mod.load_config(
        Path(__file__).parents[1] / "configs" / "e7_hopper_q2_medium_replay_v2.yaml"
    )
    rows = []
    for step in range(6):
        rows.append(
            {
                "step": step,
                "loss": 1.0,
                "positive_nll": 1.0,
                "gradient_norm": 1.0,
                "update_norm": 1.0,
                "sigma_mean": 1.0,
                "mean_boundary_fraction": 0.0,
                "log_std_min_fraction": 0.0,
                "log_std_max_fraction": 0.0,
                "mean_abs": 0.1,
                "phantom_distance_mean": 1.0,
                "rollout_status": "unavailable",
                "normalized_return": float("nan"),
            }
        )
    audit = mod.classify_actor_terminal(rows, config, None, False)
    assert audit["task_performance_status"] == "unavailable"
    assert audit["task_performance_collapse"] is None
    assert audit["normalized_return_available"] is False


def test_rollout_preflight_exercises_reset_step_episode_and_score(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeSpace:
        low = np.array([-1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)

        def seed(self, seed: int) -> None:
            self._seed = seed

        def sample(self) -> np.ndarray:
            return np.array([0.25, -0.25], dtype=np.float32)

    class FakeSpec:
        max_episode_steps = 3

    class FakeEnv:
        action_space = FakeSpace()
        spec = FakeSpec()

        def __init__(self) -> None:
            self.steps = 0

        def reset(self, seed: int | None = None):
            self.steps = 0
            return np.zeros(3, dtype=np.float32), {"seed": seed}

        def step(self, action: np.ndarray):
            self.steps += 1
            done = self.steps >= 2
            return np.ones(3, dtype=np.float32), 1.0, done, {"ok": True}

        def get_normalized_score(self, value: float) -> float:
            return value / 10.0

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        mod,
        "_open_d4rl_env",
        lambda env_id, registration_import: (
            FakeEnv(),
            {"gym_backend": "fake", "compatibility_shims": [], "make_attempts": []},
        ),
    )
    report = mod.preflight_d4rl_environment(
        env_id="fake-hopper",
        registration_import="fake_d4rl",
        expected_observation_dim=3,
        expected_action_dim=2,
        seed=7,
        max_steps=5,
        normalized_score_percent=True,
        output_dir=tmp_path / "preflight",
        required=True,
    )
    assert report["status"] == "passed"
    assert report["interaction_verified"] is True
    assert report["normalized_score_verified"] is True
    assert report["random_episode"]["steps"] == 2
    assert (tmp_path / "preflight" / "rollout_preflight.json").is_file()




def test_rollout_preflight_failure_persists_traceback_before_raising(
    tmp_path: Path, monkeypatch
) -> None:
    def fail_open(env_id: str, registration_import: str):
        raise RuntimeError("synthetic registration failure")

    monkeypatch.setattr(mod, "_open_d4rl_env", fail_open)
    output_dir = tmp_path / "preflight_failure"
    import pytest

    with pytest.raises(RuntimeError, match="rollout preflight failed"):
        mod.preflight_d4rl_environment(
            env_id="missing-hopper",
            registration_import="missing_d4rl",
            expected_observation_dim=3,
            expected_action_dim=2,
            seed=7,
            max_steps=5,
            normalized_score_percent=True,
            output_dir=output_dir,
            required=True,
        )
    report = __import__("json").loads(
        (output_dir / "rollout_preflight.json").read_text()
    )
    assert report["status"] == "failed"
    assert report["error_type"] == "RuntimeError"
    assert "synthetic registration failure" in report["error"]
    assert "Traceback" in report["traceback"]


def test_canonical_critic_is_trained_once_and_strictly_reused(tmp_path: Path) -> None:
    from dataclasses import replace

    config_path = (
        Path(__file__).parents[1] / "configs" / "e7_hopper_q2_medium_replay_v2.yaml"
    )
    config = mod.load_config(config_path)
    mode = mod.ModeConfig(
        max_transitions=None,
        seeds=(7, 8),
        canonical_critic_seed=7,
        critic_max_steps=4,
        critic_min_steps=2,
        critic_eval_interval=1,
        positive_max_steps=2,
        positive_min_steps=1,
        actor_eval_interval=1,
        branch_max_steps=2,
        branch_min_steps=1,
        matched_pairs=2,
        audit_sample_size=4,
        rollout_episodes=1,
        rollout_eval_interval=1,
    )
    config = replace(
        config,
        hidden_sizes=(4,),
        critic_batch_size=8,
        audit_windows=2,
        critic_relative_slope_tolerance=1e9,
        critic_gradient_tolerance=1e9,
        critic_update_tolerance=1e9,
    )
    episode_ids = np.repeat(np.arange(10), 3)
    size = len(episode_ids)
    observations = np.linspace(-1.0, 1.0, size * 3, dtype=np.float32).reshape(size, 3)
    actions = np.zeros((size, 2), dtype=np.float32)
    rewards = np.linspace(0.0, 1.0, size, dtype=np.float32)
    terminals = np.zeros(size, dtype=bool)
    terminals[2::3] = True
    data = mod.OfflineData(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=np.roll(observations, -1, axis=0),
        terminals=terminals,
        timeouts=np.zeros(size, dtype=bool),
        episode_ids=episode_ids,
    )
    dataset_file = tmp_path / "fake.hdf5"
    dataset_file.write_bytes(b"fake")
    dataset_manifest = {
        "basename": dataset_file.name,
        "sha256": mod.sha256_file(dataset_file),
        "size_bytes": dataset_file.stat().st_size,
    }
    root = tmp_path / "canonical"
    first = mod.prepare_canonical_critic_context(
        data=data,
        config=config,
        mode=mode,
        mode_name="pilot",
        config_path=config_path,
        dataset_manifest=dataset_manifest,
        device=torch.device("cpu"),
        artifact_root=root,
        reuse_root=None,
    )
    second = mod.prepare_canonical_critic_context(
        data=data,
        config=config,
        mode=mode,
        mode_name="pilot",
        config_path=config_path,
        dataset_manifest=dataset_manifest,
        device=torch.device("cpu"),
        artifact_root=root,
        reuse_root=None,
    )
    assert first.reused is False
    assert second.reused is True
    assert first.artifact_manifest["critic_training_count"] == 1
    assert second.artifact_manifest["shared_across_all_actor_seeds"] is True
    assert first.critic_audit["selected_checkpoint_role"] == "terminal_extension_checkpoint"
    assert first.critic_audit["final_stationarity_reconfirmed"] is True
    np.testing.assert_array_equal(first.advantages, second.advantages)


def test_pilot_terminal_audit_cannot_report_formal_gate_passed(monkeypatch) -> None:
    from types import SimpleNamespace

    mechanism_counts = {
        "natural_far_field_present": 1,
        "corrected_quadratic_branch_empirically_active": 1,
        "measurable_full_parameter_contribution": 1,
        "targeted_far_control_mitigates_dynamics": 1,
        "terminal_audit_records_complete": 1,
        "terminal_state_classification_complete": 1,
        "rollout_available_for_all_methods": 1,
        "all_mechanism_subchecks_passed": 1,
    }
    monkeypatch.setattr(
        mod,
        "aggregate_seed_summaries",
        lambda summaries: {
            "mechanism_subcheck_counts": mechanism_counts,
            "reporting_separation": {},
        },
    )
    method_audit = {
        "task_performance_status": "available",
        "state": "persistent_or_slow_drift",
        "terminal_audit_complete": True,
    }
    summary = {
        "positive_only_initialization": {
            "state": "persistent_or_slow_drift",
            "terminal_audit_complete": True,
            "task_performance_status": "available",
        },
        "methods": {method: dict(method_audit) for method in mod.METHODS},
    }
    canonical = SimpleNamespace(
        critic_audit={"optimization_terminal": False},
        artifact_manifest={"complete": True},
    )
    audit = mod.build_terminal_audit(
        summaries=[summary],
        mode_name="pilot",
        expected_seed_count=1,
        canonical=canonical,
        rollout_preflight={"status": "passed"},
        rollout_required=True,
    )
    assert audit["engineering_pipeline_complete"] is True
    assert audit["mechanism_subchecks_passed_for_completed_seeds"] is True
    assert audit["paired_seed_evidence_complete"] is False
    assert audit["formal_evidence_prerequisites_complete"] is False
    assert audit["formal_scientific_gate_passed"] is False
    assert audit["independent_validation_gate_all_seeds"] is False
