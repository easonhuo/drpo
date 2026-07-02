from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drpo import e7_bench  # noqa: E402
from drpo.e7_hopper_q2 import SquashedGaussianPolicy  # noqa: E402

CONFIG = ROOT / "configs" / "e7_bench_pilot.yaml"


def test_pilot_plan_uses_all_384_core_server_parallelism_without_serial_loops() -> None:
    config = e7_bench.load_bench_config(CONFIG)
    plan = e7_bench.build_execution_plan(config, "pilot")
    assert len(plan["critic_parallel_stage"]) == 2
    assert len(plan["positive_parallel_stage"]) == 8
    assert len(plan["branch_parallel_stage"]) == 40
    assert plan["critic_workers"] == 2
    assert plan["positive_workers"] == 8
    assert plan["branch_workers"] == 40
    assert plan["critic_workers"] * plan["critic_cpus_per_worker"] == 128
    assert plan["positive_workers"] * plan["positive_cpus_per_worker"] == 256
    assert plan["branch_workers"] * plan["branch_cpus_per_worker"] == 320
    assert plan["top_level_serial_seed_loop"] is False
    assert plan["top_level_serial_method_loop"] is False


def test_formal_plan_is_parallel_but_fail_closed_until_protocol_lock() -> None:
    config = e7_bench.load_bench_config(CONFIG)
    plan = e7_bench.build_execution_plan(config, "formal")
    assert plan["task_count"] == 9
    assert plan["parallel_unit"] == "task_seed_method"
    assert plan["serial_seed_loop_forbidden"] is True
    assert plan["serial_method_loop_forbidden"] is True
    assert plan["launch_allowed"] is False
    assert plan["protocol_locked"] is False


def test_frozen_method_list_and_coefficients_match_controlled_calibration() -> None:
    config = e7_bench.load_bench_config(CONFIG)
    assert e7_bench.PILOT_METHODS == (
        "positive_only",
        "signed",
        "global_alpha",
        "reciprocal_linear",
        "reciprocal_quadratic",
        "exponential",
    )
    assert config.methods.global_alpha == pytest.approx(0.75)
    assert config.methods.reference_distance == pytest.approx(5.0)
    assert config.methods.coefficients == {
        "reciprocal_linear": pytest.approx(0.4362580032734791),
        "reciprocal_quadratic": pytest.approx(0.5520268617673281),
        "exponential": pytest.approx(0.374162511054291),
    }
    assert config.methods.d4rl_retuning_allowed is False


def test_taper_weights_are_continuous_and_monotone() -> None:
    config = e7_bench.load_bench_config(CONFIG)
    d = torch.tensor([0.0, 2.5, 5.0, 10.0, 20.0])
    for method in e7_bench.TAPER_METHODS:
        weight = e7_bench.taper_weight(d, method, config.methods)
        assert weight[0].item() == pytest.approx(1.0)
        assert torch.all(weight[:-1] >= weight[1:])
        assert torch.all(weight > 0)
        assert torch.all(weight <= 1)
    assert torch.allclose(
        e7_bench.taper_weight(d, "global_alpha", config.methods),
        torch.full_like(d, 0.75),
    )


def test_benchmark_loss_only_tapers_negative_advantages() -> None:
    config = e7_bench.load_bench_config(CONFIG)
    policy = SquashedGaussianPolicy(3, 2, (8,), -5.0, 2.0, 1e-6)
    obs = torch.zeros(4, 3)
    actions = torch.tensor([[0.1, 0.2], [0.2, -0.3], [0.8, 0.5], [-0.9, 0.4]])
    advantage = torch.tensor([1.0, -1.0, 2.0, -2.0])
    loss, diag = e7_bench.benchmark_actor_loss(
        policy, obs, actions, advantage, "exponential", config.methods
    )
    assert torch.isfinite(loss)
    assert diag["active_fraction"] == pytest.approx(1.0)
    assert 0.0 < diag["negative_weight_mean"] <= 1.0


def test_minari_episode_loader_preserves_t_plus_one_observations(tmp_path: Path) -> None:
    path = tmp_path / "minari.hdf5"
    with h5py.File(path, "w") as handle:
        for episode, length in enumerate((3, 2, 4)):
            group = handle.create_group(f"episode_{episode}")
            group["observations"] = np.arange((length + 1) * 3).reshape(length + 1, 3)
            group["actions"] = np.zeros((length, 2), dtype=np.float32)
            group["rewards"] = np.arange(length, dtype=np.float32)
            group["terminations"] = np.array([False] * (length - 1) + [True])
            group["truncations"] = np.zeros(length, dtype=bool)
    data = e7_bench.load_minari_episode_hdf5(path, max_transitions=None)
    assert data.size == 9
    assert data.observations.shape == (9, 3)
    assert data.next_observations.shape == (9, 3)
    assert np.array_equal(data.next_observations[0], np.array([3, 4, 5], dtype=np.float32))
    assert len(np.unique(data.episode_ids)) == 3


def test_parallel_stage_rejects_accidental_serial_fallback(tmp_path: Path) -> None:
    jobs = [
        {"job_id": "a", "command": [sys.executable, "-c", "pass"], "log_path": tmp_path / "a.log"},
        {"job_id": "b", "command": [sys.executable, "-c", "pass"], "log_path": tmp_path / "b.log"},
    ]
    with pytest.raises(RuntimeError, match="serial execution is forbidden"):
        e7_bench.run_parallel_stage(
            jobs,
            max_workers=1,
            cpus_per_worker=1,
            stage="test",
            heartbeat_path=tmp_path / "heartbeat.json",
        )


def test_registry_marks_minari_medium_as_pilot_only_and_d4rl_expert_as_formal_eligible() -> None:
    registry = yaml.safe_load((ROOT / "experiments" / "registry.yaml").read_text())
    entry = next(row for row in registry["experiments"] if row["id"] == e7_bench.EXPERIMENT_ID)
    datasets = {row["id"]: row for row in entry["pilot_execution"]["datasets"]}
    assert datasets["hopper-medium-minari-v0"]["formal_nine_task_cell_eligible"] is False
    assert datasets["hopper-medium-expert-v2"]["formal_nine_task_cell_eligible"] is True
    assert entry["pilot_execution"]["scientific_status"] == "not_run"
    assert entry["pilot_execution"]["formal_evidence_allowed"] is False


def test_preflight_rejects_cpu_oversubscription_before_training(monkeypatch: pytest.MonkeyPatch) -> None:
    config = e7_bench.load_bench_config(CONFIG)
    monkeypatch.setattr(e7_bench.os, "cpu_count", lambda: 128)
    with pytest.raises(RuntimeError, match="refusing silent oversubscription"):
        e7_bench.preflight_pilot_runtime(config, [])


def test_worker_dataset_manifest_reuses_coordinator_hash_and_detects_file_change(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "data.hdf5"
    dataset.write_bytes(b"registered-data")
    spec = e7_bench.DatasetSpec(
        id="test",
        relative_path="data.hdf5",
        sha256=e7_bench.sha256_file(dataset),
        format="legacy_d4rl_hdf5",
        env_id="Hopper-v4",
        dataset_family="test",
        score_protocol="raw_return_only",
        reference_min_score=None,
        reference_max_score=None,
        formal_cell_eligible=False,
        provenance_note="test",
    )
    row = e7_bench.validate_dataset_file(tmp_path, spec)
    manifest = tmp_path / "DATASETS.json"
    import json

    manifest.write_text(json.dumps([row]))
    reused = e7_bench.worker_dataset_manifest(tmp_path, spec, str(manifest))
    assert reused["sha256"] == spec.sha256
    dataset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="dataset size changed"):
        e7_bench.worker_dataset_manifest(tmp_path, spec, str(manifest))
