from __future__ import annotations

import sys
import csv
import dataclasses
import json
import time
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
CONFIG9 = ROOT / "configs" / "e7_bench_9task_narrow.yaml"


def test_pilot_plan_uses_all_384_core_server_parallelism_without_serial_loops() -> None:
    config = e7_bench.load_bench_config(CONFIG)
    plan = e7_bench.build_execution_plan(config, "pilot")
    assert len(plan["critic_parallel_stage"]) == 2
    assert len(plan["warmstart_parallel_stage"]) == 0
    assert plan["actor_initialization"] == "direct_from_seed_no_positive_only_warmstart"
    assert len(plan["branch_parallel_stage"]) == 92
    assert sum(row["method"] == "positive_only" for row in plan["branch_parallel_stage"]) == 4
    assert plan["critic_workers"] == 2
    assert plan["warmstart_workers"] == 0
    assert plan["branch_workers"] == 92
    assert plan["critic_workers"] * plan["critic_cpus_per_worker"] == 128
    assert plan["warmstart_workers"] * plan["warmstart_cpus_per_worker"] == 0
    assert plan["branch_workers"] * plan["branch_cpus_per_worker"] == 368
    assert plan["shared_positive_warmstart_steps"] == 0
    assert plan["method_continuation_steps"] == 500000
    assert plan["total_actor_steps_per_method"] == 500000
    assert plan["top_level_serial_seed_loop"] is False
    assert plan["top_level_serial_method_loop"] is False


def test_pilot_budget_uses_direct_500k_actor_training_without_warmstart() -> None:
    config = e7_bench.load_bench_config(CONFIG)
    assert config.budget.critic_steps == 100000
    assert config.budget.shared_positive_warmstart_steps == 0
    assert config.budget.method_continuation_steps == 500000
    assert config.budget.total_actor_steps_per_method == 500000
    assert config.budget.total_actor_steps_per_method == config.budget.method_continuation_steps


def test_formal_plan_is_parallel_but_fail_closed_until_protocol_lock() -> None:
    config = e7_bench.load_bench_config(CONFIG)
    plan = e7_bench.build_execution_plan(config, "formal")
    assert plan["task_count"] == 9
    assert plan["parallel_unit"] == "task_seed_method"
    assert plan["serial_seed_loop_forbidden"] is True
    assert plan["serial_method_loop_forbidden"] is True
    assert plan["launch_allowed"] is False
    assert plan["protocol_locked"] is False


def test_parameter_sweep_keeps_families_but_expands_method_variants() -> None:
    config = e7_bench.load_bench_config(CONFIG)
    assert e7_bench.PILOT_METHOD_FAMILIES == (
        "positive_only",
        "signed",
        "global_alpha",
        "reciprocal_linear",
        "reciprocal_quadratic",
        "exponential",
    )
    assert config.budget.seeds == (200, 201)
    assert config.methods.pilot_parameter_search_enabled is True
    assert config.methods.per_task_retuning_allowed is False
    assert config.methods.d4rl_retuning_allowed is False
    assert len(config.methods.variants) == 23
    families = [variant.family for variant in config.methods.variants]
    assert families.count("positive_only") == 1
    assert families.count("signed") == 1
    assert families.count("global_alpha") == 6
    assert families.count("reciprocal_linear") == 5
    assert families.count("reciprocal_quadratic") == 5
    assert families.count("exponential") == 5
    assert "global_alpha_a0p001" in config.methods.ids
    assert "global_alpha_a0p02" in config.methods.ids
    assert "global_alpha_a0p03" in config.methods.ids
    assert "global_alpha_a0p05" not in config.methods.ids
    assert "global_alpha_a0p75" not in config.methods.ids
    assert "reciprocal_linear_c80p00" in config.methods.ids
    assert "reciprocal_quadratic_c64p00" in config.methods.ids
    assert "exponential_c14p00" in config.methods.ids
    assert "exponential_c24p00" in config.methods.ids
    assert "exponential_c0p374163" not in config.methods.ids
    assert config.methods.global_alpha == pytest.approx(0.75)
    assert config.methods.reference_distance == pytest.approx(5.0)
    assert config.methods.coefficients == {
        "reciprocal_linear": pytest.approx(0.4362580032734791),
        "reciprocal_quadratic": pytest.approx(0.5520268617673281),
        "exponential": pytest.approx(0.374162511054291),
    }


def test_recovered_network_profile_does_not_change_method_weights() -> None:
    config = e7_bench.load_bench_config(CONFIG)
    assert config.network_profile.hidden_sizes == (256, 256)
    assert config.network_profile.activation == "relu"
    assert config.network_profile.init_scheme == "orthogonal"
    assert config.network_profile.init_gain == pytest.approx(np.sqrt(2))
    assert config.network_profile.log_std_mode == "independent_global_diagonal"
    assert config.network_profile.log_std_min == pytest.approx(-5.0)
    assert config.network_profile.log_std_max == pytest.approx(2.0)
    assert config.methods.global_alpha == pytest.approx(0.75)
    assert not hasattr(config.methods, "canonical_negative_alpha")
    assert config.methods.pilot_parameter_search_enabled is True


def test_branch_worker_validates_method_without_requiring_warmstart_dir() -> None:
    parser = e7_bench.build_parser()
    branch_args = parser.parse_args(
        [
            "branch-worker",
            "--config",
            str(CONFIG),
            "--dataset-root",
            "/tmp/datasets",
            "--validated-datasets-manifest",
            "/tmp/manifest.json",
            "--dataset-id",
            "hopper-medium-expert-v2",
            "--seed",
            "200",
            "--method",
            "exponential_c14p00",
            "--critic-dir",
            "/tmp/critic",
            "--output-dir",
            "/tmp/branch",
            "--device",
            "cpu",
            "--cpus-per-worker",
            "1",
            "--expected-worker-identity-sha256",
            "0" * 64,
        ]
    )
    assert branch_args.method == "exponential_c14p00"
    assert not hasattr(branch_args, "warmstart_dir")


def test_taper_weights_are_continuous_and_monotone() -> None:
    config = e7_bench.load_bench_config(CONFIG)
    d = torch.tensor([0.0, 2.5, 5.0, 10.0, 20.0])
    for method in ("reciprocal_linear_c40p00", "reciprocal_quadratic_c32p00", "exponential_c14p00"):
        weight = e7_bench.taper_weight(d, method, config.methods)
        assert weight[0].item() == pytest.approx(1.0)
        assert torch.all(weight[:-1] >= weight[1:])
        assert torch.all(weight > 0)
        assert torch.all(weight <= 1)
    assert torch.allclose(
        e7_bench.taper_weight(d, "global_alpha_a0p02", config.methods),
        torch.full_like(d, 0.02),
    )


def test_benchmark_loss_only_tapers_negative_advantages() -> None:
    config = e7_bench.load_bench_config(CONFIG)
    policy = SquashedGaussianPolicy(3, 2, (8,), -5.0, 2.0, 1e-6)
    obs = torch.zeros(4, 3)
    actions = torch.tensor([[0.1, 0.2], [0.2, -0.3], [0.8, 0.5], [-0.9, 0.4]])
    advantage = torch.tensor([1.0, -1.0, 2.0, -2.0])
    loss, diag = e7_bench.benchmark_actor_loss(
        policy, obs, actions, advantage, "exponential_c14p00", config.methods
    )
    assert torch.isfinite(loss)
    assert diag["active_fraction"] == pytest.approx(1.0)
    assert 0.0 < diag["negative_weight_mean"] <= 1.0


def test_pilot_uses_d4rl_medium_replay_not_minari_medium() -> None:
    config = e7_bench.load_bench_config(CONFIG)
    dataset_ids = tuple(spec.id for spec in config.datasets)
    assert dataset_ids == ("hopper-medium-replay-v2", "hopper-medium-expert-v2")
    replay = config.datasets[0]
    assert replay.format == "legacy_d4rl_hdf5"
    assert replay.env_id == "Hopper-v4"
    assert replay.sha256 == "e121c5f7c9857a307baa9edc6a2c3b48e85fedb9ac316ecddd0f48ca7ef4e39b"

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


def test_parallel_stage_terminates_peer_workers_after_failure(tmp_path: Path) -> None:
    jobs = [
        {
            "job_id": "fail-fast",
            "command": [sys.executable, "-c", "raise SystemExit(7)"],
            "log_path": tmp_path / "fail.log",
        },
        {
            "job_id": "long-peer",
            "command": [sys.executable, "-c", "import time; time.sleep(30)"],
            "log_path": tmp_path / "peer.log",
        },
    ]
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="peer workers were terminated"):
        e7_bench.run_parallel_stage(
            jobs,
            max_workers=2,
            cpus_per_worker=1,
            stage="test-fail-fast",
            heartbeat_path=tmp_path / "heartbeat.json",
        )
    assert time.monotonic() - started < 10.0
    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text())
    assert heartbeat["state"] == "failed"
    assert heartbeat["failed_job_id"] == "fail-fast"


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


def test_worker_identity_binds_budget_and_taper_parameters() -> None:
    config = e7_bench.load_bench_config(CONFIG)
    spec = config.datasets[0]
    original = e7_bench.worker_identity_sha256(
        config,
        spec,
        worker="branch",
        seed=200,
        method="exponential_c14p00",
    )
    shorter_budget = dataclasses.replace(
        config,
        budget=dataclasses.replace(config.budget, method_continuation_steps=40000),
    )
    assert e7_bench.worker_identity_sha256(
        shorter_budget,
        spec,
        worker="branch",
        seed=200,
        method="exponential_c14p00",
    ) != original
    changed_variants = tuple(
        dataclasses.replace(variant, coefficient=0.5)
        if variant.id == "exponential_c14p00"
        else variant
        for variant in config.methods.variants
    )
    changed_taper = dataclasses.replace(
        config,
        methods=dataclasses.replace(config.methods, variants=changed_variants),
    )
    assert e7_bench.worker_identity_sha256(
        changed_taper,
        spec,
        worker="branch",
        seed=200,
        method="exponential_c14p00",
    ) != original


def test_nonfinite_training_failure_overrides_last_finite_audit_classification() -> None:
    terminal = {
        "state": "persistent_or_slow_drift",
        "numerical_nonfinite": False,
        "fixed_budget_completed": False,
        "explicit_terminal_classification": True,
    }
    result = e7_bench._apply_training_failure_to_terminal(
        terminal, "nonfinite_train_gradient"
    )
    assert result["state"] == "nan_inf_numerical_collapse"
    assert result["numerical_nonfinite"] is True
    assert result["fixed_budget_completed"] is False


def test_no_training_failure_preserves_terminal_classification() -> None:
    terminal = {
        "state": "fixed_horizon_inconclusive",
        "numerical_nonfinite": False,
        "fixed_budget_completed": True,
    }
    assert e7_bench._apply_training_failure_to_terminal(terminal, None) is terminal


def test_run_identity_rejects_stale_short_budget_workdir(tmp_path: Path) -> None:
    config = e7_bench.load_bench_config(CONFIG)
    work_dir = tmp_path / "run"
    work_dir.mkdir()
    identity = e7_bench.ensure_run_identity(work_dir, config, resume=False)
    assert identity == e7_bench.run_identity_sha256(config)
    stale = dataclasses.replace(
        config,
        budget=dataclasses.replace(
            config.budget,
            critic_steps=20000,
            shared_positive_warmstart_steps=0,
            method_continuation_steps=40000,
            total_actor_steps_per_method=40000,
        ),
    )
    with pytest.raises(RuntimeError, match="different E7-BENCH protocol identity"):
        e7_bench.ensure_run_identity(work_dir, stale, resume=True)


def test_worker_complete_requires_exact_identity(tmp_path: Path) -> None:
    path = tmp_path / "worker"
    path.mkdir()
    (path / "WORKER_COMPLETE.json").write_text(
        json.dumps(
            {
                "experiment_id": e7_bench.EXPERIMENT_ID,
                "dataset": {"id": "dataset"},
                "seed": 200,
                "method": "positive_only",
                "worker": "branch",
                "worker_identity_sha256": "new-budget",
            }
        )
    )
    assert e7_bench._worker_complete(
        path,
        dataset_id="dataset",
        seed=200,
        method="positive_only",
        worker="branch",
        expected_worker_identity_sha256="new-budget",
    )
    assert not e7_bench._worker_complete(
        path,
        dataset_id="dataset",
        seed=200,
        method="positive_only",
        worker="branch",
        expected_worker_identity_sha256="old-short-budget",
    )


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
    manifest.write_text(json.dumps([row]))
    reused = e7_bench.worker_dataset_manifest(tmp_path, spec, str(manifest))
    assert reused["sha256"] == spec.sha256
    dataset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="dataset size changed"):
        e7_bench.worker_dataset_manifest(tmp_path, spec, str(manifest))


def test_prepare_worker_output_archives_incomplete_resume_directory(tmp_path: Path) -> None:
    output = tmp_path / "branches" / "dataset" / "seed_200" / "signed"
    output.mkdir(parents=True)
    (output / "partial.txt").write_text("partial")

    skipped = e7_bench.prepare_worker_output(
        output,
        resume=True,
        dataset_id="dataset",
        seed=200,
        method="signed",
        worker="branch",
        expected_worker_identity_sha256="new-identity",
    )

    assert skipped is False
    assert not output.exists()
    archives = list((output.parent / "_stale_worker_outputs").iterdir())
    assert len(archives) == 1
    assert (archives[0] / "partial.txt").read_text() == "partial"


def test_aggregate_pilot_rejects_stale_branch_identity(tmp_path: Path) -> None:
    config = e7_bench.load_bench_config(CONFIG)
    for spec in config.datasets:
        for seed in config.budget.seeds:
            for method in config.methods.ids:
                branch_dir = (
                    tmp_path / "branches" / spec.id / f"seed_{seed}" / method
                )
                branch_dir.mkdir(parents=True)
                identity = e7_bench.worker_identity_sha256(
                    config,
                    spec,
                    worker="branch",
                    seed=seed,
                    method=method,
                )
                marker = {
                    "experiment_id": e7_bench.EXPERIMENT_ID,
                    "dataset": {"id": spec.id},
                    "seed": seed,
                    "method": method,
                    "worker": "branch",
                    "worker_identity_sha256": identity,
                }
                (branch_dir / "WORKER_COMPLETE.json").write_text(
                    json.dumps(marker)
                )
                (branch_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "dataset_id": spec.id,
                            "seed": seed,
                            "method": method,
                            "scheduled_total_actor_steps": (
                                config.budget.total_actor_steps_per_method
                            ),
                            "executed_total_actor_steps": (
                                config.budget.total_actor_steps_per_method
                            ),
                            "fixed_budget_completed": True,
                            "numerical_nonfinite": False,
                        }
                    )
                )

    stale = (
        tmp_path
        / "branches"
        / config.datasets[0].id
        / f"seed_{config.budget.seeds[0]}"
        / config.methods.ids[0]
        / "WORKER_COMPLETE.json"
    )
    payload = json.loads(stale.read_text())
    payload["worker_identity_sha256"] = "stale-short-budget"
    stale.write_text(json.dumps(payload))

    # Liveness-resilient aggregate: a stale-identity branch is recorded in the
    # failure inventory and PILOT_COMPLETE.json reports branch_count_failed,
    # instead of aborting the whole aggregate (spec §3.1 tolerant aggregation).
    result = e7_bench.aggregate_pilot(tmp_path, config)
    total_branches = (
        len(config.datasets) * len(config.budget.seeds) * len(config.methods.ids)
    )
    assert result["branch_count_total"] == total_branches
    assert result["branch_count_failed"] >= 1
    assert result["failure_inventory_csv"] == "failure_inventory.csv"
    inv_path = tmp_path / "failure_inventory.csv"
    assert inv_path.is_file()
    with inv_path.open(newline="") as handle:
        inventory = list(csv.DictReader(handle))
    stale_rows = [
        r for r in inventory
        if r["dataset_id"] == config.datasets[0].id
        and str(r["seed"]) == str(config.budget.seeds[0])
        and r["method"] == config.methods.ids[0]
    ]
    assert stale_rows, "stale-identity branch must appear in failure_inventory.csv"
    assert stale_rows[0]["failure_class"] == "crashed_or_incomplete"


def test_aggregate_allows_only_nan_inf_as_early_stop_exception(tmp_path: Path) -> None:
    config = e7_bench.load_bench_config(CONFIG)
    for spec in config.datasets:
        for seed in config.budget.seeds:
            for method in config.methods.ids:
                branch_dir = tmp_path / "branches" / spec.id / f"seed_{seed}" / method
                branch_dir.mkdir(parents=True)
                identity = e7_bench.worker_identity_sha256(
                    config, spec, worker="branch", seed=seed, method=method
                )
                (branch_dir / "WORKER_COMPLETE.json").write_text(
                    json.dumps(
                        {
                            "experiment_id": e7_bench.EXPERIMENT_ID,
                            "dataset": {"id": spec.id},
                            "seed": seed,
                            "method": method,
                            "worker": "branch",
                            "worker_identity_sha256": identity,
                        }
                    )
                )
                (branch_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "dataset_id": spec.id,
                            "seed": seed,
                            "method": method,
                            "scheduled_total_actor_steps": 500000,
                            "executed_total_actor_steps": 200000,
                            "fixed_budget_completed": False,
                            "numerical_nonfinite": True,
                        }
                    )
                )
    total_branches = (
        len(config.datasets) * len(config.budget.seeds) * len(config.methods.ids)
    )
    payload = e7_bench.aggregate_pilot(tmp_path, config)
    # NaN/Inf early-stop is the only registered early-termination exception:
    # such branches are kept in the summary rows AND recorded in the failure
    # inventory, with the three event types counted separately.
    assert payload["early_termination_exception"] == "nan_inf_numerical_failure_only"
    assert payload["branch_count_total"] == total_branches
    assert payload["event_counts"]["numerical_nonfinite"] == total_branches
    assert payload["branch_count_failed"] == total_branches
    assert payload["failure_inventory_csv"] == "failure_inventory.csv"

    bad = (
        tmp_path
        / "branches"
        / config.datasets[0].id
        / f"seed_{config.budget.seeds[0]}"
        / config.methods.ids[0]
        / "summary.json"
    )
    row = json.loads(bad.read_text())
    row["numerical_nonfinite"] = False
    bad.write_text(json.dumps(row))
    # A branch that stopped before the fixed budget WITHOUT a registered NaN/Inf
    # exception is no longer a valid early stop: the tolerant aggregate does not
    # raise, but flags it in the failure inventory as a distinct failure class and
    # drops it from the numerical_nonfinite event count (terminology discipline:
    # the three event types are reported separately).
    result = e7_bench.aggregate_pilot(tmp_path, config)
    assert result["event_counts"]["numerical_nonfinite"] == total_branches - 1
    assert result["branch_count_failed"] == total_branches
    with (tmp_path / "failure_inventory.csv").open(newline="") as handle:
        inventory = list(csv.DictReader(handle))
    bad_rows = [
        r for r in inventory
        if r["dataset_id"] == config.datasets[0].id
        and str(r["seed"]) == str(config.budget.seeds[0])
        and r["method"] == config.methods.ids[0]
    ]
    assert bad_rows, "non-naninf early-stop branch must appear in failure_inventory.csv"
    assert bad_rows[0]["failure_class"] == "stopped_before_budget_no_naninf"


def _write_fake_critic_artifact(
    root: Path,
    *,
    config: e7_bench.BenchConfig,
    spec: e7_bench.DatasetSpec,
) -> str:
    identity = e7_bench.worker_identity_sha256(config, spec, worker="critic")
    (root / "critic").mkdir(parents=True)
    (root / "frozen_advantage").mkdir(parents=True)
    np.savez_compressed(root / "episode_split.npz", train=np.array([0]), validation=np.array([1]), test=np.array([2]))
    np.savez_compressed(
        root / "normalizers.npz",
        observation_mean=np.zeros(1, dtype=np.float32),
        observation_std=np.ones(1, dtype=np.float32),
        target_mean=np.zeros(1, dtype=np.float32),
        target_std=np.ones(1, dtype=np.float32),
    )
    (root / "critic" / "canonical_critic.pt").write_bytes(b"checkpoint")
    (root / "critic" / "critic_terminal_audit.json").write_text("{}")
    (root / "critic" / "critic_metrics.csv").write_text("step,loss\n")
    np.savez_compressed(
        root / "frozen_advantage" / "frozen_advantages.npz",
        advantage=np.array([1.0, -1.0], dtype=np.float32),
    )
    (root / "frozen_advantage" / "advantage_manifest.json").write_text("{}")
    (root / "WORKER_COMPLETE.json").write_text(
        json.dumps(
            {
                "experiment_id": e7_bench.EXPERIMENT_ID,
                "dataset": {"id": spec.id},
                "worker": "critic",
                "worker_identity_sha256": identity,
            }
        )
    )
    return identity


def test_critic_cache_is_enabled_and_uses_actor_seed_independent_identity() -> None:
    config = e7_bench.load_bench_config(CONFIG)
    spec = config.datasets[0]
    assert config.critic_cache.enabled is True
    assert config.critic_cache.reuse is True
    assert config.critic_cache.store is True
    assert config.critic_cache.actor_seed_independent is True
    assert config.critic_cache.sweep_config_independent is True

    critic_identity = e7_bench.worker_identity_sha256(config, spec, worker="critic")
    changed_actor_seeds = dataclasses.replace(
        config,
        budget=dataclasses.replace(config.budget, seeds=(200, 201, 202, 203)),
    )
    assert e7_bench.worker_identity_sha256(changed_actor_seeds, spec, worker="critic") == critic_identity

    changed_variants = tuple(
        dataclasses.replace(variant, coefficient=22.0)
        if variant.id == "exponential_c14p00"
        else variant
        for variant in config.methods.variants
    )
    changed_sweep = dataclasses.replace(
        config,
        methods=dataclasses.replace(config.methods, variants=changed_variants),
    )
    assert e7_bench.worker_identity_sha256(changed_sweep, spec, worker="critic") == critic_identity
    assert e7_bench.worker_identity_sha256(
        changed_sweep,
        spec,
        worker="branch",
        seed=200,
        method="exponential_c14p00",
    ) != e7_bench.worker_identity_sha256(
        config,
        spec,
        worker="branch",
        seed=200,
        method="exponential_c14p00",
    )

    changed_critic_seed = dataclasses.replace(
        config,
        budget=dataclasses.replace(config.budget, canonical_critic_seed=1000),
    )
    assert e7_bench.worker_identity_sha256(changed_critic_seed, spec, worker="critic") != critic_identity


def test_critic_cache_restore_and_store_copy_artifacts_without_symlinks(tmp_path: Path) -> None:
    config = e7_bench.load_bench_config(CONFIG)
    spec = config.datasets[0]
    identity = e7_bench.worker_identity_sha256(config, spec, worker="critic")
    cache_root = tmp_path / "cache"
    entry = e7_bench.critic_cache_entry_dir(cache_root, identity)
    _write_fake_critic_artifact(entry, config=config, spec=spec)

    restored = tmp_path / "work" / "critics" / spec.id
    assert e7_bench.restore_critic_from_cache(
        config=config,
        spec=spec,
        output_dir=restored,
        cache_root=cache_root,
        expected_worker_identity_sha256=identity,
    )
    assert (restored / "CRITIC_CACHE_USED.json").is_file()
    assert e7_bench._worker_complete(
        restored,
        dataset_id=spec.id,
        worker="critic",
        expected_worker_identity_sha256=identity,
    )

    new_cache_root = tmp_path / "new_cache"
    assert e7_bench.store_critic_in_cache(
        config=config,
        spec=spec,
        source_dir=restored,
        cache_root=new_cache_root,
        expected_worker_identity_sha256=identity,
    )
    new_entry = e7_bench.critic_cache_entry_dir(new_cache_root, identity)
    assert (new_entry / "CRITIC_CACHE_STORED.json").is_file()
    assert e7_bench._worker_complete(
        new_entry,
        dataset_id=spec.id,
        worker="critic",
        expected_worker_identity_sha256=identity,
    )


# --- 9-task narrow-grid pilot (spec §7) ---------------------------------------

NINE_TASK_ENV_IDS = {
    "hopper-medium-v2", "hopper-medium-replay-v2", "hopper-medium-expert-v2",
    "walker2d-medium-v2", "walker2d-medium-replay-v2", "walker2d-medium-expert-v2",
    "halfcheetah-medium-v2", "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
}


def test_9task_narrow_grid_config_loads_with_frozen_scientific_values() -> None:
    config = e7_bench.load_bench_config(CONFIG9)
    # Nine D4RL v2 locomotion medium cells, two development seeds, 12 variants
    # => 216 task-seed-method branches (spec §3.2/§3.3).
    assert len(config.datasets) == 9
    assert {spec.id for spec in config.datasets} == NINE_TASK_ENV_IDS
    assert tuple(config.budget.seeds) == (200, 201)
    assert len(config.methods.variants) == 12
    branch_jobs = len(config.datasets) * len(config.budget.seeds) * len(config.methods.variants)
    assert branch_jobs == 216
    assert config.parallel.branch_workers >= branch_jobs
    # Frozen budget guards (only the two authorized guard edits vs legacy pilot).
    assert config.budget.critic_steps == 100000
    assert config.budget.shared_positive_warmstart_steps == 0
    assert config.budget.method_continuation_steps == 300000
    assert config.budget.total_actor_steps_per_method == 300000
    # Direct-from-seed protocol: no shared warm-start workers scheduled.
    assert config.parallel.warmstart_workers == 0
    # Network profile unchanged (spec §3.2 forbidden-to-modify).
    assert config.network_profile.hidden_sizes == (256, 256)
    assert config.network_profile.activation.lower() == "relu"
    assert config.network_profile.init_scheme.lower() == "orthogonal"
    assert config.network_profile.log_std_mode == "independent_global_diagonal"
    assert config.network_profile.log_std_min == -5.0
    assert config.network_profile.log_std_max == 2.0
    # Method families: 12 variants span exactly the six registered families.
    families = {v.family for v in config.methods.variants}
    assert families == set(e7_bench.PILOT_METHOD_FAMILIES)
    assert set(config.methods.coefficients) == set(e7_bench.TAPER_METHODS)
    # Gate unset at rest => full Gate C pilot.
    assert config.gate is None
    # Per-dataset reference scores present (D4RL v2 percent protocol).
    for spec in config.datasets:
        assert spec.score_protocol == "d4rl_v2_percent"
        assert spec.reference_min_score is not None
        assert spec.reference_max_score is not None
        assert spec.formal_cell_eligible is True


def test_9task_config_sha_and_base_config_bound() -> None:
    config = e7_bench.load_bench_config(CONFIG9)
    assert config.config_sha256
    assert config.base_config_sha256
    # config_sha256 is the sha256 of the 9-task config file on disk.
    import hashlib
    raw = CONFIG9.read_bytes()
    assert config.config_sha256 == hashlib.sha256(raw).hexdigest()


def test_liveness_gate_a_subsets_to_one_branch(tmp_path: Path) -> None:
    config = e7_bench.load_bench_config(CONFIG9)
    gated = e7_bench.apply_liveness_gate(config, "A")
    assert gated.gate == "A"
    assert [d.id for d in gated.datasets] == ["hopper-medium-replay-v2"]
    assert tuple(gated.budget.seeds) == (200,)
    assert [v.id for v in gated.methods.variants] == ["positive_only"]
    assert gated.budget.method_continuation_steps == 5000
    assert gated.budget.total_actor_steps_per_method == 5000
    # 1 dataset x 1 seed x 1 method = 1 branch.
    assert gated.parallel.branch_workers == 1
    # Frozen scientific guards survive the gate subsetting.
    assert gated.network_profile.hidden_sizes == (256, 256)
    assert gated.budget.critic_steps == 100000
    assert gated.budget.shared_positive_warmstart_steps == 0
    # Gate validation passes (no raise).
    e7_bench._validate_gate_config(gated, "A")


def test_liveness_gate_b_subsets_to_eight_branches() -> None:
    config = e7_bench.load_bench_config(CONFIG9)
    gated = e7_bench.apply_liveness_gate(config, "B")
    assert gated.gate == "B"
    assert {d.id for d in gated.datasets} == {
        "hopper-medium-replay-v2", "hopper-medium-expert-v2"
    }
    assert tuple(gated.budget.seeds) == (200, 201)
    assert {v.id for v in gated.methods.variants} == {"positive_only", "signed"}
    assert gated.budget.method_continuation_steps == 20000
    assert gated.budget.total_actor_steps_per_method == 20000
    # 2 datasets x 2 seeds x 2 methods = 8 branches.
    assert gated.parallel.branch_workers == 8
    e7_bench._validate_gate_config(gated, "B")


def test_write_gate_resolved_config_trains_gate_budget(tmp_path: Path) -> None:
    dst = tmp_path / "resolved_config.yaml"
    e7_bench._write_gate_resolved_config(CONFIG9, dst, "A")
    resolved = e7_bench.load_bench_config(dst)
    assert resolved.gate == "A"
    assert len(resolved.datasets) == 1
    assert tuple(resolved.budget.seeds) == (200,)
    assert len(resolved.methods.variants) == 1
    assert resolved.budget.method_continuation_steps == 5000
    # Frozen scientific fields copied verbatim.
    assert resolved.network_profile.hidden_sizes == (256, 256)
    assert set(resolved.methods.coefficients) == set(e7_bench.TAPER_METHODS)


def test_worker_status_writes_valid_json(tmp_path: Path) -> None:
    status = e7_bench.WorkerStatus(
        tmp_path,
        run_id="abc123",
        pid=42,
        dataset_id="hopper-medium-replay-v2",
        seed=200,
        method_id="positive_only",
    )
    path = tmp_path / "worker_status.json"
    assert path.is_file()
    payload = json.loads(path.read_text())
    assert payload["run_id"] == "abc123"
    assert payload["pid"] == 42
    assert payload["dataset_id"] == "hopper-medium-replay-v2"
    assert payload["seed"] == 200
    assert payload["method_id"] == "positive_only"
    assert payload["state"] == "running"
    assert payload["phase"] == "entered"
    assert payload["current_step"] == 0
    assert payload["nonfinite_detected"] is False
    assert payload["last_update_time_utc"]
    # Update is reflected and the file stays valid JSON.
    status.update(phase="training", current_step=5000, loss=0.25, grad_norm=1.5)
    payload = json.loads(path.read_text())
    assert payload["phase"] == "training"
    assert payload["current_step"] == 5000
    assert payload["loss"] == 0.25
    assert payload["grad_norm"] == 1.5


def test_worker_started_marker_is_valid_json(tmp_path: Path) -> None:
    e7_bench.write_worker_started(
        tmp_path,
        run_id="runid-xyz",
        pid=99,
        dataset_id="walker2d-medium-v2",
        seed=201,
        method_id="exponential_c18",
        commit="236e9d7ecb14734c3b37c9b93b85ea056b800ee3",
        config_path="configs/e7_bench_9task_narrow.yaml",
    )
    payload = json.loads((tmp_path / "WORKER_STARTED.json").read_text())
    assert payload["run_id"] == "runid-xyz"
    assert payload["pid"] == 99
    assert payload["dataset_id"] == "walker2d-medium-v2"
    assert payload["seed"] == 201
    assert payload["method_id"] == "exponential_c18"
    assert payload["commit"] == "236e9d7ecb14734c3b37c9b93b85ea056b800ee3"
    assert payload["config_path"] == "configs/e7_bench_9task_narrow.yaml"
    assert payload["start_time_utc"]


def test_write_csv_atomic_is_aggregator_compatible(tmp_path: Path) -> None:
    rows = [
        {"step": 0, "loss": 1.0, "rollout_return": None},
        {"step": 5000, "loss": 0.5, "rollout_return": 12.3},
    ]
    path = tmp_path / "metrics.csv"
    e7_bench.write_csv_atomic(path, rows)
    assert not (tmp_path / "metrics.csv.tmp").exists()
    with path.open(newline="") as handle:
        read_back = list(csv.DictReader(handle))
    assert read_back == [
        {"step": "0", "loss": "1.0", "rollout_return": ""},
        {"step": "5000", "loss": "0.5", "rollout_return": "12.3"},
    ]


def test_branch_liveness_snapshot_classifies_states(tmp_path: Path) -> None:
    config = e7_bench.apply_liveness_gate(
        e7_bench.load_bench_config(CONFIG9), "B"
    )
    branches = [
        (config.datasets[0].id, config.budget.seeds[0], "positive_only"),
        (config.datasets[0].id, config.budget.seeds[1], "positive_only"),
        (config.datasets[1].id, config.budget.seeds[0], "signed"),
        (config.datasets[1].id, config.budget.seeds[1], "signed"),
    ]
    for ds, seed, method in branches:
        bdir = tmp_path / "branches" / ds / f"seed_{seed}" / method
        bdir.mkdir(parents=True)
    # branch 0: completed (WORKER_COMPLETE present, status not failed)
    b0 = tmp_path / "branches" / branches[0][0] / f"seed_{branches[0][1]}" / branches[0][2]
    (b0 / "worker_status.json").write_text(json.dumps({"state": "completed", "current_step": 20000}))
    (b0 / "WORKER_COMPLETE.json").write_text(json.dumps({"worker": "branch"}))
    # branch 1: failed
    b1 = tmp_path / "branches" / branches[1][0] / f"seed_{branches[1][1]}" / branches[1][2]
    (b1 / "worker_status.json").write_text(json.dumps({"state": "failed", "current_step": 3000}))
    (b1 / "WORKER_COMPLETE.json").write_text(json.dumps({"worker": "branch"}))
    # branch 2: running, fresh
    b2 = tmp_path / "branches" / branches[2][0] / f"seed_{branches[2][1]}" / branches[2][2]
    (b2 / "worker_status.json").write_text(
        json.dumps({"state": "running", "current_step": 12000,
                    "last_update_time_utc": e7_bench.utc_now()})
    )
    # branch 3: stale (last update long ago)
    from datetime import datetime, timezone, timedelta
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    b3 = tmp_path / "branches" / branches[3][0] / f"seed_{branches[3][1]}" / branches[3][2]
    (b3 / "worker_status.json").write_text(
        json.dumps({"state": "running", "current_step": 8000,
                    "last_update_time_utc": old})
    )
    snap = e7_bench.branch_liveness_snapshot(tmp_path, config)
    # Gate B = 2 datasets x 2 seeds x 2 methods = 8 branches total; the four
    # branches above are classified, the other four are not-yet-started (implicit
    # in total - running - completed - failed - stale).
    assert snap["total"] == 8
    assert snap["completed"] == 1
    assert snap["failed"] == 1
    assert snap["stale"] == 1
    assert snap["running"] == 1
    assert snap["min_running_step"] == 12000
    assert snap["max_running_step"] == 12000
