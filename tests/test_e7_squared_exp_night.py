from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from drpo import e7_squared_exp_night as night
from drpo import e7_squared_exp_night_runtime_autotune as runtime
from drpo.e7_canonical_injection import NegativeControl, build_injected_agent_class
from drpo.e7_squared_exp_night_aggregate import _validate_td_gae_pair
from drpo.e7_squared_exp_night_bootstrap import (
    TrajectorySnapshotAdvantage,
    _validate_ordered_hdf5,
    compute_snapshot_tables,
)


GRID = Path("configs/e7_squared_exp_night_v1.json")
GAE_GRID = Path("configs/e7_sqexp_gae_v2.json")


@pytest.fixture(autouse=True)
def _restore_historical_profile():
    night.configure_execution(GRID)
    yield
    night.configure_execution(GRID)


def _run_spec(*, gae: bool = False) -> dict[str, object]:
    digest = "0" * 64
    return {
        "datasets": [
            {
                "id": dataset,
                "path": f"/tmp/{dataset}.hdf5",
                "sha256": digest,
            }
            for dataset in night.EXPECTED_DATASETS
        ],
        "seeds": list(night.GAE_EXPECTED_SEEDS if gae else night.EXPECTED_SEEDS),
    }


def test_grid_freezes_squared_kernel_and_blocks_historical_gae() -> None:
    grid, digest = night.load_grid(GRID)
    assert len(digest) == 64
    assert grid["steps"] == 1_000_000
    assert grid["weight_control"]["formula"] == "w(d)=w(0)*exp(-c*(d/2)^2)"
    assert grid["weight_control"]["exp_coefficients"] == [
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
    ]
    stage_c = {stage["id"]: stage for stage in grid["stages"]}["stage_c_gae"]
    assert stage_c["enabled"] is False
    assert stage_c["gae_lambda"] == 0.95
    assert stage_c["status"] == "blocked_pending_verified_trajectory_contract"


def test_build_branches_preserves_exact_historical_126_matrix() -> None:
    grid, _ = night.load_grid(GRID)
    contract = SimpleNamespace(expected_canonical_alpha=0.11)
    branches = night.build_branches(contract, _run_spec(), grid)
    assert len(branches) == 126
    assert len({branch.branch_id for branch in branches}) == 126
    assert {branch.dataset.id for branch in branches} == set(night.EXPECTED_DATASETS)
    assert {branch.seed for branch in branches} == set(night.EXPECTED_SEEDS)
    assert {
        branch.template_values["actor_update_mode"] for branch in branches
    } == set(night.EXPECTED_ACTOR_MODES)
    assert {branch.template_values["stage"] for branch in branches} == {
        "stage_a",
        "stage_b",
    }
    assert all(branch.template_values["steps"] == "1000000" for branch in branches)
    assert all(seed not in night.HELD_OUT_SEEDS for seed in {b.seed for b in branches})


def test_each_historical_actor_mode_has_anchor_plus_six_coefficients() -> None:
    grid, _ = night.load_grid(GRID)
    contract = SimpleNamespace(expected_canonical_alpha=0.11)
    branches = night.build_branches(contract, _run_spec(), grid)
    for dataset in night.EXPECTED_DATASETS:
        for seed in night.EXPECTED_SEEDS:
            for actor_mode in night.EXPECTED_ACTOR_MODES:
                selected = [
                    branch
                    for branch in branches
                    if branch.dataset.id == dataset
                    and branch.seed == seed
                    and branch.template_values["actor_update_mode"] == actor_mode
                ]
                assert len(selected) == 7
                points = {
                    (
                        branch.template_values["weight_method"],
                        float(branch.template_values["exp_coefficient"]),
                    )
                    for branch in selected
                }
                assert ("positive_only", 0.0) in points
                assert {
                    coefficient
                    for method, coefficient in points
                    if method == "squared_exponential"
                } == set(night.EXPECTED_COEFFICIENTS)


def test_historical_branch_command_exposes_no_legacy_scale(tmp_path: Path) -> None:
    grid, _ = night.load_grid(GRID)
    contract = SimpleNamespace(
        expected_canonical_alpha=0.11,
        source_root=tmp_path,
    )
    branch = night.build_branches(contract, _run_spec(), grid)[0]
    trainer_template = [
        "--dataset",
        "{dataset_path}",
        "--seed",
        "{seed}",
        "--steps",
        "{steps}",
        "--output_dir",
        "{output_dir}",
    ]
    command, config = night.branch_command(
        contract_path=tmp_path / "contract.json",
        contract=contract,
        branch=branch,
        branch_dir=tmp_path / "branch",
        trainer_argv_template=trainer_template,
    )
    assert "drpo.e7_squared_exp_night_bootstrap" in command
    assert config["weight_control"]["formula"] == "w(d)=w(0)*exp(-c*(d/2)^2)"
    assert "negative_control" not in config
    assert "negative_scale" not in str(config)
    assert "canonical_alpha" not in str(config)


def test_gae_grid_builds_exact_96_branch_matrix() -> None:
    night.configure_execution(GAE_GRID)
    grid, _ = night.load_grid(GAE_GRID)
    contract = SimpleNamespace(expected_canonical_alpha=0.11)
    branches = night.build_branches(contract, _run_spec(gae=True), grid)
    assert len(branches) == night.GAE_EXPECTED_BRANCHES == 96
    assert len({branch.branch_id for branch in branches}) == 96
    assert {branch.seed for branch in branches} == set(night.GAE_EXPECTED_SEEDS)
    assert {branch.template_values["actor_update_mode"] for branch in branches} == {
        "a2c"
    }
    assert {branch.template_values["advantage_estimator"] for branch in branches} == {
        "td",
        "gae",
    }
    assert {
        float(branch.template_values["exp_coefficient"])
        for branch in branches
        if branch.template_values["weight_method"] == "squared_exponential"
    } == set(night.GAE_COEFFICIENTS)
    assert not ({branch.seed for branch in branches} & set(night.HELD_OUT_SEEDS))


def test_gae_liveness_filters_existing_matrix_to_one_matched_pair() -> None:
    steps = 7_814
    night.configure_execution(
        GAE_GRID,
        liveness_pair=True,
        liveness_steps=steps,
    )
    grid, _ = night.load_grid(GAE_GRID)
    contract = SimpleNamespace(expected_canonical_alpha=0.11)
    branches = night.build_branches(contract, _run_spec(gae=True), grid)
    assert len(branches) == 2
    assert {branch.template_values["advantage_estimator"] for branch in branches} == {
        "td",
        "gae",
    }
    assert {branch.dataset.id for branch in branches} == {
        night.GAE_LIVENESS_DATASET
    }
    assert {branch.seed for branch in branches} == {night.GAE_LIVENESS_SEED}
    assert {
        float(branch.template_values["exp_coefficient"]) for branch in branches
    } == {night.GAE_LIVENESS_COEFFICIENT}
    assert {branch.template_values["steps"] for branch in branches} == {str(steps)}
    assert {branch.template_values["execution_mode"] for branch in branches} == {
        "liveness"
    }


def test_runtime_representative_reuses_gae_branch_matrix() -> None:
    night.configure_execution(GAE_GRID)
    grid, _ = night.load_grid(GAE_GRID)
    branches = night.build_branches(
        SimpleNamespace(expected_canonical_alpha=0.11),
        _run_spec(gae=True),
        grid,
    )
    selected = runtime._representative(branches)  # noqa: SLF001
    assert selected.dataset.id == night.GAE_LIVENESS_DATASET
    assert selected.seed == night.GAE_LIVENESS_SEED
    assert selected.template_values["actor_update_mode"] == "a2c"
    assert selected.template_values["advantage_estimator"] == "gae"
    assert float(selected.template_values["exp_coefficient"]) == 128.0


def test_ordered_hdf5_requires_explicit_timeout_and_next_observation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "replay.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("observations", data=np.zeros((2, 3), dtype=np.float32))
        handle.create_dataset("actions", data=np.zeros((2, 1), dtype=np.float32))
        handle.create_dataset("rewards", data=np.zeros(2, dtype=np.float32))
        handle.create_dataset("terminals", data=np.zeros(2, dtype=np.bool_))
    with pytest.raises(ValueError, match="timeouts.*next_observations"):
        _validate_ordered_hdf5(path)

    with h5py.File(path, "a") as handle:
        handle.create_dataset("timeouts", data=np.zeros(2, dtype=np.bool_))
        handle.create_dataset(
            "next_observations", data=np.zeros((2, 3), dtype=np.float32)
        )
    assert _validate_ordered_hdf5(path) == path.resolve()


def test_snapshot_tables_respect_terminal_timeout_and_tail() -> None:
    rewards = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    terminals = np.asarray([False, True, False, False])
    timeouts = np.asarray([False, False, True, False])
    td, gae = compute_snapshot_tables(
        rewards,
        np.zeros(4),
        np.full(4, 10.0),
        terminals,
        timeouts,
        gamma=0.9,
        gae_lambda=0.5,
    )
    np.testing.assert_allclose(td, [10.0, 2.0, 12.0, 13.0])
    np.testing.assert_allclose(gae, [10.9, 2.0, 12.0, 13.0])


def test_lambda_zero_reduces_exactly_to_td() -> None:
    td, gae = compute_snapshot_tables(
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.zeros(2),
        np.ones(2),
        np.asarray([False, False]),
        np.asarray([False, False]),
        gamma=0.99,
        gae_lambda=0.0,
    )
    np.testing.assert_array_equal(td, gae)


def test_terminal_timeout_overlap_fails_closed() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        compute_snapshot_tables(
            np.ones(2),
            np.zeros(2),
            np.ones(2),
            np.asarray([True, False]),
            np.asarray([True, False]),
            gamma=0.99,
            gae_lambda=0.95,
        )


class FixtureActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = torch.nn.Linear(2, 1, bias=False)
        self.log_std = torch.nn.Parameter(torch.zeros(1, 1))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(states)
        return mean, self.log_std.expand_as(mean)


class FixtureCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = torch.nn.Linear(2, 1, bias=False)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.value(states)


class FixtureAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.tau = 0.7
        self.alpha = 0.11
        self.actor = FixtureActor()
        self.critic = FixtureCritic()
        self.a_opt = torch.optim.SGD(self.actor.parameters(), lr=1e-2)
        self.c_opt = torch.optim.SGD(self.critic.parameters(), lr=1e-2)


def _fixture_replay() -> dict[str, np.ndarray]:
    observations = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        dtype=np.float32,
    )
    return {
        "observations": observations,
        "actions": np.zeros((4, 1), dtype=np.float32),
        "rewards": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "next_observations": observations * 0.5,
        "terminals": np.asarray([False, True, False, False]),
        "timeouts": np.asarray([False, False, True, False]),
    }


def test_snapshot_provider_uses_single_canonical_update_and_updates_critic() -> None:
    replay = _fixture_replay()
    provider = TrajectorySnapshotAdvantage(
        replay=replay,
        estimator="gae",
        batch_size=2,
    )
    injected = build_injected_agent_class(
        FixtureAgent,
        control=NegativeControl(
            method="canonical_signed",
            negative_scale=1.0,
            canonical_alpha=0.11,
        ),
        return_mode="metrics_dict",
        advantage_provider=provider,
    )
    agent = injected()
    before = [parameter.detach().clone() for parameter in agent.critic.parameters()]
    ids = torch.tensor([0.0, 2.0])
    batch = (
        torch.from_numpy(replay["observations"][[0, 2]]),
        torch.from_numpy(replay["actions"][[0, 2]]),
        torch.from_numpy(replay["rewards"][[0, 2]]),
        torch.from_numpy(replay["next_observations"][[0, 2]]),
        torch.from_numpy(replay["terminals"][[0, 2]]),
        ids,
    )
    metrics = agent.update(*batch)
    assert metrics["advantage_estimator"] == "gae"
    assert any(
        not torch.equal(left, right)
        for left, right in zip(before, agent.critic.parameters(), strict=True)
    )
    agent.update(*batch)
    agent.update(*batch)
    summary = provider.summary()
    assert summary["snapshot_count"] == 2
    assert summary["critic_evolution_observed"] is True


def test_full_pair_audit_rejects_mismatched_snapshot_hashes() -> None:
    common = {
        "dataset": "hopper-medium-expert-v2",
        "seed": 200,
        "exp_coefficient": 128.0,
        "execution_mode": "full",
        "snapshot_count": 2,
        "snapshot_refresh_interval": 7_813,
    }
    td = {**common, "advantage_estimator": "td", "snapshot_hashes": ["a", "b"]}
    gae = {**common, "advantage_estimator": "gae", "snapshot_hashes": ["a", "b"]}
    _validate_td_gae_pair(td, gae)
    gae["snapshot_hashes"] = ["a", "c"]
    with pytest.raises(RuntimeError, match="trajectories diverged"):
        _validate_td_gae_pair(td, gae)


def test_gae_refactor_has_no_parallel_python_execution_stack() -> None:
    deleted = (
        Path("src/drpo/e7_canonical_gae_injection.py"),
        Path("src/drpo/e7_sqexp_gae.py"),
        Path("scripts/run_e7_sqexp_gae_liveness.py"),
        Path("tests/test_e7_sqexp_gae.py"),
        Path("tests/test_e7_sqexp_gae_liveness.py"),
    )
    assert not any(path.exists() for path in deleted)

    canonical = Path("src/drpo/e7_canonical_injection.py").read_text()
    bootstrap = Path("src/drpo/e7_squared_exp_night_bootstrap.py").read_text()
    runner = Path("src/drpo/e7_squared_exp_night.py").read_text()
    assert canonical.count("self.a_opt.step()") == 1
    assert canonical.count("self.c_opt.step()") == 1
    assert "self.a_opt.step()" not in bootstrap
    assert "self.c_opt.step()" not in bootstrap
    assert "drpo.e7_sqexp_gae" not in runner
    assert "e7_canonical_gae_injection" not in bootstrap


def test_gae_config_remains_independent_from_historical_config() -> None:
    historical = json.loads(GRID.read_text())
    gae = json.loads(GAE_GRID.read_text())
    assert historical["experiment_id"] == night.EXPERIMENT_ID
    assert gae["experiment_id"] == night.GAE_EXPERIMENT_ID
    assert historical["development_seeds"] == [200, 201]
    assert gae["development_seeds"] == [200, 201, 202, 203]
    assert historical["expected_runnable_branches"] == 126
    assert gae["expected_total_branches"] == 96


def _write_p1_branch(
    root: Path,
    *,
    dataset: str,
    seed: int,
    method: str,
    scale: float | None,
    score: float,
) -> None:
    label = method if scale is None else f"drpo_c{scale:g}"
    branch_id = f"{dataset}__seed{seed}__gae__{label}__a2c__steps1m"
    branch = root / "branches" / branch_id
    output = branch / "trainer_output"
    output.mkdir(parents=True)
    control = {
        "method": method,
        "weight_at_zero": 0.0 if method == "positive_only" else 1.0,
        "reference_distance": 2.0,
        "formula": (
            "w(D)=w(0)*exp(-taper_lambda*relu("
            "(D-remoteness_threshold)/remoteness_scale))"
        ),
        "coordinate": "normalized_squared_standardized_distance",
        "remoteness_threshold": 0.0,
        "remoteness_scale": 1.0 if scale is None else scale,
        "taper_lambda": 1.0,
        "derived_exp_coefficient": 0.0 if scale is None else 1.0 / scale,
    }
    config = {
        "experiment_id": night.GAE_EXPERIMENT_ID,
        "profile_id": night.TUNING_PROFILE_ID,
        "branch_id": branch_id,
        "branch_kind": "injected",
        "dataset_id": dataset,
        "dataset_sha256": "0" * 64,
        "seed": seed,
        "template_values": {
            "steps": "1000000",
            "actor_update_mode": "a2c",
            "advantage_estimator": "gae",
            "execution_mode": "full",
            "weight_method": method,
        },
        "weight_control": control,
    }
    snapshot = {
        "snapshot_count": 2,
        "snapshot_refresh_interval": 100,
        "snapshot_hashes": ["a", "b"],
        "first_snapshot_critic_sha256": "a",
        "latest_snapshot_critic_sha256": "b",
        "final_critic_sha256": "c",
        "critic_evolution_observed": True,
    }
    (branch / "branch_config.json").write_text(json.dumps(config))
    (branch / "branch_manifest.json").write_text(
        json.dumps({"trajectory_snapshot": snapshot})
    )
    (branch / "COMPLETED.json").write_text(json.dumps({"return_code": 0}))
    (output / "result_summary.json").write_text(
        json.dumps(
            {
                "history": {
                    "steps": [800000, 900000, 1000000],
                    "score": [score - 1.0, score, score + 1.0],
                }
            }
        )
    )
    geometry = {
        "status": "complete",
        "update": 1000000,
        "negative_samples": 1,
        "negative_distance_mean": 1.0,
        "negative_weight_mean": 0.5,
        "negative_abs_advantage_sum": 2.0,
        "weighted_negative_abs_advantage_sum": 1.0,
        "weight_control": control,
    }
    (branch / "geometry_diagnostics.jsonl").write_text(json.dumps(geometry) + "\n")


def test_p1_full_aggregate_is_task_balanced_and_claim_bounded(tmp_path: Path) -> None:
    from drpo.e7_squared_exp_night_aggregate import aggregate

    controls = [
        ("positive_only", None, 0.0),
        *(("thresholded_exponential", scale, 2.0 - abs(math.log2(scale / 4.0))) for scale in night.TUNING_REMOTENESS_SCALES),
        ("uncontrolled", None, -2.0),
    ]
    for task_index, dataset in enumerate(night.TUNING_DATASETS):
        for seed in night.TUNING_SEEDS:
            for method, scale, delta in controls:
                _write_p1_branch(
                    tmp_path,
                    dataset=dataset,
                    seed=seed,
                    method=method,
                    scale=scale,
                    score=20.0 + task_index + (seed - 200) * 0.1 + delta,
                )

    summary = aggregate(tmp_path)
    assert summary["status"] == "PASS"
    assert summary["branch_count"] == 198
    assert summary["task_count"] == 9
    assert summary["control_count"] == 11
    aggregate_dir = tmp_path / "aggregate"
    audit = json.loads((aggregate_dir / "terminal_audit.json").read_text())
    assert audit["selected_control"] is None
    assert audit["selection_status"] == "response_curve_only_pending_protocol_freeze"
    assert audit["formal_evidence_allowed"] is False
    assert audit["fixed_horizon_is_not_convergence"] is True
    paired = (aggregate_dir / "p1_paired_deltas.csv").read_text().splitlines()
    assert len(paired) == 181
