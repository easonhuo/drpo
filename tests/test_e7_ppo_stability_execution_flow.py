from __future__ import annotations

import json
from pathlib import Path
import importlib.util

import pytest

from drpo import e7_ppo_runtime_autotune as autotune
from drpo import e7_ppo_stability_smoke as smoke
from drpo import e7_canonical_sweep as base
from drpo.e7_canonical_injection import NegativeControl, sha256_file


def _branch(mode: str) -> base.Branch:
    return base.Branch(
        branch_id=f"walker2d-medium-v2__seed200__exp__{mode}",
        branch_kind="injected",
        dataset=base.DatasetSpec(
            id="walker2d-medium-v2",
            path="/tmp/walker2d-medium-v2.hdf5",
            sha256="0" * 64,
        ),
        seed=200,
        template_values={
            "actor_update_mode": mode,
            "steps": "1000000",
            "diagnostics_interval": "1000",
        },
        negative_control=NegativeControl(
            method="exponential",
            negative_scale=1.0,
            canonical_alpha=0.11,
            reference_distance=2.0,
            exponential_coefficient=1.5,
        ),
    )


def test_smoke_selects_matched_a2c_ppo_pair_and_shortens_only_horizon() -> None:
    selected = smoke._select_smoke_branches(  # noqa: SLF001
        [_branch("a2c"), _branch("ppo_clip")]
    )
    assert [branch.template_values["actor_update_mode"] for branch in selected] == [
        "a2c",
        "ppo_clip",
    ]
    assert {branch.template_values["steps"] for branch in selected} == {"20000"}
    assert {branch.seed for branch in selected} == {200}
    assert all(branch.branch_id.startswith("smoke20k__") for branch in selected)


def _write_valid_ppo_diagnostics(branch_dir: Path) -> None:
    positions = {
        "1": {
            "ratio_mean": 1.0,
            "abs_log_ratio_mean": 0.0,
        },
        "2": {
            "ratio_mean": 1.01,
            "abs_log_ratio_mean": 0.01,
        },
        "3": {
            "ratio_mean": 0.99,
            "abs_log_ratio_mean": 0.02,
        },
        "4": {
            "ratio_mean": 1.02,
            "abs_log_ratio_mean": 0.03,
        },
    }
    latest = {
        "status": "complete",
        "update": 20_000,
        "old_policy_refresh_count": 4_999,
        "pre_update_by_block_position": positions,
        "pre_update": {
            "ratio_min": 0.7,
            "ratio_max": 1.4,
            "ratio_outside_fraction": 0.2,
            "objective_clip_fraction": 0.1,
            "positive_objective_clip_fraction": 0.08,
            "negative_objective_clip_fraction": 0.12,
        },
        "actor_gradient_norm": 2.0,
        "actor_parameter_update_norm": 0.02,
        "actor_relative_parameter_update_norm": 0.001,
        "sampled_post_update": {
            "ratio_to_old_p01": 0.8,
            "ratio_to_old_p99": 1.2,
            "single_step_ratio_min": 0.95,
            "single_step_ratio_max": 1.05,
        },
    }
    branch_dir.mkdir(parents=True)
    (branch_dir / "PPO_DIAGNOSTICS_LATEST.json").write_text(json.dumps(latest))
    line = json.dumps({"status": "running"})
    (branch_dir / "ppo_diagnostics.jsonl").write_text("\n".join([line] * 20) + "\n")


def test_ppo_smoke_diagnostics_gate_checks_ratio_and_clip_activity(
    tmp_path: Path,
) -> None:
    branch_dir = tmp_path / "branch"
    _write_valid_ppo_diagnostics(branch_dir)
    result = smoke._validate_ppo_diagnostics(branch_dir)  # noqa: SLF001
    assert result["old_policy_refresh_count"] == 4_999
    assert result["position_1_ratio_mean"] == pytest.approx(1.0)
    assert result["objective_clip_fraction"] == pytest.approx(0.1)
    assert result["negative_objective_clip_fraction"] == pytest.approx(0.12)


def test_ppo_smoke_diagnostics_reject_ratio_that_never_leaves_one(
    tmp_path: Path,
) -> None:
    branch_dir = tmp_path / "branch"
    _write_valid_ppo_diagnostics(branch_dir)
    latest_path = branch_dir / "PPO_DIAGNOSTICS_LATEST.json"
    latest = json.loads(latest_path.read_text())
    for position in ("2", "3", "4"):
        latest["pre_update_by_block_position"][position]["ratio_mean"] = 1.0
        latest["pre_update_by_block_position"][position]["abs_log_ratio_mean"] = 0.0
    latest_path.write_text(json.dumps(latest))
    with pytest.raises(smoke.SmokeGateError, match="never departed"):
        smoke._validate_ppo_diagnostics(branch_dir)  # noqa: SLF001


def test_smoke_gate_is_invalidated_by_protected_file_drift(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    smoke_dir = tmp_path / "smoke"
    for relative in smoke.PROTECTED_IMPLEMENTATION_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    fingerprints = {
        relative: sha256_file(repo / relative)
        for relative in smoke.PROTECTED_IMPLEMENTATION_PATHS
    }
    smoke_dir.mkdir()
    (smoke_dir / "SMOKE_GATE.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "experiment_id": smoke.EXPERIMENT_ID,
                "run_kind": smoke.SMOKE_RUN_KIND,
                "protected_implementation_sha256": fingerprints,
            }
        )
    )
    smoke.validate_smoke_gate(repo_root=repo, smoke_dir=smoke_dir)
    changed = repo / smoke.PROTECTED_IMPLEMENTATION_PATHS[0]
    changed.write_text("changed")
    with pytest.raises(smoke.SmokeGateError, match="changed after"):
        smoke.validate_smoke_gate(repo_root=repo, smoke_dir=smoke_dir)


def test_ppo_capacity_fingerprint_ignores_scientific_sweep_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    contract = tmp_path / "contract.json"
    run_spec_path = tmp_path / "run_spec.json"
    grid_path = tmp_path / "grid.json"
    contract.write_text("contract")
    run_spec_path.write_text("run spec")
    grid_path.write_text("grid")
    for relative in (
        "src/drpo/e7_canonical_ppo_injection.py",
        "src/drpo/e7_canonical_ppo_bootstrap.py",
        "src/drpo/e7_canonical_ppo_stability.py",
        "src/drpo/e7_canonical_ppo_stability_entry.py",
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)

    run_spec = {
        "trainer_argv_template": [
            "--batch",
            "256",
            "--lr",
            "0.0003",
            "--eval_interval",
            "50000",
            "--eval_episodes",
            "10",
        ],
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
    }
    grid = {
        "ppo": {
            "updates_per_old_policy": 4,
            "diagnostics_interval": 1000,
        },
        "exp_coefficients": [0.5, 1.0, 1.5],
        "development_seeds": [200, 201, 202, 203],
        "steps": 1_000_000,
    }
    monkeypatch.setattr(autotune, "_load_run_spec", lambda path: (run_spec, "x"))
    monkeypatch.setattr(autotune.pilot, "load_ppo_grid", lambda path: (grid, "y"))

    common = dict(
        repo_root=repo,
        contract_path=contract,
        run_spec_path=run_spec_path,
        grid_path=grid_path,
        probe_steps=20_000,
        probe_seed=990_101,
        probe_seconds=90.0,
        throughput_retention_fraction=0.97,
        fallback_workers=60,
        cpu_fraction=0.85,
        memory_headroom_fraction=0.15,
        per_worker_safety_factor=1.2,
        max_workers=None,
        max_growth_factor=3.0,
    )
    first = autotune.resource_fingerprint(**common)
    grid["exp_coefficients"] = [9.0]
    grid["development_seeds"] = [999]
    grid["steps"] = 2_000_000
    second = autotune.resource_fingerprint(**common)
    assert first == second
    assert "exp_coefficients" in first["ignored_scientific_fields"]



def test_throughput_candidate_grid_and_retained_peak_rule() -> None:
    assert autotune._candidate_workers(96, 60) == [48, 60, 72, 96]  # noqa: SLF001
    selected, rule = autotune._select_from_throughput(  # noqa: SLF001
        [
            {
                "concurrency": 48,
                "aggregate_updates_per_second": 100.0,
                "valid": True,
            },
            {
                "concurrency": 60,
                "aggregate_updates_per_second": 103.0,
                "valid": True,
            },
            {
                "concurrency": 72,
                "aggregate_updates_per_second": 104.0,
                "valid": True,
            },
            {
                "concurrency": 96,
                "aggregate_updates_per_second": 101.0,
                "valid": True,
            },
        ],
        retention_fraction=0.97,
    )
    assert selected == 60
    assert rule["peak_aggregate_updates_per_second"] == pytest.approx(104.0)


def test_throughput_selection_rejects_all_failed_candidates() -> None:
    with pytest.raises(autotune.RuntimeResourceError, match="no concurrency"):
        autotune._select_from_throughput(  # noqa: SLF001
            [
                {
                    "concurrency": 96,
                    "aggregate_updates_per_second": 0.0,
                    "valid": False,
                }
            ],
            retention_fraction=0.97,
        )

def test_pilot_identity_rejects_worker_change(tmp_path: Path) -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_e7_ppo_stability_pilot_auto.py"
    )
    spec = importlib.util.spec_from_file_location("e7_ppo_pilot_auto", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    validate_identity = module._validate_existing_run_identity  # noqa: SLF001

    work = tmp_path / "work"
    work.mkdir()
    (work / "RUN_IDENTITY.json").write_text(
        json.dumps({"plan": {"max_workers": 64}})
    )
    validate_identity(work, 64)
    with pytest.raises(Exception, match="fixes max_workers=64"):
        validate_identity(work, 96)
