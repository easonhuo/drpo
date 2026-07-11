from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from drpo import e7_canonical_exp_horizon_grid as joint
from drpo import e7_canonical_sweep as base
from tests.test_e7_canonical_sweep import contract


GRID_PATH = Path("configs/e7_canonical_exp_horizon_joint_grid_v1.json")


def _grid() -> dict:
    return json.loads(GRID_PATH.read_text())


def _run_spec(tmp_path: Path, *, source_seeds: bool = False) -> dict:
    datasets = []
    for dataset_id in joint.EXPECTED_DATASETS:
        path = tmp_path / f"{dataset_id}.hdf5"
        payload = dataset_id.encode()
        path.write_bytes(payload)
        datasets.append(
            {
                "id": dataset_id,
                "path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    seeds = joint.SOURCE_RUN_SPEC_SEEDS if source_seeds else joint.EXPECTED_SEEDS
    return {
        "run_kind": "pilot",
        "experiment_id": "EXT-H-E7-BENCH-01",
        "datasets": datasets,
        "seeds": list(seeds),
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        "trainer_argv_template": [
            "--dataset",
            "{dataset_id}",
            "--alpha",
            "0.11",
            "--steps",
            "1000000",
            "--eval_interval",
            "50000",
            "--eval_episodes",
            "10",
        ],
        "passthrough_variants": [
            {"id": "original_exp_rank_mr", "template_values": {}}
        ],
    }


def test_repository_grid_is_frozen() -> None:
    raw = _grid()
    assert raw["exp_scale1_2m_coefficients"] == [
        joint.LEGACY_EXP_COEFFICIENT,
        1.0,
        1.5,
    ]
    assert raw["exp_scale1_1m_coefficients"] == [
        0.25,
        0.5,
        0.75,
        1.25,
        2.0,
        3.0,
    ]
    assert raw["expected_total_branches"] == 432
    assert raw["fixed_max_workers"] == 60


def test_builds_432_unique_branches_with_long_jobs_first(tmp_path: Path) -> None:
    branches = joint.build_exp_horizon_branches(
        contract(tmp_path), _run_spec(tmp_path), _grid()
    )
    assert len(branches) == 432
    assert len({branch.branch_id for branch in branches}) == 432

    long = branches[:108]
    assert all(branch.template_values["steps"] == "2000000" for branch in long)
    assert all("__exponential__scale1__" in branch.branch_id for branch in long)
    assert {
        branch.negative_control.exponential_coefficient
        for branch in long
        if branch.negative_control is not None
    } == {joint.LEGACY_EXP_COEFFICIENT, 1.0, 1.5}

    remaining = branches[108:]
    assert all(branch.template_values["steps"] == "1000000" for branch in remaining)
    assert sum("__positive_only__" in branch.branch_id for branch in branches) == 36
    assert sum("__scale0p1__" in branch.branch_id for branch in branches) == 36
    assert (
        sum(
            "__baseline__original_exp_rank_mr__" in branch.branch_id
            for branch in branches
        )
        == 36
    )


def test_run_spec_loader_expands_seeds_and_replaces_only_steps(tmp_path: Path) -> None:
    source = tmp_path / "run_spec.json"
    raw = _run_spec(tmp_path, source_seeds=True)
    source.write_text(json.dumps(raw))
    loaded, digest = joint.load_exp_horizon_run_spec(str(source))
    assert loaded["seeds"] == list(joint.EXPECTED_SEEDS)
    argv = loaded["trainer_argv_template"]
    assert argv[argv.index("--steps") + 1] == "{steps}"
    assert isinstance(digest, str)

    source_argv = raw["trainer_argv_template"]
    source_argv[source_argv.index("--steps") + 1] = "2000000"
    source.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="prior 1000000"):
        joint.load_exp_horizon_run_spec(str(source))


def test_run_spec_loader_rejects_seed_thread_or_eval_drift(tmp_path: Path) -> None:
    source = tmp_path / "run_spec.json"
    raw = _run_spec(tmp_path, source_seeds=True)
    raw["seeds"] = [200, 201, 202]
    source.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="source run_spec seeds changed"):
        joint.load_exp_horizon_run_spec(str(source))

    raw = _run_spec(tmp_path, source_seeds=True)
    raw["environment"]["OMP_NUM_THREADS"] = "2"
    source.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="OMP_NUM_THREADS"):
        joint.load_exp_horizon_run_spec(str(source))

    raw = _run_spec(tmp_path, source_seeds=True)
    argv = raw["trainer_argv_template"]
    argv[argv.index("--eval_interval") + 1] = "100000"
    source.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="eval_interval"):
        joint.load_exp_horizon_run_spec(str(source))


def test_main_restores_all_generic_hooks_and_fixes_workers(monkeypatch) -> None:
    original_grid = base.load_grid
    original_run_spec = base.load_run_spec
    original_builder = base.build_branches
    observed: dict[str, object] = {}

    def fake_main(argv: list[str] | None = None) -> int:
        observed["argv"] = argv
        observed["grid"] = base.load_grid
        observed["run_spec"] = base.load_run_spec
        observed["builder"] = base.build_branches
        return 0

    monkeypatch.setattr(base, "main", fake_main)
    assert joint.main(["plan"]) == 0
    assert observed["argv"] == ["plan", "--max-workers", "60"]
    assert observed["grid"] is joint.load_exp_horizon_grid
    assert observed["run_spec"] is joint.load_exp_horizon_run_spec
    assert observed["builder"] is joint.build_exp_horizon_branches
    assert base.load_grid is original_grid
    assert base.load_run_spec is original_run_spec
    assert base.build_branches is original_builder

    with pytest.raises(ValueError, match="fixes --max-workers"):
        joint.main(["plan", "--max-workers", "80"])
