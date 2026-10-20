from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from drpo import e7_canonical_scale1_grid as scale1
from drpo import e7_canonical_sweep as base
from drpo.e7_canonical_injection import controlled_advantage
from drpo.e7_canonical_scale1_grid import (
    build_scale1_branches,
    expand_scale1_controls,
)
from tests.test_e7_canonical_sweep import contract


GRID_PATH = Path("configs/e7_canonical_scale1_coefficient_grid_v1.json")
D4RL9_GRID_PATH = Path("configs/e7_canonical_d4rl9_glq_taskwise_tuning_v1.json")


def _grid() -> dict:
    return json.loads(GRID_PATH.read_text())


def _d4rl9_grid() -> dict:
    return json.loads(D4RL9_GRID_PATH.read_text())


def _trainer_argv() -> list[str]:
    return [
        "--dataset",
        "{dataset_id}",
        "--hdf5",
        "{dataset_path}",
        "--variant",
        "iqlv_exp_rank",
        "--alpha",
        "0.11",
        "--tau",
        "0.5",
        "--temp",
        "5.0",
        "--steps",
        "1000000",
        "--batch",
        "256",
        "--lr",
        "0.0003",
        "--eval_interval",
        "50000",
        "--eval_episodes",
        "10",
        "--seed",
        "{seed}",
        "--out_dir",
        "{output_dir}",
    ]


def _d4rl9_run_spec(tmp_path: Path) -> dict:
    datasets = []
    for index, dataset_id in enumerate(scale1.D4RL9_EXPECTED_DATASETS):
        path = tmp_path / f"dataset_{index}.hdf5"
        path.write_bytes(dataset_id.encode("utf-8"))
        datasets.append(
            {
                "id": dataset_id,
                "path": str(path),
                "sha256": hashlib.sha256(dataset_id.encode("utf-8")).hexdigest(),
            }
        )
    return {
        "experiment_id": "EXT-H-E7-BENCH-01",
        "run_kind": "pilot",
        "datasets": datasets,
        "seeds": [200, 201],
        "trainer_argv_template": _trainer_argv(),
        "environment": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "passthrough_variants": [
            {"id": "original_exp_rank_mr", "template_values": {}}
        ],
    }


def test_repository_scale1_grid_has_expected_branches() -> None:
    raw = _grid()
    controls = expand_scale1_controls(raw)
    assert len(controls) == raw["branch_count_per_dataset_seed"] == 17

    taper_controls = [
        control
        for control in controls
        if control.method in {
            "reciprocal_linear",
            "reciprocal_quadratic",
            "exponential",
        }
    ]
    assert len(taper_controls) == 15
    assert {control.negative_scale for control in taper_controls} == {1.0}


def test_scale1_preserves_full_near_field_negative_alpha() -> None:
    advantage = torch.tensor([-2.0, 3.0])
    zero_distance = torch.zeros(2)
    for control in expand_scale1_controls(_grid()):
        if control.method not in {
            "reciprocal_linear",
            "reciprocal_quadratic",
            "exponential",
        }:
            continue
        adjusted, factor = controlled_advantage(advantage, zero_distance, control)
        assert torch.isclose(factor[0], torch.tensor(0.11))
        assert torch.isclose(adjusted[0], torch.tensor(-0.22))
        assert torch.isclose(factor[1], torch.tensor(1.0))
        assert torch.isclose(adjusted[1], torch.tensor(3.0))


def test_coefficient_grid_branch_ids_are_unique(tmp_path: Path) -> None:
    dataset_file = tmp_path / "dataset.hdf5"
    dataset_file.write_bytes(b"fixture")
    digest = hashlib.sha256(b"fixture").hexdigest()
    run_spec = {
        "run_kind": "pilot",
        "datasets": [
            {"id": "hopper", "path": str(dataset_file), "sha256": digest}
        ],
        "seeds": [200],
        "trainer_argv_template": [],
    }
    branches = build_scale1_branches(contract(tmp_path), run_spec, _grid())
    ids = [branch.branch_id for branch in branches]
    assert len(ids) == 17
    assert len(set(ids)) == 17
    taper_ids = [
        branch_id
        for branch_id in ids
        if "reciprocal_" in branch_id or "exponential" in branch_id
    ]
    assert taper_ids
    assert all("__scale1__coef" in branch_id for branch_id in taper_ids)


def test_d4rl9_grid_has_only_five_global_linear_quadratic_candidates() -> None:
    raw = _d4rl9_grid()
    controls = expand_scale1_controls(raw)
    assert len(controls) == raw["branch_count_per_dataset_seed"] == 15
    assert {control.method for control in controls} == {
        "global",
        "reciprocal_linear",
        "reciprocal_quadratic",
    }
    for method in ("global", "reciprocal_linear", "reciprocal_quadratic"):
        assert sum(control.method == method for control in controls) == 5
    assert all(control.reference_distance == 2.0 for control in controls)
    assert all(control.method != "exponential" for control in controls)
    assert all(control.method != "positive_only" for control in controls)


def test_d4rl9_reciprocal_families_keep_full_near_field_alpha() -> None:
    advantage = torch.tensor([-2.0, 3.0])
    zero_distance = torch.zeros(2)
    for control in expand_scale1_controls(_d4rl9_grid()):
        if control.method == "global":
            continue
        adjusted, factor = controlled_advantage(advantage, zero_distance, control)
        assert torch.isclose(factor[0], torch.tensor(0.11))
        assert torch.isclose(adjusted[0], torch.tensor(-0.22))
        assert torch.isclose(factor[1], torch.tensor(1.0))


def test_d4rl9_run_spec_expands_four_tuning_seeds_and_strips_passthrough(
    tmp_path: Path,
) -> None:
    source = tmp_path / "run_spec.json"
    source.write_text(json.dumps(_d4rl9_run_spec(tmp_path)))
    run_spec, digest = scale1.load_d4rl9_run_spec(source)
    assert run_spec["seeds"] == [200, 201, 202, 203]
    assert run_spec["passthrough_variants"] == []
    assert [item["id"] for item in run_spec["datasets"]] == list(
        scale1.D4RL9_EXPECTED_DATASETS
    )
    assert isinstance(digest, str) and len(digest) == 64


def test_d4rl9_grid_builds_exact_540_unique_branches(tmp_path: Path) -> None:
    run_spec = _d4rl9_run_spec(tmp_path)
    run_spec["seeds"] = [200, 201, 202, 203]
    run_spec["passthrough_variants"] = []
    branches = build_scale1_branches(
        contract(tmp_path), run_spec, _d4rl9_grid()
    )
    assert len(branches) == 540
    assert len({branch.branch_id for branch in branches}) == 540
    assert all(branch.branch_kind == "injected" for branch in branches)
    assert all("exponential" not in branch.branch_id for branch in branches)
    assert all("positive_only" not in branch.branch_id for branch in branches)


def test_taskwise_selection_uses_registered_tie_breaks() -> None:
    groups = []
    for dataset in scale1.D4RL9_EXPECTED_DATASETS:
        for method in ("global", "reciprocal_linear", "reciprocal_quadratic"):
            parameter_name = (
                "negative_scale"
                if method == "global"
                else f"{method}_coefficient"
            )
            for value in (0.5, 1.0, 3.0, 10.0, 30.0):
                groups.append(
                    {
                        "dataset_id": dataset,
                        "method": method,
                        "parameter_name": parameter_name,
                        "parameter_value": value,
                        "late_window_mean_across_seeds": 10.0,
                        "late_window_min_across_seeds": 5.0,
                        "best_to_late_mean_drop_mean": 1.0,
                    }
                )
    selections = scale1._select_taskwise(groups)
    assert len(selections) == 27
    assert {row["parameter_value"] for row in selections} == {0.5}


def test_main_loads_grid_without_recursion_and_restores_hooks(monkeypatch) -> None:
    original_load_grid = base.load_grid
    original_load_run_spec = base.load_run_spec
    original_build_branches = base.build_branches
    original_status = base.SCIENTIFIC_STATUS
    original_version = base.RUNNER_VERSION
    observed: dict[str, object] = {}

    def fake_main(argv: list[str] | None = None) -> int:
        raw, digest = base.load_grid(str(GRID_PATH))
        observed["argv"] = argv
        observed["metric"] = raw["primary_selection_metric"]
        observed["digest"] = digest
        observed["builder"] = base.build_branches
        return 0

    monkeypatch.setattr(base, "main", fake_main)

    assert scale1.main(["plan"]) == 0
    assert observed["argv"] == ["plan"]
    assert observed["metric"] == "final_score"
    assert isinstance(observed["digest"], str)
    assert observed["builder"] is scale1.build_scale1_branches
    assert base.load_grid is original_load_grid
    assert base.load_run_spec is original_load_run_spec
    assert base.build_branches is original_build_branches
    assert base.SCIENTIFIC_STATUS == original_status
    assert base.RUNNER_VERSION == original_version
