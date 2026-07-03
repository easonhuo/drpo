from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_du1_e6_separation_grid.py"
CONFIG = REPO_ROOT / "configs" / "du1_e6_separation_grid.yaml"


def load_runner():
    spec = importlib.util.spec_from_file_location("du1_e6_separation_grid", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_registered_grid_is_exact_and_excludes_quartic(tmp_path: Path) -> None:
    runner = load_runner()
    plan = yaml.safe_load(CONFIG.read_text())
    runner.validate_plan(plan)
    tasks = runner.build_tasks(plan, REPO_ROOT, tmp_path)
    assert len(tasks) == 120
    assert {task.seed for task in tasks} == {0, 1, 2, 3, 4}
    assert {task.method for task in tasks} == set(runner.ACTIVE_METHODS)
    assert "reciprocal_quartic_distance" not in {task.method for task in tasks}
    assert {task.negative_alpha for task in tasks} == {0.25, 0.5}
    assert {task.anchor for task in tasks} == {0.1, 0.25}
    assert {task.retention for task in tasks} == {0.25}
    assert len({task.run_id for task in tasks}) == 120


def test_formal_seed_is_rejected() -> None:
    runner = load_runner()
    plan = yaml.safe_load(CONFIG.read_text())
    plan["seeds"] = [0, 1, 2, 3, 200]
    with pytest.raises(ValueError, match="development seeds"):
        runner.validate_plan(plan)


def test_quartic_cannot_be_reintroduced() -> None:
    runner = load_runner()
    plan = yaml.safe_load(CONFIG.read_text())
    plan["methods"][-1] = "reciprocal_quartic_distance"
    with pytest.raises(ValueError, match="methods must be exactly"):
        runner.validate_plan(plan)


def test_unregistered_grid_change_is_rejected() -> None:
    runner = load_runner()
    plan = yaml.safe_load(CONFIG.read_text())
    plan["grid"]["negative_alpha"] = [0.25, 0.5, 1.0]
    with pytest.raises(ValueError, match="negative_alpha"):
        runner.validate_plan(plan)


def test_paired_task_collapse_is_assigned_per_grid_cell_and_seed() -> None:
    runner = load_runner()
    rows = [
        {
            "grid_cell_id": "cell",
            "seed": 0,
            "method": "positive_only",
            "final_expected_semantic_reward": 1.0,
        },
        {
            "grid_cell_id": "cell",
            "seed": 0,
            "method": "all_negative",
            "final_expected_semantic_reward": 0.19,
        },
        {
            "grid_cell_id": "cell",
            "seed": 0,
            "method": "global_matched",
            "final_expected_semantic_reward": 0.21,
        },
    ]
    runner.assign_task_collapse(rows, 0.2)
    assert rows[1]["task_performance_collapse"] is True
    assert rows[2]["task_performance_collapse"] is False
    assert rows[1]["paired_positive_only_reward"] == 1.0


def test_plan_only_writes_120_run_manifest(tmp_path: Path) -> None:
    output = tmp_path / "plan"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(REPO_ROOT),
            "--grid-config",
            str(CONFIG),
            "--output-root",
            str(output),
            "--plan-only",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert '"expected_runs": 120' in result.stdout
    payload = yaml.safe_load((output / "grid_plan.json").read_text())
    assert payload["expected_runs"] == 120
    assert payload["quartic_excluded_from_active_matrix"] is True
    assert payload["formal_seeds_accessed"] is False
