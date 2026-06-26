from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTOR = REPO_ROOT / "scripts" / "select_update_tests.py"
TOOL_DIR = REPO_ROOT / "tools" / "drpo-update"

sys.path.insert(0, str(TOOL_DIR))

from test_selection import (  # noqa: E402
    TestExecutionError as ExecutionError,
    TestSelectionError as SelectionError,
    execute_test_plan,
    select_test_plan,
)


def write_map(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "unknown_path_policy": "full",
        "full_commands": [["{python}", "-c", "print(\"full\")"]],
        "control_plane_patterns": ["control/**"],
        "groups": [
            {
                "id": "docs",
                "risk": "low",
                "patterns": ["docs/**"],
                "pytest_targets": [],
                "validators": [],
            },
            {
                "id": "helper",
                "risk": "medium",
                "patterns": ["tools/**", "tests/test_helper.py"],
                "pytest_targets": ["tests/test_helper.py"],
                "validators": [["{python}", "scripts/check.py"]],
            },
        ],
    }
    path.write_text(json.dumps(payload) + "\n")
    return path


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        message,
    )
    return git(repo, "rev-parse", "HEAD")


def test_known_low_and_medium_paths_select_fast_and_union_evidence(tmp_path: Path):
    impact_map = write_map(tmp_path / "map.json")
    plan = select_test_plan(
        ["docs/plan.md", "tools/helper.py", "tests/test_helper.py"],
        impact_map,
    )
    assert plan.selected_mode == "fast"
    assert plan.risk == "medium"
    assert plan.matched_groups == ("docs", "helper")
    assert plan.pytest_targets == ("tests/test_helper.py",)
    assert plan.validators == (("{python}", "scripts/check.py"),)
    assert plan.unknown_paths == ()


def test_unknown_path_escalates_to_full(tmp_path: Path):
    impact_map = write_map(tmp_path / "map.json")
    plan = select_test_plan(["new_area/runner.py"], impact_map)
    assert plan.selected_mode == "full"
    assert plan.unknown_paths == ("new_area/runner.py",)
    assert "unknown" in plan.reason


def test_control_plane_path_escalates_to_full(tmp_path: Path):
    impact_map = write_map(tmp_path / "map.json")
    plan = select_test_plan(["control/map.json"], impact_map)
    assert plan.selected_mode == "full"
    assert plan.risk == "high"
    assert "test_control_plane" in plan.matched_groups


def test_explicit_fast_cannot_override_full_requirement(tmp_path: Path):
    impact_map = write_map(tmp_path / "map.json")
    with pytest.raises(SelectionError, match="fast mode cannot override"):
        select_test_plan(["unknown.txt"], impact_map, requested_mode="fast")


def test_explicit_full_is_allowed_for_low_risk_change(tmp_path: Path):
    impact_map = write_map(tmp_path / "map.json")
    plan = select_test_plan(["docs/readme.md"], impact_map, requested_mode="full")
    assert plan.selected_mode == "full"
    assert "explicitly requested" in plan.reason


def test_invalid_map_fails_closed(tmp_path: Path):
    impact_map = tmp_path / "map.json"
    impact_map.write_text('{"schema_version": 1, "unknown_path_policy": "ignore", "groups": []}')
    with pytest.raises(SelectionError, match="fail-closed"):
        select_test_plan(["docs/a.md"], impact_map)


def test_cli_reports_plan_from_git_diff(tmp_path: Path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", str(repo)], check=True, stdout=subprocess.PIPE)
    (repo / "tools" / "drpo-update").mkdir(parents=True)
    write_map(repo / "tools" / "drpo-update" / "test_impact_map.json")
    (repo / "docs").mkdir()
    (repo / "docs" / "plan.md").write_text("base\n")
    base = commit_all(repo, "base")
    (repo / "docs" / "plan.md").write_text("changed\n")
    head = commit_all(repo, "head")
    proc = subprocess.run(
        [
            "python3",
            str(SELECTOR),
            "--repo",
            str(repo),
            "--base",
            base,
            "--head",
            head,
            "--json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["selected_mode"] == "fast"
    assert payload["changed_paths"] == ["docs/plan.md"]


def test_full_gate_aggregates_failures_and_runs_later_commands(tmp_path: Path):
    marker = tmp_path / "later-command-ran.txt"
    impact_map = tmp_path / "map.json"
    payload = {
        "schema_version": 1,
        "unknown_path_policy": "full",
        "full_commands": [
            ["{python}", "-c", "raise SystemExit(7)"],
            [
                "{python}",
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
            ],
        ],
        "control_plane_patterns": [],
        "groups": [
            {
                "id": "docs",
                "risk": "low",
                "patterns": ["docs/**"],
                "pytest_targets": [],
                "validators": [],
            }
        ],
    }
    impact_map.write_text(json.dumps(payload) + "\n")
    plan = select_test_plan(["unknown.py"], impact_map)
    log_dir = tmp_path / "logs"
    with pytest.raises(ExecutionError) as captured:
        execute_test_plan(plan, worktree=tmp_path, log_dir=log_dir)
    assert marker.read_text() == "ran"
    assert len(captured.value.outcomes) == 2
    assert captured.value.outcomes[0].returncode == 7
    assert captured.value.outcomes[1].returncode == 0
    assert len(list(log_dir.glob("*.log"))) == 2


def test_fast_gate_missing_ruff_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    impact_map = tmp_path / "map.json"
    payload = {
        "schema_version": 1,
        "unknown_path_policy": "full",
        "full_commands": [["{python}", "-c", "print('full')"]],
        "control_plane_patterns": [],
        "groups": [
            {
                "id": "python",
                "risk": "low",
                "patterns": ["*.py"],
                "pytest_targets": [],
                "validators": [],
            }
        ],
    }
    impact_map.write_text(json.dumps(payload) + "\n")
    (tmp_path / "changed.py").write_text("VALUE = 1\n")
    plan = select_test_plan(["changed.py"], impact_map)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(ExecutionError) as captured:
        execute_test_plan(plan, worktree=tmp_path, log_dir=tmp_path / "logs")
    assert [outcome.returncode for outcome in captured.value.outcomes] == [0, 127]
    assert "ruff" in (captured.value.outcomes[1].error or "")
