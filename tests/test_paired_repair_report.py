from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_SCOPE_SPEC = importlib.util.spec_from_file_location(
    "validate_update_scope", ROOT / "scripts" / "validate_update_scope.py"
)
assert _SCOPE_SPEC is not None and _SCOPE_SPEC.loader is not None
validate_update_scope = importlib.util.module_from_spec(_SCOPE_SPEC)
sys.modules[_SCOPE_SPEC.name] = validate_update_scope
_SCOPE_SPEC.loader.exec_module(validate_update_scope)

_SPEC = importlib.util.spec_from_file_location(
    "paired_repair_report", ROOT / "scripts" / "paired_repair_report.py"
)
assert _SPEC is not None and _SPEC.loader is not None
paired = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = paired
_SPEC.loader.exec_module(paired)


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def commit(repo: Path, message: str) -> str:
    git(repo, "add", ".")
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def repository(tmp_path: Path) -> tuple[Path, str, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "src" / "existing.py").write_text("def keep():\n    return 1\n")
    base = commit(repo, "base")

    (repo / "src" / "feature.py").write_text(
        "def first():\n    return 1\n\ndef duplicate():\n    return 2\n"
    )
    (repo / "tests" / "test_feature.py").write_text(
        "def test_feature():\n    assert 1 + 1 == 2\n"
    )
    a0 = commit(repo, "A0")

    (repo / "src" / "feature.py").write_text(
        "from src.existing import keep\n\ndef first():\n    return keep()\n"
    )
    b1 = commit(repo, "B1")
    return repo, base, a0, b1


def validation(path: Path, *, b1_focused: str = "pass") -> None:
    payload = {
        "schema_version": 1,
        "a0": {
            "focused_tests": "pass",
            "full_repository_pytest": "not_run",
            "ruff": "pass",
            "required_liveness": "not_applicable",
            "scientific_scope_unchanged": True,
            "reviewer_correctness": "pass",
        },
        "b1": {
            "focused_tests": b1_focused,
            "full_repository_pytest": "not_run",
            "ruff": "pass",
            "required_liveness": "not_applicable",
            "scientific_scope_unchanged": True,
            "reviewer_correctness": "pass",
        },
    }
    path.write_text(json.dumps(payload))


def test_freeze_and_close_records_real_before_after(tmp_path: Path) -> None:
    repo, base, a0, b1 = repository(tmp_path)
    record = tmp_path / "record"
    assert paired.main.__module__ == "paired_repair_report"
    freeze_args = paired.parser().parse_args(
        [
            "freeze-a0",
            "--repo-root",
            str(repo),
            "--base",
            base,
            "--a0",
            a0,
            "--claim",
            "GOV-TEST-01",
            "--worker",
            "worker-1",
            "--gate-snapshot",
            "a" * 40,
            "--record-dir",
            str(record),
        ]
    )
    assert freeze_args.function(freeze_args) == 0
    frozen = json.loads((record / "PAIR.json").read_text())
    assert frozen["a0_sha"] == a0
    assert frozen["a0_metrics"]["new_production_python_files"] == 1

    feedback = tmp_path / "feedback.md"
    feedback.write_text(
        "Reuse the existing module and remove the duplicate helper before accepting B1."
    )
    evidence = tmp_path / "validation.json"
    validation(evidence)
    close_args = paired.parser().parse_args(
        [
            "close-b1",
            "--repo-root",
            str(repo),
            "--record-dir",
            str(record),
            "--b1",
            b1,
            "--worker",
            "worker-1",
            "--feedback-file",
            str(feedback),
            "--feedback-source",
            "pr-comment:123",
            "--validation-file",
            str(evidence),
        ]
    )
    assert close_args.function(close_args) == 0
    closed = json.loads((record / "PAIR.json").read_text())
    assert closed["b1_sha"] == b1
    assert closed["evidence_verdict"] == "B1_ELIGIBLE_AND_SMALLER"
    assert closed["comparison"]["production_python_churn"] < 0
    assert (record / "GATE_FEEDBACK.md").is_file()
    assert (record / "COMPARISON.md").is_file()


def test_close_rejects_different_worker(tmp_path: Path) -> None:
    repo, base, a0, b1 = repository(tmp_path)
    record = tmp_path / "record"
    args = paired.parser().parse_args(
        [
            "freeze-a0",
            "--repo-root",
            str(repo),
            "--base",
            base,
            "--a0",
            a0,
            "--claim",
            "GOV-TEST-02",
            "--worker",
            "worker-1",
            "--gate-snapshot",
            "a" * 40,
            "--record-dir",
            str(record),
        ]
    )
    args.function(args)
    feedback = tmp_path / "feedback.md"
    feedback.write_text("This is sufficiently specific gate feedback for the same implementation.")
    evidence = tmp_path / "validation.json"
    validation(evidence)
    close_args = paired.parser().parse_args(
        [
            "close-b1",
            "--repo-root",
            str(repo),
            "--record-dir",
            str(record),
            "--b1",
            b1,
            "--worker",
            "worker-2",
            "--feedback-file",
            str(feedback),
            "--feedback-source",
            "pr-comment:123",
            "--validation-file",
            str(evidence),
        ]
    )
    with pytest.raises(validate_update_scope.ScopeError, match="worker label"):
        close_args.function(close_args)


def test_b1_losing_a0_pass_is_ineligible(tmp_path: Path) -> None:
    repo, base, a0, b1 = repository(tmp_path)
    a0_metrics = paired.metrics(repo, base, a0)
    b1_metrics = paired.metrics(repo, base, b1)
    evidence_path = tmp_path / "validation.json"
    validation(evidence_path, b1_focused="fail")
    evidence = paired.validation(evidence_path)
    verdict = paired.verdict(evidence, a0_metrics, b1_metrics)
    assert verdict == "B1_INELIGIBLE_RETAIN_A0_OR_REPAIR"


def test_freeze_rejects_non_descendant_a0(tmp_path: Path) -> None:
    repo, base, _, _ = repository(tmp_path)
    git(repo, "checkout", "--orphan", "other")
    git(repo, "rm", "-rf", ".")
    (repo / "other.py").write_text("value = 1\n")
    other = commit(repo, "other")
    with pytest.raises(validate_update_scope.ScopeError, match="not an ancestor"):
        paired.freeze(
            paired.parser().parse_args(
                [
                    "freeze-a0",
                    "--repo-root",
                    str(repo),
                    "--base",
                    base,
                    "--a0",
                    other,
                    "--claim",
                    "GOV-TEST-03",
                    "--worker",
                    "worker-1",
                    "--gate-snapshot",
                    "a" * 40,
                    "--record-dir",
                    str(tmp_path / "record"),
                ]
            )
        )
