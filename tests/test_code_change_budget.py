from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/validate_code_change_budget.py"
SPEC = importlib.util.spec_from_file_location("validate_code_change_budget", MODULE_PATH)
assert SPEC and SPEC.loader
budget = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = budget
SPEC.loader.exec_module(budget)


def _git(repo: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process.stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/base.py").write_text(
        "\n".join(f"value_{index} = {index}" for index in range(120)) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "src/helper.py").write_text(
        "def helper(value):\n    return value + 1\n",
        encoding="utf-8",
    )
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "base")
    return tmp_path


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def _replace_prefix(repo: Path, count: int, marker: str) -> None:
    path = repo / "src/base.py"
    lines = path.read_text(encoding="utf-8").splitlines()
    for index in range(count):
        lines[index] = f"{marker}_{index} = {index + 1000}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _valid_justification(changed_files: set[str], *, new_files: set[str] = set()) -> str:
    payload = {
        "purpose": "Implement the requested bounded code change without altering scientific variables.",
        "nearest_existing_modules": ["src/base.py", "src/helper.py"],
        "reuse_attempt": (
            "The existing base and helper modules were inspected first, and their public functions "
            "remain the preferred implementation path wherever they cover the requested behavior."
        ),
        "why_existing_code_is_insufficient": (
            "The requested behavior requires a distinct checked boundary that is not represented by "
            "the existing functions, while all reusable execution and helper logic remains shared."
        ),
        "why_change_cannot_be_under_100_lines": (
            "The complete change includes deterministic validation, exact-head approval binding, and "
            "regression coverage; reducing it below the threshold would omit required failure paths."
        ),
        "file_responsibilities": {
            path: "This path contains a direct part of the requested implementation or its regression coverage."
            for path in sorted(changed_files)
        },
        "new_python_files": {
            path: {
                "closest_existing_module": "src/helper.py",
                "why_new_module_required": (
                    "The new module has a separately invokable responsibility and cannot be placed in the "
                    "existing helper without coupling unrelated callers or changing its stable interface."
                ),
                "intended_reuse_by": "Future callers use this single module instead of copying its implementation.",
            }
            for path in sorted(new_files)
        },
        "tests": ["python -m pytest -q tests/test_code_change_budget.py"],
    }
    return budget.START + "\n" + json.dumps(payload, indent=2) + "\n" + budget.END


def test_exactly_100_python_churn_is_automatic(repo: Path) -> None:
    base = _head(repo)
    _replace_prefix(repo, 50, "changed")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "change fifty lines")
    analysis = budget.analyze(repo, base, _head(repo))
    assert analysis["additions"] == 50
    assert analysis["deletions"] == 50
    assert analysis["churn"] == 100
    assert analysis["requires_human_approval"] is False


def test_more_than_100_python_churn_requires_human(repo: Path) -> None:
    base = _head(repo)
    _replace_prefix(repo, 51, "changed")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "change fifty one lines")
    analysis = budget.analyze(repo, base, _head(repo))
    assert analysis["churn"] == 102
    assert analysis["requires_human_approval"] is True
    assert "python_churn_102_exceeds_100" in analysis["approval_reasons"]


def test_one_line_new_python_file_always_requires_human(repo: Path) -> None:
    base = _head(repo)
    (repo / "src/new.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "new python")
    analysis = budget.analyze(repo, base, _head(repo))
    assert analysis["churn"] == 1
    assert analysis["added_files"] == ["src/new.py"]
    assert analysis["requires_human_approval"] is True


def test_python_delete_and_rename_always_require_human(repo: Path) -> None:
    base = _head(repo)
    _git(repo, "mv", "src/helper.py", "src/helper_renamed.py")
    (repo / "src/base.py").unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "rename and delete")
    analysis = budget.analyze(repo, base, _head(repo))
    assert analysis["deleted_files"] == ["src/base.py"]
    assert analysis["renamed_files"] == [["src/helper.py", "src/helper_renamed.py"]]
    assert analysis["requires_human_approval"] is True


def test_budget_is_cumulative_across_commits(repo: Path) -> None:
    base = _head(repo)
    _replace_prefix(repo, 30, "first")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "first")
    path = repo / "src/base.py"
    lines = path.read_text(encoding="utf-8").splitlines()
    for index in range(30, 60):
        lines[index] = f"second_{index} = {index + 2000}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "second")
    analysis = budget.analyze(repo, base, _head(repo))
    assert analysis["churn"] == 120
    assert analysis["requires_human_approval"] is True


def test_incomplete_large_change_justification_is_rejected(repo: Path) -> None:
    base = _head(repo)
    _replace_prefix(repo, 51, "changed")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "large")
    body = budget.START + '\n{"purpose": "too short"}\n' + budget.END
    report = budget.evaluate(repo, base, _head(repo), body, {})
    assert report["decision"] == "REJECT_JUSTIFICATION"
    assert report["justification_errors"]


def test_approval_is_bound_to_head_and_justification_hash(repo: Path) -> None:
    base = _head(repo)
    _replace_prefix(repo, 51, "changed")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "large")
    head = _head(repo)
    body = _valid_justification({"src/base.py"})
    first = budget.evaluate(repo, base, head, body, {})
    assert first["decision"] == "NEEDS_HUMAN_APPROVAL"
    assert first["expected_human_approval_context"]

    statuses = {
        "statuses": [
            {
                "context": first["expected_human_approval_context"],
                "state": "success",
            }
        ]
    }
    approved = budget.evaluate(repo, base, head, body, statuses)
    assert approved["decision"] == "PASS_HUMAN_APPROVED"

    changed_body = body.replace("bounded code change", "bounded and separately reviewed code change")
    stale = budget.evaluate(repo, base, head, changed_body, statuses)
    assert stale["decision"] == "NEEDS_HUMAN_APPROVAL"
    assert stale["expected_human_approval_context"] != first["expected_human_approval_context"]


def test_evaluator_uses_trusted_base_and_posts_stable_status() -> None:
    source = (
        Path(__file__).resolve().parents[1] / ".github/workflows/code-change-budget.yml"
    ).read_text(encoding="utf-8")
    assert "pull_request_target:" in source
    assert "ref: ${{ env.BASE_SHA }}" in source
    assert "persist-credentials: false" in source
    assert 'git fetch --no-tags origin "$HEAD_SHA"' in source
    assert "statuses: write" in source
    assert "drpo/code-change-budget" in source
    assert "pull_request:\n" not in source


def test_manual_approval_is_exact_head_and_default_branch_bound() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / ".github/workflows/approve-large-code-change.yml"
    ).read_text(encoding="utf-8")
    assert "workflow_dispatch:" in source
    assert "expected_head_sha:" in source
    assert "actual_head" in source
    assert 'if [[ "$actual_head" != "$EXPECTED_HEAD_SHA" ]]' in source
    assert "--human-approved" in source
    assert "expected_human_approval_context" in source
    assert "ref: ${{ github.event.repository.default_branch }}" in source
