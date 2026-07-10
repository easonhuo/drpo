from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pr-gate-observer.yml"


def _load_workflow() -> dict[str, Any]:
    payload = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(payload, dict)
    return payload


def _all_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result: list[str] = []
        for key, item in value.items():
            result.extend(_all_strings(key))
            result.extend(_all_strings(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_all_strings(item))
        return result
    return []


def test_workflow_has_exact_pull_request_trigger_and_read_only_permissions() -> None:
    workflow = _load_workflow()

    assert set(workflow) == {"name", "on", "permissions", "concurrency", "jobs"}
    assert workflow["on"] == {
        "pull_request": {"types": ["opened", "synchronize", "reopened"]}
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert "push" not in workflow["on"]
    assert "pull_request_target" not in workflow["on"]
    assert "schedule" not in workflow["on"]
    assert "workflow_dispatch" not in workflow["on"]


def test_workflow_cancels_superseded_runs_and_calls_observer_with_exact_pr_shas() -> None:
    workflow = _load_workflow()

    assert workflow["concurrency"] == {
        "group": "pr-gate-observer-${{ github.event.pull_request.number }}",
        "cancel-in-progress": "true",
    }
    observe = workflow["jobs"]["observe"]
    assert observe["runs-on"] == "ubuntu-latest"
    assert observe["timeout-minutes"] == "90"

    steps = observe["steps"]
    checkout = next(step for step in steps if step.get("uses") == "actions/checkout@v4")
    assert checkout["with"]["fetch-depth"] == "0"

    collector = next(step for step in steps if step.get("id") == "observer")
    command = collector["run"]
    assert "python scripts/run_pr_gate_observer.py" in command
    assert '--base "${{ github.event.pull_request.base.sha }}"' in command
    assert '--head "${{ github.event.pull_request.head.sha }}"' in command
    assert "--output-dir" in command
    assert "--classification-json" not in command


def test_workflow_always_uploads_short_lived_artifacts_without_write_capabilities() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["observe"]["steps"]

    summary = next(step for step in steps if step.get("name") == "Publish observer summary")
    assert summary["if"] == "always()"
    assert "GITHUB_STEP_SUMMARY" in summary["run"]

    upload = next(step for step in steps if step.get("uses") == "actions/upload-artifact@v4")
    assert upload["if"] == "always()"
    assert upload["with"]["retention-days"] == "14"
    assert upload["with"]["if-no-files-found"] == "warn"
    assert upload["with"]["path"] == "${{ steps.observer.outputs.output_dir }}"

    text = "\n".join(_all_strings(workflow)).lower()
    assert "${{ secrets." not in text
    assert "pull-requests: write" not in text
    assert "contents: write" not in text
    assert "issues: write" not in text
    assert "gh pr merge" not in text
    assert "github-script" not in text
