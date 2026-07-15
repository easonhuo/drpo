from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

import dev_integration_finalize as finalizer
import dev_integration_write_path as writer
import integrate_dev_branch as planner
from prepare_dev_pilot_registration import prepare
from validate_dev_integration import json_hash

BRANCH = "dev/gov-dev-pilot-registration-fastpath-01"
CLAIM = "GOV-DEV-PILOT-REGISTRATION-FASTPATH-01"
INTEGRATION_ID = "GOV-DEV-PILOT-FASTPATH-REAL-SHADOW-01"
UPDATE_ID = "GOV-DEV-PILOT-FASTPATH-REAL-SHADOW-REGISTRATION-01"
TARGET_ID = "EXT-H-E7-BENCH-01"
MAIN_BRANCH = "shadow/gov-dev-integration-01-reg-main"
DEV_BRANCH = "shadow/gov-dev-integration-01-reg-dev"
MAIN_SHA = "ead84d39c7df8c77de82e17d6fde27028582ff15"
DEV_SHA = "17a7975c4fd0b0fb7058fd44bd6e725c6c1559ae"
SUMMARY_PREFIX = "PILOT_FASTPATH_REAL_SHADOW_JSON="
HEADING = [
    "0. 研究与执行原则（每次新会话首先阅读）",
    "0.1 当前执行门禁",
]
PATHS = {
    "configs/e7_canonical_exp_horizon_joint_grid_v1.json": "add",
    "docs/e7_canonical_exp_horizon_joint_v1.md": "add",
    "runspecs/ready/E7_EXP_HORIZON_JOINT_20260712.yaml": "add",
    "scripts/run_e7_canonical_exp_horizon_joint.py": "add",
    "scripts/run_e7_canonical_exp_horizon_joint_one_click.sh": "add",
    "src/drpo/e7_canonical_exp_horizon_grid.py": "add",
    "tests/test_e7_canonical_exp_horizon_joint.py": "add",
}


def timed(function: Callable[[], Any]) -> tuple[Any, float]:
    started = time.monotonic()
    value = function()
    return value, round(time.monotonic() - started, 6)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=1000),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remote_sha(remote: str, branch: str) -> str:
    return planner.resolve_ref(remote, f"refs/heads/{branch}")


def verify_real_refs(remote: str) -> None:
    assert remote_sha(remote, MAIN_BRANCH) == MAIN_SHA
    assert remote_sha(remote, DEV_BRANCH) == DEV_SHA


def operations_for(inspection_root: Path, remote: str) -> list[dict[str, Any]]:
    inspection_root.mkdir(parents=True)
    bare = planner.create_audit_repo(
        inspection_root,
        remote,
        f"refs/heads/{MAIN_BRANCH}",
        f"refs/heads/{DEV_BRANCH}",
    )
    actual = planner.diff_changes(bare, MAIN_SHA, DEV_SHA)
    found = {item["source_path"]: item["kind"] for item in actual}
    assert found == PATHS
    assert all(item["source_path"] == item["destination_path"] for item in actual)

    operations: list[dict[str, Any]] = []
    for path, kind in sorted(found.items()):
        old = planner.tree_entry(bare, MAIN_SHA, path)
        new = planner.tree_entry(bare, DEV_SHA, path)
        assert new is not None and new["type"] == "blob"
        assert new["mode"] in planner.MODES
        assert kind == "add" and old is None
        operations.append(
            {
                "op": kind,
                "source_path": path,
                "destination_path": path,
                "expected_blob_sha": new["sha"],
                "expected_old_blob_sha": None,
                "expected_mode": new["mode"],
            }
        )
    return operations


def clone_at(source: Path, destination: Path, commit: str) -> None:
    writer.git(
        ["clone", "--no-hardlinks", "--shared", str(source), str(destination)],
        phase="fastpath_shadow_checkout",
        code="SOURCE_UNRESOLVED",
        timeout=300,
    )
    writer.git(
        ["checkout", "--detach", commit],
        cwd=destination,
        phase="fastpath_shadow_checkout",
        code="SOURCE_UNRESOLVED",
    )


def registry_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiments = payload["experiments"]
    return {item["id"]: item for item in experiments}


def observation() -> dict[str, Any]:
    return {
        "status": "pilot",
        "scope": "local_unpublished_fastpath_v1_shadow",
        "source_commit": DEV_SHA,
        "scientific_state_changed": False,
        "method_ranking": "not_assessed",
        "convergence": "not_assessed",
        "task_performance_collapse": "not_assessed",
        "support_boundary": "not_assessed",
        "nan_inf_numerical_failure": "not_assessed",
        "publication_allowed": False,
    }


def pilot_spec(operations: list[dict[str, Any]], before: dict[str, Any]) -> dict[str, Any]:
    replacement = copy.deepcopy(before)
    replacement["fastpath_real_shadow_observation"] = observation()
    return {
        "schema_version": 1,
        "preparation_id": INTEGRATION_ID,
        "source": {
            "repository": "easonhuo/drpo",
            "remote": "origin",
            "main_ref": f"refs/heads/{MAIN_BRANCH}",
            "expected_main_sha": MAIN_SHA,
            "dev_branch": DEV_BRANCH,
            "expected_dev_sha": DEV_SHA,
            "result_commit_sha": DEV_SHA,
            "result_git_dirty": False,
        },
        "subject": {
            "experiment_id": TARGET_ID,
            "governance_claims": [CLAIM],
        },
        "implementation": {"operations": operations},
        "review": {
            "reviewer_id": "chatgpt-reviewer-gov-dev-pilot-fastpath",
            "decision_token": "real-v1-shadow-reviewed-17a7975c",
            "decision": {
                "approved": True,
                "code_integration_eligible": True,
                "evidence_level": "pilot",
                "result_status": "pilot",
                "claim_support_level": "diagnostic",
                "terminal_audit": "not_required",
                "task_performance_collapse": "not_assessed",
                "support_boundary": "not_assessed",
                "numerical_failure": "not_assessed",
            },
            "limitations": [
                "Local unpublished governance shadow only; the READY candidate must not be pushed or merged.",
                "The source pilot does not establish a method ranking, convergence, or scientific result.",
                "Only fastpath_real_shadow_observation may be added to EXT-H-E7-BENCH-01.",
            ],
            "unresolved": [
                "Measured end-to-end efficiency on a future production registration remains unknown."
            ],
        },
        "registration": {
            "mode": "replace_experiment",
            "update_id": UPDATE_ID,
            "expected_before_semantic_sha256": json_hash(before),
            "experiment": replacement,
            "handoff_operations": [
                {
                    "operation_id": "append-fastpath-real-shadow",
                    "op": "append_to_section",
                    "heading_path": HEADING,
                    "block_id": "pilot-fastpath-real-v1-shadow",
                    "content": (
                        f"- `{INTEGRATION_ID}` records a local-only pilot integration "
                        f"observation under `{TARGET_ID}` solely to validate the PR-A "
                        "preparation adapter against V1. It changes no scientific status "
                        "and supports no ranking or convergence claim."
                    ),
                }
            ],
            "registry_changes": [
                {
                    "change_id": "record-fastpath-real-shadow-observation",
                    "kind": "update_field",
                    "entity_id": TARGET_ID,
                    "field_path": ["fastpath_real_shadow_observation"],
                    "from": None,
                    "to": observation(),
                    "reason": "Validate fastpath-to-V1 compatibility without scientific change.",
                    "evidence": [
                        "experiments/registry.yaml",
                        "docs/e7_canonical_exp_horizon_joint_v1.md",
                    ],
                }
            ],
        },
    }


def gate_summary(attempt: Path) -> dict[str, Any]:
    report = read_json(attempt / "GATE_REPORT.json")
    selector = report.get("selector_plan")
    return {
        "selected_mode": (
            selector.get("selected_mode") if isinstance(selector, dict) else None
        ),
        "passed_count": report["passed_count"],
        "failed_count": report["failed_count"],
        "first_blocker": report["first_blocker"],
        "durations": {
            item["label"]: item["duration_seconds"] for item in report["outcomes"]
        },
    }


def test_real_fastpath_registration_shadow_reaches_local_ready(tmp_path: Path) -> None:
    if os.environ.get("GITHUB_HEAD_REF") != BRANCH:
        pytest.skip("real fastpath shadow runs only on the dedicated PR-A branch")

    repo = Path(__file__).resolve().parents[1]
    remote = str(
        writer.git(
            ["remote", "get-url", "origin"],
            cwd=repo,
            phase="fastpath_shadow_preflight",
            code="SOURCE_UNRESOLVED",
        )
    ).strip()
    verify_real_refs(remote)

    operations = operations_for(tmp_path / "inspection", remote)
    historical_repo = tmp_path / "historical-main"
    review_repo = tmp_path / "review-repo"
    current_head = str(
        writer.git(
            ["rev-parse", "HEAD"],
            cwd=repo,
            phase="fastpath_shadow_preflight",
            code="SOURCE_UNRESOLVED",
        )
    ).strip()
    clone_at(repo, historical_repo, MAIN_SHA)
    clone_at(repo, review_repo, current_head)

    before_registry = registry_map(historical_repo / "experiments/registry.yaml")
    before = before_registry[TARGET_ID]
    spec_path = tmp_path / "DEV_PILOT_REGISTRATION_SPEC.yaml"
    write_yaml(spec_path, pilot_spec(operations, before))

    prepared_result, prepare_seconds = timed(
        lambda: prepare(historical_repo, spec_path, tmp_path / "preparations")
    )
    assert prepared_result["state"] == "PREPARED_INPUTS"
    assert prepared_result["network_used"] is False
    assert prepared_result["repository_modified"] is False
    prepared = Path(prepared_result["preparation_dir"])
    preparation_report = read_json(prepared / "PREPARATION_REPORT.json")
    assert preparation_report["status"] == "PASS"
    assert preparation_report["mode"] == "replace_experiment"

    shutil.copytree(
        prepared / "repository_overlay",
        review_repo,
        dirs_exist_ok=True,
    )
    integration_root = review_repo / "docs" / "integrations" / INTEGRATION_ID
    request_path = integration_root / "INTEGRATION_REQUEST.yaml"
    review_path = integration_root / "REVIEW_DECISION.yaml"
    assert request_path.is_file() and review_path.is_file()

    transaction_root = tmp_path / "transactions"
    args = argparse.Namespace(
        repo_root=str(review_repo),
        request=str(request_path),
        transaction_root=str(transaction_root),
        keep_audit_repo=False,
        json=True,
    )
    with redirect_stdout(StringIO()):
        plan_result, plan_seconds = timed(lambda: planner.plan(args))
    assert plan_result == 0
    attempt = transaction_root / INTEGRATION_ID / "attempt-0001"

    prepared_tx, v1_prepare_seconds = timed(lambda: writer.prepare_transaction(attempt))
    assert prepared_tx["state"] == "PREPARED"
    for filename in ("REGISTRATION_INTENT.yaml", "REGISTRATION_APPROVAL.yaml"):
        shutil.copy2(prepared / "transaction_inputs" / filename, attempt / filename)

    source_lock = read_json(attempt / "SOURCE_LOCK.json")
    approval = yaml.safe_load(
        (attempt / "REGISTRATION_APPROVAL.yaml").read_text(encoding="utf-8")
    )
    assert source_lock["request_sha256"] == sha256(request_path)
    assert source_lock["review_decision_sha256"] == sha256(review_path)
    assert approval["request_sha256"] == source_lock["request_sha256"]
    assert approval["review_decision_sha256"] == source_lock[
        "review_decision_sha256"
    ]
    assert approval["intent_sha256"] == sha256(
        attempt / "REGISTRATION_INTENT.yaml"
    )

    normalized, normalize_seconds = timed(
        lambda: finalizer.normalize_transaction(attempt)
    )
    gated, gate_seconds = timed(lambda: finalizer.gate_transaction(attempt))
    ready, finalize_seconds = timed(lambda: finalizer.finalize_transaction(attempt))
    assert normalized["state"] == "NORMALIZED"
    assert gated["state"] == "REQUIRED_GATES_PASSED"
    assert ready["state"] == "READY"

    transaction = read_json(attempt / "TRANSACTION.json")
    normalization = read_json(attempt / "NORMALIZATION_REPORT.json")
    integration_repo = attempt / "integration-repo"
    candidate = ready["ready_commit_sha"]
    assert transaction["state"] == "READY"
    assert str(
        writer.git(
            ["rev-list", "--parents", "-n", "1", candidate],
            cwd=integration_repo,
            phase="fastpath_shadow_verify",
            code="HEAD_DRIFT",
        )
    ).strip().split() == [candidate, MAIN_SHA]
    assert not writer.git(
        ["status", "--porcelain=v1"],
        cwd=integration_repo,
        phase="fastpath_shadow_verify",
        code="WORKTREE_DIRTY",
    )

    after_registry = registry_map(integration_repo / "experiments/registry.yaml")
    assert set(after_registry) == set(before_registry)
    assert all(
        after_registry[key] == value
        for key, value in before_registry.items()
        if key != TARGET_ID
    )
    expected_target = copy.deepcopy(before)
    expected_target["fastpath_real_shadow_observation"] = observation()
    assert after_registry[TARGET_ID] == expected_target
    assert after_registry[TARGET_ID]["status"] == before["status"]

    delta = integration_repo / "docs" / "handoff_deltas" / UPDATE_ID
    assert (delta / "HANDOFF_DELTA.yaml").is_file()
    assert (delta / "MATERIALIZATION_REPORT.json").is_file()
    allowed = {
        *PATHS,
        "experiments/registry.yaml",
        "docs/handoff.md",
        f"docs/handoff_deltas/{UPDATE_ID}/HANDOFF_DELTA.yaml",
        f"docs/handoff_deltas/{UPDATE_ID}/MATERIALIZATION_REPORT.json",
    }
    changed = set(normalization["changed_paths"])
    assert allowed.issubset(changed)
    assert all(
        path.startswith("docs/handoff_shadow/stage4/minimal/generated/")
        for path in changed - allowed
    )

    verify_real_refs(remote)
    summary = {
        "schema_version": 1,
        "claim": CLAIM,
        "status": "PASS",
        "integration_id": INTEGRATION_ID,
        "locked_refs": {
            "main_branch": MAIN_BRANCH,
            "main_sha": MAIN_SHA,
            "dev_branch": DEV_BRANCH,
            "dev_sha": DEV_SHA,
        },
        "states": {
            "adapter": prepared_result["state"],
            "prepare": prepared_tx["state"],
            "normalize": normalized["state"],
            "gate": gated["state"],
            "finalize": ready["state"],
        },
        "ready_commit_sha": candidate,
        "durations_seconds": {
            "adapter_prepare": prepare_seconds,
            "v1_plan": plan_seconds,
            "v1_prepare": v1_prepare_seconds,
            "v1_normalize": normalize_seconds,
            "v1_gate": gate_seconds,
            "v1_finalize": finalize_seconds,
        },
        "gate": gate_summary(attempt),
        "acceptance": {
            "real_refs_unchanged": True,
            "local_ready": True,
            "scientific_state_preserved": True,
            "published_candidate": False,
            "task_performance_collapse_changed": False,
            "support_boundary_changed": False,
            "numerical_failure_changed": False,
            "method_ranking_created": False,
            "convergence_claim_created": False,
        },
    }
    assert all(
        value is True
        for key, value in summary["acceptance"].items()
        if key != "published_candidate"
    )
    assert summary["acceptance"]["published_candidate"] is False
    print(SUMMARY_PREFIX + json.dumps(summary, sort_keys=True, separators=(",", ":")))
