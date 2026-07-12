from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
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

BRANCH = "dev/gov-dev-branch-integration-01-batch3"
CLAIM = "GOV-DEV-BRANCH-INTEGRATION-01"
TARGET_ID = "EXT-H-E7-BENCH-01"
CODE_MAIN = "17a7975c4fd0b0fb7058fd44bd6e725c6c1559ae"
CODE_DEV = "83ae2545406cc17cdfaa9fa5f240f0dddd7e2d04"
REG_MAIN = "ead84d39c7df8c77de82e17d6fde27028582ff15"
REG_DEV = "17a7975c4fd0b0fb7058fd44bd6e725c6c1559ae"
REG_UPDATE = "GOV-DEV-INTEGRATION-B3-REGISTRATION-01"
ROLLBACK_UPDATE = "GOV-DEV-INTEGRATION-B3-ROLLBACK-01"
HEADING = [
    "0. 研究与执行原则（每次新会话首先阅读）",
    "0.1 当前执行门禁",
]

CODE_PATHS = {
    "docs/dev_branch_integration_protocol.md": "modify",
    "docs/integrations/README.md": "modify",
    "docs/scopes/GOV-DEV-BRANCH-INTEGRATION-01-BATCH2B-NOTE.md": "add",
    "docs/templates/REGISTRATION_APPROVAL.yaml": "add",
    "docs/templates/REGISTRATION_INTENT.yaml": "add",
    "scripts/dev_integration_finalize.py": "add",
    "scripts/dev_integration_finalize_core.py": "add",
    "tests/test_dev_integration_finalize.py": "add",
    "tests/test_dev_integration_finalize_recovery.py": "add",
}
REG_PATHS = {
    "configs/e7_canonical_exp_horizon_joint_grid_v1.json": "add",
    "docs/e7_canonical_exp_horizon_joint_v1.md": "add",
    "runspecs/ready/E7_EXP_HORIZON_JOINT_20260712.yaml": "add",
    "scripts/run_e7_canonical_exp_horizon_joint.py": "add",
    "scripts/run_e7_canonical_exp_horizon_joint_one_click.sh": "add",
    "src/drpo/e7_canonical_exp_horizon_grid.py": "add",
    "tests/test_e7_canonical_exp_horizon_joint.py": "add",
}

SPECS = {
    "code_only": {
        "integration_id": "GOV-DEV-INTEGRATION-B3-CODE-01",
        "main_branch": "shadow/gov-dev-integration-01-code-main",
        "main_sha": CODE_MAIN,
        "dev_branch": "shadow/gov-dev-integration-01-code-dev",
        "dev_sha": CODE_DEV,
        "paths": CODE_PATHS,
        "review": (
            "docs/integrations/GOV-DEV-INTEGRATION-B3-CODE-01/"
            "REVIEW_DECISION.yaml"
        ),
        "experiment_ids": [],
    },
    "registration": {
        "integration_id": "GOV-DEV-INTEGRATION-B3-REG-01",
        "main_branch": "shadow/gov-dev-integration-01-reg-main",
        "main_sha": REG_MAIN,
        "dev_branch": "shadow/gov-dev-integration-01-reg-dev",
        "dev_sha": REG_DEV,
        "paths": REG_PATHS,
        "review": (
            "docs/integrations/GOV-DEV-INTEGRATION-B3-REG-01/"
            "REVIEW_DECISION.yaml"
        ),
        "experiment_ids": [TARGET_ID],
    },
    "rollback": {
        "integration_id": "GOV-DEV-INTEGRATION-B3-ROLLBACK-01",
        "main_branch": "shadow/gov-dev-integration-01-reg-main",
        "main_sha": REG_MAIN,
        "dev_branch": "shadow/gov-dev-integration-01-reg-dev",
        "dev_sha": REG_DEV,
        "paths": REG_PATHS,
        "review": (
            "docs/integrations/GOV-DEV-INTEGRATION-B3-ROLLBACK-01/"
            "REVIEW_DECISION.yaml"
        ),
        "experiment_ids": [TARGET_ID],
    },
}


def timed(function: Callable[[], Any]) -> tuple[Any, float]:
    started = time.monotonic()
    value = function()
    return value, round(time.monotonic() - started, 6)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False, width=1000),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_hash(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def remote_sha(remote: str, branch: str) -> str:
    return planner.resolve_ref(remote, f"refs/heads/{branch}")


def verify_refs(remote: str, spec: dict[str, Any]) -> None:
    assert remote_sha(remote, spec["main_branch"]) == spec["main_sha"]
    assert remote_sha(remote, spec["dev_branch"]) == spec["dev_sha"]


def operations_for(
    inspection_root: Path,
    remote: str,
    name: str,
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    attempt_root = inspection_root / name
    attempt_root.mkdir(parents=True)
    bare = planner.create_audit_repo(
        attempt_root,
        remote,
        f"refs/heads/{spec['main_branch']}",
        f"refs/heads/{spec['dev_branch']}",
    )
    actual = planner.diff_changes(bare, spec["main_sha"], spec["dev_sha"])
    found = {item["source_path"]: item["kind"] for item in actual}
    assert all(item["source_path"] == item["destination_path"] for item in actual)
    assert found == spec["paths"]
    operations = []
    for path, kind in sorted(found.items()):
        old = planner.tree_entry(bare, spec["main_sha"], path)
        new = planner.tree_entry(bare, spec["dev_sha"], path)
        assert new is not None and new["type"] == "blob"
        assert new["mode"] in planner.MODES
        assert (kind == "add" and old is None) or (kind == "modify" and old)
        operations.append(
            {
                "op": kind,
                "source_path": path,
                "destination_path": path,
                "expected_blob_sha": new["sha"],
                "expected_old_blob_sha": None if old is None else old["sha"],
                "expected_mode": new["mode"],
            }
        )
    return operations


def request(spec: dict[str, Any], operations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "integration_id": spec["integration_id"],
        "source": {
            "repository": "easonhuo/drpo",
            "remote": "origin",
            "main_ref": f"refs/heads/{spec['main_branch']}",
            "expected_main_sha": spec["main_sha"],
            "dev_branch": spec["dev_branch"],
            "expected_dev_sha": spec["dev_sha"],
            "result_commit_sha": spec["dev_sha"],
            "result_git_dirty": False,
        },
        "subject": {
            "experiment_ids": spec["experiment_ids"],
            "governance_claims": [CLAIM],
        },
        "files": {"operations": operations},
        "review": {"decision_file": spec["review"]},
        "checks": {"requested_tier": "auto"},
    }


def plan_prepare(
    repo: Path,
    root: Path,
    name: str,
    spec: dict[str, Any],
    operations: list[dict[str, Any]],
) -> tuple[Path, dict[str, float]]:
    request_path = root / "requests" / spec["integration_id"] / "request.yaml"
    write_yaml(request_path, request(spec, operations))
    args = argparse.Namespace(
        repo_root=str(repo),
        request=str(request_path),
        transaction_root=str(root),
        keep_audit_repo=False,
        json=True,
    )
    with redirect_stdout(StringIO()):
        result, plan_seconds = timed(lambda: planner.plan(args))
    assert result == 0
    attempt = root / spec["integration_id"] / "attempt-0001"
    prepared, prepare_seconds = timed(lambda: writer.prepare_transaction(attempt))
    assert prepared["state"] == "PREPARED"
    return attempt, {"plan": plan_seconds, "prepare": prepare_seconds}


def registry_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiments = payload["experiments"]
    return {item["id"]: item for item in experiments}


def observation() -> dict[str, Any]:
    return {
        "status": "pilot",
        "scope": "local_unpublished_pipeline_shadow",
        "source_commit": REG_DEV,
        "scientific_state_changed": False,
        "method_ranking": "not_assessed",
        "convergence": "not_assessed",
        "task_performance_collapse": "not_assessed",
        "support_boundary": "not_assessed",
        "nan_inf_numerical_failure": "not_assessed",
        "publication_allowed": False,
    }


def install_registration(attempt: Path, *, rollback: bool) -> None:
    source = read_json(attempt / "SOURCE_LOCK.json")
    audit = read_json(attempt / "SCOPE_AUDIT.json")
    before = registry_map(attempt / "integration-repo/experiments/registry.yaml")[
        TARGET_ID
    ]
    replacement = copy.deepcopy(before)
    replacement["batch3_shadow_observation"] = observation()
    intent = {
        "schema_version": 1,
        "integration_id": source["integration_id"],
        "mode": "authoritative_delta",
        "update_id": ROLLBACK_UPDATE if rollback else REG_UPDATE,
        "registry_mutation": {
            "kind": "replace_experiment",
            "experiment_id": TARGET_ID,
            "expected_before_semantic_sha256": semantic_hash(before),
            "experiment": replacement,
        },
        "handoff_operations": [
            {
                "operation_id": "append-batch3-shadow",
                "op": "append_to_section",
                "heading_path": HEADING,
                "block_id": "batch3-shadow-pilot",
                "content": (
                    f"- Batch 3 records a local-only **pilot** integration observation "
                    f"under `{TARGET_ID}` solely to validate plumbing. It changes no "
                    "scientific status and supports no ranking or convergence claim."
                ),
            }
        ],
        "registry_changes": [
            {
                "change_id": "record-batch3-shadow-observation",
                "kind": "unsupported_batch3_fault" if rollback else "update_field",
                "entity_id": TARGET_ID,
                "field_path": ["batch3_shadow_observation"],
                "from": None,
                "to": observation(),
                "reason": "Validate local transaction registration without scientific change.",
                "evidence": [
                    "experiments/registry.yaml",
                    "docs/e7_canonical_exp_horizon_joint_v1.md",
                ],
            }
        ],
    }
    intent_path = attempt / "REGISTRATION_INTENT.yaml"
    write_yaml(intent_path, intent)
    write_yaml(
        attempt / "REGISTRATION_APPROVAL.yaml",
        {
            "schema_version": 1,
            "integration_id": source["integration_id"],
            "intent_sha256": sha256(intent_path),
            "request_sha256": source["request_sha256"],
            "review_decision_sha256": source["review_decision_sha256"],
            "reviewer": audit["review_decision"]["reviewer"],
        },
    )


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


def run_valid(
    repo: Path,
    remote: str,
    root: Path,
    inspection: Path,
    name: str,
    *,
    registration: bool,
) -> dict[str, Any]:
    spec = SPECS[name]
    verify_refs(remote, spec)
    operations = operations_for(inspection, remote, name, spec)
    attempt, durations = plan_prepare(repo, root, name, spec, operations)
    if registration:
        install_registration(attempt, rollback=False)
    normalized, durations["normalize"] = timed(
        lambda: finalizer.normalize_transaction(attempt)
    )
    gated, durations["gate"] = timed(lambda: finalizer.gate_transaction(attempt))
    ready, durations["finalize"] = timed(
        lambda: finalizer.finalize_transaction(attempt)
    )
    assert normalized["state"] == "NORMALIZED"
    assert gated["state"] == "REQUIRED_GATES_PASSED"
    assert ready["state"] == "READY"
    tx = read_json(attempt / "TRANSACTION.json")
    normalization = read_json(attempt / "NORMALIZATION_REPORT.json")
    integration_repo = attempt / "integration-repo"
    candidate = ready["ready_commit_sha"]
    assert tx["state"] == "READY"
    assert writer.git(
        ["rev-list", "--parents", "-n", "1", candidate],
        cwd=integration_repo,
        phase="batch3_verify",
        code="HEAD_DRIFT",
    ).strip().split() == [candidate, spec["main_sha"]]
    assert not writer.git(
        ["status", "--porcelain=v1"],
        cwd=integration_repo,
        phase="batch3_verify",
        code="WORKTREE_DIRTY",
    )
    if registration:
        before_text = writer.git(
            ["show", f"{spec['main_sha']}:experiments/registry.yaml"],
            cwd=integration_repo,
            phase="batch3_verify",
            code="REGISTRY_STRUCTURE_ERROR",
        )
        before_path = attempt / "base-registry.yaml"
        before_path.write_text(str(before_text), encoding="utf-8")
        before = registry_map(before_path)
        after = registry_map(integration_repo / "experiments/registry.yaml")
        assert set(before) == set(after)
        assert all(
            after[key] == value for key, value in before.items() if key != TARGET_ID
        )
        expected = copy.deepcopy(before[TARGET_ID])
        expected["batch3_shadow_observation"] = observation()
        assert after[TARGET_ID] == expected
        assert after[TARGET_ID]["status"] == before[TARGET_ID]["status"]
        delta = integration_repo / "docs/handoff_deltas" / REG_UPDATE
        assert (delta / "HANDOFF_DELTA.yaml").is_file()
        assert (delta / "MATERIALIZATION_REPORT.json").is_file()
        allowed = {
            *spec["paths"],
            "experiments/registry.yaml",
            "docs/handoff.md",
            f"docs/handoff_deltas/{REG_UPDATE}/HANDOFF_DELTA.yaml",
            f"docs/handoff_deltas/{REG_UPDATE}/MATERIALIZATION_REPORT.json",
        }
        changed = set(normalization["changed_paths"])
        assert allowed.issubset(changed)
        extra = changed - allowed
        assert all(
            path.startswith("docs/handoff_shadow/stage4/minimal/generated/")
            for path in extra
        )
    else:
        assert normalization["registration_mode"] == "code_only"
        assert normalization["changed_paths"] == sorted(spec["paths"])
    return {
        "name": name,
        "state": "READY",
        "attempt_dir": str(attempt),
        "ready_commit_sha": candidate,
        "stage_durations_seconds": durations,
        "gate": gate_summary(attempt),
    }


def run_rollback(
    repo: Path,
    remote: str,
    root: Path,
    inspection: Path,
) -> dict[str, Any]:
    spec = SPECS["rollback"]
    verify_refs(remote, spec)
    operations = operations_for(inspection, remote, "rollback", spec)
    attempt, durations = plan_prepare(repo, root, "rollback", spec, operations)
    source_commit = read_json(attempt / "PREPARE_REPORT.json")["source_commit_sha"]
    install_registration(attempt, rollback=True)
    started = time.monotonic()
    with pytest.raises(finalizer.FinalizeError):
        finalizer.normalize_transaction(attempt)
    durations["normalize_expected_failure"] = round(time.monotonic() - started, 6)
    tx = read_json(attempt / "TRANSACTION.json")
    diagnostic = read_json(attempt / "DIAGNOSTIC.json")
    integration_repo = attempt / "integration-repo"
    head = str(
        writer.git(
            ["rev-parse", "HEAD"],
            cwd=integration_repo,
            phase="batch3_rollback_verify",
            code="HEAD_DRIFT",
        )
    ).strip()
    clean = not bool(
        writer.git(
            ["status", "--porcelain=v1"],
            cwd=integration_repo,
            phase="batch3_rollback_verify",
            code="WORKTREE_DIRTY",
        )
    )
    assert tx["state"] == "BLOCKED"
    assert diagnostic["status"] == "FAIL"
    assert head == source_commit and clean
    assert not (attempt / "NORMALIZATION_REPORT.json").exists()
    verify_refs(remote, spec)
    return {
        "name": "rollback",
        "state": "BLOCKED",
        "error_code": diagnostic["error_code"],
        "phase": diagnostic["phase"],
        "source_commit_sha": source_commit,
        "restored_head_sha": head,
        "worktree_clean": clean,
        "expected_fault_intercepted": True,
        "stage_durations_seconds": durations,
    }


def test_real_batch3_dev_integration_shadows(tmp_path: Path) -> None:
    if os.environ.get("GITHUB_HEAD_REF") != BRANCH:
        pytest.skip("real-ref Batch 3 acceptance runs only on its dedicated PR branch")

    repo = Path(__file__).resolve().parents[1]
    remote = str(
        writer.git(
            ["remote", "get-url", "origin"],
            cwd=repo,
            phase="batch3_preflight",
            code="SOURCE_UNRESOLVED",
        )
    ).strip()
    root = tmp_path / "transactions"
    inspection = tmp_path / "inspection"
    root.mkdir()
    inspection.mkdir()
    code = run_valid(
        repo, remote, root, inspection, "code_only", registration=False
    )
    registration = run_valid(
        repo, remote, root, inspection, "registration", registration=True
    )
    rollback = run_rollback(repo, remote, root, inspection)
    for spec in SPECS.values():
        verify_refs(remote, spec)
    summary = {
        "schema_version": 1,
        "claim": CLAIM,
        "status": "PASS",
        "shadows": {"code_only": code, "registration": registration},
        "rollback": rollback,
        "metrics": {
            "valid_shadow_count": 2,
            "ready_shadow_count": 2,
            "expected_fault_interception_count": 1,
            "false_positive_count": 0,
            "unexpected_blocker_count": 0,
            "unique_expected_blocker_codes": [rollback["error_code"]],
            "scientific_state_upgrades": 0,
            "published_candidates": 0,
        },
        "acceptance": {
            "both_valid_shadows_ready": True,
            "rollback_restored": (
                rollback["source_commit_sha"] == rollback["restored_head_sha"]
                and rollback["worktree_clean"]
            ),
            "remote_refs_unchanged": True,
            "scientific_state_preserved": True,
            "default_merge_route_unchanged": True,
        },
    }
    assert all(summary["acceptance"].values())
    print(
        "BATCH3_SHADOW_SUMMARY_JSON="
        + json.dumps(summary, sort_keys=True, separators=(",", ":"))
    )
