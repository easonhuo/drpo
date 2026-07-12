from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

import dev_integration_finalize as finalizer


def run_git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode:
        raise AssertionError(proc.stderr or proc.stdout)
    return proc.stdout.strip()


def commit_all(repo: Path, message: str) -> str:
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-m", message)
    return run_git(repo, "rev-parse", "HEAD")


def registry_text() -> str:
    return """schema_version: 2
project:
  name: drpo
allowed_statuses: [not_run, pilot]
experiments:
- id: EXP-A
  status: not_run
  nested:
    value: 1
# preserve this unrelated comment
- id: EXP-B
  status: pilot
trailing:
  preserve: true
"""


def test_registry_add_preserves_unrelated_semantics_and_layout() -> None:
    before = registry_text()
    mutation = {
        "kind": "add_experiment",
        "experiment_id": "EXP-C",
        "expected_before_semantic_sha256": None,
        "experiment": {"id": "EXP-C", "status": "not_run", "claim": "new"},
    }
    after, report = finalizer.apply_registry_mutation(before, mutation)
    payload = yaml.safe_load(after)
    assert [item["id"] for item in payload["experiments"]] == [
        "EXP-A",
        "EXP-B",
        "EXP-C",
    ]
    assert payload["trailing"] == {"preserve": True}
    assert "# preserve this unrelated comment" in after
    assert report["experiment_id"] == "EXP-C"
    assert report["before_semantic_sha256"] is None


def test_registry_replace_uses_whole_sequence_item_span() -> None:
    before = registry_text()
    current = yaml.safe_load(before)["experiments"][0]
    replacement = {"id": "EXP-A", "status": "pilot", "nested": {"value": 2}}
    mutation = {
        "kind": "replace_experiment",
        "experiment_id": "EXP-A",
        "expected_before_semantic_sha256": finalizer.json_hash(current),
        "experiment": replacement,
    }
    after, report = finalizer.apply_registry_mutation(before, mutation)
    payload = yaml.safe_load(after)
    assert payload["experiments"][0] == replacement
    assert payload["experiments"][1] == {"id": "EXP-B", "status": "pilot"}
    assert "- - id:" not in after
    assert "# preserve this unrelated comment" in after
    assert report["before_semantic_sha256"] == finalizer.json_hash(current)


def test_registry_replace_rejects_wrong_before_hash() -> None:
    mutation = {
        "kind": "replace_experiment",
        "experiment_id": "EXP-A",
        "expected_before_semantic_sha256": "0" * 64,
        "experiment": {"id": "EXP-A", "status": "pilot"},
    }
    with pytest.raises(finalizer.FinalizeError) as caught:
        finalizer.apply_registry_mutation(registry_text(), mutation)
    assert caught.value.error_code == "IMMUTABILITY_ERROR"


def test_registration_intent_is_hash_and_subject_bound(tmp_path: Path) -> None:
    transaction = tmp_path / "attempt-0001"
    transaction.mkdir()
    context = {
        "integration_id": "GOV-TEST-INTEGRATION-01",
        "request": {
            "subject": {
                "experiment_ids": ["EXP-C"],
                "governance_claims": ["GOV-TEST-01"],
            }
        },
        "source": {
            "request_sha256": "1" * 64,
            "review_decision_sha256": "2" * 64,
        },
        "reviewer": {"id": "reviewer", "decision_token": "token"},
    }
    intent = {
        "schema_version": 1,
        "integration_id": context["integration_id"],
        "mode": "authoritative_delta",
        "update_id": "EXP-C-REGISTRATION-01",
        "registry_mutation": {
            "kind": "add_experiment",
            "experiment_id": "EXP-C",
            "expected_before_semantic_sha256": None,
            "experiment": {"id": "EXP-C", "status": "not_run"},
        },
        "handoff_operations": [],
        "registry_changes": [
            {
                "change_id": "add-exp-c",
                "kind": "add_entity",
                "entity_id": "EXP-C",
                "evidence": ["experiments/registry.yaml"],
            }
        ],
    }
    intent_path = transaction / finalizer.INTENT_FILE
    intent_path.write_text(yaml.safe_dump(intent, sort_keys=False), encoding="utf-8")
    approval = {
        "schema_version": 1,
        "integration_id": context["integration_id"],
        "intent_sha256": hashlib.sha256(intent_path.read_bytes()).hexdigest(),
        "request_sha256": context["source"]["request_sha256"],
        "review_decision_sha256": context["source"]["review_decision_sha256"],
        "reviewer": context["reviewer"],
    }
    (transaction / finalizer.APPROVAL_FILE).write_text(
        yaml.safe_dump(approval, sort_keys=False), encoding="utf-8"
    )
    validated = finalizer.validate_intent(transaction, context)
    assert validated is not None
    assert validated["mutation"]["experiment_id"] == "EXP-C"

    intent["registry_mutation"]["experiment_id"] = "EXP-OTHER"
    intent["registry_mutation"]["experiment"]["id"] = "EXP-OTHER"
    intent["registry_changes"][0]["entity_id"] = "EXP-OTHER"
    intent_path.write_text(yaml.safe_dump(intent, sort_keys=False), encoding="utf-8")
    approval["intent_sha256"] = hashlib.sha256(intent_path.read_bytes()).hexdigest()
    (transaction / finalizer.APPROVAL_FILE).write_text(
        yaml.safe_dump(approval, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(finalizer.FinalizeError) as caught:
        finalizer.validate_intent(transaction, context)
    assert caught.value.error_code == "SCOPE_VIOLATION"


def test_schema_v3_delta_hashes_exact_base() -> None:
    intent = {
        "update_id": "EXP-C-REGISTRATION-01",
        "handoff_operations": [],
        "registry_changes": [
            {
                "change_id": "add-exp-c",
                "kind": "add_entity",
                "entity_id": "EXP-C",
                "evidence": ["experiments/registry.yaml"],
            }
        ],
    }
    delta = finalizer.build_delta(
        main_sha="a" * 40,
        base_handoff="# Handoff\n",
        base_registry="schema_version: 2\nexperiments: []\n",
        after_registry="schema_version: 2\nexperiments:\n- id: EXP-C\n",
        intent=intent,
    )
    assert delta["schema_version"] == 3
    assert delta["base"]["commit"] == "a" * 40
    assert delta["registry"]["exact_base_after_sha256"] == finalizer.shadow.sha256_text(
        "schema_version: 2\nexperiments:\n- id: EXP-C\n"
    )
    assert delta["expected"][
        "exact_base_candidate_sha256"
    ] == finalizer.shadow.sha256_text("# Handoff\n")


def test_expected_final_scope_rejects_unapproved_normalizer_output() -> None:
    context = {
        "prepare": {
            "committed_changes": [
                {
                    "kind": "add",
                    "source_path": "src/example.py",
                    "destination_path": "src/example.py",
                }
            ]
        }
    }
    intent = {"update_id": "EXP-C-REGISTRATION-01", "handoff_operations": []}
    valid = [
        "src/example.py",
        "experiments/registry.yaml",
        "docs/handoff_deltas/EXP-C-REGISTRATION-01/HANDOFF_DELTA.yaml",
        "docs/handoff_deltas/EXP-C-REGISTRATION-01/MATERIALIZATION_REPORT.json",
    ]
    finalizer.expected_final_scope(context, intent, valid)
    with pytest.raises(finalizer.FinalizeError) as caught:
        finalizer.expected_final_scope(context, intent, [*valid, "docs/unapproved.md"])
    assert caught.value.error_code == "SCOPE_VIOLATION"


@contextmanager
def no_lock(_path: Path):
    yield


def test_code_only_normalize_gate_finalize_state_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "integration-repo"
    repo.mkdir()
    run_git(repo, "init", "-b", "main")
    run_git(repo, "config", "user.name", "Test User")
    run_git(repo, "config", "user.email", "test@example.com")
    (repo / "src").mkdir()
    (repo / "src/example.py").write_text("VALUE = 1\n", encoding="utf-8")
    main_sha = commit_all(repo, "base")
    run_git(repo, "checkout", "-b", "integration/test")
    (repo / "src/example.py").write_text("VALUE = 2\n", encoding="utf-8")
    source_commit = commit_all(repo, "source")

    transaction = tmp_path / "attempt-0001"
    transaction.mkdir()
    tx_path = transaction / "TRANSACTION.json"
    tx_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "PREPARED",
                "status": "PASS",
                "completed_states": [
                    "RECEIVED",
                    "SOURCE_LOCKED",
                    "REVIEWED",
                    "PREPARED",
                ],
            }
        ),
        encoding="utf-8",
    )
    context = {
        "transaction_dir": transaction,
        "transaction_path": tx_path,
        "transaction": json.loads(tx_path.read_text()),
        "source": {},
        "audit": {},
        "prepare": {
            "committed_changes": [
                {
                    "kind": "modify",
                    "source_path": "src/example.py",
                    "destination_path": "src/example.py",
                }
            ]
        },
        "request": {
            "subject": {"experiment_ids": [], "governance_claims": ["GOV-TEST"]}
        },
        "review": {},
        "reviewer": {"id": "reviewer", "decision_token": "token"},
        "integration_id": "GOV-TEST-INTEGRATION-01",
        "main_sha": main_sha,
        "dev_sha": "d" * 40,
        "source_commit_sha": source_commit,
        "repo": repo,
        "remote": "unused",
        "main_ref": "refs/heads/main",
        "dev_ref": "refs/heads/dev/test",
        "requested_gate_tier": "auto",
        "interrupted_normalization": False,
    }

    def refreshed_context(_transaction_dir: Path, _states: set[str]) -> dict:
        context["transaction"] = json.loads(tx_path.read_text(encoding="utf-8"))
        return context

    monkeypatch.setattr(finalizer, "locked", no_lock)
    monkeypatch.setattr(finalizer, "read_tx_context", refreshed_context)
    monkeypatch.setattr(
        finalizer, "trusted_main", lambda _context: tmp_path / "trusted"
    )
    monkeypatch.setattr(finalizer, "check_freshness", lambda _context, _phase: None)
    monkeypatch.setattr(
        finalizer,
        "authority_normalize",
        lambda _context, _trusted, _commit: {
            "status": "PASS",
            "mode": "delta",
            "normalization": "no_op",
        },
    )
    monkeypatch.setattr(
        finalizer,
        "authority_verify",
        lambda _context, _trusted: {"status": "PASS", "mode": "delta"},
    )

    normalized = finalizer.normalize_transaction(transaction)
    assert normalized["state"] == "NORMALIZED"
    normalized_commit = normalized["normalized_commit_sha"]
    normalized_tx = json.loads(tx_path.read_text(encoding="utf-8"))
    assert normalized_tx["state"] == "NORMALIZED"
    assert normalized_tx["normalization_report_sha256"] == finalizer.sha256(
        transaction / finalizer.NORMALIZATION_REPORT
    )
    assert run_git(
        repo, "rev-list", "--parents", "-n", "1", normalized_commit
    ).split() == [
        normalized_commit,
        main_sha,
    ]

    monkeypatch.setattr(finalizer.shutil, "which", lambda _name: "/usr/bin/ruff")

    def passing_gate(
        label: str,
        args: list[str],
        *,
        cwd: Path,
        log_dir: Path,
        timeout: int = 1800,
    ) -> dict:
        del timeout
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"{label}.log"
        stdout = ""
        if label == "test-selector-plan":
            stdout = json.dumps(
                {
                    "selected_mode": "fast",
                    "risk": "low",
                    "reason": "test",
                    "changed_paths": ["src/example.py"],
                    "matched_groups": ["test"],
                    "unknown_paths": [],
                    "pytest_targets": [],
                    "validators": [],
                    "changed_python_files": ["src/example.py"],
                    "full_commands": [],
                    "executed": [],
                }
            )
        log.write_text(stdout, encoding="utf-8")
        return {
            "label": label,
            "command": list(args),
            "cwd": str(cwd),
            "returncode": 0,
            "passed": True,
            "duration_seconds": 0.01,
            "stdout_tail": stdout,
            "stderr_tail": "",
            "log_file": str(log),
            "log_sha256": finalizer.sha256(log),
        }

    monkeypatch.setattr(finalizer, "gate_record", passing_gate)
    gated = finalizer.gate_transaction(transaction)
    assert gated["state"] == "REQUIRED_GATES_PASSED"
    gate_tx = json.loads(tx_path.read_text(encoding="utf-8"))
    assert gate_tx["gate_report_sha256"] == finalizer.sha256(
        transaction / finalizer.GATE_REPORT
    )

    ready = finalizer.finalize_transaction(transaction)
    assert ready["state"] == "READY"
    ready_tx = json.loads(tx_path.read_text(encoding="utf-8"))
    assert ready_tx["state"] == "READY"
    assert ready_tx["ready_commit_record_sha256"] == finalizer.sha256(
        transaction / finalizer.READY_COMMIT
    )
    assert run_git(repo, "status", "--porcelain=v1") == ""
