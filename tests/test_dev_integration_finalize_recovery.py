from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

import dev_integration_finalize as finalizer


@contextmanager
def no_lock(_path: Path):
    yield


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def fail_delegate(*_args, **_kwargs):
    raise AssertionError("completed-report recovery delegated to core")


def base_context(tmp_path: Path, tx_path: Path, integration_id: str) -> dict:
    return {
        "transaction_dir": tmp_path,
        "transaction": json.loads(tx_path.read_text()),
        "transaction_path": tx_path,
        "integration_id": integration_id,
        "main_sha": "1" * 40,
        "dev_sha": "2" * 40,
        "source_commit_sha": "3" * 40,
        "repo": tmp_path,
    }


def valid_log(tmp_path: Path, name: str = "gate") -> dict:
    log = tmp_path / "gate-logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("PASS\n", encoding="utf-8")
    return {
        "label": name,
        "command": ["true"],
        "cwd": str(tmp_path),
        "returncode": 0,
        "passed": True,
        "duration_seconds": 0.01,
        "stdout_tail": "PASS",
        "stderr_tail": "",
        "log_file": str(log),
        "log_sha256": finalizer.sha256(log),
    }


def test_completed_normalization_report_repairs_transaction_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tx_path = tmp_path / "TRANSACTION.json"
    integration_id = "GOV-RECOVERY-TEST-01"
    write_json(
        tx_path,
        {
            "schema_version": 1,
            "state": "PREPARED",
            "status": "PASS",
            "completed_states": ["PREPARED"],
        },
    )
    context = base_context(tmp_path, tx_path, integration_id)
    report_path = tmp_path / finalizer.NORMALIZATION_REPORT
    write_json(
        report_path,
        {
            "schema_version": 1,
            "status": "PASS",
            "state": "NORMALIZED",
            "integration_id": integration_id,
            "main_sha": context["main_sha"],
            "dev_sha": context["dev_sha"],
            "source_commit_sha": context["source_commit_sha"],
            "normalized_commit_sha": "a" * 40,
        },
    )
    monkeypatch.setattr(finalizer, "locked", no_lock)
    monkeypatch.setattr(finalizer, "read_tx_context", lambda *_: context)
    monkeypatch.setattr(finalizer, "trusted_main", lambda _context: tmp_path)
    monkeypatch.setattr(
        finalizer.core,
        "verify_normalized",
        lambda *_: {
            "status": "PASS",
            "state": "NORMALIZED",
            "integration_id": integration_id,
            "normalized_commit_sha": "a" * 40,
            "idempotent": True,
        },
    )
    monkeypatch.setattr(finalizer.core, "normalize_transaction", fail_delegate)

    result = finalizer.normalize_transaction(tmp_path)
    repaired = json.loads(tx_path.read_text())
    assert result["repaired_transaction_state"] is True
    assert repaired["state"] == "NORMALIZED"
    assert repaired["normalization_report_sha256"] == finalizer.sha256(report_path)


def test_completed_gate_report_repairs_transaction_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tx_path = tmp_path / "TRANSACTION.json"
    integration_id = "GOV-RECOVERY-TEST-02"
    normalized_commit = "b" * 40
    normalization_path = tmp_path / finalizer.NORMALIZATION_REPORT
    write_json(normalization_path, {"normalized_commit_sha": normalized_commit})
    write_json(
        tx_path,
        {
            "schema_version": 1,
            "state": "NORMALIZED",
            "status": "PASS",
            "completed_states": ["PREPARED", "NORMALIZED"],
            "normalization_report_sha256": finalizer.sha256(normalization_path),
        },
    )
    context = base_context(tmp_path, tx_path, integration_id)
    outcome = valid_log(tmp_path)
    gate_path = tmp_path / finalizer.GATE_REPORT
    write_json(
        gate_path,
        {
            "schema_version": 1,
            "status": "PASS",
            "state": "REQUIRED_GATES_PASSED",
            "integration_id": integration_id,
            "main_sha": context["main_sha"],
            "normalized_commit_sha": normalized_commit,
            "failed_count": 0,
            "first_blocker": None,
            "passed_count": 1,
            "outcomes": [outcome],
        },
    )
    monkeypatch.setattr(finalizer, "locked", no_lock)
    monkeypatch.setattr(finalizer, "read_tx_context", lambda *_: context)
    monkeypatch.setattr(finalizer.core, "ensure_clean", lambda *_: None)
    monkeypatch.setattr(finalizer.core, "gate_transaction", fail_delegate)

    result = finalizer.gate_transaction(tmp_path)
    repaired = json.loads(tx_path.read_text())
    assert result["repaired_transaction_state"] is True
    assert repaired["state"] == "REQUIRED_GATES_PASSED"
    assert repaired["gate_report_sha256"] == finalizer.sha256(gate_path)


def test_completed_ready_record_repairs_transaction_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tx_path = tmp_path / "TRANSACTION.json"
    integration_id = "GOV-RECOVERY-TEST-03"
    normalized_commit = "c" * 40
    normalization_path = tmp_path / finalizer.NORMALIZATION_REPORT
    write_json(normalization_path, {"normalized_commit_sha": normalized_commit})
    outcome = valid_log(tmp_path)
    gate_path = tmp_path / finalizer.GATE_REPORT
    write_json(
        gate_path,
        {
            "schema_version": 1,
            "status": "PASS",
            "state": "REQUIRED_GATES_PASSED",
            "integration_id": integration_id,
            "main_sha": "1" * 40,
            "normalized_commit_sha": normalized_commit,
            "failed_count": 0,
            "first_blocker": None,
            "passed_count": 1,
            "outcomes": [outcome],
        },
    )
    write_json(
        tx_path,
        {
            "schema_version": 1,
            "state": "REQUIRED_GATES_PASSED",
            "status": "PASS",
            "completed_states": [
                "PREPARED",
                "NORMALIZED",
                "REQUIRED_GATES_PASSED",
            ],
            "normalization_report_sha256": finalizer.sha256(normalization_path),
            "gate_report_sha256": finalizer.sha256(gate_path),
        },
    )
    context = base_context(tmp_path, tx_path, integration_id)
    authority = {"status": "PASS", "mode": "delta"}
    changed = ["src/example.py"]
    tree_sha = "4" * 40
    ready_path = tmp_path / finalizer.READY_COMMIT
    write_json(
        ready_path,
        {
            "schema_version": 1,
            "status": "PASS",
            "state": "READY",
            "integration_id": integration_id,
            "main_sha": context["main_sha"],
            "dev_sha": context["dev_sha"],
            "source_commit_sha": context["source_commit_sha"],
            "ready_commit_sha": normalized_commit,
            "parent_sha": context["main_sha"],
            "tree_sha": tree_sha,
            "normalization_report_sha256": finalizer.sha256(normalization_path),
            "gate_report_sha256": finalizer.sha256(gate_path),
            "authority_verify": authority,
            "changed_paths": changed,
            "publish_automation": False,
        },
    )
    monkeypatch.setattr(finalizer, "locked", no_lock)
    monkeypatch.setattr(finalizer, "read_tx_context", lambda *_: context)
    monkeypatch.setattr(finalizer.core, "ensure_clean", lambda *_: None)
    monkeypatch.setattr(finalizer.core, "ensure_one_parent", lambda *_: tree_sha)
    monkeypatch.setattr(finalizer.core, "changed_paths", lambda *_: changed)
    monkeypatch.setattr(finalizer, "trusted_main", lambda _context: tmp_path)
    monkeypatch.setattr(finalizer, "authority_verify", lambda *_: authority)
    monkeypatch.setattr(finalizer.core, "finalize_transaction", fail_delegate)

    result = finalizer.finalize_transaction(tmp_path)
    repaired = json.loads(tx_path.read_text())
    assert result["repaired_transaction_state"] is True
    assert repaired["state"] == "READY"
    assert repaired["ready_commit_record_sha256"] == finalizer.sha256(ready_path)


def test_recovery_rejects_gate_log_outside_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tx_path = tmp_path / "TRANSACTION.json"
    integration_id = "GOV-RECOVERY-TEST-04"
    normalized_commit = "d" * 40
    normalization_path = tmp_path / finalizer.NORMALIZATION_REPORT
    write_json(normalization_path, {"normalized_commit_sha": normalized_commit})
    write_json(
        tx_path,
        {
            "schema_version": 1,
            "state": "NORMALIZED",
            "status": "PASS",
            "completed_states": ["PREPARED", "NORMALIZED"],
            "normalization_report_sha256": finalizer.sha256(normalization_path),
        },
    )
    context = base_context(tmp_path, tx_path, integration_id)
    outside = tmp_path.parent / "outside.log"
    outside.write_text("PASS", encoding="utf-8")
    outcome = {
        "label": "gate",
        "passed": True,
        "log_file": str(outside),
        "log_sha256": finalizer.sha256(outside),
    }
    write_json(
        tmp_path / finalizer.GATE_REPORT,
        {
            "schema_version": 1,
            "status": "PASS",
            "state": "REQUIRED_GATES_PASSED",
            "integration_id": integration_id,
            "main_sha": context["main_sha"],
            "normalized_commit_sha": normalized_commit,
            "failed_count": 0,
            "first_blocker": None,
            "passed_count": 1,
            "outcomes": [outcome],
        },
    )
    monkeypatch.setattr(finalizer, "locked", no_lock)
    monkeypatch.setattr(finalizer, "read_tx_context", lambda *_: context)

    with pytest.raises(finalizer.FinalizeError) as caught:
        finalizer.gate_transaction(tmp_path)
    assert caught.value.error_code == "UNSAFE_PATH"
