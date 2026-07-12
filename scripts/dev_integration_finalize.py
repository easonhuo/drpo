#!/usr/bin/env python3
"""Recovery-safe CLI facade for the DRPO Batch 2B integration finalizer.

The implementation core is kept in :mod:`dev_integration_finalize_core`.  This
facade closes the three narrow crash windows where a durable stage report may
exist before ``TRANSACTION.json`` is advanced.  It verifies the completed
report and repairs only the transaction state; all other paths delegate to the
core implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import dev_integration_finalize_core as core

VERSION = f"{core.VERSION}.recovery2"

# Public aliases retained for callers and the existing test suite.
FinalizeError = core.FinalizeError
WritePathError = core.WritePathError
NORMALIZATION_REPORT = core.NORMALIZATION_REPORT
GATE_REPORT = core.GATE_REPORT
READY_COMMIT = core.READY_COMMIT
INTENT_FILE = core.INTENT_FILE
APPROVAL_FILE = core.APPROVAL_FILE
apply_registry_mutation = core.apply_registry_mutation
json_hash = core.json_hash
validate_intent = core.validate_intent
build_delta = core.build_delta
expected_final_scope = core.expected_final_scope
sha256 = core.sha256
shadow = core.shadow
shutil = core.shutil
locked = core.locked
read_tx_context = core.read_tx_context
trusted_main = core.trusted_main
check_freshness = core.check_freshness
authority_normalize = core.authority_normalize
authority_verify = core.authority_verify
gate_record = core.gate_record

_OVERRIDE_NAMES = (
    "locked",
    "read_tx_context",
    "trusted_main",
    "check_freshness",
    "authority_normalize",
    "authority_verify",
    "gate_record",
)


def __getattr__(name: str) -> Any:
    """Delegate unchanged public API to the implementation core."""

    return getattr(core, name)


def _sync_core_overrides() -> None:
    """Propagate test/integration overrides before delegating to the core."""

    namespace = globals()
    for name in _OVERRIDE_NAMES:
        setattr(core, name, namespace[name])


def _append_completed(tx: dict[str, Any], state: str) -> list[str]:
    completed = list(tx.get("completed_states", []))
    if state not in completed:
        completed.append(state)
    return completed


def _raise_with_diagnostic(
    transaction_dir: Path,
    context: dict[str, Any] | None,
    error: BaseException,
) -> None:
    if isinstance(error, core.FinalizeError):
        wrapped = error
    elif isinstance(error, core.WritePathError):
        wrapped = core.FinalizeError(
            error.error_code,
            error.phase,
            error.message,
            error.recovery,
        )
    else:
        wrapped = core.FinalizeError(
            "INTERNAL_ERROR",
            "recovery_facade",
            f"unexpected error: {error}",
        )
    core.write_diagnostic(transaction_dir, context, wrapped)
    raise wrapped from error


def _require_normalization_report_identity(
    context: dict[str, Any],
    report: dict[str, Any],
) -> None:
    expected = {
        "schema_version": core.SCHEMA,
        "status": "PASS",
        "state": "NORMALIZED",
        "integration_id": context["integration_id"],
        "main_sha": context["main_sha"],
        "dev_sha": context["dev_sha"],
        "source_commit_sha": context["source_commit_sha"],
    }
    mismatched = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatched:
        core.fail(
            "IMMUTABILITY_ERROR",
            "normalization_recovery",
            f"normalization report identity mismatch: {mismatched}",
        )


def _require_gate_report_identity(
    context: dict[str, Any],
    report: dict[str, Any],
    normalized_commit: str,
) -> None:
    expected = {
        "schema_version": core.SCHEMA,
        "status": "PASS",
        "state": "REQUIRED_GATES_PASSED",
        "integration_id": context["integration_id"],
        "main_sha": context["main_sha"],
        "normalized_commit_sha": normalized_commit,
        "failed_count": 0,
        "first_blocker": None,
    }
    mismatched = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }
    outcomes = report.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        mismatched["outcomes"] = {"expected": "non-empty list", "actual": outcomes}
    elif any(not isinstance(item, dict) or item.get("passed") is not True for item in outcomes):
        mismatched["outcomes_passed"] = {"expected": True, "actual": False}
    elif report.get("passed_count") != len(outcomes):
        mismatched["passed_count"] = {
            "expected": len(outcomes),
            "actual": report.get("passed_count"),
        }
    if mismatched:
        core.fail(
            "IMMUTABILITY_ERROR",
            "gate_recovery",
            f"gate report identity mismatch: {mismatched}",
        )
    core.verify_gate_logs(report)
    log_root = (context["transaction_dir"] / "gate-logs").resolve()
    for index, outcome in enumerate(outcomes):
        log_path = Path(
            core.text(outcome.get("log_file"), f"gate outcome {index} log_file")
        ).expanduser().resolve()
        try:
            log_path.relative_to(log_root)
        except ValueError:
            core.fail(
                "UNSAFE_PATH",
                "gate_recovery",
                f"gate log escapes transaction gate-logs: {log_path}",
            )


def _require_ready_record_identity(
    context: dict[str, Any],
    ready: dict[str, Any],
    normalized_commit: str,
    normalization_hash: str,
    gate_hash: str,
) -> None:
    tree_sha = core.ensure_one_parent(
        context["repo"],
        normalized_commit,
        context["main_sha"],
        "finalize_recovery",
    )
    changed = core.changed_paths(
        context["repo"],
        context["main_sha"],
        normalized_commit,
    )
    trusted = core.trusted_main(context)
    authority = core.authority_verify(context, trusted)
    expected = {
        "schema_version": core.SCHEMA,
        "status": "PASS",
        "state": "READY",
        "integration_id": context["integration_id"],
        "main_sha": context["main_sha"],
        "dev_sha": context["dev_sha"],
        "source_commit_sha": context["source_commit_sha"],
        "ready_commit_sha": normalized_commit,
        "parent_sha": context["main_sha"],
        "tree_sha": tree_sha,
        "normalization_report_sha256": normalization_hash,
        "gate_report_sha256": gate_hash,
        "authority_verify": authority,
        "changed_paths": changed,
        "publish_automation": False,
    }
    mismatched = {
        key: {"expected": value, "actual": ready.get(key)}
        for key, value in expected.items()
        if ready.get(key) != value
    }
    if mismatched:
        core.fail(
            "IMMUTABILITY_ERROR",
            "finalize_recovery",
            f"ready record identity mismatch: {mismatched}",
        )


def _repair_normalized_report(transaction_dir: Path) -> dict[str, Any] | None:
    report_path = transaction_dir / NORMALIZATION_REPORT
    if not report_path.is_file():
        return None
    context: dict[str, Any] | None = None
    try:
        with core.locked(transaction_dir):
            context = core.read_tx_context(
                transaction_dir,
                {"PREPARED", "NORMALIZED"},
            )
            report = core.load_json(report_path, "normalization report")
            _require_normalization_report_identity(context, report)
            trusted = core.trusted_main(context)
            result = core.verify_normalized(context, trusted)
            tx = context["transaction"]
            report_hash = core.sha256(report_path)
            recorded_hash = tx.get("normalization_report_sha256")
            if recorded_hash is not None and recorded_hash != report_hash:
                core.fail(
                    "IMMUTABILITY_ERROR",
                    "normalization_recovery",
                    "normalization report hash drifted",
                )
            tx.update(
                {
                    "tool_version": VERSION,
                    "state": "NORMALIZED",
                    "status": "PASS",
                    "completed_states": _append_completed(tx, "NORMALIZED"),
                    "normalized_commit_sha": result["normalized_commit_sha"],
                    "normalization_report_sha256": report_hash,
                    "updated_at": core.now(),
                    "next_action": "run Batch 2B gate",
                }
            )
            core.write_json(context["transaction_path"], tx)
            result["idempotent"] = True
            result["repaired_transaction_state"] = recorded_hash is None
            return result
    except Exception as error:
        _raise_with_diagnostic(transaction_dir, context, error)


def normalize_transaction(transaction_dir: Path) -> dict[str, Any]:
    transaction_dir = transaction_dir.expanduser().resolve()
    _sync_core_overrides()
    repaired = _repair_normalized_report(transaction_dir)
    if repaired is not None:
        return repaired
    return core.normalize_transaction(transaction_dir)


def _repair_gate_report(transaction_dir: Path) -> dict[str, Any] | None:
    gate_path = transaction_dir / GATE_REPORT
    if not gate_path.is_file():
        return None
    context: dict[str, Any] | None = None
    try:
        with core.locked(transaction_dir):
            context = core.read_tx_context(
                transaction_dir,
                {"NORMALIZED", "REQUIRED_GATES_PASSED"},
            )
            normalization_path = transaction_dir / NORMALIZATION_REPORT
            normalization = core.load_json(
                normalization_path,
                "normalization report",
            )
            if context["transaction"].get("normalization_report_sha256") != core.sha256(
                normalization_path
            ):
                core.fail(
                    "IMMUTABILITY_ERROR",
                    "gate_recovery",
                    "normalization report hash mismatch",
                )
            normalized_commit = core.full_sha(
                normalization.get("normalized_commit_sha"),
                "normalized_commit_sha",
            )
            report = core.load_json(gate_path, "gate report")
            report_hash = core.sha256(gate_path)
            recorded_hash = context["transaction"].get("gate_report_sha256")
            if recorded_hash is not None and recorded_hash != report_hash:
                core.fail(
                    "IMMUTABILITY_ERROR",
                    "gate_recovery",
                    "gate report hash drifted",
                )
            _require_gate_report_identity(context, report, normalized_commit)
            core.ensure_clean(
                context["repo"],
                normalized_commit,
                "gate_recovery",
            )
            tx = context["transaction"]
            tx.update(
                {
                    "tool_version": VERSION,
                    "state": "REQUIRED_GATES_PASSED",
                    "status": "PASS",
                    "completed_states": _append_completed(
                        tx,
                        "REQUIRED_GATES_PASSED",
                    ),
                    "gate_report_sha256": report_hash,
                    "updated_at": core.now(),
                    "next_action": "run Batch 2B finalize",
                }
            )
            core.write_json(context["transaction_path"], tx)
            return {
                "status": "PASS",
                "state": "REQUIRED_GATES_PASSED",
                "integration_id": context["integration_id"],
                "normalized_commit_sha": normalized_commit,
                "idempotent": True,
                "repaired_transaction_state": recorded_hash is None,
            }
    except Exception as error:
        _raise_with_diagnostic(transaction_dir, context, error)


def gate_transaction(transaction_dir: Path) -> dict[str, Any]:
    transaction_dir = transaction_dir.expanduser().resolve()
    _sync_core_overrides()
    repaired = _repair_gate_report(transaction_dir)
    if repaired is not None:
        return repaired
    return core.gate_transaction(transaction_dir)


def _repair_ready_record(transaction_dir: Path) -> dict[str, Any] | None:
    ready_path = transaction_dir / READY_COMMIT
    if not ready_path.is_file():
        return None
    context: dict[str, Any] | None = None
    try:
        with core.locked(transaction_dir):
            context = core.read_tx_context(
                transaction_dir,
                {"REQUIRED_GATES_PASSED", "READY"},
            )
            normalization_path = transaction_dir / NORMALIZATION_REPORT
            gate_path = transaction_dir / GATE_REPORT
            normalization = core.load_json(
                normalization_path,
                "normalization report",
            )
            gate_report = core.load_json(gate_path, "gate report")
            if context["transaction"].get("normalization_report_sha256") != core.sha256(
                normalization_path
            ):
                core.fail(
                    "IMMUTABILITY_ERROR",
                    "finalize_recovery",
                    "normalization report hash mismatch",
                )
            if context["transaction"].get("gate_report_sha256") != core.sha256(
                gate_path
            ):
                core.fail(
                    "IMMUTABILITY_ERROR",
                    "finalize_recovery",
                    "gate report hash mismatch",
                )
            normalized_commit = core.full_sha(
                normalization.get("normalized_commit_sha"),
                "normalized_commit_sha",
            )
            _require_gate_report_identity(context, gate_report, normalized_commit)
            ready = core.load_json(ready_path, "ready commit")
            ready_hash = core.sha256(ready_path)
            recorded_hash = context["transaction"].get("ready_commit_record_sha256")
            if recorded_hash is not None and recorded_hash != ready_hash:
                core.fail(
                    "IMMUTABILITY_ERROR",
                    "finalize_recovery",
                    "ready record hash drifted",
                )
            _require_ready_record_identity(
                context,
                ready,
                normalized_commit,
                core.sha256(normalization_path),
                core.sha256(gate_path),
            )
            core.ensure_clean(
                context["repo"],
                normalized_commit,
                "finalize_recovery",
            )
            tx = context["transaction"]
            tx.update(
                {
                    "tool_version": VERSION,
                    "state": "READY",
                    "status": "PASS",
                    "completed_states": _append_completed(tx, "READY"),
                    "ready_commit_sha": normalized_commit,
                    "ready_commit_record_sha256": ready_hash,
                    "updated_at": core.now(),
                    "next_action": "manual review/publish outside V1",
                }
            )
            core.write_json(context["transaction_path"], tx)
            return {
                "status": "PASS",
                "state": "READY",
                "integration_id": context["integration_id"],
                "ready_commit_sha": normalized_commit,
                "idempotent": True,
                "repaired_transaction_state": recorded_hash is None,
            }
    except Exception as error:
        _raise_with_diagnostic(transaction_dir, context, error)


def finalize_transaction(transaction_dir: Path) -> dict[str, Any]:
    transaction_dir = transaction_dir.expanduser().resolve()
    _sync_core_overrides()
    repaired = _repair_ready_record(transaction_dir)
    if repaired is not None:
        return repaired
    return core.finalize_transaction(transaction_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command_name", required=True)
    for name in ("normalize", "gate", "finalize"):
        child = commands.add_parser(name)
        child.add_argument("--transaction-dir", required=True)
        child.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    function = {
        "normalize": normalize_transaction,
        "gate": gate_transaction,
        "finalize": finalize_transaction,
    }[args.command_name]
    try:
        result = function(Path(args.transaction_dir))
        print(
            json.dumps(result, sort_keys=True)
            if args.json
            else f"PASS {result['integration_id']}: {result['state']}"
        )
        return 0
    except core.FinalizeError as error:
        payload = {
            "status": "FAIL",
            "error_code": error.error_code,
            "phase": error.phase,
            "message": error.message,
        }
        print(
            json.dumps(payload, sort_keys=True) if args.json else f"FAIL {error}",
            file=sys.stdout if args.json else sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
