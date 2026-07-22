"""Fail-closed ReplayAB R3 attempt-trajectory evidence validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evidence import EvidenceError, EvidenceLocator, RunIdentity, canonical_sha256
from .model import SHA40, SHA256

ATTEMPT_KINDS = frozenset({"INITIAL", "REPAIR"})
ATTEMPT_TERMINALS = frozenset({"SUCCEEDED", "FAILED", "TIMED_OUT", "INTERRUPTED", "INVALIDATED"})
DISPOSITIONS = frozenset({"NONE", "CANDIDATE", "ENVIRONMENT", "INSUFFICIENT_EVIDENCE"})
FEEDBACK_CLASSES = frozenset({"NONE", "EVALUATOR", "AUTHORITY", "EXECUTION", "OPERATOR"})
FINAL_ACCEPTANCE_VALUES = frozenset({"PASS", "REJECTED", "NOT_AVAILABLE"})
RESOURCE_DIMENSIONS = (
    "command_count",
    "active_ns",
    "retained_bytes",
    "tool_operation_count",
    "token_count",
    "message_count",
    "monetary_microunits",
)
RESOURCE_CAPABILITIES = frozenset({"OBSERVED", "UNAVAILABLE"})
TERMINAL_DISPOSITIONS = {
    "SUCCEEDED": frozenset({"NONE"}),
    "FAILED": frozenset({"CANDIDATE", "INSUFFICIENT_EVIDENCE"}),
    "TIMED_OUT": frozenset({"CANDIDATE", "ENVIRONMENT", "INSUFFICIENT_EVIDENCE"}),
    "INTERRUPTED": frozenset({"CANDIDATE", "ENVIRONMENT", "INSUFFICIENT_EVIDENCE"}),
    "INVALIDATED": frozenset({"ENVIRONMENT", "INSUFFICIENT_EVIDENCE"}),
}
MAX_ATTEMPTS, MAX_LOCATORS_PER_ATTEMPT, MAX_RUN_ARTIFACT_BYTES = 32, 8, 1 << 18
ATTEMPT_FIELDS = {
    "attempt_id",
    "ordinal",
    "kind",
    "parent_attempt_id",
    "terminal",
    "disposition",
    "input_artifact_locator",
    "output_artifact_locator",
    "event_journal_locator",
    "feedback_class",
    "feedback_locator",
    "diagnostic_codes",
    "observed_resources",
    "attempt_sha256",
}
RUN_FIELDS = {
    "schema_version",
    "run_identity",
    "base_sha",
    "toolchain_sha",
    "environment_id",
    "cache_policy",
    "backend_id",
    "resource_capabilities",
    "attempts",
    "first_attempt_id",
    "final_attempt_id",
    "final_outcome_locator",
    "final_acceptance",
    "acceptance_evidence_locator",
    "aggregate_observed_resources",
    "run_artifact_sha256",
}
BINDING_FIELDS = {"schema_version", "case_id", "arm", "run_id", "final_attempt_id"}
ACCEPTANCE_BINDING_FIELDS = BINDING_FIELDS | {"final_acceptance"}


class TrajectoryError(ValueError):
    """R3 trajectory evidence failed closed with a stable error code."""
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    ordinal: int
    kind: str
    parent_attempt_id: str | None
    terminal: str
    disposition: str
    input_artifact_locator: EvidenceLocator | None
    output_artifact_locator: EvidenceLocator | None
    event_journal_locator: EvidenceLocator
    feedback_class: str
    feedback_locator: EvidenceLocator | None
    diagnostic_codes: tuple[str, ...]
    observed_resources: tuple[tuple[str, int], ...]
    attempt_sha256: str


@dataclass(frozen=True)
class RunArtifact:
    schema_version: int
    run_identity: RunIdentity
    base_sha: str
    toolchain_sha: str
    environment_id: str
    cache_policy: str
    backend_id: str
    resource_capabilities: tuple[tuple[str, str], ...]
    attempts: tuple[AttemptRecord, ...]
    first_attempt_id: str
    final_attempt_id: str
    final_outcome_locator: EvidenceLocator | None
    final_acceptance: str
    acceptance_evidence_locator: EvidenceLocator | None
    aggregate_observed_resources: tuple[tuple[str, int], ...]
    run_artifact_sha256: str


@dataclass(frozen=True)
class TrajectorySummary:
    final_acceptance: str
    initial_terminal: str
    repair_count: int
    candidate_failure_count: int
    timeout_count: int
    interruption_count: int
    invalidation_count: int
    final_attempt_terminal: str
    trajectory_complete: bool


def _fail(code: str, message: str) -> None:
    raise TrajectoryError(code, message)


def _canonical(value: Any, code: str = "SCHEMA_INVALID") -> str:
    try:
        return canonical_sha256(value)
    except (TypeError, ValueError) as exc:
        _fail(code, f"payload is not canonically serializable: {exc}")


def _strict(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        _fail("SCHEMA_INVALID", f"{label} must contain exactly {sorted(fields)}")
    return value


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail("SCHEMA_INVALID", f"{label} must be a non-empty stripped string")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        _fail("SCHEMA_INVALID", f"{label} must be a lowercase SHA-256 digest")
    return value


def _git_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA40.fullmatch(value) is None:
        _fail("SCHEMA_INVALID", f"{label} must be a lowercase Git commit SHA")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail("SCHEMA_INVALID", f"{label} must be a non-negative integer")
    return value


def _locator(value: Any, label: str, required: bool = False) -> EvidenceLocator | None:
    if value is None:
        if required:
            _fail("SCHEMA_INVALID", f"{label} is required")
        return None
    try:
        return EvidenceLocator.from_payload(value)
    except (EvidenceError, TypeError, ValueError, KeyError) as exc:
        _fail("SCHEMA_INVALID", f"{label} is invalid: {exc}")


def _verify(locator: EvidenceLocator | None, root: str | Path) -> bytes | None:
    if locator is None:
        return None
    try:
        return locator.verify(root)
    except (EvidenceError, OSError, UnicodeError, ValueError) as exc:
        _fail("EVIDENCE_DIGEST_MISMATCH", str(exc))


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("SCHEMA_INVALID", f"{label} must be a mapping")
    return dict(value)


def _capabilities(value: Any) -> tuple[tuple[str, str], ...]:
    data = _mapping(value, "resource_capabilities")
    if set(data) != set(RESOURCE_DIMENSIONS):
        _fail("RESOURCE_CAPABILITY_INCOMPLETE", "every frozen resource dimension is required")
    if any(item not in RESOURCE_CAPABILITIES for item in data.values()):
        _fail("RESOURCE_CAPABILITY_INCOMPLETE", "resource capability values are invalid")
    return tuple((name, data[name]) for name in RESOURCE_DIMENSIONS)


def _resources(
    value: Any,
    capabilities: tuple[tuple[str, str], ...],
    label: str,
) -> tuple[tuple[str, int], ...]:
    data = _mapping(value, label)
    observed = tuple(name for name, state in capabilities if state == "OBSERVED")
    if set(data) != set(observed):
        _fail("RESOURCE_CAPABILITY_INCOMPLETE", f"{label} must cover observed dimensions exactly")
    return tuple((name, _integer(data[name], f"{label}.{name}")) for name in observed)


def _diagnostics(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        _fail("SCHEMA_INVALID", "diagnostic_codes must be a list of non-empty strings")
    result = tuple(value)
    if result != tuple(sorted(set(result))):
        _fail("SCHEMA_INVALID", "diagnostic_codes must be sorted and unique")
    return result


def validate_attempt_record(
    value: Any,
    *,
    run_id: str,
    resource_capabilities: Mapping[str, str] | tuple[tuple[str, str], ...],
    evidence_root: str | Path,
) -> AttemptRecord:
    data = _strict(value, ATTEMPT_FIELDS, "AttemptRecord")
    _sha(run_id, "run_id")
    ordinal = _integer(data["ordinal"], "ordinal")
    expected_id = _canonical({"run_id": run_id, "ordinal": ordinal})
    if data["attempt_id"] != expected_id:
        _fail("ATTEMPT_LINEAGE_INVALID", "attempt_id does not match run identity and ordinal")
    kind = data["kind"]
    terminal = data["terminal"]
    disposition = data["disposition"]
    feedback_class = data["feedback_class"]
    if kind not in ATTEMPT_KINDS or terminal not in ATTEMPT_TERMINALS:
        _fail("SCHEMA_INVALID", "attempt kind or terminal is invalid")
    if disposition not in DISPOSITIONS or disposition not in TERMINAL_DISPOSITIONS[terminal]:
        _fail("TERMINAL_DISPOSITION_INVALID", "terminal and disposition are incompatible")
    if feedback_class not in FEEDBACK_CLASSES:
        _fail("FEEDBACK_LINEAGE_INVALID", "feedback_class is invalid")
    parent = data["parent_attempt_id"]
    if parent is not None:
        _sha(parent, "parent_attempt_id")
    locators = {
        "input": _locator(data["input_artifact_locator"], "input_artifact_locator"),
        "output": _locator(data["output_artifact_locator"], "output_artifact_locator"),
        "event": _locator(data["event_journal_locator"], "event_journal_locator", True),
        "feedback": _locator(data["feedback_locator"], "feedback_locator"),
    }
    if sum(item is not None for item in locators.values()) > MAX_LOCATORS_PER_ATTEMPT:
        _fail("LIMIT_EXCEEDED", "attempt locator count exceeds the frozen limit")
    initial_has_lineage = parent is not None or feedback_class != "NONE" or locators["feedback"]
    if kind == "INITIAL" and initial_has_lineage:
        _fail("FEEDBACK_LINEAGE_INVALID", "initial attempt cannot have parent or feedback")
    if kind == "INITIAL" and ordinal != 0:
        _fail("ATTEMPT_LINEAGE_INVALID", "initial attempt must have ordinal zero")
    if kind == "REPAIR" and (ordinal == 0 or parent is None):
        _fail("ATTEMPT_LINEAGE_INVALID", "repair attempt requires a parent and positive ordinal")
    if kind == "REPAIR" and feedback_class == "NONE" and locators["feedback"] is not None:
        _fail("FEEDBACK_LINEAGE_INVALID", "independent retry cannot carry feedback evidence")
    if kind == "REPAIR" and feedback_class != "NONE" and locators["feedback"] is None:
        _fail("FEEDBACK_LINEAGE_INVALID", "feedback-bound repair requires feedback evidence")
    if data["output_artifact_locator"] is None and terminal in {"SUCCEEDED", "FAILED"}:
        _fail("SCHEMA_INVALID", "completed or failed attempt requires an output artifact")
    if locators["event"].kind != f"journal-{expected_id}":
        _fail("ATTEMPT_LINEAGE_INVALID", "event journal is bound to a different attempt")
    if locators["output"] is not None and locators["output"].kind != f"candidate-{expected_id}":
        _fail("ATTEMPT_LINEAGE_INVALID", "candidate artifact is bound to a different attempt")
    if locators["feedback"] is not None:
        feedback_binding = _canonical(
            {"parent_attempt_id": parent, "repair_attempt_id": expected_id}
        )
        if locators["feedback"].kind != f"feedback-{feedback_binding}":
            _fail("FEEDBACK_LINEAGE_INVALID", "feedback is bound to a different repair")
    capability_tuple = (
        tuple(resource_capabilities.items())
        if isinstance(resource_capabilities, Mapping)
        else tuple(resource_capabilities)
    )
    capability_tuple = _capabilities(dict(capability_tuple))
    resources = _resources(data["observed_resources"], capability_tuple, "observed_resources")
    attempt_sha = _sha(data["attempt_sha256"], "attempt_sha256")
    digest_payload = {key: item for key, item in data.items() if key != "attempt_sha256"}
    expected_sha = _canonical(digest_payload)
    if attempt_sha != expected_sha:
        _fail("ATTEMPT_DIGEST_MISMATCH", "attempt digest does not match canonical payload")
    for locator in locators.values():
        _verify(locator, evidence_root)
    return AttemptRecord(
        expected_id,
        ordinal,
        kind,
        parent,
        terminal,
        disposition,
        locators["input"],
        locators["output"],
        locators["event"],
        feedback_class,
        locators["feedback"],
        _diagnostics(data["diagnostic_codes"]),
        resources,
        attempt_sha,
    )


def _verify_binding(
    raw: bytes | None,
    *,
    run: RunIdentity,
    final_attempt_id: str,
    final_acceptance: str | None = None,
) -> None:
    if raw is None:
        return
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        _fail("FINAL_POINTER_MISMATCH", f"cannot decode binding evidence: {exc}")
    fields = ACCEPTANCE_BINDING_FIELDS if final_acceptance is not None else BINDING_FIELDS
    data = _strict(value, fields, "binding evidence")
    expected = {
        "schema_version": 1,
        "case_id": run.case_id,
        "arm": run.arm,
        "run_id": run.run_id,
        "final_attempt_id": final_attempt_id,
    }
    if final_acceptance is not None:
        expected["final_acceptance"] = final_acceptance
    if data != expected:
        _fail("FINAL_POINTER_MISMATCH", "binding evidence references a different final attempt")


def validate_r3_run_artifact(value: Any, evidence_root: str | Path) -> RunArtifact:
    try:
        size = len(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    except (TypeError, ValueError) as exc:
        _fail("SCHEMA_INVALID", f"cannot serialize RunArtifact: {exc}")
    if size > MAX_RUN_ARTIFACT_BYTES:
        _fail("LIMIT_EXCEEDED", "serialized RunArtifact exceeds the frozen limit")
    data = _strict(value, RUN_FIELDS, "RunArtifact")
    if isinstance(data["schema_version"], bool) or data["schema_version"] != 1:
        _fail("SCHEMA_INVALID", "schema_version must equal integer 1")
    try:
        run = RunIdentity.from_payload(data["run_identity"])
    except (EvidenceError, TypeError, ValueError, KeyError) as exc:
        _fail("IDENTITY_INVALID", str(exc))
    for key in ("base_sha", "toolchain_sha"):
        _git_sha(data[key], key)
    backend_id = _token(data["backend_id"], "backend_id")
    if backend_id != run.backend_id:
        _fail("IDENTITY_INVALID", "backend_id does not match run identity")
    environment_id = _token(data["environment_id"], "environment_id")
    cache_policy = _token(data["cache_policy"], "cache_policy")
    capabilities = _capabilities(data["resource_capabilities"])
    raw_attempts = data["attempts"]
    if not isinstance(raw_attempts, list) or not raw_attempts:
        _fail("MISSING_INITIAL_ATTEMPT", "attempt list must include ordinal-zero initial attempt")
    if len(raw_attempts) > MAX_ATTEMPTS:
        _fail("LIMIT_EXCEEDED", "attempt count exceeds the frozen limit")
    first_ordinal = raw_attempts[0].get("ordinal") if isinstance(raw_attempts[0], dict) else None
    if first_ordinal != 0:
        _fail("MISSING_INITIAL_ATTEMPT", "first attempt must be ordinal zero")
    ordinals = [item.get("ordinal") if isinstance(item, dict) else None for item in raw_attempts]
    if ordinals != list(range(len(raw_attempts))):
        _fail("ATTEMPT_ORDINAL_INVALID", "attempt ordinals must be contiguous and ordered")
    attempts = tuple(
        validate_attempt_record(
            item,
            run_id=run.run_id,
            resource_capabilities=capabilities,
            evidence_root=evidence_root,
        )
        for item in raw_attempts
    )
    if attempts[0].kind != "INITIAL":
        _fail("MISSING_INITIAL_ATTEMPT", "ordinal-zero attempt must be INITIAL")
    for ordinal, attempt in enumerate(attempts[1:], start=1):
        previous = attempts[ordinal - 1]
        if attempt.kind != "REPAIR" or attempt.parent_attempt_id != previous.attempt_id:
            _fail(
                "ATTEMPT_LINEAGE_INVALID",
                "repair must point to the immediately preceding attempt",
            )
    if data["first_attempt_id"] != attempts[0].attempt_id:
        _fail("MISSING_INITIAL_ATTEMPT", "first_attempt_id does not identify ordinal zero")
    if data["final_attempt_id"] != attempts[-1].attempt_id:
        _fail("FINAL_POINTER_MISMATCH", "final_attempt_id does not identify the last attempt")
    final_acceptance = data["final_acceptance"]
    if final_acceptance not in FINAL_ACCEPTANCE_VALUES:
        _fail("SCHEMA_INVALID", "final_acceptance is invalid")
    final_outcome = _locator(data["final_outcome_locator"], "final_outcome_locator")
    acceptance = _locator(data["acceptance_evidence_locator"], "acceptance_evidence_locator")
    if final_acceptance in {"PASS", "REJECTED"} and (final_outcome is None or acceptance is None):
        _fail(
            "FINAL_POINTER_MISMATCH",
            "semantic acceptance requires outcome and acceptance evidence",
        )
    if final_acceptance == "PASS" and attempts[-1].terminal != "SUCCEEDED":
        _fail("FINAL_POINTER_MISMATCH", "PASS requires a succeeded final attempt")
    non_evaluable = {"TIMED_OUT", "INTERRUPTED", "INVALIDATED"}
    if attempts[-1].terminal in non_evaluable and final_acceptance != "NOT_AVAILABLE":
        _fail("FINAL_POINTER_MISMATCH", "non-evaluable final attempt requires NOT_AVAILABLE")
    if final_acceptance == "NOT_AVAILABLE" and acceptance is not None:
        _fail("FINAL_POINTER_MISMATCH", "NOT_AVAILABLE forbids acceptance evidence")
    aggregate = _resources(
        data["aggregate_observed_resources"], capabilities, "aggregate_observed_resources"
    )
    expected_aggregate = {
        name: sum(dict(attempt.observed_resources)[name] for attempt in attempts)
        for name, state in capabilities
        if state == "OBSERVED"
    }
    if dict(aggregate) != expected_aggregate:
        _fail("RESOURCE_AGGREGATE_MISMATCH", "run resources do not equal attempt sums")
    run_sha = _sha(data["run_artifact_sha256"], "run_artifact_sha256")
    expected_sha = _canonical(
        {key: item for key, item in data.items() if key != "run_artifact_sha256"}
    )
    if run_sha != expected_sha:
        _fail("RUN_ARTIFACT_DIGEST_MISMATCH", "run artifact digest is invalid")
    final_raw = _verify(final_outcome, evidence_root)
    acceptance_raw = _verify(acceptance, evidence_root)
    _verify_binding(final_raw, run=run, final_attempt_id=attempts[-1].attempt_id)
    _verify_binding(
        acceptance_raw,
        run=run,
        final_attempt_id=attempts[-1].attempt_id,
        final_acceptance=final_acceptance if acceptance_raw is not None else None,
    )
    return RunArtifact(
        1,
        run,
        data["base_sha"],
        data["toolchain_sha"],
        environment_id,
        cache_policy,
        backend_id,
        capabilities,
        attempts,
        attempts[0].attempt_id,
        attempts[-1].attempt_id,
        final_outcome,
        final_acceptance,
        acceptance,
        aggregate,
        run_sha,
    )


def load_r3_run_artifact(
    path: str | Path,
    evidence_root: str | Path | None = None,
) -> RunArtifact:
    target = Path(path)
    if target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
        _fail("SCHEMA_INVALID", "RunArtifact path must not contain symlinks")
    try:
        raw = target.read_bytes()
    except OSError as exc:
        _fail("SCHEMA_INVALID", f"cannot read RunArtifact: {exc}")
    if not raw or len(raw) > MAX_RUN_ARTIFACT_BYTES:
        _fail("LIMIT_EXCEEDED", "serialized RunArtifact size is invalid")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        _fail("SCHEMA_INVALID", f"cannot decode RunArtifact: {exc}")
    return validate_r3_run_artifact(value, evidence_root or target.parent)


def summarize_trajectory(artifact: RunArtifact) -> TrajectorySummary:
    if not isinstance(artifact, RunArtifact) or not artifact.attempts:
        _fail("SCHEMA_INVALID", "summary requires a validated RunArtifact")
    attempts = artifact.attempts
    return TrajectorySummary(
        final_acceptance=artifact.final_acceptance,
        initial_terminal=attempts[0].terminal,
        repair_count=len(attempts) - 1,
        candidate_failure_count=sum(
            item.disposition == "CANDIDATE" and item.terminal != "SUCCEEDED" for item in attempts
        ),
        timeout_count=sum(item.terminal == "TIMED_OUT" for item in attempts),
        interruption_count=sum(item.terminal == "INTERRUPTED" for item in attempts),
        invalidation_count=sum(item.terminal == "INVALIDATED" for item in attempts),
        final_attempt_terminal=attempts[-1].terminal,
        trajectory_complete=True,
    )
