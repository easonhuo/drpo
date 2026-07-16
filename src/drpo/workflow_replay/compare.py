"""Correctness-first equivalence checks for replay outcomes."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from .model import CaseManifest

TERMINAL_STATES = {"READY", "BLOCKED", "STALE"}
GATE_STATES = {"PASS", "FAIL", "BLOCKED", "NOT_RUN"}
AUTHORITY_STATES = {"PASS", "FAIL", "BLOCKED", "NOT_RUN"}


class EquivalenceError(ValueError):
    """Correctness equivalence failed, so efficiency analysis is forbidden."""


@dataclass(frozen=True)
class OutcomeSnapshot:
    case_id: str
    terminal_state: str
    safety_boundary: str | None
    changed_paths: tuple[str, ...]
    file_modes: tuple[tuple[str, str], ...]
    output_hashes: tuple[tuple[str, str], ...]
    authority_result: str
    gate_plan: tuple[str, ...]
    gate_results: tuple[tuple[str, str], ...]
    provenance: tuple[tuple[str, str], ...]
    diagnostic_codes: tuple[str, ...] = ()
    partial_mutation: bool = False
    recovery_class: str | None = None


@dataclass(frozen=True)
class EquivalenceReport:
    case_id: str
    equivalent: bool
    mismatches: tuple[str, ...]


def _pairs(
    value: tuple[tuple[str, str], ...], label: str, required: bool = False
) -> dict[str, str]:
    invalid = not isinstance(value, tuple) or any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not all(isinstance(part, str) and part for part in item)
        for item in value
    )
    if invalid or (required and not value):
        raise EquivalenceError(f"{label} must be unique sorted string pairs")
    result = dict(value)
    if len(result) != len(value) or tuple(sorted(value)) != value:
        raise EquivalenceError(f"{label} must be unique sorted string pairs")
    return result


def _validate(snapshot: OutcomeSnapshot) -> None:
    if not isinstance(snapshot, OutcomeSnapshot):
        raise EquivalenceError("outcome must be an OutcomeSnapshot")
    if snapshot.terminal_state not in TERMINAL_STATES:
        raise EquivalenceError("terminal_state is invalid")
    if not snapshot.case_id or snapshot.authority_result not in AUTHORITY_STATES:
        raise EquivalenceError("case_id or authority_result is invalid")
    if tuple(sorted(set(snapshot.changed_paths))) != snapshot.changed_paths:
        raise EquivalenceError("changed_paths must be unique and sorted")
    modes = _pairs(snapshot.file_modes, "file_modes") if snapshot.file_modes else {}
    _pairs(snapshot.output_hashes, "output_hashes")
    gates = _pairs(snapshot.gate_results, "gate_results", True)
    _pairs(snapshot.provenance, "provenance", True)
    if set(modes) != set(snapshot.changed_paths):
        raise EquivalenceError("file_modes must cover changed_paths exactly")
    invalid_gates = tuple(gates) != snapshot.gate_plan or any(
        state not in GATE_STATES for state in gates.values()
    )
    if invalid_gates:
        raise EquivalenceError("gate_results must match gate_plan with valid states")
    if tuple(sorted(set(snapshot.diagnostic_codes))) != snapshot.diagnostic_codes:
        raise EquivalenceError("diagnostic_codes must be unique and sorted")
    if not isinstance(snapshot.partial_mutation, bool):
        raise EquivalenceError("partial_mutation must be boolean")


def compare_outcomes(
    manifest: CaseManifest, arm_a: OutcomeSnapshot, arm_b: OutcomeSnapshot
) -> EquivalenceReport:
    """Compare both arms to each other and to the frozen manifest."""
    _validate(arm_a)
    _validate(arm_b)
    expected = manifest.benchmark
    mismatches: list[str] = []

    def check(label: str, actual: Any, wanted: Any) -> None:
        if actual != wanted:
            mismatches.append(label)

    expected_paths = tuple(sorted(expected["expected_changed_paths"]))
    expected_hashes = tuple(sorted(expected["expected_final_tree_or_semantic_hashes"].items()))
    expected_gates = tuple(expected["required_gates"])
    historical = manifest.historical_task
    expected_provenance = {
        "benchmark_toolchain_sha": expected["toolchain_sha"],
        "cache_policy": expected["cache_policy"],
        "environment_id": expected["environment_id"],
        "historical_base_sha": historical["base_sha"],
        "input_spec_sha256": expected["input_spec_sha256"],
    }
    if historical["frozen_implementation_sha"] is not None:
        expected_provenance["frozen_implementation_sha"] = historical["frozen_implementation_sha"]
    for arm, snapshot in (("A", arm_a), ("B", arm_b)):
        check(f"{arm}.case_id", snapshot.case_id, manifest.case_id)
        check(f"{arm}.terminal_state", snapshot.terminal_state, expected["expected_terminal_state"])
        check(
            f"{arm}.safety_boundary",
            snapshot.safety_boundary,
            expected["expected_safety_boundary"],
        )
        check(f"{arm}.changed_paths", snapshot.changed_paths, expected_paths)
        check(f"{arm}.output_hashes", snapshot.output_hashes, expected_hashes)
        check(f"{arm}.gate_plan", snapshot.gate_plan, expected_gates)
        provenance = dict(snapshot.provenance)
        for key, value in expected_provenance.items():
            check(f"{arm}.provenance.{key}", provenance.get(key), value)
        if snapshot.terminal_state == "READY":
            check(f"{arm}.authority_result", snapshot.authority_result, "PASS")
            check(f"{arm}.gate_results", set(dict(snapshot.gate_results).values()), {"PASS"})
            check(f"{arm}.partial_mutation", snapshot.partial_mutation, False)
            check(f"{arm}.recovery_class", snapshot.recovery_class, None)
        else:
            check(f"{arm}.has_diagnostics", bool(snapshot.diagnostic_codes), True)
            check(f"{arm}.has_recovery_class", bool(snapshot.recovery_class), True)
            check(f"{arm}.partial_mutation", snapshot.partial_mutation, False)
    for field in fields(OutcomeSnapshot):
        if getattr(arm_a, field.name) != getattr(arm_b, field.name):
            mismatches.append(f"pair.{field.name}")
    return EquivalenceReport(manifest.case_id, not mismatches, tuple(dict.fromkeys(mismatches)))


def release_efficiency_payload(report: EquivalenceReport, payload: Any) -> Any:
    """Release timing data only after correctness equivalence passes."""
    if not report.equivalent:
        raise EquivalenceError("efficiency analysis blocked by correctness mismatches")
    return payload
