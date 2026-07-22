"""Deterministic dry-run planning and append-only fixture execution."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .evidence import EVENT_TERMINALS, EvidenceError, EvidenceLocator, RunIdentity
from .evidence import _validate_journal, canonical_sha256
from .model import CaseManifest
from .trajectory import FEEDBACK_CLASSES as FEEDBACK, TERMINAL_DISPOSITIONS

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_R3_FIXTURE_TERMINALS = {
    "READY": ("SUCCEEDED", "NONE"), "BLOCKED": ("FAILED", "CANDIDATE"),
    "STALE": ("INVALIDATED", "ENVIRONMENT"), "INTERRUPTED": ("INTERRUPTED", "ENVIRONMENT"),
    "INVALIDATED": ("INVALIDATED", "ENVIRONMENT"),
}

class ExecutionError(ValueError):
    """The replay execution request is unsafe or ambiguous."""


@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionPlan:
    case_id: str
    arm: str
    input_sha256: str
    environment_id: str
    cache_policy: str
    commands: tuple[CommandSpec, ...]
    plan_sha256: str


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _digest(value: Any) -> str:
    raw = json.dumps(_plain(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _commands(values: Iterable[CommandSpec]) -> tuple[CommandSpec, ...]:
    result = tuple(values)
    if not result:
        raise ExecutionError("plan must contain at least one child command")
    for command in result:
        if not isinstance(command, CommandSpec) or TOKEN.fullmatch(command.name) is None:
            raise ExecutionError("command name has invalid syntax")
        if not isinstance(command.argv, tuple) or not command.argv or any(
            not isinstance(arg, str) or not arg or "\x00" in arg for arg in command.argv
        ):
            raise ExecutionError(f"command {command.name} must have non-empty NUL-free tuple argv")
    if len({command.name for command in result}) != len(result):
        raise ExecutionError("command names must be unique")
    if len({command.argv for command in result}) != len(result):
        raise ExecutionError("duplicate child command")
    return result
def build_plan(manifest: CaseManifest, arm: str, commands: Iterable[CommandSpec]) -> ExecutionPlan:
    if arm not in {"A", "B"}:
        raise ExecutionError("arm must be A or B")
    items = _commands(commands)
    input_sha = _digest(manifest.data)
    payload = {
        "case_id": manifest.case_id,
        "arm": arm,
        "input_sha256": input_sha,
        "environment_id": manifest.benchmark["environment_id"],
        "cache_policy": manifest.benchmark["cache_policy"],
        "commands": [{"name": item.name, "argv": item.argv} for item in items],
    }
    return ExecutionPlan(
        manifest.case_id, arm, input_sha, manifest.benchmark["environment_id"],
        manifest.benchmark["cache_policy"], items, _digest(payload),
    )


def build_paired_plans(
    manifest: CaseManifest,
    arm_a_commands: Iterable[CommandSpec],
    arm_b_commands: Iterable[CommandSpec],
) -> tuple[ExecutionPlan, ExecutionPlan]:
    return build_plan(manifest, "A", arm_a_commands), build_plan(manifest, "B", arm_b_commands)


def _event_path(path: str | Path) -> Path:
    result = Path(path)
    if result.is_symlink() or any(parent.is_symlink() for parent in result.parents):
        raise ExecutionError("event path must not contain symlinks")
    if not result.parent.is_dir():
        raise ExecutionError("event parent directory must already exist")
    return result


def normalize_fixture_attempt(
    identity: RunIdentity,
    ordinal: int,
    raw_journal_locator: EvidenceLocator,
    evidence_root: str | Path,
    binding_journal_relative_path: str,
    *,
    timing: Mapping[str, int] | None = None,
    disposition: str | None = None,
    output_artifact_locator: EvidenceLocator | None = None,
    feedback_class: str = "NONE",
    feedback_locator: EvidenceLocator | None = None,
    diagnostic_codes: Iterable[str] = (),
) -> dict[str, Any]:
    """Normalize one immutable fixture JSONL execution into an R3 attempt payload."""
    invalid_ordinal = isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0
    if not isinstance(identity, RunIdentity) or invalid_ordinal:
        raise ExecutionError("fixture attempt identity or ordinal is invalid")
    if not all(item is None or isinstance(item, EvidenceLocator) for item in (raw_journal_locator, output_artifact_locator, feedback_locator)):
        raise ExecutionError("fixture locators must use EvidenceLocator")
    root = Path(evidence_root)
    try:
        raw = raw_journal_locator.verify(root)
        events = [json.loads(row) for row in raw.decode("utf-8").splitlines()]
        first, last = events[0], events[-1]
        state = EVENT_TERMINALS[last["event"]]
        _validate_journal(raw, identity, state)
    except (AttributeError, IndexError, KeyError, OSError, TypeError, ValueError) as exc:
        raise ExecutionError(f"fixture journal is invalid: {exc}") from exc
    started, finished = first["payload"], last["payload"]
    if started.get("case_id") != identity.case_id or started.get("arm") != identity.arm:
        raise ExecutionError("fixture journal start identity mismatch")
    command_count = finished.get("command_count", finished.get("child_command_count"))
    active_ns = finished.get("child_ns", dict(timing or {}).get("child_ns"))
    starts = sum(row["event"] == "command_started" for row in events)
    durations = tuple(row["payload"].get("child_elapsed_ns") for row in events if row["event"] == "command_finished")
    unfinished = starts - len(durations)
    invalid_resources = finished.get("terminal_state") != state or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in (command_count, active_ns, *durations)
    )
    completed_ns = sum(durations) if not invalid_resources else 0
    invalid_resources |= command_count != starts or unfinished not in {0, 1}
    invalid_resources |= unfinished == 1 and state != "INTERRUPTED"
    invalid_resources |= active_ns < completed_ns or (not unfinished and active_ns != completed_ns)
    if invalid_resources:
        raise ExecutionError("fixture terminal resource summary is invalid")
    terminal, default_disposition = _R3_FIXTURE_TERMINALS[state]
    disposition = default_disposition if disposition is None else disposition
    if state == "BLOCKED" and disposition == "ENVIRONMENT":
        terminal = "INVALIDATED"
    if not isinstance(disposition, str) or disposition not in TERMINAL_DISPOSITIONS[terminal]:
        raise ExecutionError("fixture terminal and disposition are incompatible")
    attempt_id = canonical_sha256({"run_id": identity.run_id, "ordinal": ordinal})
    parent_id = None if ordinal == 0 else canonical_sha256({"run_id": identity.run_id, "ordinal": ordinal - 1})
    if terminal in {"SUCCEEDED", "FAILED"} and output_artifact_locator is None:
        raise ExecutionError("completed fixture attempt requires a candidate artifact")
    if output_artifact_locator and output_artifact_locator.kind != f"candidate-{attempt_id}":
        raise ExecutionError("candidate artifact is bound to a different attempt")
    feedback_binding = canonical_sha256({
        "parent_attempt_id": parent_id, "repair_attempt_id": attempt_id})
    invalid_feedback = not isinstance(feedback_class, str) or feedback_class not in FEEDBACK or (
        ordinal == 0 and (feedback_class != "NONE" or feedback_locator is not None)
    ) or (ordinal > 0 and ((feedback_class == "NONE") != (feedback_locator is None)))
    wrong_feedback = feedback_locator and feedback_locator.kind != f"feedback-{feedback_binding}"
    if invalid_feedback or wrong_feedback:
        raise ExecutionError("fixture feedback lineage is invalid")
    if isinstance(diagnostic_codes, (str, bytes)) or not isinstance(diagnostic_codes, Iterable):
        raise ExecutionError("diagnostic codes must be a non-string iterable")
    diagnostics = tuple(diagnostic_codes)
    if any(not isinstance(item, str) or not item for item in diagnostics) or diagnostics != tuple(sorted(set(diagnostics))):
        raise ExecutionError("diagnostic codes must be sorted unique non-empty strings")
    try:
        for locator in (output_artifact_locator, feedback_locator):
            if locator is not None:
                locator.verify(root)
    except (EvidenceError, OSError, ValueError) as exc:
        raise ExecutionError(f"fixture locator is invalid: {exc}") from exc
    if not isinstance(binding_journal_relative_path, str):
        raise ExecutionError("binding journal path is unsafe")
    relative = PurePosixPath(binding_journal_relative_path)
    unsafe = (
        not relative.parts or binding_journal_relative_path.startswith(("/", "-"))
        or "\\" in binding_journal_relative_path or ".." in relative.parts
        or relative.as_posix() != binding_journal_relative_path
    )
    if unsafe:
        raise ExecutionError("binding journal path is unsafe")
    binding = {
        "schema_version": 1, "run_id": identity.run_id,
        "attempt_id": attempt_id, "ordinal": ordinal, "terminal": terminal,
        "candidate_artifact_produced": output_artifact_locator is not None,
    }
    binding_raw = json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    with _event_path(root.joinpath(*relative.parts)).open("xb") as stream:
        stream.write(binding_raw)
    binding_locator = EvidenceLocator(
        f"journal-{attempt_id}", binding_journal_relative_path,
        hashlib.sha256(binding_raw).hexdigest(), len(binding_raw))
    locators = (raw_journal_locator, output_artifact_locator, feedback_locator, binding_locator)
    retained = sum({(item.relative_path, item.sha256): item.byte_size for item in locators if item}.values())
    record = {
        "attempt_id": attempt_id, "ordinal": ordinal,
        "kind": "INITIAL" if ordinal == 0 else "REPAIR", "parent_attempt_id": parent_id,
        "terminal": terminal, "disposition": disposition,
        "input_artifact_locator": vars(raw_journal_locator),
        "output_artifact_locator": None if output_artifact_locator is None else vars(output_artifact_locator),
        "event_journal_locator": vars(binding_locator), "feedback_class": feedback_class,
        "feedback_locator": None if feedback_locator is None else vars(feedback_locator),
        "diagnostic_codes": list(diagnostics),
        "observed_resources": {
            "command_count": command_count, "active_ns": active_ns, "retained_bytes": retained,
        },
        "attempt_sha256": "",
    }
    record["attempt_sha256"] = canonical_sha256({key: value for key, value in record.items() if key != "attempt_sha256"})
    return record

def run_fixture_plan(
    plan: ExecutionPlan,
    event_path: str | Path,
    run_id: str,
    runner: Callable[[CommandSpec], int],
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> dict[str, int | str]:
    """Run fixture callbacks and separate child time from engine overhead."""
    if TOKEN.fullmatch(run_id) is None:
        raise ExecutionError("run_id has invalid syntax")
    started, child_ns, invoked, sequence = clock_ns(), 0, 0, 0
    with _event_path(event_path).open("x", encoding="utf-8") as journal:
        def record(event: str, **payload: Any) -> int:
            nonlocal sequence
            stamp = clock_ns()
            row = {
                "run_id": run_id,
                "sequence": sequence,
                "event": event,
                "monotonic_ns": stamp,
                "payload": payload,
            }
            journal.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            journal.flush()
            sequence += 1
            return stamp
        def finish(event: str, state: str, **payload: Any) -> dict[str, int | str]:
            ended = record(
                event, terminal_state=state, command_count=invoked, child_ns=child_ns, **payload
            )
            total = ended - started
            return {
                "terminal_state": state,
                "command_count": invoked,
                "total_ns": total,
                "child_ns": child_ns,
                "self_overhead_ns": max(0, total - child_ns),
            }
        record(
            "run_started", origin_ns=started, arm=plan.arm, case_id=plan.case_id,
            input_sha256=plan.input_sha256, environment_id=plan.environment_id,
            cache_policy=plan.cache_policy, plan_sha256=plan.plan_sha256,
        )
        try:
            for command in plan.commands:
                record("command_started", name=command.name, argv=command.argv)
                invoked += 1
                child_started = clock_ns()
                try:
                    status = runner(command)
                except BaseException:
                    child_ns += clock_ns() - child_started
                    raise
                elapsed = clock_ns() - child_started
                child_ns += elapsed
                if isinstance(status, bool) or not isinstance(status, int):
                    raise ExecutionError("fixture runner must return an integer exit status")
                record(
                    "command_finished",
                    name=command.name,
                    exit_status=status,
                    child_elapsed_ns=elapsed,
                )
                if status:
                    return finish("run_blocked", "BLOCKED")
        except BaseException as exc:
            finish("run_interrupted", "INTERRUPTED", exception_type=type(exc).__name__)
            raise
        return finish("run_finished", "READY")
