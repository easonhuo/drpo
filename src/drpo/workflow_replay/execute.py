"""Deterministic dry-run planning and append-only fixture execution."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import CaseManifest

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


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

        def record(event: str, **payload: Any) -> None:
            nonlocal sequence
            row = {
                "run_id": run_id,
                "sequence": sequence,
                "event": event,
                "monotonic_ns": clock_ns(),
                "payload": payload,
            }
            journal.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            journal.flush()
            sequence += 1

        def summary(state: str) -> dict[str, int | str]:
            total = clock_ns() - started
            return {
                "terminal_state": state,
                "command_count": invoked,
                "total_ns": total,
                "child_ns": child_ns,
                "self_overhead_ns": max(0, total - child_ns),
            }

        record(
            "run_started", arm=plan.arm, case_id=plan.case_id, input_sha256=plan.input_sha256,
            environment_id=plan.environment_id, cache_policy=plan.cache_policy,
            plan_sha256=plan.plan_sha256,
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
                    result = summary("BLOCKED")
                    record("run_blocked", **result)
                    return result
        except BaseException as exc:
            result = summary("INTERRUPTED")
            record("run_interrupted", exception_type=type(exc).__name__, **result)
            raise
        result = summary("READY")
        record("run_finished", **result)
        return result
