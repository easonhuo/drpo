#!/usr/bin/env python3
"""Deterministic changed-path test selection for ``drpo-update``.

The selector is intentionally standard-library only. It reads a trusted impact
map from the current repository state, classifies the candidate diff, and
returns either a focused fast gate or a fail-closed full-suite decision.

Execution is aggregate-by-default: every selected gate is attempted, every
command gets its own complete log, and failure is reported only after all
independent gates have run. This avoids one-failure-at-a-time repair cycles.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

RISK_RANK = {"low": 0, "medium": 1, "high": 2}
ALLOWED_MODES = {"auto", "fast", "full"}


class TestSelectionError(RuntimeError):
    """Invalid map, unsafe downgrade, or selected command failure."""

    __test__ = False


@dataclass(frozen=True)
class CommandOutcome:
    """Durable result for one selected integration-gate command."""

    label: str
    command: tuple[str, ...]
    returncode: int
    log_file: str | None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "command": list(self.command),
            "returncode": self.returncode,
            "passed": self.passed,
            "log_file": self.log_file,
            "error": self.error,
        }


class TestExecutionError(TestSelectionError):
    """Aggregate failure raised after all independent selected gates ran."""

    __test__ = False

    def __init__(self, message: str, outcomes: Sequence[CommandOutcome]):
        super().__init__(message)
        self.outcomes = tuple(outcomes)


@dataclass(frozen=True)
class TestPlan:
    __test__ = False

    requested_mode: str
    selected_mode: str
    risk: str
    reason: str
    changed_paths: tuple[str, ...]
    matched_groups: tuple[str, ...] = ()
    unknown_paths: tuple[str, ...] = ()
    pytest_targets: tuple[str, ...] = ()
    validators: tuple[tuple[str, ...], ...] = ()
    changed_python_files: tuple[str, ...] = ()
    full_commands: tuple[tuple[str, ...], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_mode": self.requested_mode,
            "selected_mode": self.selected_mode,
            "risk": self.risk,
            "reason": self.reason,
            "changed_paths": list(self.changed_paths),
            "matched_groups": list(self.matched_groups),
            "unknown_paths": list(self.unknown_paths),
            "pytest_targets": list(self.pytest_targets),
            "validators": [list(command) for command in self.validators],
            "changed_python_files": list(self.changed_python_files),
            "full_commands": [list(command) for command in self.full_commands],
        }


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _load_map(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TestSelectionError(f"cannot read test impact map {path}: {exc}") from exc
    if payload.get("schema_version") != 1:
        raise TestSelectionError("test impact map schema_version must equal 1")
    if payload.get("unknown_path_policy") != "full":
        raise TestSelectionError("unknown_path_policy must be fail-closed value 'full'")
    groups = payload.get("groups")
    if not isinstance(groups, list) or not groups:
        raise TestSelectionError("test impact map must contain a non-empty groups list")
    return payload


def _matches(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _normalize_changed_path(path: str) -> str:
    """Normalize separators without stripping leading dots from hidden paths."""
    normalized = path.replace(os.sep, "/").replace("\\", "/")
    return normalized.removeprefix("./")


def select_test_plan(
    changed_paths: Sequence[str],
    impact_map_path: Path,
    *,
    requested_mode: str = "auto",
) -> TestPlan:
    if requested_mode not in ALLOWED_MODES:
        raise TestSelectionError(f"unsupported test mode: {requested_mode}")
    normalized = _dedupe(
        tuple(_normalize_changed_path(path) for path in changed_paths if path)
    )
    if not normalized:
        raise TestSelectionError("candidate integration has no changed paths")

    payload = _load_map(impact_map_path)
    raw_full_commands = payload.get("full_commands")
    if not isinstance(raw_full_commands, list) or not raw_full_commands:
        raise TestSelectionError("full_commands must be a non-empty list")
    full_commands: list[tuple[str, ...]] = []
    for command in raw_full_commands:
        if not isinstance(command, list) or not command or not all(
            isinstance(item, str) and item for item in command
        ):
            raise TestSelectionError("full_commands entries must be non-empty string lists")
        full_commands.append(tuple(command))
    control_patterns = payload.get("control_plane_patterns", [])
    if not isinstance(control_patterns, list) or not all(
        isinstance(item, str) for item in control_patterns
    ):
        raise TestSelectionError("control_plane_patterns must be a list of strings")

    risk = "low"
    matched_groups: list[str] = []
    matched_paths: set[str] = set()
    pytest_targets: list[str] = []
    validators: list[tuple[str, ...]] = []

    for group in payload["groups"]:
        if not isinstance(group, dict):
            raise TestSelectionError("each test impact group must be an object")
        group_id = group.get("id")
        group_risk = group.get("risk")
        patterns = group.get("patterns")
        if not isinstance(group_id, str) or not group_id:
            raise TestSelectionError("each test impact group requires a non-empty id")
        if group_risk not in RISK_RANK:
            raise TestSelectionError(f"group {group_id} has invalid risk: {group_risk}")
        if not isinstance(patterns, list) or not patterns or not all(
            isinstance(item, str) for item in patterns
        ):
            raise TestSelectionError(f"group {group_id} requires string patterns")
        group_matches = [path for path in normalized if _matches(path, patterns)]
        if not group_matches:
            continue
        matched_groups.append(group_id)
        matched_paths.update(group_matches)
        if RISK_RANK[group_risk] > RISK_RANK[risk]:
            risk = group_risk
        targets = group.get("pytest_targets", [])
        if not isinstance(targets, list) or not all(
            isinstance(item, str) and item for item in targets
        ):
            raise TestSelectionError(f"group {group_id} pytest_targets must be strings")
        pytest_targets.extend(targets)
        raw_validators = group.get("validators", [])
        if not isinstance(raw_validators, list):
            raise TestSelectionError(f"group {group_id} validators must be a list")
        for command in raw_validators:
            if not isinstance(command, list) or not command or not all(
                isinstance(item, str) for item in command
            ):
                raise TestSelectionError(
                    f"group {group_id} validator commands must be non-empty string lists"
                )
            validators.append(tuple(command))

    control_paths = [path for path in normalized if _matches(path, control_patterns)]
    if control_paths:
        risk = "high"
        matched_paths.update(control_paths)
        if "test_control_plane" not in matched_groups:
            matched_groups.append("test_control_plane")

    unknown = tuple(path for path in normalized if path not in matched_paths)
    requires_full = bool(unknown) or risk == "high"
    if requested_mode == "full":
        selected_mode = "full"
        reason = "full mode explicitly requested"
    elif requested_mode == "fast":
        if requires_full:
            detail = "unknown paths" if unknown else "high-risk control or shared code"
            raise TestSelectionError(
                f"fast mode cannot override fail-closed full selection: {detail}"
            )
        selected_mode = "fast"
        reason = "fast mode explicitly requested and selector found no high-risk or unknown paths"
    elif requires_full:
        selected_mode = "full"
        reason = "unknown paths require full suite" if unknown else "high-risk path requires full suite"
    else:
        selected_mode = "fast"
        reason = "all changed paths matched low/medium-risk groups"

    changed_python = tuple(path for path in normalized if path.endswith(".py"))
    return TestPlan(
        requested_mode=requested_mode,
        selected_mode=selected_mode,
        risk=risk,
        reason=reason,
        changed_paths=normalized,
        matched_groups=_dedupe(matched_groups),
        unknown_paths=unknown,
        pytest_targets=_dedupe(pytest_targets),
        validators=tuple(dict.fromkeys(validators)),
        changed_python_files=changed_python,
        full_commands=tuple(dict.fromkeys(full_commands)),
    )


def _safe_log_name(index: int, label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-._")
    return f"{index:02d}-{(normalized or 'command')[:80]}.log"


def _resolve_command(
    raw_command: Sequence[str],
    *,
    python_executable: str,
) -> tuple[tuple[str, ...] | None, str | None]:
    resolved: list[str] = []
    for token in raw_command:
        if token == "{python}":
            resolved.append(python_executable)
        elif token == "{ruff}":
            ruff = shutil.which("ruff")
            if not ruff:
                return None, "required executable 'ruff' was not found on PATH"
            resolved.append(ruff)
        else:
            resolved.append(token)
    return tuple(resolved), None


def _execute_command(
    command: Sequence[str] | None,
    *,
    label: str,
    cwd: Path,
    log_path: Path | None,
    setup_error: str | None = None,
) -> CommandOutcome:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    command_tuple = tuple(command or ())
    header = f"$ {' '.join(command_tuple) if command_tuple else label}\n"
    if setup_error:
        text = header + setup_error + "\n"
        if log_path is not None:
            log_path.write_text(text)
        print(setup_error, file=sys.stderr)
        return CommandOutcome(
            label=label,
            command=command_tuple,
            returncode=127,
            log_file=str(log_path) if log_path else None,
            error=setup_error,
        )

    assert command is not None
    try:
        proc = subprocess.Popen(
            list(command),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        text = header + f"command start failed: {exc}\n"
        if log_path is not None:
            log_path.write_text(text)
        print(text.rstrip(), file=sys.stderr)
        return CommandOutcome(
            label=label,
            command=command_tuple,
            returncode=127,
            log_file=str(log_path) if log_path else None,
            error=str(exc),
        )

    log_handle = log_path.open("w") if log_path is not None else None
    try:
        if log_handle:
            log_handle.write(header)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            if log_handle:
                log_handle.write(line)
        returncode = proc.wait()
        trailer = f"\n[exit_code] {returncode}\n"
        if log_handle:
            log_handle.write(trailer)
    finally:
        if log_handle:
            log_handle.close()
    return CommandOutcome(
        label=label,
        command=command_tuple,
        returncode=returncode,
        log_file=str(log_path) if log_path else None,
    )


def execute_test_plan(
    plan: TestPlan,
    *,
    worktree: Path,
    python_executable: str = sys.executable,
    log_dir: Path | None = None,
    outcome_callback: Callable[[CommandOutcome], None] | None = None,
) -> list[str]:
    """Execute every selected gate, then fail once with the complete failure set."""

    outcomes: list[CommandOutcome] = []
    labels: list[str] = []

    def record(
        label: str,
        command: Sequence[str] | None,
        *,
        setup_error: str | None = None,
    ) -> None:
        index = len(outcomes) + 1
        log_path = log_dir / _safe_log_name(index, label) if log_dir else None
        outcome = _execute_command(
            command,
            label=label,
            cwd=worktree,
            log_path=log_path,
            setup_error=setup_error,
        )
        outcomes.append(outcome)
        labels.append(label)
        if outcome_callback:
            outcome_callback(outcome)

    if plan.selected_mode == "full":
        for raw_command in plan.full_commands:
            label = " ".join(raw_command) + " [full]"
            command, setup_error = _resolve_command(
                raw_command,
                python_executable=python_executable,
            )
            record(label, command, setup_error=setup_error)
    else:
        existing_python = [
            path for path in plan.changed_python_files if (worktree / path).is_file()
        ]
        if existing_python:
            record(
                "python -m py_compile [changed Python files]",
                (python_executable, "-m", "py_compile", *existing_python),
            )
            ruff = shutil.which("ruff")
            record(
                "ruff check [changed Python files]",
                (ruff, "check", *existing_python) if ruff else None,
                setup_error=None if ruff else "required executable 'ruff' was not found on PATH",
            )

        for raw_command in plan.validators:
            command, setup_error = _resolve_command(
                raw_command,
                python_executable=python_executable,
            )
            record(" ".join(raw_command), command, setup_error=setup_error)

        existing_targets = [
            target
            for target in plan.pytest_targets
            if (worktree / target.split("::", 1)[0]).exists()
        ]
        missing_targets = [
            target for target in plan.pytest_targets if target not in existing_targets
        ]
        if missing_targets:
            record(
                "validate selected pytest targets",
                None,
                setup_error=(
                    "selected pytest targets are missing from candidate worktree: "
                    + ", ".join(missing_targets)
                ),
            )
        if existing_targets:
            label = "python -m pytest -q " + " ".join(existing_targets)
            record(
                label,
                (python_executable, "-m", "pytest", "-q", *existing_targets),
            )

    failed = [outcome for outcome in outcomes if not outcome.passed]
    if failed:
        summary = "; ".join(
            f"{outcome.label} (exit {outcome.returncode})" for outcome in failed
        )
        raise TestExecutionError(
            f"{len(failed)} of {len(outcomes)} selected test commands failed: {summary}",
            outcomes,
        )
    return labels
