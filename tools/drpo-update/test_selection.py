#!/usr/bin/env python3
"""Deterministic changed-path test selection for ``drpo-update``.

The selector is intentionally standard-library only.  It reads a trusted impact
map from the current repository state, classifies the candidate diff, and
returns either a focused fast gate or a fail-closed full-suite decision.
"""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

RISK_RANK = {"low": 0, "medium": 1, "high": 2}
ALLOWED_MODES = {"auto", "fast", "full"}


class TestSelectionError(RuntimeError):
    """Invalid map, unsafe downgrade, or selected command failure."""


@dataclass(frozen=True)
class TestPlan:
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


def select_test_plan(
    changed_paths: Sequence[str],
    impact_map_path: Path,
    *,
    requested_mode: str = "auto",
) -> TestPlan:
    if requested_mode not in ALLOWED_MODES:
        raise TestSelectionError(f"unsupported test mode: {requested_mode}")
    normalized = _dedupe(tuple(path.replace(os.sep, "/").lstrip("./") for path in changed_paths if path))
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
    if not isinstance(control_patterns, list) or not all(isinstance(item, str) for item in control_patterns):
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
        if not isinstance(patterns, list) or not patterns or not all(isinstance(item, str) for item in patterns):
            raise TestSelectionError(f"group {group_id} requires string patterns")
        group_matches = [path for path in normalized if _matches(path, patterns)]
        if not group_matches:
            continue
        matched_groups.append(group_id)
        matched_paths.update(group_matches)
        if RISK_RANK[group_risk] > RISK_RANK[risk]:
            risk = group_risk
        targets = group.get("pytest_targets", [])
        if not isinstance(targets, list) or not all(isinstance(item, str) and item for item in targets):
            raise TestSelectionError(f"group {group_id} pytest_targets must be strings")
        pytest_targets.extend(targets)
        raw_validators = group.get("validators", [])
        if not isinstance(raw_validators, list):
            raise TestSelectionError(f"group {group_id} validators must be a list")
        for command in raw_validators:
            if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
                raise TestSelectionError(f"group {group_id} validator commands must be non-empty string lists")
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
            raise TestSelectionError(f"fast mode cannot override fail-closed full selection: {detail}")
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


def _run_command(command: Sequence[str], *, cwd: Path) -> None:
    proc = subprocess.run(list(command), cwd=cwd, text=True, check=False)
    if proc.returncode != 0:
        raise TestSelectionError(
            f"selected test command failed ({proc.returncode}): {' '.join(command)}"
        )


def _resolve_command(
    raw_command: Sequence[str],
    *,
    python_executable: str,
) -> tuple[str, ...] | None:
    resolved: list[str] = []
    for token in raw_command:
        if token == "{python}":
            resolved.append(python_executable)
        elif token == "{ruff}":
            ruff = shutil.which("ruff")
            if not ruff:
                print("WARNING: ruff is not installed; selected Ruff command skipped", file=sys.stderr)
                return None
            resolved.append(ruff)
        else:
            resolved.append(token)
    return tuple(resolved)


def execute_test_plan(
    plan: TestPlan,
    *,
    worktree: Path,
    python_executable: str = sys.executable,
) -> list[str]:
    """Execute a selected plan and return human-readable command labels."""
    executed: list[str] = []
    if plan.selected_mode == "full":
        for raw_command in plan.full_commands:
            command = _resolve_command(raw_command, python_executable=python_executable)
            if command is None:
                continue
            _run_command(command, cwd=worktree)
            executed.append(" ".join(raw_command) + " [full]")
        return executed

    existing_python = [
        path for path in plan.changed_python_files if (worktree / path).is_file()
    ]
    if existing_python:
        command = (python_executable, "-m", "py_compile", *existing_python)
        _run_command(command, cwd=worktree)
        executed.append("python -m py_compile [changed Python files]")
        ruff = shutil.which("ruff")
        if ruff:
            _run_command((ruff, "check", *existing_python), cwd=worktree)
            executed.append("ruff check [changed Python files]")
        else:
            print("WARNING: ruff is not installed; changed-file Ruff gate skipped", file=sys.stderr)

    for raw_command in plan.validators:
        command = _resolve_command(raw_command, python_executable=python_executable)
        if command is None:
            continue
        _run_command(command, cwd=worktree)
        executed.append(" ".join(raw_command))

    existing_targets = [target for target in plan.pytest_targets if (worktree / target.split("::", 1)[0]).exists()]
    missing_targets = [target for target in plan.pytest_targets if target not in existing_targets]
    if missing_targets:
        raise TestSelectionError(
            "selected pytest targets are missing from candidate worktree: " + ", ".join(missing_targets)
        )
    if existing_targets:
        command = (python_executable, "-m", "pytest", "-q", *existing_targets)
        _run_command(command, cwd=worktree)
        executed.append("python -m pytest -q " + " ".join(existing_targets))
    return executed
