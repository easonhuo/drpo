"""Thin Arm-B composition over the accepted fastpath and V1 commands."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .execute import CommandSpec, TOKEN

SHA40 = re.compile(r"[0-9a-f]{40}")


class OrchestrationError(RuntimeError):
    """The candidate path stopped before local READY."""

    def __init__(self, step: str, message: str):
        super().__init__(f"{step}: {message}")
        self.step = step
        self.message = message


@dataclass(frozen=True)
class ProcessResult:
    """Shell-free child result returned by an injected process invoker."""

    returncode: int
    stdout: str
    stderr: str = ""


@dataclass(frozen=True)
class CandidateOutcome:
    """Successful local READY identity and the exact work performed."""

    preparation_id: str
    preparation_dir: str
    transaction_dir: str
    ready_commit_sha: str
    commands: tuple[CommandSpec, ...]
    placements: tuple[str, ...]


Invoker = Callable[[CommandSpec], ProcessResult]


def _plain_path(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = path.absolute()
    current = path
    while True:
        if current.is_symlink():
            raise OrchestrationError(label, f"path contains a symlink: {current}")
        if current.parent == current:
            return path
        current = current.parent


def _existing_dir(value: Any, root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise OrchestrationError(label, "child output did not provide an absolute directory")
    raw = _plain_path(value, label)
    resolved_root = root.expanduser().resolve()
    resolved = raw.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise OrchestrationError(label, f"directory escapes declared root: {raw}") from exc
    if not resolved.is_dir():
        raise OrchestrationError(label, f"directory does not exist: {resolved}")
    current = raw
    while True:
        if current.is_symlink():
            raise OrchestrationError(label, f"directory path contains a symlink: {current}")
        if current.resolve() == resolved_root:
            break
        if current.parent == current:
            raise OrchestrationError(label, "directory is not under the declared root")
        current = current.parent
    return resolved


def _payload(command: CommandSpec, result: ProcessResult, expected_state: str) -> dict[str, Any]:
    if not isinstance(result, ProcessResult):
        raise OrchestrationError(command.name, "invoker returned an invalid result object")
    if isinstance(result.returncode, bool) or not isinstance(result.returncode, int):
        raise OrchestrationError(command.name, "returncode must be an integer")
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise OrchestrationError(command.name, f"child exit {result.returncode}: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise OrchestrationError(command.name, "child stdout is not one JSON object") from exc
    if not isinstance(payload, dict):
        raise OrchestrationError(command.name, "child JSON must be an object")
    if payload.get("status") != "PASS" or payload.get("state") != expected_state:
        raise OrchestrationError(
            command.name,
            f"expected PASS/{expected_state}, got {payload.get('status')}/{payload.get('state')}",
        )
    return payload


def _copy_exact_tree(source_root: Path, destination_root: Path, label: str) -> tuple[str, ...]:
    if source_root.is_symlink() or not source_root.is_dir():
        raise OrchestrationError(label, f"source tree is missing or unsafe: {source_root}")
    placements: list[str] = []
    for current, directories, filenames in os.walk(source_root, followlinks=False):
        directories.sort()
        filenames.sort()
        base = Path(current)
        for directory in directories:
            if (base / directory).is_symlink():
                raise OrchestrationError(label, f"source tree contains a symlink: {base / directory}")
        for filename in filenames:
            source = base / filename
            if source.is_symlink() or not source.is_file():
                raise OrchestrationError(label, f"source entry is not a regular file: {source}")
            relative = source.relative_to(source_root)
            destination = destination_root / relative
            parent = destination.parent
            while parent != destination_root:
                if parent.is_symlink():
                    raise OrchestrationError(label, f"destination parent is a symlink: {parent}")
                parent = parent.parent
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.is_symlink() or not destination.is_file():
                    raise OrchestrationError(label, f"destination is not a regular file: {destination}")
                if destination.read_bytes() != source.read_bytes():
                    raise OrchestrationError(label, f"destination conflicts with prepared bytes: {destination}")
            else:
                shutil.copyfile(source, destination)
                os.chmod(destination, stat.S_IMODE(source.stat().st_mode))
            placements.append(f"{label}:{relative.as_posix()}")
    if not placements:
        raise OrchestrationError(label, "prepared tree contains no files")
    return tuple(placements)


def run_candidate(
    *,
    repo_root: str | Path,
    spec_path: str | Path,
    preparation_root: str | Path,
    transaction_root: str | Path,
    python_executable: str,
    invoke: Invoker,
) -> CandidateOutcome:
    """Run the accepted preparation and V1 stages once, without repair or publication."""
    repository = _plain_path(repo_root, "repo_root").resolve()
    spec = _plain_path(spec_path, "spec_path").resolve()
    preparations = _plain_path(preparation_root, "preparation_root")
    transactions = _plain_path(transaction_root, "transaction_root")
    if not repository.is_dir():
        raise OrchestrationError("repo_root", f"repository directory is unavailable: {repository}")
    if not spec.is_file() or spec.is_symlink():
        raise OrchestrationError("spec_path", f"spec is not a regular file: {spec}")
    if not isinstance(python_executable, str) or not python_executable or "\x00" in python_executable:
        raise OrchestrationError("python_executable", "python executable must be non-empty and NUL-free")

    commands: list[CommandSpec] = []
    placements: list[str] = []

    def call(name: str, argv: tuple[str, ...], state: str) -> dict[str, Any]:
        command = CommandSpec(name, argv)
        commands.append(command)
        return _payload(command, invoke(command), state)

    prepared = call(
        "prepare-inputs",
        (
            python_executable,
            "scripts/prepare_dev_pilot_registration.py",
            "--repo-root",
            str(repository),
            "--spec",
            str(spec),
            "--output-root",
            str(preparations),
            "--json",
        ),
        "PREPARED_INPUTS",
    )
    preparation_id = prepared.get("preparation_id")
    if not isinstance(preparation_id, str) or TOKEN.fullmatch(preparation_id) is None:
        raise OrchestrationError("prepare-inputs", "prepared identity is invalid")
    preparation_dir = _existing_dir(
        prepared.get("preparation_dir"), preparations, "preparation_dir"
    )
    placements.extend(
        _copy_exact_tree(preparation_dir / "repository_overlay", repository, "repository")
    )
    request = repository / "docs" / "integrations" / preparation_id / "INTEGRATION_REQUEST.yaml"
    if not request.is_file() or request.is_symlink():
        raise OrchestrationError("repository", f"prepared request was not placed: {request}")

    reviewed = call(
        "v1-plan",
        (
            python_executable,
            "scripts/integrate_dev_branch.py",
            "plan",
            "--repo-root",
            str(repository),
            "--request",
            str(request),
            "--transaction-root",
            str(transactions),
            "--json",
        ),
        "REVIEWED",
    )
    transaction_dir = _existing_dir(
        reviewed.get("attempt_dir"), transactions, "transaction_dir"
    )
    call(
        "v1-prepare",
        (
            python_executable,
            "scripts/dev_integration_write_path.py",
            "--transaction-dir",
            str(transaction_dir),
            "--json",
        ),
        "PREPARED",
    )

    transaction_inputs = preparation_dir / "transaction_inputs"
    if transaction_inputs.exists():
        placements.extend(
            _copy_exact_tree(transaction_inputs, transaction_dir, "transaction")
        )

    for name, state in (
        ("normalize", "NORMALIZED"),
        ("gate", "REQUIRED_GATES_PASSED"),
        ("finalize", "READY"),
    ):
        final = call(
            f"v1-{name}",
            (
                python_executable,
                "scripts/dev_integration_finalize.py",
                name,
                "--transaction-dir",
                str(transaction_dir),
                "--json",
            ),
            state,
        )
    ready_sha = final.get("ready_commit_sha")
    if not isinstance(ready_sha, str) or SHA40.fullmatch(ready_sha) is None:
        raise OrchestrationError("v1-finalize", "READY output lacks a full commit SHA")
    if len({command.name for command in commands}) != len(commands):
        raise OrchestrationError("internal", "candidate generated duplicate command names")
    return CandidateOutcome(
        preparation_id,
        str(preparation_dir),
        str(transaction_dir),
        ready_sha,
        tuple(commands),
        tuple(placements),
    )
