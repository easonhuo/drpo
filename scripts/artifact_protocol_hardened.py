#!/usr/bin/env python3
"""Hardened implementation for DRPO experiment provenance and artifact delivery.

This module is intentionally imported by the three public command wrappers:
``package_experiment_hardened.py``, ``verify_experiment_package_hardened.py``, and
``run_experiment_guard_hardened.py``.  Keeping the public entry points stable avoids
breaking existing local automation while allowing the safety policy to evolve in
one place.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import queue
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator

FINAL_KINDS = {"governance", "experiment-final"}
RESULT_KINDS = {
    "experiment-checkpoint",
    "experiment-failed",
    "experiment-raw-complete",
    "experiment-final",
}
ALL_KINDS = FINAL_KINDS | RESULT_KINDS
RECOVERY_KINDS = {
    "experiment-checkpoint",
    "experiment-failed",
    "experiment-raw-complete",
}
REQUIRED_TOP_LEVEL = {
    "update.patch",
    "BASE_COMMIT.txt",
    "CHANGE_SUMMARY.md",
    "TEST_COMMANDS.sh",
    "ARTIFACT_MANIFEST.json",
    "SHA256SUMS.txt",
}
SHA_RE = re.compile(r"[0-9a-f]{40}")
EXPERIMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
MODEL_STATE_SUFFIXES = {
    ".ckpt",
    ".pt",
    ".pth",
    ".safetensors",
}
GENERIC_BINARY_SUFFIXES = {
    ".bin",
    ".npz",
    ".npy",
}
MODEL_OR_OPTIMIZER_NAMES = (
    "adapter_model",
    "pytorch_model",
    "model-",
    "optimizer",
    "scheduler",
    "rng_state",
    "trainer_state",
)
FOUNDATION_MODEL_PATH_TOKENS = {
    "base_model",
    "foundation_model",
    "pretrained_model",
    "hf_cache",
    "huggingface_cache",
}
SIDECAR_PURPOSES = {
    "cross_machine_transfer",
    "restart",
    "independent_audit",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(
    cmd: list[str],
    cwd: Path,
    *,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def git_output(repo: Path, *args: str, check: bool = True) -> str:
    return run(["git", *args], repo, check=check).stdout.strip()


def ensure_git_repo(repo: Path) -> None:
    result = run(["git", "rev-parse", "--is-inside-work-tree"], repo, check=False)
    if result.returncode != 0 or result.stdout.strip() != "true":
        raise ValueError(f"Not a Git work tree: {repo}")


def validate_sha(value: str) -> str:
    value = value.strip().lower()
    if not SHA_RE.fullmatch(value):
        raise ValueError(f"Expected a full 40-character Git SHA, got: {value!r}")
    return value


def ensure_commit_exists(repo: Path, commit: str) -> None:
    commit = validate_sha(commit)
    result = run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], repo, check=False)
    if result.returncode != 0:
        raise ValueError(f"Commit {commit} is not available in repository {repo}")



def validate_experiment_id(value: str) -> str:
    if not EXPERIMENT_ID_RE.fullmatch(value):
        raise ValueError(
            "experiment_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}; "
            f"got {value!r}"
        )
    return value


def first_symlink_component(path: Path) -> Path | None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for part in parts:
        current /= part
        if current.is_symlink():
            return current
    return None


def reject_symlink_path(path: Path, label: str) -> None:
    component = first_symlink_component(path)
    if component is not None:
        raise ValueError(f"{label} may not use a symbolic link path component: {component}")


def tracked_paths_under(repo: Path, path: Path) -> list[str]:
    try:
        rel = path.absolute().relative_to(repo.resolve())
    except ValueError:
        return []
    if not rel.parts:
        return git_output(repo, "ls-files", check=False).splitlines()
    result = run(["git", "ls-files", "--", rel.as_posix()], repo, check=False)
    return [line for line in result.stdout.splitlines() if line.strip()]


def reject_tracked_runtime_path(repo: Path, path: Path, label: str, *, subtree: bool) -> None:
    tracked = tracked_paths_under(repo, path)
    if not tracked:
        return
    if subtree or path.is_file():
        preview = ", ".join(tracked[:5])
        extra = " ..." if len(tracked) > 5 else ""
        raise ValueError(
            f"{label} overlaps tracked repository content ({preview}{extra}); "
            "runtime paths may not hide tracked modifications"
        )


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def require_identity(
    payload: dict[str, Any],
    *,
    label: str,
    experiment_id: str,
    base_commit: str,
) -> None:
    if payload.get("experiment_id") != experiment_id:
        raise ValueError(
            f"{label} experiment_id must be {experiment_id!r}, got {payload.get('experiment_id')!r}"
        )
    if payload.get("base_commit") != base_commit:
        raise ValueError(
            f"{label} base_commit must be {base_commit}, got {payload.get('base_commit')!r}"
        )


def git_head(repo: Path) -> str:
    return validate_sha(git_output(repo, "rev-parse", "HEAD"))


def git_branch(repo: Path) -> str:
    return git_output(repo, "branch", "--show-current", check=False) or "DETACHED"


def git_status(repo: Path) -> list[str]:
    text = git_output(repo, "status", "--porcelain", "--untracked-files=all", check=False)
    return [line for line in text.splitlines() if line.strip()]


def git_status_excluding(repo: Path, ignored: Iterable[Path]) -> list[str]:
    ignored_rel: list[Path] = []
    for path in ignored:
        try:
            ignored_rel.append(path.resolve().relative_to(repo.resolve()))
        except ValueError:
            continue
    rows: list[str] = []
    for line in git_status(repo):
        raw = line[3:] if len(line) >= 4 else line
        # Renames are rendered as ``old -> new``; conservatively inspect the destination.
        raw = raw.split(" -> ")[-1].strip().strip('"')
        rel = Path(raw)
        if is_generated_artifact(rel):
            continue
        if any(rel == base or base in rel.parents for base in ignored_rel):
            continue
        rows.append(line)
    return rows


def resolve_origin_main(repo: Path, timeout: float = 5.0) -> dict[str, Any]:
    """Resolve origin/main without trusting a web page or cached search result.

    Priority is an authoritative ``git ls-remote`` query.  If networking is not
    available, the local remote-tracking ref is reported as a fallback, but the
    result is explicitly marked as non-authoritative.
    """
    configured = run(["git", "remote", "get-url", "origin"], repo, check=False)
    if configured.returncode != 0:
        return {
            "sha": None,
            "source": "unresolved",
            "authoritative": False,
            "error": "origin remote is not configured",
        }
    remote_error = ""
    try:
        remote = run(
            ["git", "ls-remote", "--exit-code", "origin", "refs/heads/main"],
            repo,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        remote = None
        remote_error = f"git ls-remote timed out after {timeout:.1f}s"
    if remote is not None and remote.returncode == 0 and remote.stdout.strip():
        sha = validate_sha(remote.stdout.split()[0])
        return {
            "sha": sha,
            "source": "git_ls_remote_origin_main",
            "authoritative": True,
            "error": None,
        }
    if remote is not None:
        remote_error = (remote.stderr or remote.stdout).strip() or "ls-remote unavailable"
    fallback = run(
        ["git", "rev-parse", "refs/remotes/origin/main"],
        repo,
        check=False,
    )
    if fallback.returncode == 0 and fallback.stdout.strip():
        return {
            "sha": validate_sha(fallback.stdout),
            "source": "local_remote_tracking_ref",
            "authoritative": False,
            "error": remote_error,
        }
    return {
        "sha": None,
        "source": "unresolved",
        "authoritative": False,
        "error": remote_error or "origin/main unavailable",
    }


def resolve_commit(
    repo: Path,
    expected_sha: str | None,
    *,
    require_origin_match: bool = False,
) -> dict[str, Any]:
    ensure_git_repo(repo)
    local = git_head(repo)
    expected = validate_sha(expected_sha) if expected_sha else local
    if local != expected:
        raise ValueError(f"Local HEAD {local} does not match expected commit {expected}")
    origin = resolve_origin_main(repo)
    origin_matches_local = bool(origin["authoritative"] and origin["sha"] == local)
    if require_origin_match and not origin["authoritative"]:
        raise ValueError(
            "Could not authoritatively verify origin/main with git ls-remote: "
            f"{origin['error']}"
        )
    if require_origin_match and not origin_matches_local:
        raise ValueError(
            f"Authoritative origin/main {origin['sha']} does not match local HEAD {local}"
        )
    return {
        "local_head": local,
        "expected_sha": expected,
        "origin_main": origin,
        "origin_matches_local": origin_matches_local,
        "verified": origin_matches_local,
    }


def safe_repo_rel(value: str | Path) -> Path:
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"Path must be repository-relative and may not escape: {value}")
    return rel


def is_generated_artifact(path: Path) -> bool:
    ignored_parts = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    return bool(ignored_parts.intersection(path.parts)) or path.suffix in {".pyc", ".pyo"}


def changed_paths(repo: Path) -> list[Path]:
    tracked = git_output(repo, "diff", "--name-only", "HEAD", "--").splitlines()
    staged = git_output(repo, "diff", "--cached", "--name-only", "HEAD", "--").splitlines()
    untracked = git_output(repo, "ls-files", "--others", "--exclude-standard").splitlines()
    paths = sorted({Path(x) for x in tracked + staged + untracked if x.strip()})
    return [
        p
        for p in paths
        if (repo / p).is_file() and not (repo / p).is_symlink() and not is_generated_artifact(p)
    ]


def patch_for_repo(repo: Path, paths: list[Path]) -> str:
    tracked_names = set(git_output(repo, "ls-files").splitlines())
    tracked = [p.as_posix() for p in paths if p.as_posix() in tracked_names]
    untracked = [p.as_posix() for p in paths if p.as_posix() not in tracked_names]
    chunks: list[str] = []
    if tracked:
        result = run(
            ["git", "diff", "--binary", "--full-index", "HEAD", "--", *tracked],
            repo,
        )
        if result.stdout:
            chunks.append(result.stdout)
    for name in untracked:
        result = run(
            ["git", "diff", "--no-index", "--binary", "--", "/dev/null", name],
            repo,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        chunks.append(result.stdout)
    return "".join(chunks)


def copy_changed_files(repo: Path, paths: Iterable[Path], destination: Path) -> None:
    for rel in paths:
        source = repo / rel
        if source.is_symlink():
            raise ValueError(f"Modified file may not be a symlink: {rel}")
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def copy_source_snapshot(
    repo: Path,
    paths: Iterable[Path],
    destination: Path,
    *,
    base_commit: str,
    from_commit: bool,
) -> None:
    for rel in paths:
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if from_commit:
            ensure_commit_exists(repo, base_commit)
            result = subprocess.run(
                ["git", "show", f"{base_commit}:{rel.as_posix()}"],
                cwd=repo,
                capture_output=True,
            )
            if result.returncode != 0:
                message = result.stderr.decode("utf-8", errors="replace").strip()
                raise ValueError(
                    f"Could not read source file {rel} from launch commit {base_commit}: {message}"
                )
            target.write_bytes(result.stdout)
        else:
            source = repo / rel
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"Missing or unsafe source file: {rel}")
            shutil.copy2(source, target)


def validate_source_snapshot_inputs(
    repo: Path,
    paths: Iterable[Path],
    *,
    base_commit: str,
) -> None:
    """Fail before launch when a requested source snapshot is not commit-bound."""
    ensure_commit_exists(repo, base_commit)
    for rel in paths:
        result = run(
            ["git", "cat-file", "-e", f"{base_commit}:{rel.as_posix()}"],
            repo,
            check=False,
        )
        if result.returncode != 0:
            raise ValueError(
                "Requested --source-file is unavailable at launch commit "
                f"{base_commit}: {rel.as_posix()}"
            )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_checksums(stage: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(p for p in stage.rglob("*") if p.is_file() and not p.is_symlink()):
        rel = path.relative_to(stage).as_posix()
        if rel == "SHA256SUMS.txt":
            continue
        checksums[rel] = file_sha256(path)
    (stage / "SHA256SUMS.txt").write_text(
        "".join(f"{digest} {name}\n" for name, digest in checksums.items())
    )
    return checksums


def find_terminal_audit(result_dir: Path) -> Path | None:
    for candidate in (
        result_dir / "TERMINAL_AUDIT.json",
        result_dir / "terminal_audit.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def validate_result_markers(
    kind: str,
    result_dir: Path | None,
    *,
    experiment_id: str,
    base_commit: str,
) -> None:
    if kind not in RESULT_KINDS:
        return
    if result_dir is None or not result_dir.is_dir():
        raise ValueError(f"{kind} requires an existing --result-dir")

    run_manifest_path = result_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise ValueError(f"{kind} requires run_manifest.json")
    run_manifest = read_json_object(run_manifest_path, "run_manifest.json")
    require_identity(
        run_manifest,
        label="run_manifest.json",
        experiment_id=experiment_id,
        base_commit=base_commit,
    )

    logs_dir = result_dir / "logs"
    log_files = []
    if logs_dir.is_dir():
        log_files = [
            path
            for path in logs_dir.rglob("*")
            if path.is_file() and not path.is_symlink()
        ]
    if not log_files:
        raise ValueError(f"{kind} requires at least one regular file under logs/")

    if kind == "experiment-final":
        complete_path = result_dir / "RUN_COMPLETE.json"
        if not complete_path.is_file():
            raise ValueError("experiment-final requires RUN_COMPLETE.json")
        audit_path = find_terminal_audit(result_dir)
        if audit_path is None:
            raise ValueError("experiment-final requires a terminal audit")
        complete = read_json_object(complete_path, "RUN_COMPLETE.json")
        audit = read_json_object(audit_path, audit_path.name)
        require_identity(
            complete,
            label="RUN_COMPLETE.json",
            experiment_id=experiment_id,
            base_commit=base_commit,
        )
        require_identity(
            audit,
            label=audit_path.name,
            experiment_id=experiment_id,
            base_commit=base_commit,
        )
    elif kind == "experiment-failed":
        marker = result_dir / "RUN_FAILED.json"
        if not marker.is_file():
            raise ValueError("experiment-failed requires RUN_FAILED.json")
        failed = read_json_object(marker, "RUN_FAILED.json")
        require_identity(
            failed,
            label="RUN_FAILED.json",
            experiment_id=experiment_id,
            base_commit=base_commit,
        )
    elif kind == "experiment-raw-complete":
        marker = result_dir / "RUN_RAW_COMPLETE.json"
        if not marker.is_file():
            raise ValueError("experiment-raw-complete requires RUN_RAW_COMPLETE.json")
        raw_complete = read_json_object(marker, "RUN_RAW_COMPLETE.json")
        require_identity(
            raw_complete,
            label="RUN_RAW_COMPLETE.json",
            experiment_id=experiment_id,
            base_commit=base_commit,
        )


@dataclass
class ResultEntry:
    path: str
    kind: str
    size_bytes: int
    sha256: str | None
    target: str | None
    include_main: bool
    include_sidecar: bool
    reason: str
    role: str
    storage_path: str | None
    persistence_status: str


def _is_checkpoint_like(rel: Path) -> bool:
    lower = rel.as_posix().lower()
    suffix = rel.suffix.lower()
    named_model_state = any(token in lower for token in MODEL_OR_OPTIMIZER_NAMES)
    if suffix in MODEL_STATE_SUFFIXES:
        return True
    if suffix == ".bin":
        return named_model_state
    return named_model_state


def _is_foundation_model_weight(rel: Path) -> bool:
    # A filename such as model.safetensors can be a legitimate full-finetune output,
    # so classification must use an explicit base/pretrained-cache path marker rather
    # than the filename alone. Imported foundation weights should normally live outside
    # result_dir entirely; this catches accidental copies under a clearly marked path.
    parts = [part.lower() for part in rel.parts[:-1]]
    return any(
        part in FOUNDATION_MODEL_PATH_TOKENS
        or part.startswith("models--")
        or "foundation" in part
        or "pretrained" in part
        for part in parts
    )


def scan_result_tree(
    result_dir: Path,
    *,
    package_kind: str,
    max_single_bytes: int,
    selected_sidecar_paths: set[str],
    large_file_persistence: str,
) -> list[ResultEntry]:
    root = result_dir.resolve()
    entries: list[ResultEntry] = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        kept_dirs: list[str] = []
        for name in sorted(dirnames):
            path = current_path / name
            rel = path.relative_to(root)
            if path.is_symlink():
                resolved = path.resolve(strict=False)
                if not resolved.exists():
                    raise ValueError(f"Result directory contains broken symlink: {rel} -> {resolved}")
                try:
                    resolved.relative_to(root)
                    internal = True
                except ValueError:
                    internal = False
                entries.append(
                    ResultEntry(
                        path=rel.as_posix(),
                        kind="symlink_dir",
                        size_bytes=0,
                        sha256=None,
                        target=str(resolved),
                        include_main=False,
                        include_sidecar=False,
                        reason="internal_reference" if internal else "external_symlink_rejected",
                        role="reference",
                        storage_path=str(path),
                        persistence_status="reference_only",
                    )
                )
                if not internal:
                    raise ValueError(f"Result directory contains external symlink: {rel} -> {resolved}")
                continue
            kept_dirs.append(name)
        dirnames[:] = kept_dirs
        for name in sorted(filenames):
            path = current_path / name
            rel = path.relative_to(root)
            rel_name = rel.as_posix()
            if path.is_symlink():
                resolved = path.resolve(strict=False)
                if not resolved.exists():
                    raise ValueError(f"Result directory contains broken symlink: {rel} -> {resolved}")
                try:
                    resolved.relative_to(root)
                    internal = True
                except ValueError:
                    internal = False
                entries.append(
                    ResultEntry(
                        path=rel_name,
                        kind="symlink_file",
                        size_bytes=0,
                        sha256=None,
                        target=str(resolved),
                        include_main=False,
                        include_sidecar=False,
                        reason="internal_reference" if internal else "external_symlink_rejected",
                        role="reference",
                        storage_path=str(path),
                        persistence_status="reference_only",
                    )
                )
                if not internal:
                    raise ValueError(f"Result directory contains external symlink: {rel} -> {resolved}")
                continue
            if not path.is_file():
                continue
            size = path.stat().st_size
            digest = file_sha256(path)
            checkpoint_like = _is_checkpoint_like(rel)
            foundation_model = _is_foundation_model_weight(rel)
            too_large = size > max_single_bytes
            # Model/optimizer state is never embedded in the main evidence ZIP, even when
            # it is smaller than the generic single-file threshold.
            include_main = not too_large and not checkpoint_like and not foundation_model
            include_sidecar = rel_name in selected_sidecar_paths
            textual = rel.suffix.lower() in {".log", ".txt", ".json", ".jsonl", ".csv", ".md"}
            if include_sidecar and foundation_model:
                raise ValueError(
                    f"Foundation-model weights may not be copied into a sidecar: {rel_name}"
                )
            if include_sidecar and not (checkpoint_like or too_large):
                raise ValueError(
                    f"--sidecar-file is only valid for checkpoint-like or oversized files: {rel_name}"
                )
            if foundation_model:
                reason = "foundation_model_forbidden_index_only"
            elif include_main:
                reason = "included_main"
            elif too_large and textual and not checkpoint_like:
                reason = "text_tail_in_main"
            elif include_sidecar:
                reason = "explicit_sidecar"
            else:
                reason = "large_or_checkpoint_index_only"
            if foundation_model:
                role = "foundation_model_weight"
            elif checkpoint_like:
                role = "checkpoint_or_model_state"
            elif too_large and textual:
                role = "large_text_or_log"
            elif too_large:
                role = "large_result"
            else:
                role = "result_file"
            if include_main:
                persistence_status = "embedded_main"
                storage_path = None
            elif include_sidecar:
                persistence_status = "sidecar_candidate"
                storage_path = str(path.resolve())
            else:
                persistence_status = large_file_persistence
                storage_path = str(path.resolve())
            entries.append(
                ResultEntry(
                    path=rel_name,
                    kind="file",
                    size_bytes=size,
                    sha256=digest,
                    target=None,
                    include_main=include_main,
                    include_sidecar=include_sidecar,
                    reason=reason,
                    role=role,
                    storage_path=storage_path,
                    persistence_status=persistence_status,
                )
            )
    return entries


def copy_result_entries(
    result_dir: Path,
    destination: Path,
    experiment_id: str,
    entries: list[ResultEntry],
    *,
    sidecar: bool,
) -> None:
    root = result_dir.resolve()
    target_root = destination / experiment_id
    target_root.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        should_copy = entry.include_sidecar if sidecar else entry.include_main
        if entry.kind != "file":
            continue
        source = root / entry.path
        if not sidecar and not should_copy and entry.reason == "text_tail_in_main":
            target = target_root / (entry.path + ".tail.txt")
            target.parent.mkdir(parents=True, exist_ok=True)
            current_size = source.stat().st_size
            current_digest = file_sha256(source)
            if current_size != entry.size_bytes or current_digest != entry.sha256:
                raise ValueError(
                    f"Result file changed while packaging: {entry.path}; "
                    f"scanned size/hash={entry.size_bytes}/{entry.sha256}, "
                    f"current size/hash={current_size}/{current_digest}"
                )
            tail_bytes = 1024 * 1024
            with source.open("rb") as handle:
                if entry.size_bytes > tail_bytes:
                    handle.seek(-tail_bytes, os.SEEK_END)
                payload = handle.read()
            target.write_bytes(
                (
                    f"# Truncated evidence tail for {entry.path}\n"
                    f"# Original bytes: {entry.size_bytes}; SHA256: {entry.sha256}\n"
                ).encode("utf-8")
                + payload
            )
            continue
        if not should_copy:
            continue
        target = target_root / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied_size = target.stat().st_size
        copied_digest = file_sha256(target)
        if copied_size != entry.size_bytes or copied_digest != entry.sha256:
            target.unlink(missing_ok=True)
            raise ValueError(
                f"Result file changed while packaging: {entry.path}; "
                f"scanned size/hash={entry.size_bytes}/{entry.sha256}, "
                f"copied size/hash={copied_size}/{copied_digest}"
            )


def build_summary(
    experiment_id: str,
    package_kind: str,
    base_commit: str,
    paths: list[Path],
    result_dir: Path | None,
    supplied: Path | None,
) -> str:
    if supplied:
        return supplied.read_text()
    changed = "\n".join(f"- `{p.as_posix()}`" for p in paths) or "- None"
    return f"""# Change Summary

- Governance/experiment ID: `{experiment_id}`
- Package kind: `{package_kind}`
- Base commit: `{base_commit}`
- Generated UTC: `{utc_now()}`
- Result directory: `{result_dir if result_dir else 'None'}`

## Modified files

{changed}

## Purpose

Durably package the registered change or experiment evidence. Scientific acceptance is
controlled by `docs/handoff.md`, `experiments/registry.yaml`, and the package kind.

## Remaining items

Review the artifact manifest, execute `TEST_COMMANDS.sh`, and apply `update.patch` only
against the stated base commit.
"""


def executable_test_script(commands: list[str]) -> str:
    if not commands:
        commands = ["python3 -m pytest -q"]
    return "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n"


def write_zip_from_stage(stage: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(stage.rglob("*")):
            if path.is_file() and not path.is_symlink():
                zf.write(path, path.relative_to(stage).as_posix())


def _safe_zip_member(name: str) -> None:
    if "\\" in name:
        raise ValueError(f"ZIP member uses backslashes: {name}")
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"Unsafe ZIP member path: {name}")
    first = pure.parts[0]
    if re.fullmatch(r"[A-Za-z]:", first):
        raise ValueError(f"Unsafe drive-qualified ZIP member: {name}")


def parse_checksums(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        digest, sep, name = raw.partition(" ")
        if not sep or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid checksum row: {raw!r}")
        _safe_zip_member(name)
        if name in rows:
            raise ValueError(f"Duplicate checksum entry: {name}")
        rows[name] = digest
    return rows


def validate_test_commands(text: str) -> None:
    lowered = text.lower()
    forbidden = ["/abs/path", "todo", "placeholder", "replace_me"]
    bad = [token for token in forbidden if token in lowered]
    if bad:
        raise ValueError(f"TEST_COMMANDS.sh contains placeholder tokens: {bad}")
    if "set -euo pipefail" not in text:
        raise ValueError("TEST_COMMANDS.sh must use 'set -euo pipefail'")


def read_zip_json(zf: zipfile.ZipFile, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(zf.read(name))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def verify_result_markers(
    zf: zipfile.ZipFile,
    names: set[str],
    manifest: dict[str, Any],
    base_commit: str,
) -> None:
    kind = str(manifest.get("package_kind"))
    experiment_id = str(manifest.get("experiment_id"))
    prefix = f"results/{experiment_id}/"
    if kind not in RESULT_KINDS:
        return
    run_manifest_name = prefix + "run_manifest.json"
    if run_manifest_name not in names:
        raise ValueError(f"{kind} is missing run_manifest.json")
    if not any(name.startswith(prefix + "logs/") for name in names):
        raise ValueError(f"{kind} is missing a log file under logs/")
    run_manifest = read_zip_json(zf, run_manifest_name)
    require_identity(
        run_manifest,
        label=run_manifest_name,
        experiment_id=experiment_id,
        base_commit=base_commit,
    )
    if kind == "experiment-final":
        complete_name = prefix + "RUN_COMPLETE.json"
        if complete_name not in names:
            raise ValueError("experiment-final is missing RUN_COMPLETE.json")
        audits = [prefix + "TERMINAL_AUDIT.json", prefix + "terminal_audit.json"]
        audit_name = next((name for name in audits if name in names), None)
        if audit_name is None:
            raise ValueError("experiment-final is missing a terminal audit")
        complete = read_zip_json(zf, complete_name)
        audit = read_zip_json(zf, audit_name)
        require_identity(
            complete,
            label=complete_name,
            experiment_id=experiment_id,
            base_commit=base_commit,
        )
        require_identity(
            audit,
            label=audit_name,
            experiment_id=experiment_id,
            base_commit=base_commit,
        )
    elif kind == "experiment-failed":
        marker = prefix + "RUN_FAILED.json"
        if marker not in names:
            raise ValueError("experiment-failed is missing RUN_FAILED.json")
        failed = read_zip_json(zf, marker)
        require_identity(
            failed,
            label=marker,
            experiment_id=experiment_id,
            base_commit=base_commit,
        )
    elif kind == "experiment-raw-complete":
        marker = prefix + "RUN_RAW_COMPLETE.json"
        if marker not in names:
            raise ValueError("experiment-raw-complete is missing RUN_RAW_COMPLETE.json")
        raw_complete = read_zip_json(zf, marker)
        require_identity(
            raw_complete,
            label=marker,
            experiment_id=experiment_id,
            base_commit=base_commit,
        )


def validate_modified_files_manifest(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("Manifest modified_files must be a JSON list")
    rows: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError("Manifest modified_files entries must be strings")
        rel = safe_repo_rel(item).as_posix()
        if rel != item:
            raise ValueError(f"Manifest modified_files path is not canonical: {item!r}")
        if rel in seen:
            raise ValueError(f"Manifest modified_files contains a duplicate: {rel}")
        seen.add(rel)
        rows.append(rel)
    return rows


def run_git_apply_check(repo: Path, patch: bytes, expected_sha: str) -> None:
    actual = git_head(repo)
    if actual != expected_sha:
        raise ValueError(f"Repository HEAD {actual} does not match BASE_COMMIT {expected_sha}")
    if not patch.strip():
        return
    # Verify against an isolated index loaded from the immutable base commit. The
    # caller's real index may contain staged edits that are themselves part of the
    # package, so using it directly would create a false apply conflict.
    with tempfile.TemporaryDirectory(prefix="drpo_apply_index_") as tmp:
        index_path = Path(tmp) / "index"
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(index_path)
        read_tree = subprocess.run(
            ["git", "read-tree", expected_sha],
            cwd=repo,
            text=True,
            capture_output=True,
            env=env,
        )
        if read_tree.returncode != 0:
            raise ValueError(f"Could not initialize isolated Git index:\n{read_tree.stderr}")
        patch_path = Path(tmp) / "update.patch"
        patch_path.write_bytes(patch)
        result = subprocess.run(
            ["git", "apply", "--check", "--cached", str(patch_path)],
            cwd=repo,
            text=True,
            capture_output=True,
            env=env,
        )
        if result.returncode != 0:
            raise ValueError(f"git apply --check failed:\n{result.stderr}")


def verify_package(
    package: Path,
    *,
    repo_root: Path | None,
    skip_head_match: bool,
    hard_limit_mib: float,
) -> dict[str, Any]:
    package = package.resolve()
    if not package.is_file() or not zipfile.is_zipfile(package):
        raise ValueError(f"Not a valid ZIP: {package}")
    size_mib = package.stat().st_size / (1024 * 1024)
    if size_mib > hard_limit_mib:
        raise ValueError(
            f"Main package is {size_mib:.3f} MiB, exceeding hard limit {hard_limit_mib:.3f} MiB"
        )
    with zipfile.ZipFile(package) as zf:
        raw_names = zf.namelist()
        for name in raw_names:
            _safe_zip_member(name)
        if len(raw_names) != len(set(raw_names)):
            raise ValueError("ZIP contains duplicate member names")
        names = {name for name in raw_names if not name.endswith("/")}
        missing = REQUIRED_TOP_LEVEL - names
        if missing:
            raise ValueError(f"Missing required top-level files: {sorted(missing)}")
        try:
            base_text = zf.read("BASE_COMMIT.txt").decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"BASE_COMMIT.txt is not UTF-8: {exc}") from exc
        if not re.fullmatch(r"[0-9a-f]{40}\n", base_text):
            raise ValueError("BASE_COMMIT.txt must contain exactly one lowercase full SHA")
        base_sha = base_text.strip()
        manifest = read_zip_json(zf, "ARTIFACT_MANIFEST.json")
        if manifest.get("base_commit") != base_sha:
            raise ValueError("Manifest base_commit does not match BASE_COMMIT.txt")
        kind = manifest.get("package_kind")
        if not isinstance(kind, str) or kind not in ALL_KINDS:
            raise ValueError(f"Unknown or invalid package_kind: {kind!r}")
        experiment_id_raw = manifest.get("experiment_id")
        if not isinstance(experiment_id_raw, str):
            raise ValueError("Manifest experiment_id must be a string")
        experiment_id = validate_experiment_id(experiment_id_raw)
        modified_files = validate_modified_files_manifest(manifest.get("modified_files"))
        declared_modified = {f"modified_files/{name}" for name in modified_files}
        actual_modified = {
            name for name in names if name.startswith("modified_files/")
        }
        if actual_modified != declared_modified:
            raise ValueError(
                "modified_files inventory mismatch; "
                f"missing={sorted(declared_modified - actual_modified)}, "
                f"extra={sorted(actual_modified - declared_modified)}"
            )
        patch = zf.read("update.patch")
        if kind in FINAL_KINDS and not patch.strip():
            raise ValueError(f"{kind} requires a non-empty update.patch")
        if kind in FINAL_KINDS and not modified_files:
            raise ValueError("Final packages require complete files under modified_files/")
        checksums = parse_checksums(zf.read("SHA256SUMS.txt").decode("utf-8"))
        expected_names = names - {"SHA256SUMS.txt"}
        if set(checksums) != expected_names:
            raise ValueError(
                "Checksum inventory mismatch; "
                f"missing={sorted(expected_names - set(checksums))}, "
                f"extra={sorted(set(checksums) - expected_names)}"
            )
        for name, expected in checksums.items():
            if sha256_bytes(zf.read(name)) != expected:
                raise ValueError(f"Checksum mismatch for {name}")
        validate_test_commands(zf.read("TEST_COMMANDS.sh").decode("utf-8"))
        verify_result_markers(zf, names, manifest, base_sha)
        index_name = "LARGE_FILE_INDEX.json"
        if index_name in names:
            index = read_zip_json(zf, index_name)
            entries = index.get("entries")
            if not isinstance(entries, list):
                raise ValueError("LARGE_FILE_INDEX.json entries must be a list")
            for row in entries:
                if not isinstance(row, dict):
                    raise ValueError("LARGE_FILE_INDEX.json entries must be JSON objects")
                path = row.get("path")
                if not isinstance(path, str):
                    raise ValueError("LARGE_FILE_INDEX.json entry path must be a string")
                safe_repo_rel(path)
            if index.get("sidecar_required") and not index.get("sidecar"):
                raise ValueError("Large-file index requires a sidecar but none is declared")
    git_apply_checked = False
    evidence_commit_present = False
    if repo_root is not None and not skip_head_match:
        repo = repo_root.resolve()
        if kind in FINAL_KINDS or patch.strip():
            run_git_apply_check(repo, patch, base_sha)
            git_apply_checked = True
        else:
            ensure_git_repo(repo)
            ensure_commit_exists(repo, base_sha)
            evidence_commit_present = True
    return {
        "package": str(package),
        "package_kind": kind,
        "experiment_id": experiment_id,
        "base_commit": base_sha,
        "size_mib": round(size_mib, 3),
        "hard_limit_mib": hard_limit_mib,
        "checksum_files": len(checksums),
        "git_apply_check": git_apply_checked,
        "evidence_commit_present": evidence_commit_present,
        "verified": True,
    }


def package_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and atomically verify a DRPO artifact")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--package-kind", choices=sorted(ALL_KINDS), required=True)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-commit")
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--test-command", action="append", default=[])
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--no-repository-changes", action="store_true")
    parser.add_argument("--source-file", action="append", default=[])
    parser.add_argument("--warning-mib", type=float, default=25.0)
    parser.add_argument("--max-package-mib", type=float, default=25.0)
    parser.add_argument("--max-single-file-mib", type=float, default=10.0)
    parser.add_argument("--sidecar-output", type=Path)
    parser.add_argument("--sidecar-file", action="append", default=[])
    parser.add_argument("--sidecar-purpose", choices=sorted(SIDECAR_PURPOSES))
    parser.add_argument("--max-sidecar-mib", type=float, default=1024.0)
    parser.add_argument("--max-sidecar-files", type=int, default=2)
    parser.add_argument(
        "--large-file-persistence",
        choices=["persistent_local", "external_durable", "ephemeral", "unknown"],
        default="persistent_local",
    )
    parser.add_argument("--allow-recovery-base-mismatch", action="store_true")
    parser.add_argument("--require-origin-main-match", action="store_true")
    return parser.parse_args(argv)


def verify_sidecar_candidate(package: Path, *, hard_limit_mib: float) -> dict[str, Any]:
    size_mib = package.stat().st_size / (1024 * 1024)
    if size_mib > hard_limit_mib:
        raise ValueError(
            f"Sidecar is {size_mib:.3f} MiB, exceeding hard limit {hard_limit_mib:.3f} MiB"
        )
    with zipfile.ZipFile(package) as zf:
        raw_names = zf.namelist()
        for name in raw_names:
            _safe_zip_member(name)
        if len(raw_names) != len(set(raw_names)):
            raise ValueError("Sidecar ZIP contains duplicate member names")
        names = {name for name in raw_names if not name.endswith("/")}
        required = {"SIDECAR_MANIFEST.json", "SHA256SUMS.txt"}
        missing = required - names
        if missing:
            raise ValueError(f"Sidecar is missing required files: {sorted(missing)}")
        manifest = read_zip_json(zf, "SIDECAR_MANIFEST.json")
        experiment_id_raw = manifest.get("experiment_id")
        if not isinstance(experiment_id_raw, str):
            raise ValueError("Sidecar manifest experiment_id must be a string")
        experiment_id = validate_experiment_id(experiment_id_raw)
        base_commit_raw = manifest.get("base_commit")
        if not isinstance(base_commit_raw, str):
            raise ValueError("Sidecar manifest base_commit must be a string")
        validate_sha(base_commit_raw)
        if manifest.get("selection_mode") != "explicit_only":
            raise ValueError("Sidecar selection_mode must be explicit_only")
        if manifest.get("purpose") not in SIDECAR_PURPOSES:
            raise ValueError("Sidecar manifest has an invalid purpose")
        if not isinstance(manifest.get("files"), list) or not manifest["files"]:
            raise ValueError("Sidecar manifest must list at least one selected file")
        expected_payload: set[str] = set()
        seen_paths: set[str] = set()
        for row in manifest["files"]:
            if not isinstance(row, dict):
                raise ValueError("Sidecar manifest file entries must be JSON objects")
            rel_raw = row.get("path")
            if not isinstance(rel_raw, str):
                raise ValueError("Sidecar manifest file path must be a string")
            rel = safe_repo_rel(rel_raw).as_posix()
            if rel != rel_raw:
                raise ValueError(f"Sidecar manifest path is not canonical: {rel_raw!r}")
            if rel in seen_paths:
                raise ValueError(f"Sidecar manifest contains a duplicate file: {rel}")
            seen_paths.add(rel)
            if row.get("kind") != "file" or row.get("include_sidecar") is not True:
                raise ValueError(f"Sidecar manifest entry is not explicitly selected: {rel}")
            size_bytes = row.get("size_bytes")
            digest = row.get("sha256")
            if not isinstance(size_bytes, int) or size_bytes < 0:
                raise ValueError(f"Sidecar manifest has invalid size for {rel}")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Sidecar manifest has invalid SHA-256 for {rel}")
            member = f"results/{experiment_id}/{rel}"
            expected_payload.add(member)
            try:
                info = zf.getinfo(member)
            except KeyError as exc:
                raise ValueError(f"Sidecar is missing selected file: {member}") from exc
            if info.file_size != size_bytes:
                raise ValueError(f"Sidecar size does not match manifest for {member}")
            if sha256_bytes(zf.read(member)) != digest:
                raise ValueError(f"Sidecar SHA-256 does not match manifest for {member}")
        actual_payload = names - required
        if actual_payload != expected_payload:
            raise ValueError(
                "Sidecar payload inventory mismatch; "
                f"missing={sorted(expected_payload - actual_payload)}, "
                f"extra={sorted(actual_payload - expected_payload)}"
            )
        checksums = parse_checksums(zf.read("SHA256SUMS.txt").decode("utf-8"))
        expected = names - {"SHA256SUMS.txt"}
        if set(checksums) != expected:
            raise ValueError("Sidecar checksum inventory mismatch")
        for name, digest in checksums.items():
            if sha256_bytes(zf.read(name)) != digest:
                raise ValueError(f"Sidecar checksum mismatch for {name}")
    return {
        "size_mib": round(size_mib, 3),
        "files": len(manifest["files"]),
        "experiment_id": experiment_id,
        "base_commit": base_commit_raw,
        "purpose": manifest["purpose"],
    }


def package_main(argv: list[str] | None = None) -> int:
    args = package_parse_args(argv)
    try:
        validate_experiment_id(args.experiment_id)
        reject_symlink_path(args.output, "--output")
        if args.sidecar_output is not None:
            reject_symlink_path(args.sidecar_output, "--sidecar-output")
        if args.result_dir is not None:
            reject_symlink_path(args.result_dir, "--result-dir")
        if args.max_package_mib <= 0 or args.max_single_file_mib <= 0:
            raise ValueError("Main-package size limits must be positive")
        if args.max_sidecar_mib <= 0 or args.max_sidecar_files <= 0:
            raise ValueError("Sidecar size and file-count limits must be positive")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    repo = args.repo_root.resolve()
    candidate: Path | None = None
    candidate_sidecar: Path | None = None
    try:
        ensure_git_repo(repo)
        output = args.output.resolve()
        sidecar_output = args.sidecar_output.resolve() if args.sidecar_output else None
        reject_tracked_runtime_path(repo, output, "--output", subtree=False)
        if sidecar_output is not None:
            reject_tracked_runtime_path(repo, sidecar_output, "--sidecar-output", subtree=False)
        if args.result_dir is not None:
            reject_tracked_runtime_path(repo, args.result_dir.resolve(), "--result-dir", subtree=True)
        packaging_head = git_head(repo)
        if args.allow_recovery_base_mismatch:
            if args.package_kind not in RECOVERY_KINDS:
                raise ValueError("--allow-recovery-base-mismatch is only valid for recovery packages")
            if not args.base_commit:
                raise ValueError("Recovery base mismatch mode requires --base-commit")
            base_commit = validate_sha(args.base_commit)
            ensure_commit_exists(repo, base_commit)
            origin = resolve_origin_main(repo)
            resolution = {
                "local_head": packaging_head,
                "expected_sha": base_commit,
                "origin_main": origin,
                "origin_matches_local": bool(origin.get("authoritative") and origin.get("sha") == packaging_head),
                "verified": bool(origin.get("authoritative") and origin.get("sha") == packaging_head),
                "head_matches_evidence_base": packaging_head == base_commit,
                "recovery_evidence_mode": True,
            }
        else:
            resolution = resolve_commit(
                repo,
                args.base_commit,
                require_origin_match=args.require_origin_main_match,
            )
            base_commit = resolution["local_head"]

        result_dir = args.result_dir.resolve() if args.result_dir else None
        validate_result_markers(
            args.package_kind,
            result_dir,
            experiment_id=args.experiment_id,
            base_commit=base_commit,
        )
        if args.no_repository_changes:
            if args.package_kind not in RECOVERY_KINDS:
                raise ValueError("--no-repository-changes is only valid for recovery packages")
            paths: list[Path] = []
        elif args.changed_file:
            paths = [safe_repo_rel(x) for x in args.changed_file]
            missing = [p for p in paths if not (repo / p).is_file()]
            if missing:
                raise ValueError(f"Missing changed files: {missing}")
        else:
            paths = changed_paths(repo)
        if result_dir is not None:
            try:
                result_relative = result_dir.relative_to(repo)
            except ValueError:
                result_relative = None
            if result_relative is not None:
                paths = [
                    p for p in paths if p != result_relative and result_relative not in p.parents
                ]
        patch = patch_for_repo(repo, paths)
        if args.package_kind in FINAL_KINDS and not patch.strip():
            raise ValueError(f"{args.package_kind} requires a non-empty update.patch")

        source_files = [safe_repo_rel(x) for x in args.source_file]
        selected_sidecar_paths = {safe_repo_rel(x).as_posix() for x in args.sidecar_file}
        if sidecar_output is not None and sidecar_output == output:
            raise ValueError("--sidecar-output must differ from --output")
        if sidecar_output is not None and not selected_sidecar_paths:
            raise ValueError("--sidecar-output requires at least one explicit --sidecar-file")
        if sidecar_output is not None and not args.sidecar_purpose:
            raise ValueError("--sidecar-output requires an explicit --sidecar-purpose")
        if args.sidecar_purpose and sidecar_output is None:
            raise ValueError("--sidecar-purpose requires --sidecar-output")
        if selected_sidecar_paths and sidecar_output is None:
            raise ValueError("--sidecar-file requires --sidecar-output")
        if sidecar_output is not None and sidecar_output.exists():
            raise ValueError(
                "Refusing to overwrite an existing sidecar; use a new versioned --sidecar-output path"
            )
        if len(selected_sidecar_paths) > args.max_sidecar_files:
            raise ValueError(
                f"Selected {len(selected_sidecar_paths)} sidecar files, exceeding limit {args.max_sidecar_files}"
            )

        max_single_bytes = max(1, int(args.max_single_file_mib * 1024 * 1024))
        entries: list[ResultEntry] = []
        if result_dir is not None:
            entries = scan_result_tree(
                result_dir,
                package_kind=args.package_kind,
                max_single_bytes=max_single_bytes,
                selected_sidecar_paths=selected_sidecar_paths,
                large_file_persistence=args.large_file_persistence,
            )
        discovered = {entry.path for entry in entries if entry.kind == "file"}
        missing_selected = selected_sidecar_paths - discovered
        if missing_selected:
            raise ValueError(f"Selected sidecar files do not exist: {sorted(missing_selected)}")

        with tempfile.TemporaryDirectory(prefix="drpo_artifact_") as tmp:
            tmp_root = Path(tmp)
            stage = tmp_root / "main"
            stage.mkdir()
            (stage / "modified_files").mkdir()
            (stage / "update.patch").write_text(patch)
            (stage / "BASE_COMMIT.txt").write_text(base_commit + "\n")
            (stage / "CHANGE_SUMMARY.md").write_text(
                build_summary(
                    args.experiment_id,
                    args.package_kind,
                    base_commit,
                    paths,
                    result_dir,
                    args.summary_file,
                )
            )
            test_script = stage / "TEST_COMMANDS.sh"
            test_script.write_text(executable_test_script(args.test_command))
            test_script.chmod(test_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            copy_changed_files(repo, paths, stage / "modified_files")
            if result_dir is not None:
                copy_result_entries(
                    result_dir,
                    stage / "results",
                    args.experiment_id,
                    entries,
                    sidecar=False,
                )
            copy_source_snapshot(
                repo,
                source_files,
                stage / "source_snapshot",
                base_commit=base_commit,
                from_commit=args.package_kind in RECOVERY_KINDS or packaging_head != base_commit,
            )

            sidecar_meta: dict[str, Any] | None = None
            if sidecar_output is not None:
                selected_entries = [entry for entry in entries if entry.include_sidecar]
                if not selected_entries:
                    raise ValueError("No result files matched the explicit --sidecar-file selection")
                side_stage = tmp_root / "sidecar"
                side_stage.mkdir()
                copy_result_entries(
                    result_dir,  # type: ignore[arg-type]
                    side_stage / "results",
                    args.experiment_id,
                    entries,
                    sidecar=True,
                )
                side_manifest = {
                    "schema_version": 2,
                    "experiment_id": args.experiment_id,
                    "base_commit": base_commit,
                    "generated_utc": utc_now(),
                    "selection_mode": "explicit_only",
                    "purpose": args.sidecar_purpose,
                    "files": [asdict(entry) for entry in selected_entries],
                }
                (side_stage / "SIDECAR_MANIFEST.json").write_text(
                    json.dumps(side_manifest, indent=2, ensure_ascii=False) + "\n"
                )
                write_checksums(side_stage)
                sidecar_output.parent.mkdir(parents=True, exist_ok=True)
                candidate_sidecar = sidecar_output.with_name(
                    f".{sidecar_output.name}.candidate-{os.getpid()}-{time.time_ns()}"
                )
                write_zip_from_stage(side_stage, candidate_sidecar)
                side_report = verify_sidecar_candidate(
                    candidate_sidecar,
                    hard_limit_mib=args.max_sidecar_mib,
                )
                sidecar_meta = {
                    "path": sidecar_output.name,
                    "sha256": file_sha256(candidate_sidecar),
                    "size_bytes": candidate_sidecar.stat().st_size,
                    "status": "verified_candidate",
                    "selection_mode": "explicit_only",
                    "purpose": args.sidecar_purpose,
                    "selected_files": sorted(selected_sidecar_paths),
                    "verification": side_report,
                }

            excluded_entries = [
                entry for entry in entries if entry.kind != "file" or not entry.include_main
            ]
            large_index = {
                "schema_version": 2,
                "generated_utc": utc_now(),
                "entries": [asdict(entry) for entry in excluded_entries],
                "default_delivery": "persistent_local_index",
                "sidecar_default": False,
                "sidecar_required": bool(sidecar_meta),
                "sidecar": sidecar_meta,
            }
            (stage / "LARGE_FILE_INDEX.json").write_text(
                json.dumps(large_index, indent=2, ensure_ascii=False) + "\n"
            )
            manifest = {
                "schema_version": 3,
                "generated_utc": utc_now(),
                "experiment_id": args.experiment_id,
                "package_kind": args.package_kind,
                "base_commit": base_commit,
                "packaging_head": packaging_head,
                "repository": "easonhuo/drpo",
                "branch": git_branch(repo),
                "commit_resolution": resolution,
                "modified_files": [p.as_posix() for p in paths],
                "result_dir_name": args.experiment_id if result_dir else None,
                "result_storage_root": str(result_dir) if result_dir else None,
                "source_files": [p.as_posix() for p in source_files],
                "source_snapshot_commit": base_commit if source_files else None,
                "scientific_completion_claim": args.package_kind == "experiment-final",
                "durable_delivery_pending": True,
                "lightweight_recovery": args.package_kind in RECOVERY_KINDS,
                "main_package_hard_limit_mib": args.max_package_mib,
                "max_single_file_mib": args.max_single_file_mib,
                "large_file_persistence": args.large_file_persistence,
                "sidecar_default": False,
                "sidecar": sidecar_meta,
                "symlink_policy": "never_follow",
            }
            (stage / "ARTIFACT_MANIFEST.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
            )
            write_checksums(stage)
            output.parent.mkdir(parents=True, exist_ok=True)
            candidate = output.with_name(
                f".{output.name}.candidate-{os.getpid()}-{time.time_ns()}"
            )
            write_zip_from_stage(stage, candidate)
            report = verify_package(
                candidate,
                repo_root=repo,
                skip_head_match=False,
                hard_limit_mib=args.max_package_mib,
            )
            published_sidecar: Path | None = None
            try:
                if candidate_sidecar is not None and sidecar_output is not None:
                    # Sidecars use a new versioned filename and are never overwritten.
                    # Publish it first; if the main atomic replace fails, remove the newly
                    # published orphan and leave the previous main package untouched.
                    os.replace(candidate_sidecar, sidecar_output)
                    candidate_sidecar = None
                    published_sidecar = sidecar_output
                os.replace(candidate, output)
                candidate = None
            except Exception:
                if published_sidecar is not None:
                    published_sidecar.unlink(missing_ok=True)
                raise
            report.update(
                {
                    "output": str(output),
                    "atomic_publish": True,
                    "previous_output_preserved_until_publish": True,
                    "sidecar_output": str(sidecar_output) if sidecar_output and sidecar_output.exists() else None,
                }
            )
            print(json.dumps(report, indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if candidate is not None:
            candidate.unlink(missing_ok=True)
        if candidate_sidecar is not None:
            candidate_sidecar.unlink(missing_ok=True)


def verify_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a DRPO durable artifact")
    parser.add_argument("package", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--warning-mib", type=float, default=25.0)
    parser.add_argument("--max-package-mib", type=float, default=25.0)
    parser.add_argument("--skip-head-match", action="store_true")
    return parser.parse_args(argv)


def verify_main(argv: list[str] | None = None) -> int:
    args = verify_parse_args(argv)
    try:
        report = verify_package(
            args.package,
            repo_root=args.repo_root,
            skip_head_match=args.skip_head_match,
            hard_limit_mib=args.max_package_mib,
        )
        report["warning_threshold_mib"] = args.warning_mib
        report["over_warning_threshold"] = report["size_mib"] > args.warning_mib
        print(json.dumps(report, indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def latest_mtime(root: Path) -> float | None:
    latest: float | None = None
    for path in root.rglob("*"):
        try:
            if path.name == "heartbeat.json":
                continue
            if path.is_file() and not path.is_symlink():
                value = path.stat().st_mtime
                latest = value if latest is None else max(latest, value)
        except FileNotFoundError:
            continue
    return latest


def progress_counts(root: Path, patterns: list[str]) -> dict[str, int]:
    return {
        pattern: sum(1 for p in root.glob(pattern) if p.is_file() and not p.is_symlink())
        for pattern in patterns
    }


def stream_reader(pipe: Any, events: queue.Queue[tuple[float, str]]) -> None:
    try:
        for line in iter(pipe.readline, ""):
            events.put((time.time(), line))
    finally:
        pipe.close()


def capture_dirty_snapshot(repo: Path, destination: Path, max_file_mib: float = 10.0) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    tracked = run(["git", "diff", "--binary", "--full-index", "HEAD"], repo).stdout
    staged = run(["git", "diff", "--cached", "--binary", "--full-index", "HEAD"], repo).stdout
    (destination / "working_tree.patch").write_text(tracked)
    (destination / "staged.patch").write_text(staged)
    untracked = git_output(repo, "ls-files", "--others", "--exclude-standard").splitlines()
    max_bytes = int(max_file_mib * 1024 * 1024)
    indexed: list[dict[str, Any]] = []
    for raw in untracked:
        rel = safe_repo_rel(raw)
        source = repo / rel
        if is_generated_artifact(rel) or not source.is_file() or source.is_symlink():
            continue
        size = source.stat().st_size
        row = {"path": rel.as_posix(), "size_bytes": size, "sha256": file_sha256(source)}
        if size <= max_bytes:
            target = destination / "untracked_files" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            row["captured"] = True
        else:
            row["captured"] = False
            row["reason"] = "exceeds_launch_snapshot_limit"
        indexed.append(row)
    snapshot = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "base_commit": git_head(repo),
        "status": git_status(repo),
        "untracked": indexed,
    }
    atomic_json(destination / "LAUNCH_SNAPSHOT_MANIFEST.json", snapshot)
    return snapshot


def guard_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DRPO experiment under provenance guard")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    parser.add_argument("--sidecar-output", type=Path)
    parser.add_argument("--sidecar-file", action="append", default=[])
    parser.add_argument("--sidecar-purpose", choices=sorted(SIDECAR_PURPOSES))
    parser.add_argument("--max-sidecar-mib", type=float, default=1024.0)
    parser.add_argument("--max-sidecar-files", type=int, default=2)
    parser.add_argument(
        "--large-file-persistence",
        choices=["persistent_local", "external_durable", "ephemeral", "unknown"],
        default="persistent_local",
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--stale-seconds", type=float, default=600.0)
    parser.add_argument("--fail-on-stale", action="store_true")
    parser.add_argument("--termination-grace-seconds", type=float, default=30.0)
    parser.add_argument("--progress-glob", action="append", default=[])
    parser.add_argument("--required-output", action="append", default=[])
    parser.add_argument("--source-file", action="append", default=[])
    parser.add_argument("--run-class", choices=["formal", "pilot"], default="formal")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--expected-commit")
    parser.add_argument("--require-origin-main-match", action="store_true")
    parser.add_argument("--max-package-mib", type=float, default=25.0)
    parser.add_argument("--max-single-file-mib", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("Supply the experiment command after --")
    if (
        args.heartbeat_seconds <= 0
        or args.stale_seconds <= 0
        or args.termination_grace_seconds <= 0
    ):
        parser.error("Heartbeat, stale, and termination-grace intervals must be positive")
    if args.max_package_mib <= 0 or args.max_single_file_mib <= 0:
        parser.error("Main-package size limits must be positive")
    if args.max_sidecar_mib <= 0 or args.max_sidecar_files <= 0:
        parser.error("Sidecar size and file-count limits must be positive")
    if args.run_class == "formal" and args.allow_dirty:
        parser.error("--allow-dirty is only valid with --run-class pilot")
    if args.sidecar_output is not None and not args.sidecar_file:
        parser.error("--sidecar-output requires at least one --sidecar-file")
    if args.sidecar_output is not None and not args.sidecar_purpose:
        parser.error("--sidecar-output requires --sidecar-purpose")
    if args.sidecar_purpose and args.sidecar_output is None:
        parser.error("--sidecar-purpose requires --sidecar-output")
    return args


def package_recovery(
    repo: Path,
    experiment_id: str,
    output_root: Path,
    artifact_output: Path,
    package_kind: str,
    source_files: list[str],
    *,
    launch_commit: str,
    sidecar_output: Path | None,
    sidecar_files: list[str],
    sidecar_purpose: str | None,
    max_package_mib: float,
    max_single_file_mib: float,
    max_sidecar_mib: float,
    max_sidecar_files: int,
    large_file_persistence: str,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "--repo-root",
        str(repo),
        "--experiment-id",
        experiment_id,
        "--package-kind",
        package_kind,
        "--result-dir",
        str(output_root),
        "--output",
        str(artifact_output),
        "--base-commit",
        launch_commit,
        "--allow-recovery-base-mismatch",
        "--no-repository-changes",
        "--large-file-persistence",
        large_file_persistence,
        "--max-package-mib",
        str(max_package_mib),
        "--max-single-file-mib",
        str(max_single_file_mib),
        "--max-sidecar-mib",
        str(max_sidecar_mib),
        "--max-sidecar-files",
        str(max_sidecar_files),
        "--test-command",
        "python3 -m pytest -q tests/test_experiment_artifact_protocol.py tests/test_experiment_artifact_hardening.py",
    ]
    if sidecar_output is not None:
        cmd.extend(["--sidecar-output", str(sidecar_output)])
        if sidecar_purpose is not None:
            cmd.extend(["--sidecar-purpose", sidecar_purpose])
    for sidecar_file in sidecar_files:
        cmd.extend(["--sidecar-file", sidecar_file])
    for source in source_files:
        cmd.extend(["--source-file", source])
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        returncode = package_main(cmd)
    return subprocess.CompletedProcess(
        args=["package_experiment_hardened.py", *cmd],
        returncode=returncode,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
    )


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def guard_main(argv: list[str] | None = None) -> int:
    args = guard_parse_args(argv)
    try:
        validate_experiment_id(args.experiment_id)
        reject_symlink_path(args.output_root, "--output-root")
        reject_symlink_path(args.artifact_output, "--artifact-output")
        if args.sidecar_output is not None:
            reject_symlink_path(args.sidecar_output, "--sidecar-output")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    repo = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    artifact_output = args.artifact_output.resolve()
    sidecar_output = args.sidecar_output.resolve() if args.sidecar_output else None
    if sidecar_output is not None and sidecar_output == artifact_output:
        print("ERROR: --sidecar-output must differ from --artifact-output", file=sys.stderr)
        return 2
    if _path_is_within(artifact_output, output_root):
        print("ERROR: --artifact-output must be outside --output-root", file=sys.stderr)
        return 2
    if sidecar_output is not None and _path_is_within(sidecar_output, output_root):
        print("ERROR: --sidecar-output must be outside --output-root", file=sys.stderr)
        return 2
    if sidecar_output is not None and sidecar_output.exists():
        print(
            "ERROR: --sidecar-output already exists; use a new versioned path",
            file=sys.stderr,
        )
        return 2
    try:
        ensure_git_repo(repo)
        reject_tracked_runtime_path(repo, output_root, "--output-root", subtree=True)
        reject_tracked_runtime_path(repo, artifact_output, "--artifact-output", subtree=False)
        if sidecar_output is not None:
            reject_tracked_runtime_path(repo, sidecar_output, "--sidecar-output", subtree=False)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if output_root.exists():
        try:
            has_existing_entries = any(output_root.iterdir())
        except OSError as exc:
            print(f"ERROR: cannot inspect --output-root: {exc}", file=sys.stderr)
            return 2
        if has_existing_entries:
            print(
                "ERROR: --output-root must be new or empty so stale files cannot satisfy "
                "required-output or contaminate the artifact",
                file=sys.stderr,
            )
            return 2
    try:
        # A formal run must be anchored either by an explicit full expected SHA
        # (suitable for an offline clone or Git bundle) or by a live authoritative
        # origin/main check.  Silently trusting the current local HEAD is not enough.
        require_origin_match = args.require_origin_main_match or (
            args.run_class == "formal" and args.expected_commit is None
        )
        resolution = resolve_commit(
            repo,
            args.expected_commit,
            require_origin_match=require_origin_match,
        )
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    launch_head = resolution["local_head"]
    try:
        source_files = [safe_repo_rel(value) for value in args.source_file]
        validate_source_snapshot_inputs(repo, source_files, base_commit=launch_head)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    ignored_runtime_paths = [output_root, artifact_output]
    if sidecar_output:
        ignored_runtime_paths.append(sidecar_output)
    launch_status = git_status_excluding(repo, ignored_runtime_paths)
    dirty = bool(launch_status)
    if args.run_class == "formal" and dirty:
        print(
            "ERROR: formal experiments require a clean worktree. Dirty entries:\n"
            + "\n".join(launch_status),
            file=sys.stderr,
        )
        return 2
    if args.run_class == "pilot" and dirty and not args.allow_dirty:
        print("ERROR: dirty pilot requires explicit --allow-dirty", file=sys.stderr)
        return 2
    output_root.mkdir(parents=True, exist_ok=True)
    launch_snapshot = None
    if dirty:
        launch_snapshot = capture_dirty_snapshot(repo, output_root / "provenance_launch")
    logs = output_root / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / "supervised_run.log"
    heartbeat_path = output_root / "heartbeat.json"
    start_wall = time.time()
    start_utc = utc_now()
    run_manifest = {
        "schema_version": 3,
        "experiment_id": args.experiment_id,
        "execution_state": "running",
        "run_class": args.run_class,
        "start_utc": start_utc,
        "repo_root": str(repo),
        "branch": git_branch(repo),
        "base_commit": launch_head,
        "commit_resolution": resolution,
        "git_dirty_at_launch": dirty,
        "git_status_at_launch": launch_status,
        "launch_snapshot": bool(launch_snapshot),
        "command": args.command,
        "cwd": str(repo),
        "python": sys.version,
        "platform": platform.platform(),
        "pid": None,
        "heartbeat_seconds": args.heartbeat_seconds,
        "stale_seconds": args.stale_seconds,
        "termination_grace_seconds": args.termination_grace_seconds,
        "progress_globs": args.progress_glob,
        "required_outputs": args.required_output,
        "artifact_output": str(artifact_output),
        "large_file_persistence": args.large_file_persistence,
        "sidecar_default": False,
        "sidecar_files": args.sidecar_file,
        "sidecar_purpose": args.sidecar_purpose,
    }
    atomic_json(output_root / "run_manifest.json", run_manifest)

    try:
        process = subprocess.Popen(
            args.command,
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        trace = traceback.format_exc()
        log_path.write_text(
            f"[{start_utc}] START_FAILED command={args.command!r}\n{trace}"
        )
        try:
            end_head = git_head(repo)
            end_status = git_status_excluding(repo, ignored_runtime_paths)
        except Exception:
            end_head = launch_head
            end_status = ["unable_to_resolve_end_git_state"]
        common = {
            "schema_version": 3,
            "experiment_id": args.experiment_id,
            "run_class": args.run_class,
            "start_utc": start_utc,
            "end_utc": utc_now(),
            "elapsed_seconds": round(time.time() - start_wall, 3),
            "pid": None,
            "returncode": None,
            "supervisor_error_type": type(exc).__name__,
            "supervisor_error": str(exc),
            "traceback_file": "logs/supervised_run.log",
            "missing_required_outputs": args.required_output,
            "base_commit": launch_head,
            "end_commit": end_head,
            "git_status_at_end": end_status,
            "provenance_compromised": end_head != launch_head or bool(end_status),
            "branch": git_branch(repo),
            "command": args.command,
            "scientific_acceptance_pending": True,
            "execution_state": "failed",
        }
        atomic_json(output_root / "RUN_FAILED.json", common)
        run_manifest.update(common)
        atomic_json(output_root / "run_manifest.json", run_manifest)
        packaged = package_recovery(
            repo,
            args.experiment_id,
            output_root,
            artifact_output,
            "experiment-failed",
            args.source_file,
            launch_commit=launch_head,
            sidecar_output=sidecar_output,
            sidecar_files=args.sidecar_file,
            sidecar_purpose=args.sidecar_purpose,
            max_package_mib=args.max_package_mib,
            max_single_file_mib=args.max_single_file_mib,
            max_sidecar_mib=args.max_sidecar_mib,
            max_sidecar_files=args.max_sidecar_files,
            large_file_persistence=args.large_file_persistence,
        )
        atomic_json(
            output_root / "recovery_package_status.json",
            {
                "command": packaged.args,
                "returncode": packaged.returncode,
                "stdout": packaged.stdout,
                "stderr": packaged.stderr,
                "artifact_output": str(artifact_output),
                "artifact_exists": artifact_output.is_file(),
            },
        )
        if packaged.returncode != 0:
            print(packaged.stdout, end="")
            print(packaged.stderr, file=sys.stderr, end="")
            return 3
        print(packaged.stdout, end="")
        print("The experiment command did not start; failed-run evidence was preserved.")
        return 2

    run_manifest["pid"] = process.pid
    atomic_json(output_root / "run_manifest.json", run_manifest)
    forwarded_signal: int | None = None
    previous_handlers: dict[int, Any] = {}
    events: queue.Queue[tuple[float, str]] = queue.Queue()
    reader: threading.Thread | None = None
    last_console_activity = time.time()
    last_fs_activity = latest_mtime(output_root) or start_wall
    last_heartbeat = 0.0
    stale_detected = False
    stale_term_sent_at: float | None = None
    stale_kill_sent = False
    supervisor_error: dict[str, str] | None = None
    returncode = -1

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal forwarded_signal
        forwarded_signal = signum
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, handle_signal)
        if process.stdout is None:
            raise RuntimeError("Child stdout pipe was not created")
        reader = threading.Thread(
            target=stream_reader,
            args=(process.stdout, events),
            daemon=True,
        )
        reader.start()
        with log_path.open("a", buffering=1) as log:
            log.write(f"[{start_utc}] START pid={process.pid} command={args.command!r}\n")
            while True:
                now = time.time()
                try:
                    while True:
                        event_time, line = events.get_nowait()
                        last_console_activity = event_time
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        log.write(line)
                except queue.Empty:
                    pass
                current_mtime = latest_mtime(output_root)
                if current_mtime is not None:
                    last_fs_activity = max(last_fs_activity, current_mtime)
                last_activity = max(last_console_activity, last_fs_activity)
                stale_for = max(0.0, now - last_activity)
                if stale_for >= args.stale_seconds:
                    stale_detected = True
                    if (
                        args.fail_on_stale
                        and process.poll() is None
                        and stale_term_sent_at is None
                    ):
                        log.write(
                            f"[{utc_now()}] STALE timeout={stale_for:.1f}s; "
                            "sending SIGTERM\n"
                        )
                        os.killpg(process.pid, signal.SIGTERM)
                        stale_term_sent_at = now
                if (
                    args.fail_on_stale
                    and stale_term_sent_at is not None
                    and process.poll() is None
                    and not stale_kill_sent
                    and now - stale_term_sent_at >= args.termination_grace_seconds
                ):
                    log.write(
                        f"[{utc_now()}] STALE child ignored SIGTERM for "
                        f"{args.termination_grace_seconds:.1f}s; sending SIGKILL\n"
                    )
                    os.killpg(process.pid, signal.SIGKILL)
                    stale_kill_sent = True
                if now - last_heartbeat >= args.heartbeat_seconds:
                    atomic_json(
                        heartbeat_path,
                        {
                            "experiment_id": args.experiment_id,
                            "execution_state": "running" if process.poll() is None else "exited",
                            "utc": utc_now(),
                            "pid": process.pid,
                            "elapsed_seconds": round(now - start_wall, 3),
                            "process_returncode": process.poll(),
                            "seconds_since_activity": round(stale_for, 3),
                            "stale_detected": stale_detected,
                            "stale_term_sent": stale_term_sent_at is not None,
                            "stale_kill_sent": stale_kill_sent,
                            "progress": progress_counts(output_root, args.progress_glob),
                            "latest_output_mtime_utc": datetime.fromtimestamp(
                                last_fs_activity, timezone.utc
                            ).isoformat(),
                        },
                    )
                    last_heartbeat = now
                if (
                    process.poll() is not None
                    and events.empty()
                    and reader is not None
                    and not reader.is_alive()
                ):
                    break
                time.sleep(min(1.0, args.heartbeat_seconds / 4.0))
            if reader is not None:
                reader.join(timeout=2)
            try:
                while True:
                    _event_time, line = events.get_nowait()
                    sys.stdout.write(line)
                    log.write(line)
            except queue.Empty:
                pass
            returncode = process.wait()
            log.write(f"[{utc_now()}] EXIT returncode={returncode}\n")
    except Exception as exc:
        supervisor_error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            with log_path.open("a") as log:
                log.write(
                    f"[{utc_now()}] SUPERVISOR_EXCEPTION "
                    f"{supervisor_error['type']}: {supervisor_error['message']}\n"
                )
                log.write(supervisor_error["traceback"])
        except OSError:
            pass
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=args.termination_grace_seconds)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        polled = process.poll()
        returncode = polled if polled is not None else -1
    finally:
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass

    try:
        end_head = git_head(repo)
        end_status = git_status_excluding(repo, ignored_runtime_paths)
        end_branch = git_branch(repo)
        provenance_resolution_failed = False
    except Exception as exc:
        provenance_resolution_failed = True
        end_head = launch_head
        end_status = [f"unable_to_resolve_end_git_state: {type(exc).__name__}: {exc}"]
        end_branch = "UNRESOLVED"
        if supervisor_error is None:
            supervisor_error = {
                "type": type(exc).__name__,
                "message": f"End-of-run provenance resolution failed: {exc}",
                "traceback": traceback.format_exc(),
            }
        try:
            with log_path.open("a") as log:
                log.write(supervisor_error["traceback"])
        except OSError:
            pass
    provenance_compromised = args.run_class == "formal" and (
        provenance_resolution_failed or end_head != launch_head or bool(end_status)
    )
    missing_outputs = [name for name in args.required_output if not (output_root / name).exists()]
    success = (
        returncode == 0
        and not missing_outputs
        and not (args.fail_on_stale and stale_detected)
        and not provenance_compromised
        and supervisor_error is None
    )
    common = {
        "schema_version": 3,
        "experiment_id": args.experiment_id,
        "run_class": args.run_class,
        "start_utc": start_utc,
        "end_utc": utc_now(),
        "elapsed_seconds": round(time.time() - start_wall, 3),
        "pid": process.pid,
        "returncode": returncode,
        "supervisor_error_type": supervisor_error["type"] if supervisor_error else None,
        "supervisor_error": supervisor_error["message"] if supervisor_error else None,
        "traceback_file": "logs/supervised_run.log" if supervisor_error else None,
        "forwarded_signal": forwarded_signal,
        "stale_detected": stale_detected,
        "stale_term_sent": stale_term_sent_at is not None,
        "stale_kill_sent": stale_kill_sent,
        "missing_required_outputs": missing_outputs,
        "base_commit": launch_head,
        "end_commit": end_head,
        "git_status_at_end": end_status,
        "provenance_compromised": provenance_compromised,
        "branch": end_branch,
        "command": args.command,
        "scientific_acceptance_pending": True,
    }
    if success:
        atomic_json(output_root / "RUN_RAW_COMPLETE.json", {**common, "execution_state": "raw_complete"})
        package_kind = "experiment-raw-complete"
    else:
        atomic_json(output_root / "RUN_FAILED.json", {**common, "execution_state": "failed"})
        package_kind = "experiment-failed"
    run_manifest.update(
        {
            "execution_state": "raw_complete" if success else "failed",
            "end_utc": common["end_utc"],
            "returncode": returncode,
            "missing_required_outputs": missing_outputs,
            "stale_detected": stale_detected,
            "end_commit": end_head,
            "git_status_at_end": end_status,
            "provenance_compromised": provenance_compromised,
        }
    )
    atomic_json(output_root / "run_manifest.json", run_manifest)
    packaged = package_recovery(
        repo,
        args.experiment_id,
        output_root,
        artifact_output,
        package_kind,
        args.source_file,
        launch_commit=launch_head,
        sidecar_output=sidecar_output,
        sidecar_files=args.sidecar_file,
        sidecar_purpose=args.sidecar_purpose,
        max_package_mib=args.max_package_mib,
        max_single_file_mib=args.max_single_file_mib,
        max_sidecar_mib=args.max_sidecar_mib,
        max_sidecar_files=args.max_sidecar_files,
        large_file_persistence=args.large_file_persistence,
    )
    package_record = {
        "command": packaged.args,
        "returncode": packaged.returncode,
        "stdout": packaged.stdout,
        "stderr": packaged.stderr,
        "artifact_output": str(artifact_output),
        "artifact_exists": artifact_output.is_file(),
    }
    atomic_json(output_root / "recovery_package_status.json", package_record)
    if packaged.returncode != 0:
        print(packaged.stdout, end="")
        print(packaged.stderr, file=sys.stderr, end="")
        return 3
    print(packaged.stdout, end="")
    if success:
        print(
            "Raw computation completed and an internally verified recovery artifact was "
            "published. Terminal audit and final scientific packaging remain required."
        )
        return 0
    print("The run failed; lightweight evidence was preserved in a verified failed-run artifact.")
    return returncode if returncode != 0 else 2


def resolve_main_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve and verify the DRPO main commit")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-sha")
    parser.add_argument("--require-origin-main-match", action="store_true")
    return parser.parse_args(argv)


def resolve_main_cli(argv: list[str] | None = None) -> int:
    args = resolve_main_parse_args(argv)
    try:
        report = resolve_commit(
            args.repo_root.resolve(),
            args.expected_sha,
            require_origin_match=args.require_origin_main_match,
        )
        print(json.dumps(report, indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
