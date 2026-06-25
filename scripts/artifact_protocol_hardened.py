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
import hashlib
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
LARGE_BINARY_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".pt",
    ".pth",
    ".safetensors",
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


def validate_result_markers(kind: str, result_dir: Path | None) -> None:
    if kind not in RESULT_KINDS:
        return
    if result_dir is None or not result_dir.is_dir():
        raise ValueError(f"{kind} requires an existing --result-dir")
    if kind == "experiment-final":
        if not (result_dir / "RUN_COMPLETE.json").is_file():
            raise ValueError("experiment-final requires RUN_COMPLETE.json")
        if find_terminal_audit(result_dir) is None:
            raise ValueError("experiment-final requires a terminal audit")
    elif kind == "experiment-failed":
        if not (result_dir / "RUN_FAILED.json").is_file():
            raise ValueError("experiment-failed requires RUN_FAILED.json")
    elif kind == "experiment-raw-complete":
        if not (result_dir / "RUN_RAW_COMPLETE.json").is_file():
            raise ValueError("experiment-raw-complete requires RUN_RAW_COMPLETE.json")


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


def _is_checkpoint_like(rel: Path) -> bool:
    lower = rel.as_posix().lower()
    return rel.suffix.lower() in LARGE_BINARY_SUFFIXES or any(
        token in lower for token in MODEL_OR_OPTIMIZER_NAMES
    )


def scan_result_tree(
    result_dir: Path,
    *,
    package_kind: str,
    max_single_bytes: int,
    sidecar_enabled: bool,
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
                        kind="symlink_file",
                        size_bytes=0,
                        sha256=None,
                        target=str(resolved),
                        include_main=False,
                        include_sidecar=False,
                        reason="internal_reference" if internal else "external_symlink_rejected",
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
            too_large = size > max_single_bytes
            lightweight = package_kind in RECOVERY_KINDS
            include_main = not too_large and not (lightweight and checkpoint_like)
            include_sidecar = bool(sidecar_enabled and (checkpoint_like or too_large))
            textual = rel.suffix.lower() in {".log", ".txt", ".json", ".jsonl", ".csv", ".md"}
            if include_main:
                reason = "included_main"
            elif too_large and textual and not checkpoint_like:
                reason = "text_tail_in_main"
            elif include_sidecar:
                reason = "large_or_checkpoint_sidecar"
            else:
                reason = "large_or_checkpoint_index_only"
            entries.append(
                ResultEntry(
                    path=rel.as_posix(),
                    kind="file",
                    size_bytes=size,
                    sha256=digest,
                    target=None,
                    include_main=include_main,
                    include_sidecar=include_sidecar,
                    reason=reason,
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


def verify_result_markers(names: set[str], manifest: dict[str, Any]) -> None:
    kind = str(manifest.get("package_kind"))
    experiment_id = str(manifest.get("experiment_id"))
    prefix = f"results/{experiment_id}/"
    if kind == "experiment-final":
        if prefix + "RUN_COMPLETE.json" not in names:
            raise ValueError("experiment-final is missing RUN_COMPLETE.json")
        audits = {prefix + "TERMINAL_AUDIT.json", prefix + "terminal_audit.json"}
        if not names.intersection(audits):
            raise ValueError("experiment-final is missing a terminal audit")
    elif kind == "experiment-failed" and prefix + "RUN_FAILED.json" not in names:
        raise ValueError("experiment-failed is missing RUN_FAILED.json")
    elif kind == "experiment-raw-complete" and prefix + "RUN_RAW_COMPLETE.json" not in names:
        raise ValueError("experiment-raw-complete is missing RUN_RAW_COMPLETE.json")


def run_git_apply_check(repo: Path, patch: bytes, expected_sha: str) -> None:
    actual = git_head(repo)
    if actual != expected_sha:
        raise ValueError(f"Repository HEAD {actual} does not match BASE_COMMIT {expected_sha}")
    if not patch.strip():
        return
    with tempfile.NamedTemporaryFile(suffix=".patch") as handle:
        handle.write(patch)
        handle.flush()
        result = subprocess.run(
            ["git", "apply", "--check", "--cached", handle.name],
            cwd=repo,
            text=True,
            capture_output=True,
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
        base_text = zf.read("BASE_COMMIT.txt").decode("utf-8")
        if not re.fullmatch(r"[0-9a-f]{40}\n", base_text):
            raise ValueError("BASE_COMMIT.txt must contain exactly one lowercase full SHA")
        base_sha = base_text.strip()
        manifest = json.loads(zf.read("ARTIFACT_MANIFEST.json"))
        if manifest.get("base_commit") != base_sha:
            raise ValueError("Manifest base_commit does not match BASE_COMMIT.txt")
        kind = str(manifest.get("package_kind"))
        patch = zf.read("update.patch")
        if kind in FINAL_KINDS and not patch.strip():
            raise ValueError(f"{kind} requires a non-empty update.patch")
        if kind in FINAL_KINDS and not any(name.startswith("modified_files/") for name in names):
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
        verify_result_markers(names, manifest)
        for name in list(manifest.get("modified_files") or []):
            path = f"modified_files/{name}"
            if path not in names:
                raise ValueError(f"Manifest modified file is missing: {path}")
        index_name = "LARGE_FILE_INDEX.json"
        if index_name in names:
            index = json.loads(zf.read(index_name))
            if not isinstance(index.get("entries"), list):
                raise ValueError("LARGE_FILE_INDEX.json is malformed")
            if index.get("sidecar_required") and not index.get("sidecar"):
                raise ValueError("Large-file index requires a sidecar but none is declared")
    if repo_root is not None and not skip_head_match:
        run_git_apply_check(repo_root.resolve(), patch, base_sha)
    return {
        "package": str(package),
        "package_kind": kind,
        "base_commit": base_sha,
        "size_mib": round(size_mib, 3),
        "hard_limit_mib": hard_limit_mib,
        "checksum_files": len(checksums),
        "git_apply_check": bool(repo_root is not None and not skip_head_match),
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
    parser.add_argument("--source-file", action="append", default=[])
    parser.add_argument("--warning-mib", type=float, default=25.0)
    parser.add_argument("--max-package-mib", type=float, default=25.0)
    parser.add_argument("--max-single-file-mib", type=float, default=10.0)
    parser.add_argument("--sidecar-output", type=Path)
    parser.add_argument("--require-origin-main-match", action="store_true")
    return parser.parse_args(argv)


def package_main(argv: list[str] | None = None) -> int:
    args = package_parse_args(argv)
    repo = args.repo_root.resolve()
    try:
        resolution = resolve_commit(
            repo,
            args.base_commit,
            require_origin_match=args.require_origin_main_match,
        )
        base_commit = resolution["local_head"]
        result_dir = args.result_dir.resolve() if args.result_dir else None
        validate_result_markers(args.package_kind, result_dir)
        if args.changed_file:
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
        max_single_bytes = max(1, int(args.max_single_file_mib * 1024 * 1024))
        sidecar_output = args.sidecar_output.resolve() if args.sidecar_output else None
        entries: list[ResultEntry] = []
        if result_dir is not None:
            entries = scan_result_tree(
                result_dir,
                package_kind=args.package_kind,
                max_single_bytes=max_single_bytes,
                sidecar_enabled=sidecar_output is not None,
            )
        excluded = [entry for entry in entries if entry.kind == "file" and not entry.include_main]
        if args.package_kind == "experiment-final" and excluded and sidecar_output is None:
            raise ValueError(
                "experiment-final contains large/checkpoint files. Supply --sidecar-output "
                "or reduce the result directory."
            )
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
            for rel in source_files:
                source = repo / rel
                if not source.is_file() or source.is_symlink():
                    raise ValueError(f"Missing or unsafe source file: {rel}")
                target = stage / "source_snapshot" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            sidecar_meta: dict[str, Any] | None = None
            if sidecar_output is not None and any(e.include_sidecar for e in entries):
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
                    "schema_version": 1,
                    "experiment_id": args.experiment_id,
                    "base_commit": base_commit,
                    "generated_utc": utc_now(),
                    "files": [asdict(e) for e in entries if e.include_sidecar],
                }
                (side_stage / "SIDECAR_MANIFEST.json").write_text(
                    json.dumps(side_manifest, indent=2, ensure_ascii=False) + "\n"
                )
                write_checksums(side_stage)
                candidate_sidecar = sidecar_output.with_suffix(sidecar_output.suffix + ".candidate")
                candidate_sidecar.parent.mkdir(parents=True, exist_ok=True)
                candidate_sidecar.unlink(missing_ok=True)
                write_zip_from_stage(side_stage, candidate_sidecar)
                sidecar_meta = {
                    "path": sidecar_output.name,
                    "sha256": file_sha256(candidate_sidecar),
                    "size_bytes": candidate_sidecar.stat().st_size,
                    "status": "candidate_verified_by_checksum",
                }
            large_index = {
                "schema_version": 1,
                "generated_utc": utc_now(),
                "entries": [asdict(e) for e in entries if e.kind != "file" or not e.include_main],
                "sidecar_required": bool(args.package_kind == "experiment-final" and excluded),
                "sidecar": sidecar_meta,
            }
            (stage / "LARGE_FILE_INDEX.json").write_text(
                json.dumps(large_index, indent=2, ensure_ascii=False) + "\n"
            )
            manifest = {
                "schema_version": 2,
                "generated_utc": utc_now(),
                "experiment_id": args.experiment_id,
                "package_kind": args.package_kind,
                "base_commit": base_commit,
                "repository": "easonhuo/drpo",
                "branch": git_branch(repo),
                "commit_resolution": resolution,
                "modified_files": [p.as_posix() for p in paths],
                "result_dir_name": args.experiment_id if result_dir else None,
                "source_files": [p.as_posix() for p in source_files],
                "scientific_completion_claim": args.package_kind == "experiment-final",
                "durable_delivery_pending": True,
                "lightweight_recovery": args.package_kind in RECOVERY_KINDS,
                "main_package_hard_limit_mib": args.max_package_mib,
                "max_single_file_mib": args.max_single_file_mib,
                "sidecar": sidecar_meta,
                "symlink_policy": "never_follow",
            }
            (stage / "ARTIFACT_MANIFEST.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
            )
            write_checksums(stage)
            output = args.output.resolve()
            candidate = output.with_suffix(output.suffix + ".candidate")
            output.parent.mkdir(parents=True, exist_ok=True)
            candidate.unlink(missing_ok=True)
            output.unlink(missing_ok=True)
            write_zip_from_stage(stage, candidate)
            try:
                report = verify_package(
                    candidate,
                    repo_root=repo,
                    skip_head_match=False,
                    hard_limit_mib=args.max_package_mib,
                )
            except Exception:
                candidate.unlink(missing_ok=True)
                if sidecar_output is not None:
                    sidecar_output.with_suffix(sidecar_output.suffix + ".candidate").unlink(missing_ok=True)
                raise
            os.replace(candidate, output)
            if sidecar_output is not None:
                candidate_sidecar = sidecar_output.with_suffix(sidecar_output.suffix + ".candidate")
                if candidate_sidecar.exists():
                    os.replace(candidate_sidecar, sidecar_output)
            report.update(
                {
                    "output": str(output),
                    "atomic_publish": True,
                    "sidecar_output": str(sidecar_output) if sidecar_output and sidecar_output.exists() else None,
                }
            )
            print(json.dumps(report, indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


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
            if path.is_file():
                value = path.stat().st_mtime
                latest = value if latest is None else max(latest, value)
        except FileNotFoundError:
            continue
    return latest


def progress_counts(root: Path, patterns: list[str]) -> dict[str, int]:
    return {pattern: sum(1 for p in root.glob(pattern) if p.is_file()) for pattern in patterns}


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
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--stale-seconds", type=float, default=600.0)
    parser.add_argument("--fail-on-stale", action="store_true")
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
    if args.heartbeat_seconds <= 0 or args.stale_seconds <= 0:
        parser.error("Heartbeat and stale intervals must be positive")
    if args.run_class == "formal" and args.allow_dirty:
        parser.error("--allow-dirty is only valid with --run-class pilot")
    return args


def package_recovery(
    repo: Path,
    experiment_id: str,
    output_root: Path,
    artifact_output: Path,
    package_kind: str,
    source_files: list[str],
    *,
    sidecar_output: Path | None,
    max_package_mib: float,
    max_single_file_mib: float,
) -> subprocess.CompletedProcess[str]:
    script = repo / "scripts" / "package_experiment_hardened.py"
    cmd = [
        sys.executable,
        str(script),
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
        git_head(repo),
        "--max-package-mib",
        str(max_package_mib),
        "--max-single-file-mib",
        str(max_single_file_mib),
        "--test-command",
        "python3 -m pytest -q tests/test_experiment_artifact_protocol.py tests/test_experiment_artifact_hardening.py",
    ]
    if sidecar_output is not None:
        cmd.extend(["--sidecar-output", str(sidecar_output)])
    for source in source_files:
        cmd.extend(["--source-file", source])
    return subprocess.run(cmd, cwd=repo, text=True, capture_output=True)


def guard_main(argv: list[str] | None = None) -> int:
    args = guard_parse_args(argv)
    repo = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    artifact_output = args.artifact_output.resolve()
    try:
        resolution = resolve_commit(
            repo,
            args.expected_commit,
            require_origin_match=args.require_origin_main_match,
        )
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    launch_head = resolution["local_head"]
    ignored_runtime_paths = [output_root, artifact_output]
    if args.sidecar_output:
        ignored_runtime_paths.append(args.sidecar_output.resolve())
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
        "schema_version": 2,
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
        "progress_globs": args.progress_glob,
        "required_outputs": args.required_output,
        "artifact_output": str(artifact_output),
    }
    atomic_json(output_root / "run_manifest.json", run_manifest)
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
    run_manifest["pid"] = process.pid
    atomic_json(output_root / "run_manifest.json", run_manifest)
    forwarded_signal: int | None = None

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal forwarded_signal
        forwarded_signal = signum
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    previous_handlers = {sig: signal.signal(sig, handle_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    events: queue.Queue[tuple[float, str]] = queue.Queue()
    assert process.stdout is not None
    reader = threading.Thread(target=stream_reader, args=(process.stdout, events), daemon=True)
    reader.start()
    last_console_activity = time.time()
    last_fs_activity = latest_mtime(output_root) or start_wall
    last_heartbeat = 0.0
    stale_detected = False
    try:
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
                    if args.fail_on_stale and process.poll() is None:
                        log.write(f"[{utc_now()}] STALE timeout={stale_for:.1f}s; terminating\n")
                        os.killpg(process.pid, signal.SIGTERM)
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
                            "progress": progress_counts(output_root, args.progress_glob),
                            "latest_output_mtime_utc": datetime.fromtimestamp(
                                last_fs_activity, timezone.utc
                            ).isoformat(),
                        },
                    )
                    last_heartbeat = now
                if process.poll() is not None and events.empty() and not reader.is_alive():
                    break
                time.sleep(min(1.0, args.heartbeat_seconds / 4.0))
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
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    end_head = git_head(repo)
    end_status = git_status_excluding(repo, ignored_runtime_paths)
    provenance_compromised = args.run_class == "formal" and (
        end_head != launch_head or bool(end_status)
    )
    missing_outputs = [name for name in args.required_output if not (output_root / name).exists()]
    success = (
        returncode == 0
        and not missing_outputs
        and not (args.fail_on_stale and stale_detected)
        and not provenance_compromised
    )
    common = {
        "schema_version": 2,
        "experiment_id": args.experiment_id,
        "run_class": args.run_class,
        "start_utc": start_utc,
        "end_utc": utc_now(),
        "elapsed_seconds": round(time.time() - start_wall, 3),
        "pid": process.pid,
        "returncode": returncode,
        "forwarded_signal": forwarded_signal,
        "stale_detected": stale_detected,
        "missing_required_outputs": missing_outputs,
        "base_commit": launch_head,
        "end_commit": end_head,
        "git_status_at_end": end_status,
        "provenance_compromised": provenance_compromised,
        "branch": git_branch(repo),
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
        sidecar_output=args.sidecar_output.resolve() if args.sidecar_output else None,
        max_package_mib=args.max_package_mib,
        max_single_file_mib=args.max_single_file_mib,
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
