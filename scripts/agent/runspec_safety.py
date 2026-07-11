#!/usr/bin/env python3
"""Fail-closed safety wrappers for RunSpec execution and publication."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from runspec_lib import (
    CLAIMED_DIR,
    DONE_DIR,
    FAILED_DIR,
    RUNNING_DIR,
    RunSpecError,
    command_script_path,
    current_commit,
    git_text,
    is_ancestor,
    iter_ready_specs,
    now_utc,
    package_artifacts,
    safe_relpath,
    state_path,
    validate_runspec,
)

PUBLISHED_DIR = Path(".runspec_state") / "published"
ACTIVE_STATE_DIRS = (CLAIMED_DIR, RUNNING_DIR, DONE_DIR, FAILED_DIR, PUBLISHED_DIR)
FIXED_WORKSPACE_BRANCHES = {"e7": "dev/server-e7", "e8": "dev/server-e8"}


def existing_run_states(repo: Path, run_id: str) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for state_dir in ACTIVE_STATE_DIRS:
        path = state_path(repo, state_dir, run_id)
        if path.exists():
            rows.append((state_dir.name, path))
    return rows


def _state_error(run_id: str, states: list[tuple[str, Path]]) -> RunSpecError:
    names = sorted({name for name, _ in states})
    if any(name in {"done", "published"} for name in names):
        return RunSpecError(
            f"ALREADY_COMPLETED run_id={run_id} states={names}; use a new run_id for a rerun"
        )
    if "failed" in names:
        return RunSpecError(
            f"FAILED_RUN_ID_REUSE_FORBIDDEN run_id={run_id}; create a new run_id for retry"
        )
    return RunSpecError(f"ALREADY_ACTIVE run_id={run_id} states={names}")


def _protected_paths(spec: dict[str, Any]) -> list[str]:
    provenance = spec.get("provenance") or {}
    if not isinstance(provenance, dict):
        raise RunSpecError("provenance must be a mapping")
    values = provenance.get("protected_paths") or []
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise RunSpecError("provenance.protected_paths must be a list of exact paths")
    command = str((spec.get("entrypoint") or {}).get("command") or "")
    script = command_script_path(command)
    paths = list(values)
    if script and script not in paths:
        paths.append(script)
    normalized: list[str] = []
    for value in paths:
        rel = safe_relpath(value).as_posix()
        if any(ch in rel for ch in "*?[]"):
            raise RunSpecError("provenance.protected_paths must use exact files, not globs")
        if rel not in normalized:
            normalized.append(rel)
    return normalized


def validate_provenance(repo: Path, spec: dict[str, Any]) -> None:
    repo_commit = str(spec.get("repo_commit") or "").strip()
    if not repo_commit:
        # Compatibility for legacy tests/specs. Production templates always pin a commit.
        return
    head = current_commit(repo)
    provenance = spec.get("provenance") or {}
    policy = str(provenance.get("commit_policy") or "protected_paths_unchanged")
    if policy not in {"protected_paths_unchanged", "exact_head"}:
        raise RunSpecError(f"unsupported provenance.commit_policy: {policy}")
    if policy == "exact_head":
        if repo_commit != head:
            raise RunSpecError(
                "repo_commit must equal HEAD under exact_head: "
                f"{repo_commit} != {head}"
            )
        return
    if repo_commit == head:
        return
    if not is_ancestor(repo, repo_commit, "HEAD"):
        raise RunSpecError(
            f"repo_commit {repo_commit} is not an ancestor of current HEAD {head}"
        )
    paths = _protected_paths(spec)
    if not paths:
        raise RunSpecError(
            "protected_paths_unchanged requires entrypoint script or provenance.protected_paths"
        )
    changed = git_text(repo, "diff", "--name-only", f"{repo_commit}..HEAD", "--", *paths)
    dirty = git_text(repo, "diff", "--name-only", "HEAD", "--", *paths)
    staged = git_text(repo, "diff", "--cached", "--name-only", "HEAD", "--", *paths)
    drift = sorted(
        {
            line
            for text in (changed, dirty, staged)
            for line in text.splitlines()
            if line
        }
    )
    if drift:
        raise RunSpecError(
            f"protected experiment files changed since repo_commit {repo_commit}: {drift}"
        )


def validate_runspec_safe(
    repo: Path,
    spec_path: Path,
    *,
    lane_config: dict[str, Any] | None = None,
    require_registry: bool = True,
) -> dict[str, Any]:
    spec = validate_runspec(
        repo,
        spec_path,
        lane_config=lane_config,
        require_registry=require_registry,
    )
    validate_provenance(repo, spec)
    return spec


def claim_next_runspec_safe(
    repo: Path,
    *,
    lane_config: dict[str, Any],
    run_id: str | None = None,
) -> Path:
    candidates: list[tuple[int, str, str, Path, dict[str, Any]]] = []
    errors: list[str] = []
    for path in iter_ready_specs(repo):
        try:
            spec = validate_runspec_safe(repo, path, lane_config=lane_config)
            if spec["lane"] != lane_config["lane"]:
                continue
            if run_id and spec["run_id"] != run_id:
                continue
            states = existing_run_states(repo, spec["run_id"])
            if states:
                if run_id:
                    raise _state_error(spec["run_id"], states)
                continue
            priority = int(spec.get("priority", 0) or 0)
            created_at = str(spec.get("created_at") or "")
            candidates.append((-priority, created_at, spec["run_id"], path, spec))
        except RunSpecError as exc:
            errors.append(f"{path}: {exc}")
    if not candidates:
        target = f" run_id={run_id}" if run_id else ""
        detail = "\nRejected candidates:\n" + "\n".join(errors[:10]) if errors else ""
        raise RunSpecError(f"NO_READY_TASK lane={lane_config['lane']}{target}{detail}")
    _, _, selected_run_id, selected_path, spec = sorted(candidates)[0]
    claimed = state_path(repo, CLAIMED_DIR, selected_run_id)
    claimed.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(spec)
    payload["claim"] = {
        "claimed_at": now_utc(),
        "source_path": selected_path.relative_to(repo).as_posix(),
        "lane": lane_config["lane"],
    }
    try:
        fd = os.open(claimed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RunSpecError(f"ALREADY_ACTIVE run_id={selected_run_id} state=claimed") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return claimed


def _has_symlink_component(repo: Path, path: Path) -> bool:
    try:
        rel = path.relative_to(repo)
    except ValueError:
        return True
    cursor = repo
    for part in rel.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return True
    return False


def reject_symlink_artifacts(repo: Path, spec: dict[str, Any]) -> None:
    for pattern in list((spec.get("artifacts") or {}).get("include") or []):
        safe_relpath(pattern.replace("*", "x") or "x")
        for path in repo.glob(pattern):
            if _has_symlink_component(repo, path):
                rel = path.relative_to(repo).as_posix() if path.is_relative_to(repo) else str(path)
                raise RunSpecError(f"artifact include matched symlink path: {rel}")
            resolved = path.resolve(strict=False)
            if not resolved.is_relative_to(repo.resolve()):
                raise RunSpecError(f"artifact path escapes repository: {path}")


def package_artifacts_safe(
    repo: Path,
    spec_path: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    spec = validate_runspec_safe(repo, spec_path, require_registry=False)
    reject_symlink_artifacts(repo, spec)
    return package_artifacts(repo, spec_path, output_dir=output_dir)


def expected_workspace_branch(lane: str) -> str:
    try:
        return FIXED_WORKSPACE_BRANCHES[lane]
    except KeyError as exc:
        raise RunSpecError(f"no fixed workspace branch for lane={lane}") from exc


def _git_ok(repo: Path, *args: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def ensure_workspace_branch(repo: Path, lane: str) -> str:
    expected = expected_workspace_branch(lane)
    if not _git_ok(repo, "diff", "--quiet") or not _git_ok(
        repo, "diff", "--cached", "--quiet"
    ):
        raise RunSpecError(
            "tracked worktree/index must be clean before configuring executor branch"
        )
    current = git_text(repo, "branch", "--show-current")
    if current == expected:
        return expected
    if not _git_ok(repo, "rev-parse", "--verify", "HEAD"):
        git_text(repo, "switch", "-c", expected)
        return expected
    source_head = current_commit(repo)
    if _git_ok(repo, "show-ref", "--verify", "--quiet", f"refs/heads/{expected}"):
        git_text(repo, "switch", expected)
    elif _git_ok(repo, "show-ref", "--verify", "--quiet", f"refs/remotes/origin/{expected}"):
        git_text(repo, "switch", "--track", "-c", expected, f"origin/{expected}")
    else:
        git_text(repo, "switch", "-c", expected)
        return expected
    branch_head = current_commit(repo)
    if branch_head == source_head or is_ancestor(repo, source_head, "HEAD"):
        return expected
    if is_ancestor(repo, branch_head, source_head):
        git_text(repo, "merge", "--ff-only", source_head)
        return expected
    raise RunSpecError(
        f"existing {expected} diverged from source HEAD {source_head}; "
        "review/reset the lane branch before configuring"
    )


def validate_fixed_publish_branch(spec: dict[str, Any], lane_config: dict[str, Any]) -> None:
    expected = str(lane_config.get("publish_branch") or "").strip()
    if not expected:
        return
    publish = spec.get("publish") or {}
    actual = str(publish.get("dev_branch") or "").strip()
    if actual != expected:
        raise RunSpecError(
            f"publish.dev_branch={actual!r} must equal workspace publish_branch={expected!r}"
        )
