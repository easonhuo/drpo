#!/usr/bin/env python3
"""Validation and delivery-manifest helpers for RunSpec result publication."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from runspec_lib import (
    ARTIFACT_DIRNAME,
    RunSpecError,
    is_model_like,
    now_utc,
    safe_relpath,
    sha256_file,
)

BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
DEFAULT_MAX_FILE_MB = 10
DEFAULT_MAX_TOTAL_MB = 25
DELIVERY_ROOT = Path("runspec_deliveries")


def validate_branch(value: str, *, field: str) -> str:
    branch = value.strip()
    if not BRANCH_RE.fullmatch(branch) or branch.startswith("-") or ".." in branch:
        raise RunSpecError(f"invalid {field}: {branch!r}")
    if branch in {"main", "master"}:
        raise RunSpecError(f"{field} must be a dev branch, not {branch}")
    return branch


def validate_publish_block(spec: dict[str, Any], lane: str) -> dict[str, Any]:
    publish = spec.get("publish") or {}
    if not isinstance(publish, dict):
        raise RunSpecError("publish must be a mapping")
    if publish.get("enabled") is not True:
        raise RunSpecError("publish.enabled must be true")
    branch = validate_branch(str(publish.get("dev_branch") or ""), field="publish.dev_branch")
    if not branch.startswith("dev/"):
        raise RunSpecError("publish.dev_branch must start with dev/")
    lane_token = lane.replace("_", "-")
    if lane_token not in branch.lower():
        raise RunSpecError(
            f"publish.dev_branch={branch} must identify workspace lane={lane}"
        )
    base_branch = str(publish.get("base_branch") or "main").strip()
    if not BRANCH_RE.fullmatch(base_branch):
        raise RunSpecError(f"invalid publish.base_branch: {base_branch!r}")
    commit_paths = publish.get("commit_paths") or []
    if not isinstance(commit_paths, list) or not commit_paths:
        raise RunSpecError("publish.commit_paths must be a non-empty list")
    normalized: list[str] = []
    for value in commit_paths:
        if not isinstance(value, str):
            raise RunSpecError("publish.commit_paths entries must be strings")
        rel = safe_relpath(value).as_posix()
        if any(ch in rel for ch in "*?[]"):
            raise RunSpecError("publish.commit_paths must contain exact files, not globs")
        if rel.startswith((".git/", ".runspec_state/", f"{ARTIFACT_DIRNAME}/")):
            raise RunSpecError(f"publish.commit_paths may not include local state/artifacts: {rel}")
        if is_model_like(rel):
            raise RunSpecError(f"publish.commit_paths contains model/checkpoint-like file: {rel}")
        normalized.append(rel)
    if len(set(normalized)) != len(normalized):
        raise RunSpecError("publish.commit_paths contains duplicates")
    result = dict(publish)
    result["dev_branch"] = branch
    result["base_branch"] = base_branch
    result["commit_paths"] = normalized
    result.setdefault("auto", False)
    result.setdefault("remote", "origin")
    result.setdefault("create_draft_pr", True)
    result.setdefault("max_commit_file_size_mb", DEFAULT_MAX_FILE_MB)
    result.setdefault("max_commit_total_size_mb", DEFAULT_MAX_TOTAL_MB)
    result.setdefault("commit_message", f"Publish {spec['run_id']} results")
    result.setdefault("pr_title", f"[{lane.upper()}] {spec['run_id']} result delivery")
    return result


def validate_commit_files(
    repo: Path,
    paths: Iterable[str],
    publish: dict[str, Any],
) -> list[dict[str, Any]]:
    max_file = int(publish["max_commit_file_size_mb"]) * 1024 * 1024
    max_total = int(publish["max_commit_total_size_mb"]) * 1024 * 1024
    if max_file <= 0 or max_total <= 0 or max_file > max_total:
        raise RunSpecError("publish size limits must be positive and file <= total")
    rows: list[dict[str, Any]] = []
    total = 0
    for rel in paths:
        path = repo / rel
        if path.is_symlink():
            raise RunSpecError(f"publish file may not be a symlink: {rel}")
        if not path.is_file():
            raise RunSpecError(f"publish file does not exist: {rel}")
        size = path.stat().st_size
        if size > max_file:
            raise RunSpecError(f"publish file too large: {rel} ({size} > {max_file} bytes)")
        total += size
        rows.append({"path": rel, "size_bytes": size, "sha256": sha256_file(path)})
    if total > max_total:
        raise RunSpecError(f"publish files too large in total: {total} > {max_total} bytes")
    return rows


def artifact_manifest(repo: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    path = repo / ARTIFACT_DIRNAME / f"{run_id}_manifest.json"
    if not path.is_file():
        raise RunSpecError(f"artifact manifest is missing: {path.relative_to(repo)}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunSpecError(f"invalid artifact manifest JSON: {exc}") from exc
    if payload.get("run_id") != run_id:
        raise RunSpecError("artifact manifest run_id mismatch")
    zip_rel = str(payload.get("zip_path") or "")
    zip_path = repo / safe_relpath(zip_rel)
    if not zip_path.is_file():
        raise RunSpecError(f"artifact ZIP is missing: {zip_rel}")
    actual = sha256_file(zip_path)
    if actual != payload.get("zip_sha256"):
        raise RunSpecError("artifact ZIP checksum does not match artifact manifest")
    return path, payload


def write_delivery_manifest(
    repo: Path,
    *,
    spec: dict[str, Any],
    publish: dict[str, Any],
    commit_files: list[dict[str, Any]],
    artifact: dict[str, Any],
    parent_commit: str,
) -> Path:
    delivery_dir = repo / DELIVERY_ROOT / spec["run_id"]
    delivery_dir.mkdir(parents=True, exist_ok=True)
    path = delivery_dir / "DELIVERY_MANIFEST.json"
    payload = {
        "schema_version": 1,
        "created_at": now_utc(),
        "run_id": spec["run_id"],
        "lane": spec["lane"],
        "experiment_id": spec["experiment_id"],
        "run_repo_commit": spec.get("repo_commit"),
        "publish_parent_commit": parent_commit,
        "dev_branch": publish["dev_branch"],
        "base_branch": publish["base_branch"],
        "artifact_zip": {
            "path": artifact["zip_path"],
            "sha256": artifact["zip_sha256"],
            "size_bytes": (repo / safe_relpath(artifact["zip_path"])).stat().st_size,
            "persistence": "training_server_local",
            "committed_to_git": False,
        },
        "committed_result_files": commit_files,
        "review_contract": {
            "merge_mode": "selective_integration_from_current_main",
            "automatic_merge_allowed": False,
            "reviewer_must_check": [
                "scientific-variable drift",
                "result provenance",
                "artifact completeness",
                "result-status classification",
                "registry/handoff delta",
                "required gates",
            ],
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
