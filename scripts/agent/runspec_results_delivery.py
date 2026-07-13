#!/usr/bin/env python3
"""Export completed RunSpec evidence and append it to a results-only Git repo."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from runspec_lib import (
    ARTIFACT_DIRNAME,
    DONE_DIR,
    RunSpecError,
    is_model_like,
    now_utc,
    read_yaml,
    safe_relpath,
    sha256_file,
    state_path,
    validate_runspec,
)

DELIVERY_ROOT = Path(".runspec_state") / "delivery"
RESULTS_CACHE_ROOT = Path(".runspec_state") / "results_repo"
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RunSpecError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{detail}")
    return proc


def _git(repo: Path, *args: str, check: bool = True) -> str:
    return _run(["git", "-C", str(repo), *args], cwd=repo, check=check).stdout.strip()


def validate_delivery_block(spec: dict[str, Any], lane: str) -> dict[str, Any]:
    """Validate and normalize the optional results-repository delivery block."""

    raw = spec.get("delivery") or {}
    if not isinstance(raw, dict):
        raise RunSpecError("delivery must be a mapping")
    enabled = raw.get("enabled", False)
    auto = raw.get("auto", False)
    if not isinstance(enabled, bool) or not isinstance(auto, bool):
        raise RunSpecError("delivery.enabled and delivery.auto must be booleans")
    if auto and not enabled:
        raise RunSpecError("delivery.auto=true requires delivery.enabled=true")
    publish = spec.get("publish") or {}
    if (
        enabled
        and isinstance(publish, dict)
        and publish.get("enabled") is True
    ):
        raise RunSpecError("delivery and legacy publish may not both be enabled")
    if not enabled:
        return {"enabled": False, "auto": False}

    mode = str(raw.get("mode") or "")
    if mode != "results_repo":
        raise RunSpecError("delivery.mode must be results_repo")
    repository = str(raw.get("repository") or "")
    if not REPOSITORY_RE.match(repository):
        raise RunSpecError("delivery.repository must use owner/repository form")
    branch = str(raw.get("branch") or "")
    expected_branch = f"ingest/{lane}"
    if branch != expected_branch:
        raise RunSpecError(
            f"delivery.branch must be {expected_branch} for lane={lane}"
        )
    profile = str(raw.get("export_profile") or "manifest_text_v1")
    if profile != "manifest_text_v1":
        raise RunSpecError(
            "only delivery.export_profile=manifest_text_v1 is supported in V1"
        )
    max_total_size_mb = int(raw.get("max_total_size_mb", 30))
    max_file_size_mb = int(raw.get("max_file_size_mb", 10))
    if not 1 <= max_total_size_mb <= 100:
        raise RunSpecError("delivery.max_total_size_mb must be between 1 and 100")
    if not 1 <= max_file_size_mb <= max_total_size_mb:
        raise RunSpecError(
            "delivery.max_file_size_mb must be positive and no larger than total size"
        )
    return {
        "enabled": True,
        "auto": auto,
        "mode": mode,
        "repository": repository,
        "branch": branch,
        "export_profile": profile,
        "max_total_size_mb": max_total_size_mb,
        "max_file_size_mb": max_file_size_mb,
    }


def _has_symlink_component(repo: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(repo)
    except ValueError:
        return True
    cursor = repo
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return True
    return False


def _load_artifact_manifest(repo: Path, run_id: str) -> dict[str, Any]:
    path = repo / ARTIFACT_DIRNAME / f"{run_id}_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunSpecError(f"artifact manifest is missing: {path.relative_to(repo)}") from exc
    except json.JSONDecodeError as exc:
        raise RunSpecError(f"artifact manifest is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunSpecError("artifact manifest must contain a JSON object")
    return payload


def _verified_source_rows(
    repo: Path,
    spec: dict[str, Any],
    artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    for key in ("run_id", "lane", "experiment_id"):
        if str(artifact.get(key) or "") != str(spec.get(key) or ""):
            raise RunSpecError(f"artifact manifest {key} does not match RunSpec")
    rows = artifact.get("included") or []
    if not isinstance(rows, list) or not rows:
        raise RunSpecError("artifact manifest included list is empty")
    verified: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            raise RunSpecError("artifact manifest included row must be an object")
        rel = safe_relpath(str(raw.get("path") or "")).as_posix()
        path = repo / rel
        if not path.is_file():
            raise RunSpecError(f"artifact source file is missing: {rel}")
        if _has_symlink_component(repo, path):
            raise RunSpecError(f"artifact source uses a symlink path: {rel}")
        if not path.resolve().is_relative_to(repo.resolve()):
            raise RunSpecError(f"artifact source escapes repository: {rel}")
        if is_model_like(rel):
            raise RunSpecError(f"model-like artifact may not be delivered: {rel}")
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if int(raw.get("size_bytes", -1)) != actual_size:
            raise RunSpecError(f"artifact source size drift after packaging: {rel}")
        if str(raw.get("sha256") or "") != actual_sha:
            raise RunSpecError(f"artifact source SHA-256 drift after packaging: {rel}")
        verified.append(
            {
                "path": rel,
                "size_bytes": actual_size,
                "sha256": actual_sha,
            }
        )
    return verified


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _result_readme(
    spec: dict[str, Any],
    artifact: dict[str, Any],
    copied: list[str],
    compacted: list[str],
    omitted: list[str],
) -> str:
    return "\n".join(
        [
            f"# RunSpec result: `{spec['run_id']}`",
            "",
            f"- Lane: `{spec['lane']}`",
            f"- Experiment: `{spec['experiment_id']}`",
            f"- Source code commit: `{artifact.get('repo_commit', '')}`",
            f"- Source artifact ZIP SHA-256: `{artifact.get('zip_sha256', '')}`",
            f"- Directly copied text files: `{len(copied)}`",
            f"- Compacted branch JSON files: `{len(compacted)}`",
            f"- Omitted non-text files: `{len(omitted)}`",
            "",
            "`RESULT_MANIFEST.json` is the authoritative file inventory.",
            "`READY_FOR_REVIEW.json` is written only after the review package is complete.",
            "",
        ]
    )


def export_review_package(
    repo: Path,
    spec: dict[str, Any],
    artifact: dict[str, Any],
    delivery: dict[str, Any],
) -> dict[str, Any]:
    """Create a deterministic, text-first review package for the online reviewer."""

    rows = _verified_source_rows(repo, spec, artifact)
    run_id = str(spec["run_id"])
    package_dir = repo / DELIVERY_ROOT / run_id / "review_package"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    copied: list[str] = []
    compacted: list[str] = []
    omitted: list[str] = []
    branch_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    max_file_bytes = int(delivery["max_file_size_mb"]) * 1024 * 1024

    for row in rows:
        rel = row["path"]
        source = repo / rel
        suffix = source.suffix.lower()
        if suffix not in TEXT_SUFFIXES:
            omitted.append(rel)
            continue
        if source.stat().st_size > max_file_bytes:
            raise RunSpecError(
                f"delivery source file exceeds max_file_size_mb: {rel}"
            )
        parts = [part.lower() for part in Path(rel).parts]
        if "branches" in parts and suffix == ".json":
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RunSpecError(f"branch JSON is invalid: {rel}: {exc}") from exc
            compact_row = {
                "source_path": rel,
                "source_sha256": row["sha256"],
                "payload": payload,
            }
            if source.name.upper() == "FAILED.JSON":
                failure_rows.append(compact_row)
            else:
                branch_rows.append(compact_row)
            compacted.append(rel)
            continue
        target = package_dir / "files" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.append(rel)

    if not copied and not branch_rows and not failure_rows:
        raise RunSpecError("delivery review package contains no readable result files")
    _write_jsonl(package_dir / "BRANCH_RESULTS.jsonl", branch_rows)
    _write_jsonl(package_dir / "FAILURES.jsonl", failure_rows)
    source_manifest = {
        "schema_version": 1,
        "artifact_manifest": artifact,
        "copied": copied,
        "compacted": compacted,
        "omitted_non_text": omitted,
    }
    (package_dir / "SOURCE_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (package_dir / "README.md").write_text(
        _result_readme(spec, artifact, copied, compacted, omitted),
        encoding="utf-8",
    )

    payload_files: list[dict[str, Any]] = []
    total_size = 0
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(package_dir).as_posix()
        size = path.stat().st_size
        total_size += size
        payload_files.append(
            {"path": rel, "size_bytes": size, "sha256": sha256_file(path)}
        )
    max_total_bytes = int(delivery["max_total_size_mb"]) * 1024 * 1024
    if total_size > max_total_bytes:
        raise RunSpecError(
            f"delivery review package is too large: {total_size} > {max_total_bytes} bytes"
        )
    manifest_base = {
        "schema_version": 1,
        "created_at": artifact.get("created_at") or now_utc(),
        "run_id": run_id,
        "lane": spec["lane"],
        "experiment_id": spec["experiment_id"],
        "source_repo_commit": artifact.get("repo_commit"),
        "source_artifact_zip_sha256": artifact.get("zip_sha256"),
        "export_profile": delivery["export_profile"],
        "total_size_bytes": total_size,
        "files": payload_files,
    }
    manifest_sha = _canonical_sha256(manifest_base)
    result_manifest = {**manifest_base, "manifest_sha256": manifest_sha}
    (package_dir / "RESULT_MANIFEST.json").write_text(
        json.dumps(result_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ready = {
        "schema_version": 1,
        "status": "READY_FOR_REVIEW",
        "run_id": run_id,
        "lane": spec["lane"],
        "manifest_sha256": manifest_sha,
    }
    (package_dir / "READY_FOR_REVIEW.json").write_text(
        json.dumps(ready, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "package_dir": package_dir,
        "manifest_sha256": manifest_sha,
        "total_size_bytes": total_size,
        "copied_count": len(copied),
        "compacted_count": len(compacted),
        "omitted_count": len(omitted),
    }


def _remote_url(repository: str) -> str:
    override = os.environ.get("DRPO_RESULTS_REMOTE_URL", "").strip()
    if override:
        return override
    return f"git@github.com:{repository}.git"


def _cache_dir(repo: Path, repository: str) -> Path:
    override = os.environ.get("DRPO_RESULTS_CACHE_DIR", "").strip()
    root = Path(override).expanduser().resolve() if override else repo / RESULTS_CACHE_ROOT
    return root / repository.replace("/", "__")


def _prepare_results_checkout(
    source_repo: Path,
    repository: str,
    branch: str,
) -> Path:
    remote = _remote_url(repository)
    checkout = _cache_dir(source_repo, repository)
    if not (checkout / ".git").is_dir():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        if checkout.exists():
            shutil.rmtree(checkout)
        _run(["git", "clone", "--no-checkout", remote, str(checkout)], cwd=checkout.parent)
    else:
        existing_remote = _git(checkout, "remote", "get-url", "origin")
        if existing_remote != remote:
            raise RunSpecError(
                "cached results repository remote does not match configured repository"
            )
    _git(checkout, "fetch", "origin", "--prune")
    remote_ref = f"refs/remotes/origin/{branch}"
    if _git(checkout, "show-ref", "--verify", "--quiet", remote_ref, check=False) == "":
        probe = _run(
            ["git", "-C", str(checkout), "show-ref", "--verify", "--quiet", remote_ref],
            cwd=checkout,
            check=False,
        )
        has_remote_branch = probe.returncode == 0
    else:
        has_remote_branch = True
    if has_remote_branch:
        _git(checkout, "checkout", "-B", branch, f"origin/{branch}")
        _git(checkout, "reset", "--hard", f"origin/{branch}")
    else:
        _git(checkout, "checkout", "--orphan", branch)
        _git(checkout, "rm", "-rf", ".", check=False)
    _git(checkout, "clean", "-fd")
    _git(checkout, "config", "user.name", "DRPO Results Executor")
    _git(
        checkout,
        "config",
        "user.email",
        "drpo-results@users.noreply.github.com",
    )
    return checkout


def _existing_manifest_sha(target: Path) -> str | None:
    manifest = target / "RESULT_MANIFEST.json"
    ready = target / "READY_FOR_REVIEW.json"
    if not manifest.exists() and not ready.exists():
        return None
    if not manifest.is_file() or not ready.is_file():
        raise RunSpecError("existing result directory is incomplete")
    try:
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        ready_payload = json.loads(ready.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunSpecError(f"existing result metadata is invalid JSON: {exc}") from exc
    manifest_sha = str(manifest_payload.get("manifest_sha256") or "")
    if ready_payload.get("manifest_sha256") != manifest_sha:
        raise RunSpecError("existing READY_FOR_REVIEW does not match result manifest")
    return manifest_sha


def upload_review_package(
    source_repo: Path,
    spec: dict[str, Any],
    delivery: dict[str, Any],
    exported: dict[str, Any],
) -> dict[str, Any]:
    checkout = _prepare_results_checkout(
        source_repo,
        delivery["repository"],
        delivery["branch"],
    )
    result_rel = Path("runs") / str(spec["lane"]) / str(spec["run_id"])
    target = checkout / result_rel
    existing_sha = _existing_manifest_sha(target)
    expected_sha = str(exported["manifest_sha256"])
    if existing_sha is not None:
        if existing_sha != expected_sha:
            raise RunSpecError(
                "RESULT_CONFLICT: run_id already exists with different manifest SHA-256"
            )
        return {
            "status": "ALREADY_DELIVERED",
            "idempotent": True,
            "results_commit": _git(checkout, "rev-parse", "HEAD"),
            "result_path": result_rel.as_posix(),
            "manifest_sha256": expected_sha,
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(exported["package_dir"], target)
    _git(checkout, "add", "--", result_rel.as_posix())
    staged = [
        line
        for line in _git(checkout, "diff", "--cached", "--name-only", "--").splitlines()
        if line
    ]
    expected_prefix = result_rel.as_posix() + "/"
    if not staged or any(not path.startswith(expected_prefix) for path in staged):
        raise RunSpecError("results upload staged files outside the run_id directory")
    _git(checkout, "commit", "-m", f"results {spec['run_id']}")
    commit = _git(checkout, "rev-parse", "HEAD")
    _git(
        checkout,
        "push",
        "-u",
        "origin",
        f"HEAD:refs/heads/{delivery['branch']}",
    )
    return {
        "status": "PASS",
        "idempotent": False,
        "results_commit": commit,
        "result_path": result_rel.as_posix(),
        "manifest_sha256": expected_sha,
    }


def deliver_completed_run(repo: Path, run_id: str) -> dict[str, Any]:
    """Export and upload a completed RunSpec without modifying the code repo."""

    done_path = state_path(repo, DONE_DIR, run_id)
    if not done_path.is_file():
        raise RunSpecError(f"completed RunSpec state is missing: {done_path.relative_to(repo)}")
    spec = validate_runspec(repo, done_path, require_registry=False)
    delivery = validate_delivery_block(spec, str(spec["lane"]))
    if not delivery["enabled"]:
        raise RunSpecError("RunSpec delivery is not enabled")
    artifact = _load_artifact_manifest(repo, run_id)
    exported = export_review_package(repo, spec, artifact, delivery)
    report_path = repo / DELIVERY_ROOT / run_id / "DELIVERY_REPORT.json"
    try:
        uploaded = upload_review_package(repo, spec, delivery, exported)
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "FAIL",
            "run_id": run_id,
            "lane": spec["lane"],
            "repository": delivery["repository"],
            "branch": delivery["branch"],
            "manifest_sha256": exported["manifest_sha256"],
            "error": str(exc),
            "reported_at": now_utc(),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "lane": spec["lane"],
        "experiment_id": spec["experiment_id"],
        "repository": delivery["repository"],
        "branch": delivery["branch"],
        "total_size_bytes": exported["total_size_bytes"],
        "delivered_at": now_utc(),
        **uploaded,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
