#!/usr/bin/env python3
"""Formal delivery gate, V1 size policy, and graceful oversize reporting."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from runspec_lib import (
    ARTIFACT_DIRNAME,
    DONE_DIR,
    RunSpecError,
    now_utc,
    read_yaml,
    state_path,
)

MAX_REVIEW_PACKAGE_MB = 30
MAX_REVIEW_FILE_MB = 10
FORMAL_RESULTS_REPOSITORY = "easonhuo/drpo-results"
RESULT_TOO_LARGE = "RESULT_TOO_LARGE"
_TOO_LARGE_MARKERS = (
    "delivery source file exceeds max_file_size_mb",
    "generated delivery file exceeds max_file_size_mb",
    "delivery review package is too large",
)


def formal_delivery_required(spec: dict[str, Any]) -> bool:
    """Return whether the RunSpec must use automatic durable delivery.

    ``policy.formal_evidence_allowed: false`` is the only local-only opt-out.
    Missing declarations fail closed as delivery-required so an omitted field
    cannot silently downgrade a potentially formal run.
    """

    policy = spec.get("policy") or {}
    if not isinstance(policy, dict):
        raise RunSpecError("policy must be a mapping")
    declared = policy.get("formal_evidence_allowed")
    if declared is not None and not isinstance(declared, bool):
        raise RunSpecError("policy.formal_evidence_allowed must be a boolean")
    return declared is not False


def validate_formal_delivery_policy(spec: dict[str, Any]) -> bool:
    """Fail closed unless a delivery-required RunSpec uses the canonical channel."""

    required = formal_delivery_required(spec)
    if not required:
        return False

    raw = spec.get("delivery") or {}
    if not isinstance(raw, dict):
        raise RunSpecError("delivery must be a mapping")
    if raw.get("enabled") is not True or raw.get("auto") is not True:
        raise RunSpecError(
            "formal RunSpec requires delivery.enabled=true and delivery.auto=true; "
            "set policy.formal_evidence_allowed=false only for non-formal local runs"
        )
    if raw.get("mode") != "results_repo":
        raise RunSpecError("formal RunSpec requires delivery.mode=results_repo")
    if raw.get("repository") != FORMAL_RESULTS_REPOSITORY:
        raise RunSpecError(
            "formal RunSpec requires delivery.repository="
            f"{FORMAL_RESULTS_REPOSITORY}"
        )
    lane = str(spec.get("lane") or "")
    if not lane:
        raise RunSpecError("formal RunSpec requires a lane before delivery validation")
    expected_branch = f"ingest/{lane}"
    if raw.get("branch") != expected_branch:
        raise RunSpecError(
            f"formal RunSpec requires delivery.branch={expected_branch}"
        )
    profile = raw.get("export_profile", "manifest_text_v1")
    if profile != "manifest_text_v1":
        raise RunSpecError(
            "formal RunSpec requires delivery.export_profile=manifest_text_v1"
        )
    return True


def validate_simple_size_policy(spec: dict[str, Any]) -> dict[str, int] | None:
    """Enforce formal delivery first, then the intentionally small V1 limits."""

    validate_formal_delivery_policy(spec)
    raw = spec.get("delivery") or {}
    if not isinstance(raw, dict):
        raise RunSpecError("delivery must be a mapping")
    if raw.get("enabled") is not True:
        return None
    try:
        total_mb = int(raw.get("max_total_size_mb", MAX_REVIEW_PACKAGE_MB))
        file_mb = int(raw.get("max_file_size_mb", MAX_REVIEW_FILE_MB))
    except (TypeError, ValueError) as exc:
        raise RunSpecError("delivery size limits must be integers") from exc
    if not 1 <= total_mb <= MAX_REVIEW_PACKAGE_MB:
        raise RunSpecError(
            "delivery.max_total_size_mb must be between 1 and "
            f"{MAX_REVIEW_PACKAGE_MB} for results-repository V1"
        )
    if not 1 <= file_mb <= MAX_REVIEW_FILE_MB:
        raise RunSpecError(
            "delivery.max_file_size_mb must be between 1 and "
            f"{MAX_REVIEW_FILE_MB} for results-repository V1"
        )
    if file_mb > total_mb:
        raise RunSpecError(
            "delivery.max_file_size_mb may not exceed delivery.max_total_size_mb"
        )
    return {"max_total_size_mb": total_mb, "max_file_size_mb": file_mb}


def is_result_too_large_error(exc: Exception) -> bool:
    text = str(exc)
    return any(marker in text for marker in _TOO_LARGE_MARKERS)


def record_result_too_large(
    repo: Path,
    spec: dict[str, Any],
    artifact: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    """Record a normal delivery downgrade without changing the completed run."""

    limits = validate_simple_size_policy(spec) or {
        "max_total_size_mb": MAX_REVIEW_PACKAGE_MB,
        "max_file_size_mb": MAX_REVIEW_FILE_MB,
    }
    run_id = str(spec["run_id"])
    delivery_root = repo / ".runspec_state" / "delivery" / run_id
    shutil.rmtree(delivery_root / "review_package", ignore_errors=True)
    delivery = spec.get("delivery") or {}
    report = {
        "schema_version": 1,
        "status": RESULT_TOO_LARGE,
        "run_id": run_id,
        "lane": spec.get("lane"),
        "experiment_id": spec.get("experiment_id"),
        "repository": delivery.get("repository"),
        "branch": delivery.get("branch"),
        "upload_attempted": False,
        "experiment_state": "done",
        "artifact_zip": artifact.get("zip_path"),
        "artifact_zip_sha256": artifact.get("zip_sha256"),
        "max_total_size_mb": limits["max_total_size_mb"],
        "max_file_size_mb": limits["max_file_size_mb"],
        "reason": str(exc),
        "reported_at": now_utc(),
    }
    delivery_root.mkdir(parents=True, exist_ok=True)
    (delivery_root / "DELIVERY_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def record_completed_result_too_large(
    repo: Path,
    run_id: str,
    exc: Exception,
) -> dict[str, Any]:
    done_path = state_path(repo, DONE_DIR, run_id)
    if not done_path.is_file():
        raise RunSpecError(f"completed RunSpec state is missing: {done_path.relative_to(repo)}")
    spec = read_yaml(done_path)
    artifact_path = repo / ARTIFACT_DIRNAME / f"{run_id}_manifest.json"
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except FileNotFoundError as missing:
        raise RunSpecError(
            f"artifact manifest is missing: {artifact_path.relative_to(repo)}"
        ) from missing
    except json.JSONDecodeError as invalid:
        raise RunSpecError(f"artifact manifest is invalid JSON: {invalid}") from invalid
    if not isinstance(artifact, dict):
        raise RunSpecError("artifact manifest must contain a JSON object")
    return record_result_too_large(repo, spec, artifact, exc)
