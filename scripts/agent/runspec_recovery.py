#!/usr/bin/env python3
"""Bounded, RunSpec-declared recovery for transient server run failures."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from runspec_lib import (
    RunSpecError,
    STATE_DIRNAME,
    read_yaml,
    run_entrypoint,
    safe_relpath,
    validate_existing_script,
    write_yaml,
)

_ALLOWED_KEYS = {
    "enabled",
    "max_attempts",
    "resume_command",
    "retryable_exit_codes",
    "checkpoint_globs",
    "backoff_seconds",
}
_HARD_STOP_PATTERNS = (
    (
        "out_of_memory",
        re.compile(
            r"cuda\s+out\s+of\s+memory|torch\.cuda\.outofmemoryerror|"
            r"\bout\s+of\s+memory\b|\boom[-_ ]?(?:kill|killed|killer)\b|outofmemoryerror",
            re.IGNORECASE,
        ),
    ),
    (
        "nan_or_nonfinite",
        re.compile(r"\bnan\b|non[- ]finite|not\s+finite", re.IGNORECASE),
    ),
)


def validate_recovery_policy(repo: Path, spec: dict[str, Any]) -> dict[str, Any] | None:
    """Validate the optional recovery block and return a normalized policy."""
    raw = spec.get("recovery")
    if raw is None or raw is False:
        return None
    if not isinstance(raw, dict):
        raise RunSpecError("recovery must be a mapping")
    unknown = sorted(set(raw) - _ALLOWED_KEYS)
    if unknown:
        raise RunSpecError(f"unsupported recovery fields: {unknown}")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise RunSpecError("recovery.enabled must be boolean")
    if not enabled:
        return None

    max_attempts = raw.get("max_attempts")
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool):
        raise RunSpecError("recovery.max_attempts must be an integer")
    if max_attempts < 2 or max_attempts > 3:
        raise RunSpecError("recovery.max_attempts must be between 2 and 3 inclusive")

    resume_command = str(raw.get("resume_command") or "").strip()
    if not resume_command:
        raise RunSpecError("recovery.resume_command is required when recovery is enabled")
    validate_existing_script(repo, resume_command, field="recovery.resume_command")

    codes = raw.get("retryable_exit_codes")
    if not isinstance(codes, list) or not codes:
        raise RunSpecError("recovery.retryable_exit_codes must be a non-empty list")
    normalized_codes: list[int] = []
    for value in codes:
        if not isinstance(value, int) or isinstance(value, bool) or value == 0:
            raise RunSpecError("recovery.retryable_exit_codes must contain non-zero integers")
        if value < -255 or value > 255:
            raise RunSpecError("recovery retryable exit codes must be in [-255, 255]")
        if value not in normalized_codes:
            normalized_codes.append(value)

    globs = raw.get("checkpoint_globs")
    if not isinstance(globs, list) or not globs or not all(isinstance(x, str) for x in globs):
        raise RunSpecError("recovery.checkpoint_globs must be a non-empty list of strings")
    normalized_globs: list[str] = []
    for pattern in globs:
        text = pattern.strip()
        if not text:
            raise RunSpecError("recovery.checkpoint_globs cannot contain empty patterns")
        safe_relpath(text.replace("**/", "").replace("*", "x").replace("?", "x") or "x")
        normalized_globs.append(text)

    backoff = raw.get("backoff_seconds", 0)
    if not isinstance(backoff, (int, float)) or isinstance(backoff, bool):
        raise RunSpecError("recovery.backoff_seconds must be numeric")
    backoff = float(backoff)
    if backoff < 0 or backoff > 600:
        raise RunSpecError("recovery.backoff_seconds must be between 0 and 600")

    return {
        "enabled": True,
        "max_attempts": max_attempts,
        "resume_command": resume_command,
        "retryable_exit_codes": normalized_codes,
        "checkpoint_globs": normalized_globs,
        "backoff_seconds": backoff,
    }


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


def _fresh_checkpoints(repo: Path, patterns: list[str], *, not_before: float) -> list[str]:
    found: list[str] = []
    repo_resolved = repo.resolve()
    for pattern in patterns:
        for path in sorted(repo.glob(pattern)):
            if _has_symlink_component(repo, path):
                raise RunSpecError(f"recovery checkpoint matched symlink path: {path}")
            resolved = path.resolve(strict=False)
            if not resolved.is_relative_to(repo_resolved):
                raise RunSpecError(f"recovery checkpoint escapes repository: {path}")
            if not path.is_file():
                continue
            # Tolerate coarse filesystem timestamp resolution while rejecting stale leftovers.
            if path.stat().st_mtime + 2.0 < not_before:
                continue
            rel = path.relative_to(repo).as_posix()
            if rel not in found:
                found.append(rel)
    return found


def _read_status(attempt_dir: Path) -> dict[str, Any]:
    path = attempt_dir / "RUN_STATUS.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_attempt_text(repo: Path, status: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in ("stdout", "stderr"):
        value = status.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = repo / safe_relpath(value)
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if not resolved.is_relative_to(repo.resolve()) or not resolved.is_file():
            continue
        try:
            chunks.append(resolved.read_text(encoding="utf-8", errors="replace")[-200_000:])
        except OSError:
            continue
    return "\n".join(chunks)


def _hard_stop_reason(text: str) -> str | None:
    for name, pattern in _HARD_STOP_PATTERNS:
        if pattern.search(text):
            return name
    return None


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_entrypoint_with_recovery(repo: Path, spec_path: Path) -> dict[str, Any]:
    """Run the declared entrypoint with bounded checkpoint-gated recovery."""
    spec = read_yaml(spec_path)
    policy = validate_recovery_policy(repo, spec)
    if policy is None:
        result = run_entrypoint(repo, spec_path)
        result.update({"attempts": 1, "recovery_used": False, "recovery_report": None})
        return result

    run_id = str(spec.get("run_id") or "")
    log_root = repo / STATE_DIRNAME / "logs" / run_id
    report_path = log_root / "RECOVERY_REPORT.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "policy": policy,
        "status": "running",
        "attempts": [],
    }
    _write_report(report_path, report)

    last_error: Exception | None = None
    for attempt in range(1, int(policy["max_attempts"]) + 1):
        command_kind = "initial" if attempt == 1 else "resume"
        attempt_dir = log_root / f"attempt-{attempt:02d}"
        attempt_spec_path = spec_path
        if attempt > 1:
            attempt_spec = dict(spec)
            attempt_spec["entrypoint"] = dict(spec.get("entrypoint") or {})
            attempt_spec["entrypoint"]["command"] = policy["resume_command"]
            attempt_spec_path = log_root / f"attempt-{attempt:02d}-runspec.yaml"
            write_yaml(attempt_spec_path, attempt_spec)

        started = time.time()
        try:
            result = run_entrypoint(repo, attempt_spec_path, log_dir=attempt_dir)
        except Exception as exc:  # noqa: BLE001 - classify using durable RUN_STATUS.json.
            last_error = exc
            status = _read_status(attempt_dir)
            returncode = status.get("returncode")
            row: dict[str, Any] = {
                "attempt": attempt,
                "command_kind": command_kind,
                "returncode": returncode,
                "outcome": "failed",
                "error": str(exc),
            }
            report["attempts"].append(row)

            retry_reason: str | None = None
            checkpoints: list[str] = []
            if not isinstance(returncode, int) or returncode == 0:
                retry_reason = "not_a_nonzero_entrypoint_failure"
            elif returncode not in policy["retryable_exit_codes"]:
                retry_reason = f"exit_code_not_retryable:{returncode}"
            else:
                hard_stop = _hard_stop_reason(_read_attempt_text(repo, status))
                if hard_stop:
                    retry_reason = f"hard_stop:{hard_stop}"
                else:
                    checkpoints = _fresh_checkpoints(
                        repo,
                        policy["checkpoint_globs"],
                        not_before=started,
                    )
                    if not checkpoints:
                        retry_reason = "no_fresh_checkpoint"
            row["checkpoint_paths"] = checkpoints

            if retry_reason is not None:
                row["retry_decision"] = "stop"
                row["stop_reason"] = retry_reason
                report["status"] = "failed"
                report["final_reason"] = retry_reason
                _write_report(report_path, report)
                raise RunSpecError(
                    f"RunSpec recovery stopped after attempt {attempt}: {retry_reason}; "
                    f"report={report_path.relative_to(repo).as_posix()}"
                ) from exc

            if attempt >= int(policy["max_attempts"]):
                row["retry_decision"] = "stop"
                row["stop_reason"] = "max_attempts_exhausted"
                report["status"] = "failed"
                report["final_reason"] = "max_attempts_exhausted"
                _write_report(report_path, report)
                raise RunSpecError(
                    f"RunSpec recovery exhausted {attempt} attempts; "
                    f"report={report_path.relative_to(repo).as_posix()}"
                ) from exc

            row["retry_decision"] = "retry"
            row["next_attempt"] = attempt + 1
            _write_report(report_path, report)
            if policy["backoff_seconds"]:
                time.sleep(float(policy["backoff_seconds"]))
            continue

        report["attempts"].append(
            {
                "attempt": attempt,
                "command_kind": command_kind,
                "returncode": result.get("returncode"),
                "outcome": "passed",
            }
        )
        report["status"] = "passed"
        report["completed_attempt"] = attempt
        _write_report(report_path, report)
        result.update(
            {
                "attempts": attempt,
                "recovery_used": attempt > 1,
                "recovery_report": report_path.relative_to(repo).as_posix(),
            }
        )
        return result

    raise RunSpecError(f"unreachable recovery state: {last_error}")
