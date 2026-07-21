#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "usage: $0 REPO_ROOT WORK_DIR MAX_WORKERS APPROVAL_FILE CONTRACT RUN_SPEC GRID" >&2
  exit 2
fi

python3 - "$@" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


class ApprovalError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ApprovalError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def resolved(path_text: str, repo: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = repo / path
    return path.resolve()


def canonical_relative(path: Path, repo: Path) -> str:
    try:
        return path.relative_to(repo).as_posix()
    except ValueError:
        return str(path)


def require_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be a regular non-symlink file: {path}")


def require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} must be a non-empty string")
    return value.strip()


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApprovalError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        fail(f"{label} root must be an object: {path}")
    return value


def write_or_verify_identity(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_json_object(path, "worker-cap identity")
        if existing != payload:
            fail(
                "worker-cap policy changed inside an existing work directory; "
                "use a new user approval and a new run/work directory"
            )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    repo = Path(sys.argv[1]).expanduser().resolve()
    work_dir = resolved(sys.argv[2], repo)
    max_workers_raw = sys.argv[3].strip()
    approval_raw = sys.argv[4].strip()
    contract = resolved(sys.argv[5], repo)
    run_spec = resolved(sys.argv[6], repo)
    grid = resolved(sys.argv[7], repo)

    if not (repo / ".git").exists():
        fail(f"repository root is not a Git checkout: {repo}")
    if git(repo, "status", "--porcelain").stdout.strip():
        fail("worker-cap validation requires a clean checkout")
    for path, label in (
        (contract, "contract"),
        (run_spec, "run spec"),
        (grid, "grid"),
    ):
        require_file(path, label)

    grid_payload = load_json_object(grid, "grid")
    experiment_id = require_nonempty_string(
        grid_payload.get("experiment_id"), "grid.experiment_id"
    )
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    affinity_cpu_ids = sorted(int(value) for value in os.sched_getaffinity(0))
    common_scope = {
        "experiment_id": experiment_id,
        "work_dir": str(work_dir),
        "affinity_cpu_ids": affinity_cpu_ids,
        "contract_sha256": sha256_file(contract),
        "run_spec_sha256": sha256_file(run_spec),
        "grid_sha256": sha256_file(grid),
    }

    if max_workers_raw:
        try:
            max_workers = int(max_workers_raw)
        except ValueError as exc:
            raise ApprovalError("MAX_WORKERS must be an integer") from exc
        if max_workers < 1:
            fail("MAX_WORKERS must be positive")
        if not approval_raw:
            fail(
                "MAX_WORKERS is user-governed and requires an explicit approval file; "
                "AI agents may recommend a value but may not set or change it"
            )

        approval = resolved(approval_raw, repo)
        require_file(approval, "worker-cap approval")
        authorization_root = (repo / "docs" / "runtime_worker_cap_authorizations").resolve()
        try:
            approval_relative = approval.relative_to(authorization_root)
        except ValueError as exc:
            raise ApprovalError(
                "worker-cap approval must live under "
                "docs/runtime_worker_cap_authorizations/"
            ) from exc
        repo_relative = approval.relative_to(repo).as_posix()
        if approval_relative.name == "README.md":
            fail("README.md is policy documentation, not an approval record")
        if git(repo, "ls-files", "--error-unmatch", "--", repo_relative, check=False).returncode:
            fail("worker-cap approval must be tracked by Git")
        if git(repo, "status", "--porcelain", "--", repo_relative).stdout.strip():
            fail("worker-cap approval must be clean and committed")

        record = load_json_object(approval, "worker-cap approval")
        if record.get("schema_version") != 1:
            fail("worker-cap approval schema_version must equal 1")
        if record.get("status") != "approved":
            fail("worker-cap approval status must equal approved")
        if record.get("approved_by") != "repository_owner":
            fail("worker-cap approval approved_by must equal repository_owner")
        authorization_id = require_nonempty_string(
            record.get("authorization_id"), "authorization_id"
        )
        approval_reference = require_nonempty_string(
            record.get("approval_reference"), "approval_reference"
        )
        reason = require_nonempty_string(record.get("reason"), "reason")
        scope = record.get("scope")
        if not isinstance(scope, dict):
            fail("worker-cap approval scope must be an object")

        expected_scope = {
            **common_scope,
            "max_workers": max_workers,
        }
        for key, expected in expected_scope.items():
            if scope.get(key) != expected:
                fail(
                    f"worker-cap approval scope mismatch for {key}: "
                    f"expected={expected!r} actual={scope.get(key)!r}"
                )

        approved_code_commit = require_nonempty_string(
            scope.get("approved_code_commit"), "scope.approved_code_commit"
        )
        if len(approved_code_commit) != 40:
            fail("scope.approved_code_commit must be a full 40-character commit SHA")
        if git(
            repo,
            "merge-base",
            "--is-ancestor",
            approved_code_commit,
            head,
            check=False,
        ).returncode:
            fail("approved_code_commit must be an ancestor of the launch commit")

        protected_paths = [
            "scripts/validate_user_approved_worker_cap.sh",
            "scripts/run_e7_squared_exp_night_one_click.sh",
            "scripts/run_e7_squared_exp_night_resume_one_click.sh",
            "scripts/run_e7_squared_exp_night_liveness_one_click.sh",
            "scripts/run_e7_squared_exp_night_auto.py",
            "src/drpo/e7_squared_exp_night_runtime_autotune.py",
            "src/drpo/e7_squared_exp_night.py",
        ]
        grid_relative = canonical_relative(grid, repo)
        if not grid_relative.startswith("/"):
            protected_paths.append(grid_relative)
        if git(
            repo,
            "diff",
            "--quiet",
            f"{approved_code_commit}..{head}",
            "--",
            *protected_paths,
            check=False,
        ).returncode:
            fail(
                "worker-cap protected runtime paths changed after approved_code_commit; "
                "a new user approval is required"
            )

        payload = {
            "schema_version": 1,
            "mode": "user_approved_hard_cap",
            "max_workers": max_workers,
            "authorization": {
                "authorization_id": authorization_id,
                "approval_reference": approval_reference,
                "approved_by": "repository_owner",
                "reason": reason,
                "approval_path": repo_relative,
                "approval_sha256": sha256_file(approval),
                "approved_code_commit": approved_code_commit,
            },
            "launch_commit": head,
            "scope": common_scope,
        }
    else:
        if approval_raw:
            fail("an approval file may not be supplied when MAX_WORKERS is unset")
        payload = {
            "schema_version": 1,
            "mode": "unset_autotune_controls_concurrency",
            "max_workers": None,
            "authorization": None,
            "launch_commit": head,
            "scope": common_scope,
        }

    identity_path = work_dir / "USER_APPROVED_WORKER_CAP.json"
    write_or_verify_identity(identity_path, payload)
    print(json.dumps({"path": str(identity_path), **payload}, sort_keys=True))
    return 0


try:
    raise SystemExit(main())
except ApprovalError as exc:
    print(f"worker-cap approval: FAIL: {exc}", file=sys.stderr)
    raise SystemExit(2)
PY
