#!/usr/bin/env python3
"""Fail closed when a PR exceeds DRPO's initial Python change budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

LIMIT = 100
BUDGET_CONTEXT = "drpo/code-change-budget"
APPROVAL_PREFIX = "drpo/human-code-approval/"
START = "<!-- DRPO_CODE_CHANGE_JUSTIFICATION_START -->"
END = "<!-- DRPO_CODE_CHANGE_JUSTIFICATION_END -->"


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def analyze(repo: Path, base: str, head: str) -> dict[str, Any]:
    git(repo, "cat-file", "-e", f"{base}^{{commit}}")
    git(repo, "cat-file", "-e", f"{head}^{{commit}}")
    revision = f"{base}...{head}"

    additions = deletions = 0
    for line in git(repo, "diff", "--numstat", revision, "--", "*.py").splitlines():
        if not line:
            continue
        added, deleted, _path = line.split("\t", 2)
        if "-" in (added, deleted):
            raise RuntimeError(f"binary Python file is forbidden: {line}")
        additions += int(added)
        deletions += int(deleted)

    changed: set[str] = set()
    added_files: set[str] = set()
    deleted_files: set[str] = set()
    renamed: list[list[str]] = []
    copied: list[list[str]] = []
    output = git(
        repo,
        "diff",
        "--name-status",
        "--find-renames=50%",
        "--find-copies=50%",
        revision,
        "--",
        "*.py",
    )
    for line in output.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        kind = parts[0][0]
        if kind in {"R", "C"}:
            old, new = parts[1], parts[2]
            changed.update((old, new))
            (renamed if kind == "R" else copied).append([old, new])
        else:
            path = parts[1]
            changed.add(path)
            if kind == "A":
                added_files.add(path)
            elif kind == "D":
                deleted_files.add(path)

    churn = additions + deletions
    structural = bool(added_files or deleted_files or renamed or copied)
    reasons: list[str] = []
    if churn > LIMIT:
        reasons.append(f"python_churn_{churn}_exceeds_{LIMIT}")
    if added_files:
        reasons.append("python_file_added")
    if deleted_files:
        reasons.append("python_file_deleted")
    if renamed:
        reasons.append("python_file_renamed")
    if copied:
        reasons.append("python_file_copied")
    return {
        "additions": additions,
        "deletions": deletions,
        "churn": churn,
        "changed_files": sorted(changed),
        "added_files": sorted(added_files),
        "deleted_files": sorted(deleted_files),
        "renamed_files": sorted(renamed),
        "copied_files": sorted(copied),
        "requires_human_approval": churn > LIMIT or structural,
        "approval_reasons": reasons,
    }


def extract_justification(body: str) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    start = body.find(START)
    end = body.find(END)
    if start < 0 and end < 0:
        return None, None, ["structured justification block is missing"]
    if start < 0 or end <= start:
        return None, None, ["justification markers are malformed"]
    raw = body[start + len(START) : end].strip()
    digest = hashlib.sha256(raw.encode()).hexdigest() if raw else None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        return None, digest, [f"justification is not valid JSON: {error}"]
    if not isinstance(payload, dict):
        return None, digest, ["justification root must be an object"]
    return payload, digest, []


def base_python_paths(repo: Path, base: str) -> set[str]:
    return {
        path
        for path in git(repo, "ls-tree", "-r", "--name-only", base).splitlines()
        if path.endswith(".py")
    }


def require_text(
    payload: Mapping[str, Any], key: str, minimum: int, errors: list[str]
) -> None:
    value = payload.get(key)
    if not isinstance(value, str) or len(value.strip()) < minimum:
        errors.append(f"{key} must contain at least {minimum} characters")


def validate_justification(
    repo: Path, base: str, diff: Mapping[str, Any], payload: dict[str, Any] | None
) -> list[str]:
    if payload is None:
        return []
    errors: list[str] = []
    require_text(payload, "purpose", 20, errors)
    require_text(payload, "reuse_attempt", 80, errors)
    require_text(payload, "why_existing_code_is_insufficient", 80, errors)
    if int(diff["churn"]) > LIMIT:
        require_text(payload, "why_change_cannot_be_under_100_lines", 80, errors)

    existing = base_python_paths(repo, base)
    nearest = payload.get("nearest_existing_modules")
    if not isinstance(nearest, list) or not nearest:
        errors.append("nearest_existing_modules must be a non-empty list")
    else:
        for path in nearest:
            if not isinstance(path, str) or path not in existing:
                errors.append(f"nearest existing module not found at base: {path!r}")

    responsibilities = payload.get("file_responsibilities")
    expected = set(diff["changed_files"])
    if not isinstance(responsibilities, dict):
        errors.append("file_responsibilities must be an object")
    else:
        actual = set(responsibilities)
        if expected != actual:
            errors.append(
                "file_responsibilities must exactly cover changed Python files; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        for path, reason in responsibilities.items():
            if not isinstance(reason, str) or len(reason.strip()) < 30:
                errors.append(f"file responsibility is too short: {path}")

    new_paths = set(diff["added_files"])
    new_paths.update(pair[1] for pair in diff["copied_files"])
    new_records = payload.get("new_python_files", {})
    if not isinstance(new_records, dict) or set(new_records) != new_paths:
        errors.append("new_python_files must exactly cover added/copied Python files")
    elif isinstance(new_records, dict):
        for path, record in new_records.items():
            if not isinstance(record, dict):
                errors.append(f"new_python_files[{path!r}] must be an object")
                continue
            closest = record.get("closest_existing_module")
            if not isinstance(closest, str) or closest not in existing:
                errors.append(f"closest_existing_module does not exist at base: {path}")
            require_text(record, "why_new_module_required", 80, errors)
            require_text(record, "intended_reuse_by", 30, errors)

    tests = payload.get("tests")
    if not isinstance(tests, list) or not tests or any(
        not isinstance(item, str) or not item.strip() for item in tests
    ):
        errors.append("tests must be a non-empty list")
    return errors


def approval_context(digest: str | None) -> str | None:
    return f"{APPROVAL_PREFIX}{digest[:16]}" if digest else None


def has_approval(statuses: Mapping[str, Any], context: str | None) -> bool:
    if context is None:
        return False
    return any(
        isinstance(item, Mapping)
        and item.get("context") == context
        and item.get("state") == "success"
        for item in statuses.get("statuses", [])
    )


def evaluate(
    repo: Path,
    base: str,
    head: str,
    body: str,
    statuses: Mapping[str, Any],
    *,
    human_approved: bool = False,
) -> dict[str, Any]:
    diff = analyze(repo, base, head)
    report: dict[str, Any] = {
        "schema_version": 1,
        "base_sha": base,
        "head_sha": head,
        "python_diff": diff,
    }
    if not diff["requires_human_approval"]:
        report.update(decision="PASS_AUTOMATIC")
        return report

    payload, digest, parse_errors = extract_justification(body)
    errors = parse_errors + validate_justification(repo, base, diff, payload)
    context = approval_context(digest)
    approved = human_approved or has_approval(statuses, context)
    if errors:
        decision = "REJECT_JUSTIFICATION"
    elif approved:
        decision = "PASS_HUMAN_APPROVED"
    else:
        decision = "NEEDS_HUMAN_APPROVAL"
    report.update(
        decision=decision,
        justification_sha256=digest,
        expected_human_approval_context=context,
        human_approval_present=approved,
        justification_errors=errors,
    )
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo-root", type=Path, default=Path("."))
    value.add_argument("--base", required=True)
    value.add_argument("--head", required=True)
    value.add_argument("--pr-body-file", type=Path, required=True)
    value.add_argument("--status-json", type=Path)
    value.add_argument("--report", type=Path, required=True)
    value.add_argument("--human-approved", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        statuses = (
            json.loads(args.status_json.read_text()) if args.status_json else {}
        )
        report = evaluate(
            args.repo_root.resolve(),
            args.base,
            args.head,
            args.pr_body_file.read_text(),
            statuses,
            human_approved=args.human_approved,
        )
    except Exception as error:  # fail closed at the workflow boundary
        report = {
            "schema_version": 1,
            "decision": "ERROR",
            "base_sha": args.base,
            "head_sha": args.head,
            "error": str(error),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if str(report["decision"]).startswith("PASS_") else 3


if __name__ == "__main__":
    raise SystemExit(main())
