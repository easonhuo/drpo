#!/usr/bin/env python3
"""Freeze and compare one real A0 -> feedback -> B1 repair trajectory."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from validate_update_scope import ScopeError, git_text, run_git

PAIR = "PAIR.json"
CHECKS = ("focused_tests", "full_repository_pytest", "ruff", "required_liveness")
CHECK_STATUS = {"pass", "fail", "not_run", "not_applicable"}
REVIEW_STATUS = {"pass", "fail", "pending"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def commit(repo: Path, value: str) -> str:
    sha = git_text(repo, "rev-parse", "--verify", f"{value}^{{commit}}")
    if len(sha) != 40:
        raise ScopeError(f"could not resolve commit: {value}")
    return sha


def full_sha(value: str, label: str) -> str:
    value = value.lower()
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise ScopeError(f"{label} must be a full Git SHA")
    return value


def require_ancestor(repo: Path, before: str, after: str, label: str) -> None:
    result = run_git(repo, "merge-base", "--is-ancestor", before, after, check=False)
    if result.returncode:
        raise ScopeError(f"{label}: {before} is not an ancestor of {after}")


def final_path(path: str) -> str:
    if " => " not in path:
        return path
    if "{" in path and "}" in path:
        prefix, rest = path.split("{", 1)
        changed, suffix = rest.split("}", 1)
        return prefix + changed.split(" => ", 1)[1] + suffix
    return path.split(" => ", 1)[1]


def is_test(path: str) -> bool:
    return path.endswith(".py") and path.startswith("tests/")


def is_production(path: str) -> bool:
    return path.endswith(".py") and not is_test(path)


def metrics(repo: Path, base: str, head: str) -> dict[str, int]:
    prod_add = prod_del = test_add = test_del = 0
    for line in git_text(repo, "diff", "--numstat", "-M", "-C", base, head, "--").splitlines():
        if not line:
            continue
        added, deleted, raw_path = line.split("\t", 2)
        path = final_path(raw_path)
        add = 0 if added == "-" else int(added)
        delete = 0 if deleted == "-" else int(deleted)
        if is_test(path):
            test_add += add
            test_del += delete
        elif is_production(path):
            prod_add += add
            prod_del += delete

    statuses = [
        line.split("\t")
        for line in git_text(
            repo,
            "diff",
            "--name-status",
            "-M",
            "-C",
            "--find-copies-harder",
            base,
            head,
            "--",
        ).splitlines()
        if line
    ]
    new_prod = sum(
        row[0].startswith(("A", "C")) and is_production(row[-1]) for row in statuses
    )
    return {
        "production_python_additions": prod_add,
        "production_python_deletions": prod_del,
        "production_python_churn": prod_add + prod_del,
        "test_python_additions": test_add,
        "test_python_deletions": test_del,
        "test_python_churn": test_add + test_del,
        "changed_files": len(statuses),
        "new_production_python_files": int(new_prod),
    }


def load_pair(record_dir: Path) -> dict[str, object]:
    path = record_dir / PAIR
    if not path.is_file():
        raise ScopeError(f"missing paired-repair record: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ScopeError("unsupported paired-repair record")
    return value


def validation(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ScopeError("validation must be a schema_version=1 object")
    for phase in ("a0", "b1"):
        block = value.get(phase)
        if not isinstance(block, dict):
            raise ScopeError(f"validation lacks {phase}")
        if any(block.get(field) not in CHECK_STATUS for field in CHECKS):
            raise ScopeError(f"validation has an invalid {phase} check status")
        if block.get("scientific_scope_unchanged") is not True:
            raise ScopeError(f"{phase}.scientific_scope_unchanged must be true")
        if block.get("reviewer_correctness") not in REVIEW_STATUS:
            raise ScopeError(f"{phase}.reviewer_correctness has invalid status")
    return value


def verdict(checks: dict[str, object], a0: dict[str, int], b1: dict[str, int]) -> str:
    a0_checks = checks["a0"]
    b1_checks = checks["b1"]
    assert isinstance(a0_checks, dict) and isinstance(b1_checks, dict)
    eligible = b1_checks["reviewer_correctness"] == "pass"
    eligible &= all(b1_checks[field] != "fail" for field in CHECKS)
    eligible &= all(
        a0_checks[field] != "pass" or b1_checks[field] == "pass" for field in CHECKS
    )
    if not eligible:
        return "B1_INELIGIBLE_RETAIN_A0_OR_REPAIR"
    smaller = (
        b1["production_python_churn"] < a0["production_python_churn"]
        or b1["new_production_python_files"] < a0["new_production_python_files"]
    )
    return "B1_ELIGIBLE_AND_SMALLER" if smaller else "B1_ELIGIBLE_NO_SIZE_GAIN"


def freeze(args: argparse.Namespace) -> int:
    repo = args.repo_root.resolve()
    record = args.record_dir.resolve()
    if record.exists() and any(record.iterdir()):
        raise ScopeError(f"record directory must be new or empty: {record}")
    base, a0 = commit(repo, args.base), commit(repo, args.a0)
    require_ancestor(repo, base, a0, "A0 lineage")
    if base == a0:
        raise ScopeError("A0 must contain a real change")
    value: dict[str, object] = {
        "schema_version": 1,
        "status": "a0_frozen_waiting_feedback",
        "claim_id": args.claim,
        "worker_label": args.worker,
        "gate_snapshot_sha": full_sha(args.gate_snapshot, "gate snapshot"),
        "base_sha": base,
        "a0_sha": a0,
        "a0_metrics": metrics(repo, base, a0),
        "frozen_at_utc": now(),
    }
    write_json(record / PAIR, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def comparison_markdown(value: dict[str, object]) -> str:
    a0 = value["a0_metrics"]
    b1 = value["b1_metrics"]
    change = value["comparison"]
    assert isinstance(a0, dict) and isinstance(b1, dict) and isinstance(change, dict)
    rows = [
        (
            "Production Python churn",
            a0["production_python_churn"],
            b1["production_python_churn"],
            change["production_python_churn"],
        ),
        (
            "New production Python files",
            a0["new_production_python_files"],
            b1["new_production_python_files"],
            change["new_production_python_files"],
        ),
        (
            "Test Python churn",
            a0["test_python_churn"],
            b1["test_python_churn"],
            change["test_python_churn"],
        ),
        ("Changed files", a0["changed_files"], b1["changed_files"], change["changed_files"]),
    ]
    table = "\n".join(
        f"| {name} | {before} | {after} | {delta} |"
        for name, before, after, delta in rows
    )
    return f"""# Paired Repair Comparison

- claim: `{value['claim_id']}`
- base: `{value['base_sha']}`
- A0: `{value['a0_sha']}`
- B1: `{value['b1_sha']}`
- gate: `{value['gate_snapshot_sha']}`
- verdict: **{value['evidence_verdict']}**

| Metric | A0 | B1 | Change |
|---|---:|---:|---:|
{table}

Correctness remains primary. This report does not authorize merge and is not a causal
two-worker A/B result.
"""


def close(args: argparse.Namespace) -> int:
    repo, record = args.repo_root.resolve(), args.record_dir.resolve()
    value = load_pair(record)
    if value.get("status") != "a0_frozen_waiting_feedback":
        raise ScopeError("record is not waiting for B1")
    if value.get("worker_label") != args.worker:
        raise ScopeError("B1 worker label must match A0")
    feedback = args.feedback_file.read_text(encoding="utf-8").strip()
    if len(feedback) < 40:
        raise ScopeError("gate feedback is too short")
    checks = validation(args.validation_file)
    b1 = commit(repo, args.b1)
    a0 = str(value["a0_sha"])
    base = str(value["base_sha"])
    require_ancestor(repo, a0, b1, "B1 lineage")
    if a0 == b1:
        raise ScopeError("B1 must differ from A0")
    a0_metrics = value["a0_metrics"]
    assert isinstance(a0_metrics, dict)
    b1_metrics = metrics(repo, base, b1)
    fields = (
        "production_python_churn",
        "new_production_python_files",
        "test_python_churn",
        "changed_files",
    )
    value.update(
        {
            "status": "closed_waiting_human_merge_decision",
            "b1_sha": b1,
            "b1_metrics": b1_metrics,
            "repair_metrics": metrics(repo, a0, b1),
            "comparison": {field: b1_metrics[field] - int(a0_metrics[field]) for field in fields},
            "gate_feedback": {
                "sha256": hashlib.sha256(feedback.encode()).hexdigest(),
                "source": args.feedback_source,
            },
            "validation": checks,
            "evidence_verdict": verdict(checks, a0_metrics, b1_metrics),
            "closed_at_utc": now(),
        }
    )
    (record / "GATE_FEEDBACK.md").write_text(feedback + "\n", encoding="utf-8")
    write_json(record / "VALIDATION.json", checks)
    write_json(record / PAIR, value)
    (record / "COMPARISON.md").write_text(comparison_markdown(value), encoding="utf-8")
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    a0 = commands.add_parser("freeze-a0")
    a0.add_argument("--repo-root", type=Path, default=Path("."))
    a0.add_argument("--base", required=True)
    a0.add_argument("--a0", required=True)
    a0.add_argument("--claim", required=True)
    a0.add_argument("--worker", required=True)
    a0.add_argument("--gate-snapshot", required=True)
    a0.add_argument("--record-dir", type=Path, required=True)
    a0.set_defaults(function=freeze)
    b1 = commands.add_parser("close-b1")
    b1.add_argument("--repo-root", type=Path, default=Path("."))
    b1.add_argument("--record-dir", type=Path, required=True)
    b1.add_argument("--b1", required=True)
    b1.add_argument("--worker", required=True)
    b1.add_argument("--feedback-file", type=Path, required=True)
    b1.add_argument("--feedback-source", required=True)
    b1.add_argument("--validation-file", type=Path, required=True)
    b1.set_defaults(function=close)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return int(args.function(args))
    except (OSError, ValueError, json.JSONDecodeError, ScopeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
