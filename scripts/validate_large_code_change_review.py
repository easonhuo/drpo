#!/usr/bin/env python3
"""Verify evidence before a large/structural Python PR reaches a human."""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

from validate_update_scope import git_text, run_git

START = "<!-- DRPO_LARGE_CHANGE_REVIEW_START -->"
END = "<!-- DRPO_LARGE_CHANGE_REVIEW_END -->"


def exists(repo: Path, commit: str, path: str) -> bool:
    return run_git(repo, "cat-file", "-e", f"{commit}:{path}", check=False).returncode == 0


def blob(repo: Path, commit: str, path: str) -> str:
    return git_text(repo, "show", f"{commit}:{path}")


def named_nodes(source: str, *, tests_only: bool = False) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if tests_only and not node.name.startswith("test_"):
            continue
        if not tests_only and node.name.startswith("_"):
            continue
        result[node.name] = ast.dump(node, include_attributes=False)
    return result


def validate(repo: Path, base: str, head: str, body: str) -> list[str]:
    errors: list[str] = []
    if body.count(START) != 1 or body.count(END) != 1:
        return ["PR body needs exactly one DRPO_LARGE_CHANGE_REVIEW block"]
    data = json.loads(body.split(START, 1)[1].split(END, 1)[0].strip())
    if not isinstance(data, dict):
        return ["large-change review block must be one JSON object"]

    rows = [
        tuple(line.split("\t"))
        for line in git_text(
            repo, "diff", "--name-status", "-M", "-C", "--find-copies-harder",
            base, head, "--", "*.py",
        ).splitlines()
        if line.strip()
    ]
    changed = {row[-1] for row in rows}
    added = {row[-1] for row in rows if row[0].startswith(("A", "C"))}

    for key in ("why_over_100", "smaller_alternative", "why_smaller_fails"):
        if not isinstance(data.get(key), str) or len(data[key].strip()) < 60:
            errors.append(f"{key} must be a specific explanation of at least 60 characters")

    responsibilities = data.get("changed_python_files")
    if not isinstance(responsibilities, dict) or set(responsibilities) != changed:
        errors.append("changed_python_files must exactly cover actual Python paths")
    elif any(not isinstance(v, str) or len(v.strip()) < 20 for v in responsibilities.values()):
        errors.append("every changed Python path needs a concrete responsibility")

    nearest = data.get("nearest_existing")
    if not isinstance(nearest, list) or not nearest:
        errors.append("nearest_existing must list inspected base Python modules")
    else:
        for path in nearest:
            if not isinstance(path, str) or not path.endswith(".py") or not exists(repo, base, path):
                errors.append(f"invalid nearest_existing path: {path!r}")

    new_files = data.get("new_python_files", {})
    if not isinstance(new_files, dict) or set(new_files) != added:
        errors.append("new_python_files must exactly cover added/copied Python paths")
    else:
        for path, item in new_files.items():
            if not isinstance(item, dict):
                errors.append(f"{path}: new-file explanation must be an object")
                continue
            closest, reason = item.get("closest_existing"), item.get("why_not_extend")
            if not isinstance(closest, str) or not exists(repo, base, closest):
                errors.append(f"{path}: closest_existing is not present at base")
            if not isinstance(reason, str) or len(reason.strip()) < 60:
                errors.append(f"{path}: why_not_extend is not specific enough")

    tests: set[str] = set()
    requirements = data.get("core_requirements")
    if not isinstance(requirements, list) or not requirements:
        errors.append("core_requirements must enumerate required functionality")
    else:
        for item in requirements:
            if not isinstance(item, dict) or not item.get("id") or not item.get("statement"):
                errors.append("each core requirement needs id and statement")
                continue
            if not isinstance(item.get("tests"), list) or not item["tests"]:
                errors.append(f"core requirement {item.get('id')} has no test evidence")
                continue
            for nodeid in item["tests"]:
                path, _, test_name = str(nodeid).partition("::")
                tests.add(path)
                if not path.startswith("tests/") or not exists(repo, head, path):
                    errors.append(f"missing core-function test at head: {nodeid}")
                elif not test_name or test_name not in named_nodes(blob(repo, head, path), tests_only=True):
                    errors.append(f"declared core-function test does not exist: {nodeid}")
    if tests and not (tests & changed):
        errors.append("at least one declared core-function test file must change")

    reuse = data.get("reuse_evidence")
    if not isinstance(reuse, list) or not reuse:
        errors.append("reuse_evidence must identify existing symbols actually reused")
    else:
        for item in reuse:
            if not isinstance(item, dict):
                errors.append("reuse_evidence entries must be objects")
                continue
            path, name, used_by = item.get("path"), item.get("symbol"), item.get("used_by")
            if not all(isinstance(v, str) and v for v in (path, name, used_by)):
                errors.append("reuse_evidence requires path, symbol, and used_by")
            elif not exists(repo, base, path) or name not in blob(repo, base, path):
                errors.append(f"reused symbol is absent at base: {path}:{name}")
            elif used_by not in changed or not exists(repo, head, used_by) or name not in blob(repo, head, used_by):
                errors.append(f"reuse is not visible in changed file: {used_by}:{name}")

    for row in rows:
        status = row[0][0]
        if status == "C":
            errors.append(f"copy detected: {row[1]} -> {row[2]}; reuse the source")
        elif status == "D":
            errors.append(f"Python deletion needs a separate user-approved removal plan: {row[1]}")
        elif status in {"M", "R"}:
            old, new = (row[1], row[2]) if status == "R" else (row[1], row[1])
            before, after = blob(repo, base, old), blob(repo, head, new)
            removed = set(named_nodes(before)) - set(named_nodes(after))
            if removed:
                errors.append(f"public symbols removed from {old}: {sorted(removed)}")
            if old.startswith("tests/"):
                old_tests, new_tests = named_nodes(before, tests_only=True), named_nodes(after, tests_only=True)
                rewritten = sorted(name for name, tree in old_tests.items() if new_tests.get(name) != tree)
                if rewritten:
                    errors.append(
                        f"existing regression tests rewritten in {old}: {rewritten}; add tests instead"
                    )
    return errors


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--base", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--pr-body-file", type=Path, required=True)
    a = p.parse_args()
    try:
        errors = validate(a.repo_root.resolve(), a.base, a.head, a.pr_body_file.read_text())
    except (OSError, ValueError, json.JSONDecodeError, SyntaxError) as exc:
        errors = [str(exc)]
    print(json.dumps({"status": "FAIL" if errors else "PASS", "errors": errors}, indent=2))
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
