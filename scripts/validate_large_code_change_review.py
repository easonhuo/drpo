#!/usr/bin/env python3
"""Verify evidence before a large/structural Python PR reaches a human."""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

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


def test_nodes(source: str) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def defined_symbols(source: str) -> set[str]:
    result: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            result.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Assign):
            result.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            result.add(node.target.id)
    return result


def referenced_symbols(source: str) -> set[str]:
    tree = ast.parse(source)
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }


def _anchors(node: ast.AST) -> set[str]:
    ignored = {
        "assert",
        "bool",
        "dict",
        "float",
        "int",
        "len",
        "list",
        "range",
        "set",
        "str",
        "tuple",
        "pytest",
        "True",
        "False",
        "None",
    }
    return (
        {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
        | {item.attr for item in ast.walk(node) if isinstance(item, ast.Attribute)}
    ) - ignored


def _meaningful_checks(node: ast.AST) -> tuple[int, bool]:
    checks = 0
    trivial = False
    for item in ast.walk(node):
        if isinstance(item, ast.Assert):
            dynamic = any(
                isinstance(part, (ast.Name, ast.Attribute, ast.Call, ast.Subscript))
                for part in ast.walk(item.test)
            )
            if dynamic:
                checks += 1
            else:
                trivial = True
        elif isinstance(item, ast.With):
            for context in item.items:
                call = context.context_expr
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "raises"
                    and call.args
                ):
                    checks += 1
        elif isinstance(item, ast.Call):
            name = (
                item.func.attr
                if isinstance(item.func, ast.Attribute)
                else item.func.id
                if isinstance(item.func, ast.Name)
                else ""
            )
            if name.startswith("assert") or any(
                keyword.arg == "check"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in item.keywords
            ):
                checks += 1
    return checks, trivial


def _head_test(
    repo: Path, head: str, nodeid: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    path, sep, name = nodeid.partition("::")
    if not sep or not path.startswith("tests/") or not exists(repo, head, path):
        return None
    return test_nodes(blob(repo, head, path)).get(name)


def _replacement_changed(repo: Path, base: str, head: str, nodeid: str) -> bool:
    path, _, name = nodeid.partition("::")
    head_node = _head_test(repo, head, nodeid)
    if head_node is None:
        return False
    if not exists(repo, base, path):
        return True
    base_node = test_nodes(blob(repo, base, path)).get(name)
    return base_node is None or ast.dump(
        base_node, include_attributes=False
    ) != ast.dump(head_node, include_attributes=False)


def _validate_test_replacements(
    repo: Path,
    base: str,
    head: str,
    data: dict[str, Any],
    rewritten: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    core_nodeids: set[str],
) -> list[str]:
    errors: list[str] = []
    declarations = data.get("test_replacements", {})
    if not isinstance(declarations, dict):
        return ["test_replacements must be an object"]
    if set(declarations) != set(rewritten):
        missing = sorted(set(rewritten) - set(declarations))
        extra = sorted(set(declarations) - set(rewritten))
        if missing:
            errors.append(f"rewritten tests lack replacement evidence: {missing}")
        if extra:
            errors.append(f"test_replacements declares unchanged tests: {extra}")
        return errors

    used_replacements: set[str] = set()
    for old_nodeid, old_node in rewritten.items():
        item = declarations[old_nodeid]
        if not isinstance(item, dict):
            errors.append(f"{old_nodeid}: replacement evidence must be an object")
            continue
        reason = item.get("reason")
        preserved = item.get("preserved_behavior")
        replacements = item.get("replacements")
        anchors = item.get("anchors")
        if not isinstance(reason, str) or len(reason.strip()) < 40:
            errors.append(f"{old_nodeid}: replacement reason is not specific enough")
        if not isinstance(preserved, str) or len(preserved.strip()) < 30:
            errors.append(f"{old_nodeid}: preserved_behavior is not specific enough")
        if not isinstance(replacements, list) or not replacements or any(
            not isinstance(value, str) or not value for value in replacements
        ):
            errors.append(f"{old_nodeid}: replacements must be a non-empty string list")
            continue
        if len(replacements) != len(set(replacements)):
            errors.append(f"{old_nodeid}: replacements contain duplicates")
            continue
        overlap = used_replacements & set(replacements)
        if overlap:
            errors.append(
                f"replacement tests cannot cover multiple old tests: {sorted(overlap)}"
            )
        used_replacements.update(replacements)
        if not isinstance(anchors, list) or not anchors:
            errors.append(f"{old_nodeid}: at least one preserved anchor is required")
            anchors = []

        old_anchors = _anchors(old_node)
        replacement_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        for nodeid in replacements:
            node = _head_test(repo, head, nodeid)
            if node is None:
                errors.append(f"{old_nodeid}: replacement test does not exist: {nodeid}")
                continue
            replacement_nodes.append(node)
            if nodeid not in core_nodeids:
                errors.append(f"{old_nodeid}: replacement test is not core evidence: {nodeid}")
            if not _replacement_changed(repo, base, head, nodeid):
                errors.append(f"{old_nodeid}: replacement test is not new or changed: {nodeid}")
        if not replacement_nodes:
            continue

        replacement_anchors = set().union(*(_anchors(node) for node in replacement_nodes))
        for anchor in anchors:
            if isinstance(anchor, str):
                before_name = after_name = anchor
            elif isinstance(anchor, dict) and set(anchor) == {"before", "after"}:
                before_name, after_name = anchor["before"], anchor["after"]
            else:
                errors.append(f"{old_nodeid}: invalid anchor declaration: {anchor!r}")
                continue
            if not all(
                isinstance(value, str) and value for value in (before_name, after_name)
            ):
                errors.append(f"{old_nodeid}: anchor names must be non-empty strings")
            elif before_name not in old_anchors or after_name not in replacement_anchors:
                errors.append(
                    f"{old_nodeid}: anchor continuity not proven: "
                    f"{before_name!r}->{after_name!r}"
                )

        old_checks, _ = _meaningful_checks(old_node)
        replacement_metrics = [_meaningful_checks(node) for node in replacement_nodes]
        replacement_checks = sum(count for count, _ in replacement_metrics)
        if any(trivial for _, trivial in replacement_metrics):
            errors.append(f"{old_nodeid}: replacement contains a constant/trivial assertion")
        if replacement_checks < max(1, old_checks):
            errors.append(
                f"{old_nodeid}: replacement weakens executable checks "
                f"({replacement_checks} < {max(1, old_checks)})"
            )
    return errors


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
            repo,
            "diff",
            "--name-status",
            "-M",
            "-C",
            "--find-copies-harder",
            base,
            head,
            "--",
            "*.py",
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
    elif any(
        not isinstance(value, str) or len(value.strip()) < 20
        for value in responsibilities.values()
    ):
        errors.append("every changed Python path needs a concrete responsibility")

    nearest = data.get("nearest_existing")
    if not isinstance(nearest, list) or not nearest:
        errors.append("nearest_existing must list inspected base Python modules")
    else:
        for path in nearest:
            if (
                not isinstance(path, str)
                or not path.endswith(".py")
                or not exists(repo, base, path)
            ):
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
    core_nodeids: set[str] = set()
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
                nodeid = str(nodeid)
                path, _, test_name = nodeid.partition("::")
                tests.add(path)
                core_nodeids.add(nodeid)
                if not path.startswith("tests/") or not exists(repo, head, path):
                    errors.append(f"missing core-function test at head: {nodeid}")
                elif (
                    not test_name
                    or test_name
                    not in named_nodes(blob(repo, head, path), tests_only=True)
                ):
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
            if not all(isinstance(value, str) and value for value in (path, name, used_by)):
                errors.append("reuse_evidence requires path, symbol, and used_by")
                continue
            if not exists(repo, base, path) or name not in defined_symbols(
                blob(repo, base, path)
            ):
                errors.append(f"reused symbol is not defined at base: {path}:{name}")
                continue
            if used_by not in changed or not exists(repo, head, used_by):
                errors.append(f"reuse target is not changed code: {used_by}:{name}")
                continue
            head_source = blob(repo, head, used_by)
            reused_in_place = used_by == path and name in defined_symbols(head_source)
            if not reused_in_place and name not in referenced_symbols(head_source):
                errors.append(
                    f"reused symbol is not referenced by changed code: {used_by}:{name}"
                )

    rewritten: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for row in rows:
        status = row[0][0]
        if status == "C":
            errors.append(f"copy detected: {row[1]} -> {row[2]}; reuse the source")
        elif status == "D":
            errors.append(
                f"Python deletion needs a separate user-approved removal plan: {row[1]}"
            )
        elif status in {"M", "R"}:
            old, new = (row[1], row[2]) if status == "R" else (row[1], row[1])
            before, after = blob(repo, base, old), blob(repo, head, new)
            if not old.startswith("tests/"):
                removed = set(named_nodes(before)) - set(named_nodes(after))
                if removed:
                    errors.append(f"public symbols removed from {old}: {sorted(removed)}")
            else:
                old_tests, new_tests = test_nodes(before), test_nodes(after)
                new_dumps = {
                    name: ast.dump(node, include_attributes=False)
                    for name, node in new_tests.items()
                }
                for name, node in old_tests.items():
                    if new_dumps.get(name) != ast.dump(node, include_attributes=False):
                        rewritten[f"{old}::{name}"] = node

    errors.extend(
        _validate_test_replacements(repo, base, head, data, rewritten, core_nodeids)
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--pr-body-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        errors = validate(
            args.repo_root.resolve(),
            args.base,
            args.head,
            args.pr_body_file.read_text(),
        )
    except (OSError, ValueError, json.JSONDecodeError, SyntaxError) as exc:
        errors = [str(exc)]
    print(json.dumps({"status": "FAIL" if errors else "PASS", "errors": errors}, indent=2))
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
