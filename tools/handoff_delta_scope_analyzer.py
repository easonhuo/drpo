#!/usr/bin/env python3
"""Read-only DRPO handoff delta scope analyzer.

Round-1 tool for the schema-v4 scoped experiment delta plan.  The analyzer does
not modify the repository and does not enable scoped merge.  It only reports the
experiment scope a delta appears to touch and, for schema-v4 candidate deltas,
compares the declared preimage hash with the current repository scope hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover - dependency error path
    yaml = None  # type: ignore
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None

SCOPE_SCHEMA = "drpo-scoped-experiment-v1"
MISSING = "<MISSING>"
EXPERIMENT_ID_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9_.]+){2,}\b")


class AnalyzerError(Exception):
    """Controlled analyzer failure."""


def load_yaml(path: Path) -> Any:
    if yaml is None:
        raise AnalyzerError(f"PyYAML is required to read {path}: {YAML_IMPORT_ERROR}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AnalyzerError(f"missing file: {path}") from exc
    except Exception as exc:
        raise AnalyzerError(f"failed to parse YAML {path}: {exc}") from exc


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def repo_dirty(repo: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return False
    return bool(result.stdout.strip())


def registry_experiments(registry: Any) -> list[dict[str, Any]]:
    if not isinstance(registry, dict):
        return []
    items = registry.get("experiments")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def registry_entry_for(repo: Path, experiment_id: str) -> Any:
    registry_path = repo / "experiments" / "registry.yaml"
    registry = load_yaml(registry_path)
    matches = [item for item in registry_experiments(registry) if item.get("id") == experiment_id]
    if len(matches) > 1:
        raise AnalyzerError(f"registry has duplicate experiment id {experiment_id}")
    return matches[0] if matches else MISSING


def handoff_material_for(repo: Path, experiment_id: str, *, context: int = 2) -> Any:
    handoff_path = repo / "docs" / "handoff.md"
    if not handoff_path.is_file():
        return MISSING
    lines = handoff_path.read_text(encoding="utf-8").splitlines()
    hit_indices = [i for i, line in enumerate(lines) if experiment_id in line]
    if not hit_indices:
        return MISSING

    ranges: list[tuple[int, int]] = []
    for idx in hit_indices:
        start = max(0, idx - context)
        end = min(len(lines), idx + context + 1)
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))

    snippets = []
    for start, end in ranges:
        snippets.append(
            {
                "start_line": start + 1,
                "end_line": end,
                "text": "\n".join(lines[start:end]),
            }
        )
    return {"match_count": len(hit_indices), "snippets": snippets}


def experiment_scope_material(repo: Path, experiment_id: str) -> dict[str, Any]:
    return {
        "schema": SCOPE_SCHEMA,
        "experiment_id": experiment_id,
        "registry_entry": registry_entry_for(repo, experiment_id),
        "handoff_entry": handoff_material_for(repo, experiment_id),
    }


def experiment_scope_hash(repo: Path, experiment_id: str) -> tuple[str, dict[str, Any]]:
    material = experiment_scope_material(repo, experiment_id)
    return sha256_text(canonical_json(material)), material


def ids_from_any(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        found.update(EXPERIMENT_ID_RE.findall(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in {"experiment_id", "id"} and isinstance(item, str):
                found.add(item)
            found.update(ids_from_any(item))
    elif isinstance(value, list):
        for item in value:
            found.update(ids_from_any(item))
    return found


def schema_v4_experiment_ids(delta: dict[str, Any]) -> tuple[set[str], list[str]]:
    warnings: list[str] = []
    ids: set[str] = set()

    scope = delta.get("scope")
    if isinstance(scope, dict):
        if scope.get("type") != "experiment":
            warnings.append(f"unsupported scope.type={scope.get('type')!r}")
        if isinstance(scope.get("id"), str):
            ids.add(scope["id"])

    operation = delta.get("operation")
    if isinstance(operation, dict):
        if operation.get("type") not in {None, "register_or_update_experiment"}:
            warnings.append(f"unsupported operation.type={operation.get('type')!r}")
        if isinstance(operation.get("experiment_id"), str):
            ids.add(operation["experiment_id"])

    payload = delta.get("payload")
    payload_ids = ids_from_any(payload)
    if payload_ids:
        ids.update(payload_ids)

    return ids, warnings


def schema_v3_experiment_ids(delta: dict[str, Any]) -> tuple[set[str], list[str]]:
    warnings: list[str] = []
    ids: set[str] = set()

    registry = delta.get("registry")
    if isinstance(registry, dict):
        for key in ("changes", "transitions"):
            changes = registry.get(key)
            if isinstance(changes, list):
                ids.update(ids_from_any(changes))

    # For v3 deltas with unchanged registry, operation content sometimes mentions
    # an experiment ID.  This is only a diagnostic inference, not a merge contract.
    operations = delta.get("operations")
    if isinstance(operations, list):
        ids.update(ids_from_any(operations))

    if not ids:
        warnings.append("no experiment id inferred from schema-v3 delta")
    return ids, warnings


def analyze(repo: Path, delta_path: Path) -> dict[str, Any]:
    repo = repo.resolve()
    delta_path = delta_path.resolve()
    delta = load_yaml(delta_path)
    if not isinstance(delta, dict):
        raise AnalyzerError("delta root must be a mapping")

    schema_version = delta.get("schema_version")
    warnings: list[str] = []
    errors: list[str] = []
    ids: set[str]

    if schema_version == 4:
        ids, schema_warnings = schema_v4_experiment_ids(delta)
        warnings.extend(schema_warnings)
        if delta.get("delta_kind") != "scoped_experiment_update":
            errors.append(f"unsupported schema-v4 delta_kind={delta.get('delta_kind')!r}")
    elif schema_version in {1, 2, 3}:
        ids, schema_warnings = schema_v3_experiment_ids(delta)
        warnings.extend(schema_warnings)
    else:
        ids = set()
        errors.append(f"unsupported schema_version={schema_version!r}")

    sorted_ids = sorted(ids)
    if len(sorted_ids) == 0:
        errors.append("no experiment scope identified")
    elif len(sorted_ids) > 1:
        errors.append("multiple experiment scopes identified; fail closed")

    scopes = []
    for experiment_id in sorted_ids:
        current_hash, material = experiment_scope_hash(repo, experiment_id)
        preimage_hash = None
        preimage = delta.get("preimage") if isinstance(delta.get("preimage"), dict) else {}
        if isinstance(preimage, dict):
            maybe_hash = preimage.get("experiment_scope_sha256")
            if isinstance(maybe_hash, str):
                preimage_hash = maybe_hash
        hash_match = None if preimage_hash is None else (preimage_hash == current_hash)
        scopes.append(
            {
                "type": "experiment",
                "id": experiment_id,
                "current_scope_sha256": current_hash,
                "preimage_scope_sha256": preimage_hash,
                "preimage_matches_current": hash_match,
                "registry_entry_present": material["registry_entry"] != MISSING,
                "handoff_entry_present": material["handoff_entry"] != MISSING,
                "handoff_match_count": 0
                if material["handoff_entry"] == MISSING
                else material["handoff_entry"].get("match_count", 0),
            }
        )

    if schema_version == 4:
        if len(scopes) == 1:
            if scopes[0]["preimage_scope_sha256"] is None:
                errors.append("schema-v4 delta is missing preimage.experiment_scope_sha256")
            elif scopes[0]["preimage_matches_current"] is False:
                errors.append("schema-v4 preimage hash differs from current experiment scope")
        if not os.environ.get("DRPO_ENABLE_SCHEMA_V4_SCOPED_DELTA"):
            warnings.append("schema-v4 scoped merge is not enabled; analyzer is read-only")

    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "schema_version": schema_version,
        "delta_path": str(delta_path),
        "repo": str(repo),
        "repository_dirty": repo_dirty(repo),
        "scope_count": len(scopes),
        "scopes": scopes,
        "errors": errors,
        "warnings": warnings,
        "read_only": True,
    }


def run_self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="drpo_scope_analyzer_") as tmp:
        root = Path(tmp)
        repo = root / "repo"
        (repo / "experiments").mkdir(parents=True)
        (repo / "docs").mkdir(parents=True)
        subprocess.run(["git", "init"], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        (repo / "experiments" / "registry.yaml").write_text(
            "schema_version: 2\nexperiments:\n"
            "- id: EXT-C-E8-TEST-01\n  status: not_run\n  name: test\n",
            encoding="utf-8",
        )
        (repo / "docs" / "handoff.md").write_text(
            "# Handoff\n\n## Experiments\n\nEXT-C-E8-TEST-01 is registered.\n",
            encoding="utf-8",
        )
        scope_hash, _ = experiment_scope_hash(repo, "EXT-C-E8-TEST-01")

        good_delta = root / "good.yaml"
        good_delta.write_text(
            "schema_version: 4\n"
            "delta_kind: scoped_experiment_update\n"
            "operation:\n  type: register_or_update_experiment\n  experiment_id: EXT-C-E8-TEST-01\n"
            "scope:\n  type: experiment\n  id: EXT-C-E8-TEST-01\n"
            "preimage:\n  scope_exists: true\n  experiment_scope_sha256: \"" + scope_hash + "\"\n"
            "payload:\n  registry_entry:\n    id: EXT-C-E8-TEST-01\n    status: pilot\n",
            encoding="utf-8",
        )
        bad_hash_delta = root / "bad_hash.yaml"
        bad_hash_delta.write_text(good_delta.read_text(encoding="utf-8").replace(scope_hash, "0" * 64), encoding="utf-8")
        multi_delta = root / "multi.yaml"
        multi_delta.write_text(
            good_delta.read_text(encoding="utf-8")
            + "  related: EXT-C-E7-OTHER-01\n",
            encoding="utf-8",
        )

        good = analyze(repo, good_delta)
        bad_hash = analyze(repo, bad_hash_delta)
        multi = analyze(repo, multi_delta)
        return {
            "status": "PASS"
            if good["status"] == "PASS" and bad_hash["status"] == "FAIL" and multi["status"] == "FAIL"
            else "FAIL",
            "good_status": good["status"],
            "bad_hash_status": bad_hash["status"],
            "multi_scope_status": multi["status"],
            "network_used": False,
            "repository_modified": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--delta", type=Path, help="HANDOFF_DELTA.yaml to analyze")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--self-test", action="store_true", help="run built-in self tests")
    args = parser.parse_args(argv)

    try:
        if args.self_test:
            result = run_self_test()
        else:
            if args.delta is None:
                raise AnalyzerError("--delta is required unless --self-test is used")
            result = analyze(args.repo, args.delta)
    except AnalyzerError as exc:
        result = {"status": "FAIL", "errors": [str(exc)], "read_only": True}

    if args.json or args.self_test:
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(f"status: {result.get('status')}")
        for scope in result.get("scopes", []):
            print(f"scope: {scope['type']}:{scope['id']} {scope['current_scope_sha256']}")
        for warning in result.get("warnings", []):
            print(f"warning: {warning}")
        for error in result.get("errors", []):
            print(f"error: {error}")
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
