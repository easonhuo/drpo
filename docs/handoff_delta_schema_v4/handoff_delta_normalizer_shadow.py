#!/usr/bin/env python3
"""Schema-v4 scoped-delta normalizer shadow decision helper.

Round-3 bridge for the schema-v4 scoped experiment delta plan.  This helper does
not apply, normalize, or merge a delta.  It evaluates the same conservative
``would_merge`` / ``would_reject`` decision that a future trusted normalizer path
would use, then exits without modifying the repository.  schema-v3 remains the
only production merge path.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover - dependency error path
    yaml = None  # type: ignore
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None

import handoff_delta_scope_analyzer as analyzer  # noqa: E402

SHADOW_SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSION = 4
SUPPORTED_DELTA_KIND = "scoped_experiment_update"
SUPPORTED_OPERATION_TYPE = "register_or_update_experiment"
SUPPORTED_SCOPE_TYPE = "experiment"


class ShadowError(Exception):
    """Controlled shadow decision failure."""


def require_yaml() -> None:
    if yaml is None:
        raise ShadowError(f"PyYAML is required: {YAML_IMPORT_ERROR}")


def load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    require_yaml()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ShadowError(f"missing {label}: {path}") from exc
    except Exception as exc:
        raise ShadowError(f"failed to parse {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ShadowError(f"{label} must be a YAML mapping")
    return payload


def _reject(reason: str, *, analysis: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": "PASS",
        "decision": "would_reject",
        "reason": reason,
        "analysis_status": None if analysis is None else analysis.get("status"),
        "analysis_errors": [] if analysis is None else analysis.get("errors", []),
        "analysis_warnings": [] if analysis is None else analysis.get("warnings", []),
        "schema_v4_merge_enabled": False,
        "repository_modified": False,
        "network_used": False,
    }


def _single_scope(analysis: dict[str, Any]) -> dict[str, Any] | None:
    scopes = analysis.get("scopes")
    if not isinstance(scopes, list) or len(scopes) != 1:
        return None
    scope = scopes[0]
    return scope if isinstance(scope, dict) else None


def shadow_decision(repo: Path, delta_path: Path) -> dict[str, Any]:
    """Return a conservative schema-v4 would-merge/would-reject decision.

    ``status`` reports whether the shadow helper itself ran to completion.  The
    future merge outcome is carried in ``decision``.  Malformed, ambiguous, or
    unsupported deltas are successful shadow evaluations with ``would_reject``;
    only tool/runtime problems raise ``ShadowError``.
    """

    repo = repo.expanduser().resolve()
    delta_path = delta_path.expanduser().resolve()
    delta = load_yaml_mapping(delta_path, "delta")

    if delta.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        return _reject(
            f"unsupported schema_version={delta.get('schema_version')!r}; schema-v3 remains production default"
        )
    if delta.get("delta_kind") != SUPPORTED_DELTA_KIND:
        return _reject(f"unsupported schema-v4 delta_kind={delta.get('delta_kind')!r}")

    operation = delta.get("operation")
    if not isinstance(operation, dict):
        return _reject("schema-v4 delta.operation must be a mapping")
    if operation.get("type") != SUPPORTED_OPERATION_TYPE:
        return _reject(f"unsupported operation.type={operation.get('type')!r}")
    operation_experiment_id = operation.get("experiment_id")
    if not isinstance(operation_experiment_id, str) or not operation_experiment_id:
        return _reject("operation.experiment_id must be a non-empty string")

    scope = delta.get("scope")
    if not isinstance(scope, dict):
        return _reject("schema-v4 delta.scope must be a mapping")
    if scope.get("type") != SUPPORTED_SCOPE_TYPE:
        return _reject(f"unsupported scope.type={scope.get('type')!r}")
    if scope.get("id") != operation_experiment_id:
        return _reject("scope.id must exactly match operation.experiment_id")

    preimage = delta.get("preimage")
    if not isinstance(preimage, dict):
        return _reject("schema-v4 delta.preimage must be a mapping")
    if preimage.get("schema") not in {None, analyzer.SCOPE_SCHEMA}:
        return _reject(f"unsupported preimage.schema={preimage.get('schema')!r}")
    preimage_hash = preimage.get("experiment_scope_sha256")
    if not isinstance(preimage_hash, str) or not preimage_hash:
        return _reject("preimage.experiment_scope_sha256 must be a non-empty string")

    analysis = analyzer.analyze(repo, delta_path)
    shadow_scope = _single_scope(analysis)
    if analysis.get("status") != "PASS":
        return _reject("scope analyzer rejected schema-v4 delta", analysis=analysis)
    if shadow_scope is None:
        return _reject("scope analyzer did not return exactly one scope", analysis=analysis)
    if shadow_scope.get("type") != SUPPORTED_SCOPE_TYPE:
        return _reject(f"unsupported analyzed scope.type={shadow_scope.get('type')!r}", analysis=analysis)
    if shadow_scope.get("id") != operation_experiment_id:
        return _reject("analyzed scope id differs from operation.experiment_id", analysis=analysis)
    if shadow_scope.get("preimage_matches_current") is not True:
        return _reject("schema-v4 preimage does not match current experiment scope", analysis=analysis)

    return {
        "status": "PASS",
        "decision": "would_merge",
        "reason": "single schema-v4 experiment scope preimage matches current repository state",
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "delta_kind": SUPPORTED_DELTA_KIND,
        "operation_type": SUPPORTED_OPERATION_TYPE,
        "scope": {
            "type": SUPPORTED_SCOPE_TYPE,
            "id": operation_experiment_id,
            "current_scope_sha256": shadow_scope.get("current_scope_sha256"),
            "preimage_scope_sha256": shadow_scope.get("preimage_scope_sha256"),
            "registry_entry_present": shadow_scope.get("registry_entry_present"),
            "handoff_entry_present": shadow_scope.get("handoff_entry_present"),
        },
        "analysis_status": analysis.get("status"),
        "analysis_errors": analysis.get("errors", []),
        "analysis_warnings": analysis.get("warnings", []),
        "schema_v4_merge_enabled": False,
        "repository_modified": False,
        "network_used": False,
    }


def run_self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="drpo_normalizer_shadow_") as tmp:
        root = Path(tmp)
        repo = root / "repo"
        (repo / "experiments").mkdir(parents=True)
        (repo / "docs").mkdir(parents=True)
        subprocess.run(["git", "init"], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        (repo / "experiments" / "registry.yaml").write_text(
            "schema_version: 2\nexperiments:\n"
            "- id: EXT-C-E8-SHADOW-TEST-01\n  status: not_run\n  name: shadow test\n"
            "- id: EXT-C-E7-UNRELATED-01\n  status: not_run\n  name: unrelated test\n",
            encoding="utf-8",
        )
        (repo / "docs" / "handoff.md").write_text(
            "# Handoff\n\n"
            "EXT-C-E8-SHADOW-TEST-01 is registered.\n"
            "EXT-C-E7-UNRELATED-01 is registered.\n",
            encoding="utf-8",
        )
        subprocess.run(
            ["git", "add", "experiments/registry.yaml", "docs/handoff.md"],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=DRPO Shadow Self-Test",
                "-c",
                "user.email=drpo-shadow-self-test@local.invalid",
                "commit",
                "-m",
                "seed shadow self-test repo",
            ],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        scope_hash, _ = analyzer.experiment_scope_hash(repo, "EXT-C-E8-SHADOW-TEST-01")

        good_delta = root / "good.yaml"
        good_delta.write_text(
            "schema_version: 4\n"
            "delta_kind: scoped_experiment_update\n"
            "update_id: SCHEMA-V4-SHADOW-GOOD\n"
            "mode: scoped_candidate\n"
            "operation:\n  type: register_or_update_experiment\n  experiment_id: EXT-C-E8-SHADOW-TEST-01\n"
            "scope:\n  type: experiment\n  id: EXT-C-E8-SHADOW-TEST-01\n"
            "preimage:\n  schema: drpo-scoped-experiment-v1\n  scope_exists: true\n  experiment_scope_sha256: \""
            + scope_hash
            + "\"\n"
            "payload:\n  registry_entry:\n    id: EXT-C-E8-SHADOW-TEST-01\n    status: pilot\n",
            encoding="utf-8",
        )
        bad_hash_delta = root / "bad_hash.yaml"
        bad_hash_delta.write_text(
            good_delta.read_text(encoding="utf-8").replace(scope_hash, "0" * 64),
            encoding="utf-8",
        )
        multi_delta = root / "multi.yaml"
        multi_delta.write_text(
            good_delta.read_text(encoding="utf-8") + "  related: EXT-C-E7-OTHER-01\n",
            encoding="utf-8",
        )
        schema_v3_delta = root / "v3.yaml"
        schema_v3_delta.write_text(
            "schema_version: 3\nupdate_id: EXT-C-E8-SHADOW-V3\n",
            encoding="utf-8",
        )

        good = shadow_decision(repo, good_delta)
        bad_hash = shadow_decision(repo, bad_hash_delta)
        multi = shadow_decision(repo, multi_delta)
        v3 = shadow_decision(repo, schema_v3_delta)

        registry_path = repo / "experiments" / "registry.yaml"
        registry_path.write_text(
            registry_path.read_text(encoding="utf-8").replace(
                "EXT-C-E7-UNRELATED-01\n  status: not_run",
                "EXT-C-E7-UNRELATED-01\n  status: pilot",
            ),
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "experiments/registry.yaml"], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=DRPO Shadow Self-Test",
                "-c",
                "user.email=drpo-shadow-self-test@local.invalid",
                "commit",
                "-m",
                "change unrelated experiment scope",
            ],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        different_scope_drift = shadow_decision(repo, good_delta)

        registry_path.write_text(
            registry_path.read_text(encoding="utf-8").replace(
                "EXT-C-E8-SHADOW-TEST-01\n  status: not_run",
                "EXT-C-E8-SHADOW-TEST-01\n  status: pilot",
            ),
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "experiments/registry.yaml"], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=DRPO Shadow Self-Test",
                "-c",
                "user.email=drpo-shadow-self-test@local.invalid",
                "commit",
                "-m",
                "change target experiment scope",
            ],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        same_scope_drift = shadow_decision(repo, good_delta)
        dirty_after = analyzer.repo_dirty(repo)

        return {
            "status": "PASS"
            if good.get("decision") == "would_merge"
            and bad_hash.get("decision") == "would_reject"
            and multi.get("decision") == "would_reject"
            and v3.get("decision") == "would_reject"
            and different_scope_drift.get("decision") == "would_merge"
            and same_scope_drift.get("decision") == "would_reject"
            and not dirty_after
            else "FAIL",
            "good_decision": good.get("decision"),
            "bad_hash_decision": bad_hash.get("decision"),
            "multi_scope_decision": multi.get("decision"),
            "schema_v3_decision": v3.get("decision"),
            "different_scope_drift_decision": different_scope_drift.get("decision"),
            "same_scope_drift_decision": same_scope_drift.get("decision"),
            "repository_modified": dirty_after,
            "schema_v4_merge_enabled": False,
            "network_used": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--delta", type=Path, help="HANDOFF_DELTA.yaml to evaluate")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--self-test", action="store_true", help="run built-in self tests")
    args = parser.parse_args(argv)

    try:
        if args.self_test:
            result = run_self_test()
        else:
            if args.delta is None:
                raise ShadowError("--delta is required unless --self-test is used")
            result = shadow_decision(args.repo, args.delta)
    except (ShadowError, analyzer.AnalyzerError) as exc:
        result = {
            "status": "FAIL",
            "decision": "would_reject",
            "errors": [str(exc)],
            "schema_v4_merge_enabled": False,
            "repository_modified": False,
            "network_used": False,
        }

    if args.json or args.self_test:
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(f"status: {result.get('status')}")
        print(f"decision: {result.get('decision')}")
        print(f"reason: {result.get('reason')}")
        scope = result.get("scope")
        if isinstance(scope, dict):
            print(f"scope: {scope.get('type')}:{scope.get('id')}")
            print(f"current_scope_sha256: {scope.get('current_scope_sha256')}")
        for error in result.get("analysis_errors", []) + result.get("errors", []):
            print(f"error: {error}")
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
