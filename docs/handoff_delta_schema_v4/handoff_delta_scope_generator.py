#!/usr/bin/env python3
"""Generate schema-v4 scoped experiment handoff delta candidates.

Round-2 companion to ``tools/handoff_delta_scope_analyzer.py``.  This helper is
kept under ``docs/handoff_delta_schema_v4`` intentionally: it is an experimental
schema-v4 authoring aid, not a production update/normalization tool.  It does
not apply, normalize, or merge a delta.  It only writes a schema-v4 candidate
YAML file for exactly one experiment scope and then re-runs the read-only
analyzer against the generated file.  schema-v3 remains the production default.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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

GENERATOR_SCHEMA_VERSION = 1
SUPPORTED_DELTA_KIND = "scoped_experiment_update"
SUPPORTED_OPERATION_TYPE = "register_or_update_experiment"
EXPERIMENT_ID_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:-[A-Z0-9_.]+){2,}$")


class GeneratorError(Exception):
    """Controlled generator failure."""


def require_yaml() -> None:
    if yaml is None:
        raise GeneratorError(f"PyYAML is required: {YAML_IMPORT_ERROR}")


def load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    require_yaml()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GeneratorError(f"missing {label}: {path}") from exc
    except Exception as exc:
        raise GeneratorError(f"failed to parse {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GeneratorError(f"{label} must be a YAML mapping")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    require_yaml()
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temp.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def validate_experiment_id(experiment_id: str) -> None:
    if not EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise GeneratorError(
            "experiment id must be a DRPO-style uppercase scoped identifier; "
            f"got {experiment_id!r}"
        )


def payload_experiment_ids(payload: dict[str, Any]) -> set[str]:
    return analyzer.ids_from_any(payload)


def validate_payload_scope(payload: dict[str, Any], experiment_id: str) -> None:
    ids = payload_experiment_ids(payload)
    extra = sorted(ids - {experiment_id})
    if extra:
        raise GeneratorError(
            "payload references experiment IDs outside the target scope: "
            + ", ".join(extra)
        )


def build_delta(
    repo: Path,
    *,
    experiment_id: str,
    payload: dict[str, Any],
    update_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_experiment_id(experiment_id)
    validate_payload_scope(payload, experiment_id)
    scope_hash, material = analyzer.experiment_scope_hash(repo, experiment_id)
    scope_exists = material["registry_entry"] != analyzer.MISSING or material["handoff_entry"] != analyzer.MISSING
    delta = {
        "schema_version": 4,
        "delta_kind": SUPPORTED_DELTA_KIND,
        "update_id": update_id,
        "mode": "scoped_candidate",
        "operation": {
            "type": SUPPORTED_OPERATION_TYPE,
            "experiment_id": experiment_id,
        },
        "scope": {
            "type": "experiment",
            "id": experiment_id,
        },
        "preimage": {
            "schema": analyzer.SCOPE_SCHEMA,
            "scope_exists": bool(scope_exists),
            "experiment_scope_sha256": scope_hash,
        },
        "payload": payload,
        "generator": {
            "schema_version": GENERATOR_SCHEMA_VERSION,
            "tool": "docs/handoff_delta_schema_v4/handoff_delta_scope_generator.py",
            "read_only_base": True,
            "applies_delta": False,
        },
    }
    return delta, material


def generate(
    repo: Path,
    *,
    experiment_id: str,
    payload_path: Path,
    output: Path,
    update_id: str | None,
) -> dict[str, Any]:
    repo = repo.expanduser().resolve()
    payload = load_yaml_mapping(payload_path.expanduser(), "payload")
    resolved_update_id = update_id or f"SCHEMA-V4-SCOPED-{experiment_id}"
    delta, material = build_delta(
        repo,
        experiment_id=experiment_id,
        payload=payload,
        update_id=resolved_update_id,
    )
    write_yaml(output, delta)
    analysis = analyzer.analyze(repo, output)
    if analysis.get("status") != "PASS":
        raise GeneratorError(
            "generated delta failed analyzer validation: "
            + json.dumps(analysis.get("errors", []), ensure_ascii=False)
        )
    return {
        "status": "PASS",
        "output": str(output.expanduser().resolve()),
        "repo": str(repo),
        "experiment_id": experiment_id,
        "update_id": resolved_update_id,
        "scope_exists": material["registry_entry"] != analyzer.MISSING
        or material["handoff_entry"] != analyzer.MISSING,
        "current_scope_sha256": delta["preimage"]["experiment_scope_sha256"],
        "analyzer_status": analysis["status"],
        "read_only_merge_behavior": True,
        "schema_v4_merge_enabled": False,
    }


def run_self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="drpo_scope_generator_") as tmp:
        root = Path(tmp)
        repo = root / "repo"
        (repo / "experiments").mkdir(parents=True)
        (repo / "docs").mkdir(parents=True)
        subprocess.run(["git", "init"], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        (repo / "experiments" / "registry.yaml").write_text(
            "schema_version: 2\nexperiments:\n"
            "- id: EXT-C-E8-GENERATOR-TEST-01\n  status: not_run\n  name: generator test\n",
            encoding="utf-8",
        )
        (repo / "docs" / "handoff.md").write_text(
            "# Handoff\n\nEXT-C-E8-GENERATOR-TEST-01 is registered.\n",
            encoding="utf-8",
        )
        good_payload = root / "payload.yaml"
        good_payload.write_text(
            "registry_entry:\n"
            "  id: EXT-C-E8-GENERATOR-TEST-01\n"
            "  status: pilot\n"
            "handoff_entry:\n"
            "  text: EXT-C-E8-GENERATOR-TEST-01 pilot registration candidate\n",
            encoding="utf-8",
        )
        output = root / "HANDOFF_DELTA.yaml"
        good = generate(
            repo,
            experiment_id="EXT-C-E8-GENERATOR-TEST-01",
            payload_path=good_payload,
            output=output,
            update_id="SCHEMA-V4-GENERATOR-SELFTEST",
        )
        generated_delta = load_yaml_mapping(output, "generated delta")
        generated_analysis = analyzer.analyze(repo, output)

        bad_payload = root / "bad_payload.yaml"
        bad_payload.write_text(
            "registry_entry:\n"
            "  id: EXT-C-E8-GENERATOR-TEST-01\n"
            "  related: EXT-C-E7-OTHER-01\n",
            encoding="utf-8",
        )
        try:
            generate(
                repo,
                experiment_id="EXT-C-E8-GENERATOR-TEST-01",
                payload_path=bad_payload,
                output=root / "BAD_HANDOFF_DELTA.yaml",
                update_id="SCHEMA-V4-GENERATOR-BAD",
            )
        except GeneratorError:
            bad_rejected = True
        else:
            bad_rejected = False

        return {
            "status": "PASS"
            if good["status"] == "PASS"
            and generated_analysis["status"] == "PASS"
            and generated_delta.get("schema_version") == 4
            and bad_rejected
            else "FAIL",
            "good_status": good["status"],
            "generated_analysis_status": generated_analysis["status"],
            "bad_payload_rejected": bad_rejected,
            "network_used": False,
            "schema_v4_merge_enabled": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--experiment-id", help="single target experiment ID")
    parser.add_argument("--payload", type=Path, help="YAML payload mapping for the schema-v4 delta")
    parser.add_argument("--output", type=Path, help="output HANDOFF_DELTA.yaml path")
    parser.add_argument("--update-id", help="optional immutable update id")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--self-test", action="store_true", help="run built-in self tests")
    args = parser.parse_args(argv)

    try:
        if args.self_test:
            result = run_self_test()
        else:
            if not args.experiment_id:
                raise GeneratorError("--experiment-id is required unless --self-test is used")
            if args.payload is None:
                raise GeneratorError("--payload is required unless --self-test is used")
            if args.output is None:
                raise GeneratorError("--output is required unless --self-test is used")
            result = generate(
                args.repo,
                experiment_id=args.experiment_id,
                payload_path=args.payload,
                output=args.output,
                update_id=args.update_id,
            )
    except (GeneratorError, analyzer.AnalyzerError) as exc:
        result = {"status": "FAIL", "errors": [str(exc)], "schema_v4_merge_enabled": False}

    if args.json or args.self_test:
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(f"status: {result.get('status')}")
        if result.get("status") == "PASS":
            print(f"output: {result.get('output')}")
            print(f"experiment_id: {result.get('experiment_id')}")
            print(f"current_scope_sha256: {result.get('current_scope_sha256')}")
        for error in result.get("errors", []):
            print(f"error: {error}")
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
