#!/usr/bin/env python3
"""Experimental schema-v4 scoped experiment materializer.

Round-4 controlled bridge for schema-v4 scoped delta.  This helper is still not
wired into the production ``drpo-update`` normalizer path.  It can materialize a
single schema-v4 experiment scoped delta only when all of these are true:

* the Round-3 shadow decision is ``would_merge``;
* the caller passes ``--apply``;
* ``DRPO_ENABLE_SCHEMA_V4_SCOPED_DELTA=1`` is present in the environment.

Without ``--apply`` it is a dry-run verifier.  With ``--apply`` but without the
environment switch it fails closed.  schema-v3 remains the production default.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "docs" / "handoff_delta_schema_v4"))

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover - dependency error path
    yaml = None  # type: ignore
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None

import handoff_delta_scope_analyzer as analyzer  # noqa: E402
import handoff_delta_normalizer_shadow as shadow  # noqa: E402

MERGER_SCHEMA_VERSION = 1
ENABLE_ENV = "DRPO_ENABLE_SCHEMA_V4_SCOPED_DELTA"
SUPPORTED_PAYLOAD_KEYS = {"registry_entry", "handoff_entry"}
HANDOFF_BLOCK_PREFIX = "SCHEMA-V4-SCOPED-EXPERIMENT"


class MergeError(Exception):
    """Controlled scoped merge failure."""


def require_yaml() -> None:
    if yaml is None:
        raise MergeError(f"PyYAML is required: {YAML_IMPORT_ERROR}")


def load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    require_yaml()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MergeError(f"missing {label}: {path}") from exc
    except Exception as exc:
        raise MergeError(f"failed to parse {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MergeError(f"{label} must be a YAML mapping")
    return payload


def write_yaml_atomic(path: Path, value: dict[str, Any]) -> None:
    require_yaml()
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temp.write_text(
            yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def validate_payload(delta: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    payload = delta.get("payload")
    if not isinstance(payload, dict):
        raise MergeError("schema-v4 delta.payload must be a mapping")
    unknown = sorted(set(payload) - SUPPORTED_PAYLOAD_KEYS)
    if unknown:
        raise MergeError("unsupported payload keys for Round-4 materializer: " + ", ".join(unknown))
    if not any(key in payload for key in SUPPORTED_PAYLOAD_KEYS):
        raise MergeError("payload must include registry_entry and/or handoff_entry")
    ids = analyzer.ids_from_any(payload)
    extra = sorted(ids - {experiment_id})
    if extra:
        raise MergeError("payload references experiment IDs outside target scope: " + ", ".join(extra))

    registry_entry = payload.get("registry_entry")
    if registry_entry is not None:
        if not isinstance(registry_entry, dict):
            raise MergeError("payload.registry_entry must be a mapping")
        if registry_entry.get("id") != experiment_id:
            raise MergeError("payload.registry_entry.id must exactly match operation.experiment_id")

    handoff_entry = payload.get("handoff_entry")
    if handoff_entry is not None:
        handoff_text(handoff_entry, experiment_id)  # validates representation
    return payload


def registry_document(repo: Path) -> tuple[Path, dict[str, Any]]:
    path = repo / "experiments" / "registry.yaml"
    registry = load_yaml_mapping(path, "registry")
    experiments = registry.get("experiments")
    if not isinstance(experiments, list):
        raise MergeError("experiments/registry.yaml must contain an experiments list")
    for item in experiments:
        if not isinstance(item, dict):
            raise MergeError("experiments/registry.yaml experiments entries must be mappings")
    return path, registry


def upsert_registry_entry(repo: Path, experiment_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    path, registry = registry_document(repo)
    experiments = registry["experiments"]
    matches = [idx for idx, item in enumerate(experiments) if item.get("id") == experiment_id]
    if len(matches) > 1:
        raise MergeError(f"registry has duplicate experiment id {experiment_id}")
    action = "updated" if matches else "inserted"
    if matches:
        experiments[matches[0]] = entry
    else:
        experiments.append(entry)
    write_yaml_atomic(path, registry)
    return {"path": str(path), "action": action}


def handoff_text(value: Any, experiment_id: str) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, dict):
        candidate = value.get("text", value.get("markdown"))
        if not isinstance(candidate, str):
            raise MergeError("payload.handoff_entry mapping must include text or markdown string")
        text = candidate
    else:
        raise MergeError("payload.handoff_entry must be a string or mapping")
    text = text.strip()
    if not text:
        raise MergeError("payload.handoff_entry text must be non-empty")
    if experiment_id not in text:
        raise MergeError("payload.handoff_entry text must mention operation.experiment_id")
    return text


def handoff_markers(experiment_id: str) -> tuple[str, str]:
    start = f"<!-- {HANDOFF_BLOCK_PREFIX}:{experiment_id}:START -->"
    end = f"<!-- {HANDOFF_BLOCK_PREFIX}:{experiment_id}:END -->"
    return start, end


def upsert_handoff_entry(repo: Path, experiment_id: str, value: Any) -> dict[str, Any]:
    path = repo / "docs" / "handoff.md"
    if not path.is_file():
        raise MergeError("docs/handoff.md is missing")
    text = path.read_text(encoding="utf-8")
    body = handoff_text(value, experiment_id)
    start, end = handoff_markers(experiment_id)
    block = f"{start}\n{body}\n{end}"

    start_idx = text.find(start)
    end_idx = text.find(end)
    if (start_idx == -1) != (end_idx == -1):
        raise MergeError(f"handoff block markers for {experiment_id} are unbalanced")
    if start_idx != -1 and end_idx != -1:
        if end_idx < start_idx:
            raise MergeError(f"handoff block markers for {experiment_id} are out of order")
        end_idx += len(end)
        new_text = text[:start_idx] + block + text[end_idx:]
        action = "updated"
    else:
        separator = "" if text.endswith("\n") else "\n"
        new_text = text + separator + "\n" + block + "\n"
        action = "inserted"
    path.write_text(new_text, encoding="utf-8")
    return {"path": str(path), "action": action}


def target_experiment_id(delta: dict[str, Any]) -> str:
    operation = delta.get("operation")
    if not isinstance(operation, dict):
        raise MergeError("schema-v4 delta.operation must be a mapping")
    experiment_id = operation.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise MergeError("operation.experiment_id must be a non-empty string")
    return experiment_id


def materialize(repo: Path, delta_path: Path, *, apply: bool = False) -> dict[str, Any]:
    repo = repo.expanduser().resolve()
    delta_path = delta_path.expanduser().resolve()
    delta = load_yaml_mapping(delta_path, "delta")
    experiment_id = target_experiment_id(delta)
    decision = shadow.shadow_decision(repo, delta_path)
    if decision.get("decision") != "would_merge":
        return {
            "status": "FAIL",
            "reason": "Round-3 shadow decision is not would_merge",
            "decision": decision,
            "applied": False,
            "schema_v4_merge_enabled": os.environ.get(ENABLE_ENV) == "1",
            "repository_modified": False,
            "network_used": False,
        }
    payload = validate_payload(delta, experiment_id)
    if not apply:
        return {
            "status": "PASS",
            "reason": "dry-run only; pass --apply and set DRPO_ENABLE_SCHEMA_V4_SCOPED_DELTA=1 to materialize",
            "decision": decision,
            "experiment_id": experiment_id,
            "applied": False,
            "schema_v4_merge_enabled": os.environ.get(ENABLE_ENV) == "1",
            "repository_modified": False,
            "network_used": False,
        }
    if os.environ.get(ENABLE_ENV) != "1":
        raise MergeError(f"refusing to materialize unless {ENABLE_ENV}=1")

    changes: list[dict[str, Any]] = []
    if "registry_entry" in payload:
        changes.append(upsert_registry_entry(repo, experiment_id, payload["registry_entry"]))
    if "handoff_entry" in payload:
        changes.append(upsert_handoff_entry(repo, experiment_id, payload["handoff_entry"]))

    new_scope_sha256, material = analyzer.experiment_scope_hash(repo, experiment_id)
    return {
        "status": "PASS",
        "reason": "schema-v4 scoped experiment delta materialized under explicit Round-4 gate",
        "decision": decision,
        "experiment_id": experiment_id,
        "applied": True,
        "changes": changes,
        "new_scope_sha256": new_scope_sha256,
        "registry_entry_present": material["registry_entry"] != analyzer.MISSING,
        "handoff_entry_present": material["handoff_entry"] != analyzer.MISSING,
        "schema_v4_merge_enabled": True,
        "repository_modified": True,
        "network_used": False,
    }


def _write_delta(path: Path, experiment_id: str, scope_hash: str, payload: str) -> None:
    path.write_text(
        "schema_version: 4\n"
        "delta_kind: scoped_experiment_update\n"
        "update_id: SCHEMA-V4-MATERIALIZER-SELF-TEST\n"
        "mode: scoped_candidate\n"
        f"operation:\n  type: register_or_update_experiment\n  experiment_id: {experiment_id}\n"
        f"scope:\n  type: experiment\n  id: {experiment_id}\n"
        "preimage:\n  schema: drpo-scoped-experiment-v1\n  scope_exists: true\n"
        f"  experiment_scope_sha256: \"{scope_hash}\"\n"
        "payload:\n" + payload,
        encoding="utf-8",
    )


def run_self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="drpo_scoped_merge_") as tmp:
        root = Path(tmp)
        repo = root / "repo"
        (repo / "experiments").mkdir(parents=True)
        (repo / "docs").mkdir(parents=True)
        subprocess.run(["git", "init"], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        (repo / "experiments" / "registry.yaml").write_text(
            "schema_version: 2\nexperiments:\n"
            "- id: EXT-C-E8-MERGE-TEST-01\n  status: not_run\n  name: merge test\n"
            "- id: EXT-C-E7-UNRELATED-01\n  status: not_run\n  name: unrelated test\n",
            encoding="utf-8",
        )
        (repo / "docs" / "handoff.md").write_text(
            "# Handoff\n\nEXT-C-E8-MERGE-TEST-01 is registered.\n"
            "EXT-C-E7-UNRELATED-01 is registered.\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "experiments/registry.yaml", "docs/handoff.md"], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=DRPO Scoped Merge Self-Test",
                "-c",
                "user.email=drpo-scoped-merge-self-test@local.invalid",
                "commit",
                "-m",
                "seed scoped merge self-test repo",
            ],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        scope_hash, _ = analyzer.experiment_scope_hash(repo, "EXT-C-E8-MERGE-TEST-01")
        payload = (
            "  registry_entry:\n"
            "    id: EXT-C-E8-MERGE-TEST-01\n"
            "    status: pilot\n"
            "    name: merge test updated\n"
            "  handoff_entry:\n"
            "    text: 'EXT-C-E8-MERGE-TEST-01 materialized by schema-v4 scoped merge self-test.'\n"
        )
        good_delta = root / "good.yaml"
        _write_delta(good_delta, "EXT-C-E8-MERGE-TEST-01", scope_hash, payload)
        dry_run = materialize(repo, good_delta, apply=False)

        no_env_status = "UNSET"
        old_env = os.environ.pop(ENABLE_ENV, None)
        try:
            try:
                materialize(repo, good_delta, apply=True)
            except MergeError:
                no_env_status = "FAIL"
            else:
                no_env_status = "PASS"
        finally:
            if old_env is not None:
                os.environ[ENABLE_ENV] = old_env

        os.environ[ENABLE_ENV] = "1"
        try:
            applied = materialize(repo, good_delta, apply=True)
        finally:
            if old_env is None:
                os.environ.pop(ENABLE_ENV, None)
            else:
                os.environ[ENABLE_ENV] = old_env

        # Replaying the same stale delta must fail after the target scope changed.
        replay = materialize(repo, good_delta, apply=False)

        # Different-scope drift should still be allowed for the original target.
        repo2 = root / "repo2"
        subprocess.run(["git", "clone", str(repo), str(repo2)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(["git", "checkout", "HEAD~0"], cwd=repo2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # Reset repo2 to the seed state by rebuilding it from scratch for clarity.
        repo2 = root / "repo2_seed"
        (repo2 / "experiments").mkdir(parents=True)
        (repo2 / "docs").mkdir(parents=True)
        subprocess.run(["git", "init"], cwd=repo2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        (repo2 / "experiments" / "registry.yaml").write_text(
            "schema_version: 2\nexperiments:\n"
            "- id: EXT-C-E8-MERGE-TEST-01\n  status: not_run\n  name: merge test\n"
            "- id: EXT-C-E7-UNRELATED-01\n  status: pilot\n  name: unrelated changed\n",
            encoding="utf-8",
        )
        (repo2 / "docs" / "handoff.md").write_text(
            "# Handoff\n\nEXT-C-E8-MERGE-TEST-01 is registered.\n"
            "EXT-C-E7-UNRELATED-01 is now pilot.\n",
            encoding="utf-8",
        )
        different_scope_drift = materialize(repo2, good_delta, apply=False)

        bad_payload_delta = root / "bad_payload.yaml"
        _write_delta(
            bad_payload_delta,
            "EXT-C-E8-MERGE-TEST-01",
            scope_hash,
            "  registry_entry:\n    id: EXT-C-E8-MERGE-TEST-01\n    related: EXT-C-E7-OTHER-01\n",
        )
        try:
            bad_payload = materialize(repo, bad_payload_delta, apply=False)
        except MergeError:
            bad_payload_status = "FAIL"
        else:
            bad_payload_status = str(bad_payload.get("status"))

        return {
            "status": "PASS"
            if dry_run.get("status") == "PASS"
            and dry_run.get("applied") is False
            and no_env_status == "FAIL"
            and applied.get("status") == "PASS"
            and applied.get("applied") is True
            and replay.get("status") == "FAIL"
            and different_scope_drift.get("status") == "PASS"
            and different_scope_drift.get("decision", {}).get("decision") == "would_merge"
            and bad_payload_status == "FAIL"
            else "FAIL",
            "dry_run_status": dry_run.get("status"),
            "apply_without_env_status": no_env_status,
            "apply_with_env_status": applied.get("status"),
            "replay_stale_delta_status": replay.get("status"),
            "different_scope_drift_status": different_scope_drift.get("status"),
            "different_scope_drift_decision": different_scope_drift.get("decision", {}).get("decision"),
            "bad_payload_status": bad_payload_status,
            "schema_v4_merge_enabled_default": os.environ.get(ENABLE_ENV) == "1",
            "network_used": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--delta", type=Path, help="schema-v4 HANDOFF_DELTA.yaml to dry-run/apply")
    parser.add_argument("--apply", action="store_true", help="materialize the delta; also requires DRPO_ENABLE_SCHEMA_V4_SCOPED_DELTA=1")
    parser.add_argument("--self-test", action="store_true", help="run built-in isolated self-test")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    try:
        if args.self_test:
            result = run_self_test()
        else:
            if args.delta is None:
                raise MergeError("--delta is required unless --self-test is used")
            result = materialize(args.repo, args.delta, apply=args.apply)
    except MergeError as exc:
        result = {
            "status": "FAIL",
            "error": str(exc),
            "schema_v4_merge_enabled": os.environ.get(ENABLE_ENV) == "1",
            "repository_modified": False,
            "network_used": False,
        }
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
