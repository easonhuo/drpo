#!/usr/bin/env python3
"""Validate immutable post-run evidence locators in experiments/registry.yaml.

The validator is transition-aware when --base/--head are supplied. Existing historical
experiments without a locator are grandfathered until their registry entry is changed.
Any changed or newly added delivered experiment must carry a complete locator, and an
existing locator may only be extended by appending immutable records.

The same pull-request transition also fails closed when a newly added authoritative
schema-v3 handoff delta has not been fully materialized into the PR head.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

REGISTRY_PATH = "experiments/registry.yaml"
HANDOFF_PATH = "docs/handoff.md"
HANDOFF_DELTA_ROOT = "docs/handoff_deltas"
HANDOFF_DELTA_NAME = "HANDOFF_DELTA.yaml"
MATERIALIZATION_REPORT_NAME = "MATERIALIZATION_REPORT.json"
SCHEMA_VERSION = 1
CANONICAL_RESULTS_REPOSITORY = "easonhuo/drpo-results"
CANONICAL_EXPORT_PROFILE = "manifest_text_v1"
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}")
LANE_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
SHA_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
RECORD_KEYS = {
    "run_id",
    "lane",
    "source_commit",
    "results_repository",
    "results_branch",
    "results_commit",
    "result_path",
    "manifest_sha256",
    "export_profile",
}
LOCATOR_KEYS = {"schema_version", "primary_run_id", "records"}


class EvidenceLocatorError(ValueError):
    """Raised when an evidence-locator or closure-materialization contract is violated."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> None:
    raise EvidenceLocatorError(code, message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("EVIDENCE_LOCATOR_INVALID", f"{label} must be a mapping")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("EVIDENCE_LOCATOR_INVALID", f"{label} must be a non-empty string")
    return value.strip()


def _safe_relative_path(value: Any, label: str) -> str:
    text = _string(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
        _fail("EVIDENCE_LOCATOR_INVALID", f"{label} must be a safe repository-relative path")
    normalized = path.as_posix()
    if normalized != text or text.endswith("/"):
        _fail("EVIDENCE_LOCATOR_INVALID", f"{label} must be normalized without a trailing slash")
    return normalized


def validate_locator(experiment_id: str, raw: Any) -> dict[str, Any]:
    locator = _mapping(raw, f"{experiment_id}.evidence_locator")
    unknown = sorted(set(locator) - LOCATOR_KEYS)
    missing = sorted(LOCATOR_KEYS - set(locator))
    if unknown or missing:
        _fail(
            "EVIDENCE_LOCATOR_INVALID",
            f"{experiment_id}.evidence_locator keys mismatch; missing={missing}, unknown={unknown}",
        )
    if locator["schema_version"] != SCHEMA_VERSION:
        _fail(
            "EVIDENCE_LOCATOR_INVALID",
            f"{experiment_id}.evidence_locator.schema_version must equal {SCHEMA_VERSION}",
        )
    primary_run_id = _string(
        locator["primary_run_id"], f"{experiment_id}.evidence_locator.primary_run_id"
    )
    if not ID_RE.fullmatch(primary_run_id):
        _fail("EVIDENCE_LOCATOR_INVALID", f"{experiment_id} has an invalid primary_run_id")
    records = locator["records"]
    if not isinstance(records, list) or not records:
        _fail(
            "EVIDENCE_LOCATOR_INVALID",
            f"{experiment_id}.evidence_locator.records must be non-empty",
        )

    normalized_records: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw_record in enumerate(records):
        label = f"{experiment_id}.evidence_locator.records[{index}]"
        record = _mapping(raw_record, label)
        unknown_record = sorted(set(record) - RECORD_KEYS)
        missing_record = sorted(RECORD_KEYS - set(record))
        if unknown_record or missing_record:
            _fail(
                "EVIDENCE_LOCATOR_INVALID",
                f"{label} keys mismatch; missing={missing_record}, unknown={unknown_record}",
            )
        run_id = _string(record["run_id"], f"{label}.run_id")
        lane = _string(record["lane"], f"{label}.lane")
        source_commit = _string(record["source_commit"], f"{label}.source_commit")
        repository = _string(record["results_repository"], f"{label}.results_repository")
        branch = _string(record["results_branch"], f"{label}.results_branch")
        results_commit = _string(record["results_commit"], f"{label}.results_commit")
        result_path = _safe_relative_path(record["result_path"], f"{label}.result_path")
        manifest_sha256 = _string(record["manifest_sha256"], f"{label}.manifest_sha256")
        export_profile = _string(record["export_profile"], f"{label}.export_profile")

        if not ID_RE.fullmatch(run_id) or run_id in seen:
            _fail("EVIDENCE_LOCATOR_INVALID", f"{label}.run_id is invalid or duplicated")
        if not LANE_RE.fullmatch(lane):
            _fail("EVIDENCE_LOCATOR_INVALID", f"{label}.lane is invalid")
        if not SHA_RE.fullmatch(source_commit):
            _fail(
                "EVIDENCE_LOCATOR_INVALID",
                f"{label}.source_commit must be a full lowercase Git SHA",
            )
        if repository != CANONICAL_RESULTS_REPOSITORY:
            _fail(
                "EVIDENCE_LOCATOR_INVALID",
                f"{label}.results_repository must be {CANONICAL_RESULTS_REPOSITORY}",
            )
        if branch != f"ingest/{lane}":
            _fail("EVIDENCE_LOCATOR_INVALID", f"{label}.results_branch must be ingest/{lane}")
        if not SHA_RE.fullmatch(results_commit):
            _fail(
                "EVIDENCE_LOCATOR_INVALID",
                f"{label}.results_commit must be a full lowercase Git SHA",
            )
        expected_path = f"runs/{lane}/{run_id}"
        if result_path != expected_path:
            _fail("EVIDENCE_LOCATOR_INVALID", f"{label}.result_path must be {expected_path}")
        if not SHA256_RE.fullmatch(manifest_sha256):
            _fail(
                "EVIDENCE_LOCATOR_INVALID",
                f"{label}.manifest_sha256 must be lowercase SHA-256",
            )
        if export_profile != CANONICAL_EXPORT_PROFILE:
            _fail(
                "EVIDENCE_LOCATOR_INVALID",
                f"{label}.export_profile must be {CANONICAL_EXPORT_PROFILE}",
            )
        seen.add(run_id)
        normalized_records.append(
            {
                "run_id": run_id,
                "lane": lane,
                "source_commit": source_commit,
                "results_repository": repository,
                "results_branch": branch,
                "results_commit": results_commit,
                "result_path": result_path,
                "manifest_sha256": manifest_sha256,
                "export_profile": export_profile,
            }
        )

    if primary_run_id != normalized_records[-1]["run_id"]:
        _fail(
            "EVIDENCE_LOCATOR_INVALID",
            f"{experiment_id}.evidence_locator.primary_run_id must name the last appended record",
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "primary_run_id": primary_run_id,
        "records": normalized_records,
    }


def delivery_markers(experiment: dict[str, Any]) -> list[str]:
    markers: list[str] = []
    execution = experiment.get("execution")
    if isinstance(execution, dict) and execution.get("state") == "delivered":
        markers.append("execution.state=delivered")
    if experiment.get("formal_run_status") == "delivered":
        markers.append("formal_run_status=delivered")
    evidence = experiment.get("evidence")
    if isinstance(evidence, dict):
        if evidence.get("delivered_to_user") is True:
            markers.append("evidence.delivered_to_user=true")
        legacy_result_keys = {
            "results_repository",
            "results_branch",
            "results_commit",
            "result_path",
            "manifest_sha256",
        }
        present = sorted(legacy_result_keys & set(evidence))
        if present:
            markers.append("evidence contains legacy result-locator fields: " + ",".join(present))
    return markers


def _load_registry_bytes(data: bytes, label: str) -> dict[str, dict[str, Any]]:
    try:
        payload = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        _fail("REGISTRY_INVALID", f"cannot parse {label}: {exc}")
    root = _mapping(payload, label)
    if root.get("schema_version") != 2:
        _fail("REGISTRY_INVALID", f"{label} schema_version must equal 2")
    experiments = root.get("experiments")
    if not isinstance(experiments, list):
        _fail("REGISTRY_INVALID", f"{label}.experiments must be a list")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(experiments):
        experiment = _mapping(raw, f"{label}.experiments[{index}]")
        experiment_id = _string(experiment.get("id"), f"{label}.experiments[{index}].id")
        if experiment_id in result:
            _fail("REGISTRY_INVALID", f"{label} contains duplicate experiment ID {experiment_id}")
        result[experiment_id] = experiment
    return result


def _git_show(repo: Path, ref: str, path: str) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        _fail("REGISTRY_UNAVAILABLE", f"cannot read {path} at {ref}: {detail}")
    return proc.stdout


def _git_object_exists(repo: Path, ref: str, path: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{ref}:{path}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def _added_handoff_delta_paths(repo: Path, base: str, head: str) -> list[str]:
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--name-only",
            "--diff-filter=A",
            base,
            head,
            "--",
            HANDOFF_DELTA_ROOT,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        _fail(
            "HANDOFF_MATERIALIZATION_UNAVAILABLE",
            f"cannot inspect added handoff deltas between {base} and {head}: {detail}",
        )
    paths: list[str] = []
    for raw in proc.stdout.decode("utf-8", errors="strict").splitlines():
        path = PurePosixPath(raw)
        if (
            len(path.parts) == 4
            and path.parts[0] == "docs"
            and path.parts[1] == "handoff_deltas"
            and path.parts[-1] == HANDOFF_DELTA_NAME
        ):
            paths.append(path.as_posix())
    return sorted(paths)


def _load_handoff_delta(data: bytes, label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        _fail("HANDOFF_MATERIALIZATION_INVALID", f"cannot parse {label}: {exc}")
    if not isinstance(payload, dict):
        _fail("HANDOFF_MATERIALIZATION_INVALID", f"{label} must be a mapping")
    return payload


def _validate_materialized_operations(
    update_id: str,
    operations: list[Any],
    handoff_text: str,
) -> None:
    for index, raw in enumerate(operations):
        label = f"{update_id}.operations[{index}]"
        if not isinstance(raw, dict):
            _fail("HANDOFF_MATERIALIZATION_INVALID", f"{label} must be a mapping")
        op = raw.get("op")
        if not isinstance(op, str):
            _fail("HANDOFF_MATERIALIZATION_INVALID", f"{label}.op must be a string")
        if op == "replace_heading":
            new_heading = raw.get("new_heading")
            if not isinstance(new_heading, str) or not new_heading.strip():
                _fail(
                    "HANDOFF_MATERIALIZATION_INVALID",
                    f"{label}.new_heading must be a non-empty string",
                )
            heading_pattern = re.compile(
                rf"^#{{1,6}}\s+{re.escape(new_heading.strip())}\s*$",
                re.MULTILINE,
            )
            if not heading_pattern.search(handoff_text):
                _fail(
                    "HANDOFF_MATERIALIZATION_MISSING",
                    f"{update_id} replacement heading is absent from {HANDOFF_PATH}: {new_heading}",
                )
            continue
        if op in {"insert_after_heading", "append_to_section"}:
            block_id = raw.get("block_id")
            if not isinstance(block_id, str) or not block_id.strip():
                _fail(
                    "HANDOFF_MATERIALIZATION_INVALID",
                    f"{label}.block_id must be a non-empty string",
                )
            location = "after_heading" if op == "insert_after_heading" else "section_end"
            start = f"<!-- HANDOFF-DELTA-BLOCK:{location}:{block_id}:START -->"
            end = f"<!-- HANDOFF-DELTA-BLOCK:{location}:{block_id}:END -->"
            if start not in handoff_text or end not in handoff_text:
                _fail(
                    "HANDOFF_MATERIALIZATION_MISSING",
                    f"{update_id} block {block_id} is absent from materialized {HANDOFF_PATH}",
                )
            continue
        _fail(
            "HANDOFF_MATERIALIZATION_INVALID",
            f"{label}.op is unsupported: {op!r}",
        )


def validate_handoff_materialization_transition(
    repo: Path,
    base: str,
    head: str,
) -> dict[str, Any]:
    checked: list[str] = []
    for delta_path in _added_handoff_delta_paths(repo, base, head):
        delta = _load_handoff_delta(
            _git_show(repo, head, delta_path),
            f"{delta_path} at {head}",
        )
        if delta.get("schema_version") != 3 or delta.get("mode") != "authoritative":
            continue
        update_id = delta.get("update_id")
        if not isinstance(update_id, str) or not update_id.strip():
            _fail(
                "HANDOFF_MATERIALIZATION_INVALID",
                f"{delta_path}.update_id must be a non-empty string",
            )
        expected_parent = PurePosixPath(HANDOFF_DELTA_ROOT) / update_id
        if PurePosixPath(delta_path).parent != expected_parent:
            _fail(
                "HANDOFF_MATERIALIZATION_INVALID",
                f"{delta_path} directory does not match update_id {update_id}",
            )
        report_path = (expected_parent / MATERIALIZATION_REPORT_NAME).as_posix()
        if not _git_object_exists(repo, head, report_path):
            _fail(
                "HANDOFF_MATERIALIZATION_MISSING",
                f"{update_id} has no sibling {MATERIALIZATION_REPORT_NAME} in the PR head",
            )
        operations = delta.get("operations")
        if not isinstance(operations, list):
            _fail(
                "HANDOFF_MATERIALIZATION_INVALID",
                f"{update_id}.operations must be a list",
            )
        if operations:
            base_handoff = _git_show(repo, base, HANDOFF_PATH)
            head_handoff = _git_show(repo, head, HANDOFF_PATH)
            if base_handoff == head_handoff:
                _fail(
                    "HANDOFF_MATERIALIZATION_MISSING",
                    f"{update_id} declares handoff operations but {HANDOFF_PATH} is unchanged",
                )
            try:
                handoff_text = head_handoff.decode("utf-8")
            except UnicodeError as exc:
                _fail(
                    "HANDOFF_MATERIALIZATION_INVALID",
                    f"cannot decode materialized {HANDOFF_PATH}: {exc}",
                )
            _validate_materialized_operations(update_id, operations, handoff_text)
        checked.append(update_id)
    return {
        "handoff_materialization_policy_id": "GOV-RESULT-CLOSURE-MATERIALIZATION-GATE-01",
        "checked_handoff_materialization_count": len(checked),
        "checked_handoff_materialization_ids": checked,
    }


def validate_current(registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    locator_count = 0
    grandfathered: list[str] = []
    for experiment_id, experiment in sorted(registry.items()):
        raw = experiment.get("evidence_locator")
        if raw is not None:
            validate_locator(experiment_id, raw)
            locator_count += 1
        elif delivery_markers(experiment):
            grandfathered.append(experiment_id)
    return {
        "mode": "current",
        "experiment_count": len(registry),
        "locator_count": locator_count,
        "grandfathered_missing_count": len(grandfathered),
        "grandfathered_missing_ids": grandfathered,
    }


def validate_transition(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    # Compare the parsed YAML values directly. PyYAML may resolve timestamps to
    # date/datetime objects, which are valid comparable values but are not JSON-serializable.
    changed_ids = sorted(
        experiment_id
        for experiment_id in set(before) | set(after)
        if before.get(experiment_id) != after.get(experiment_id)
    )
    checked: list[str] = []
    for experiment_id in changed_ids:
        old = before.get(experiment_id)
        new = after.get(experiment_id)
        if new is None:
            continue
        old_raw = old.get("evidence_locator") if old else None
        new_raw = new.get("evidence_locator")
        if old_raw is not None and new_raw is None:
            _fail("EVIDENCE_LOCATOR_REMOVED", f"{experiment_id} removed its evidence_locator")

        old_locator = validate_locator(experiment_id, old_raw) if old_raw is not None else None
        new_locator = validate_locator(experiment_id, new_raw) if new_raw is not None else None
        markers = delivery_markers(new)
        if markers and new_locator is None:
            _fail(
                "EVIDENCE_LOCATOR_MISSING",
                f"{experiment_id} is a changed delivered experiment but has no evidence_locator; "
                + "; ".join(markers),
            )
        if old_locator is not None and new_locator is not None:
            old_records = old_locator["records"]
            new_records = new_locator["records"]
            if len(new_records) < len(old_records) or new_records[: len(old_records)] != old_records:
                _fail(
                    "EVIDENCE_LOCATOR_MUTATED",
                    f"{experiment_id} changed or removed an immutable evidence-locator record",
                )
            if (
                old_locator["primary_run_id"] != new_locator["primary_run_id"]
                and len(new_records) == len(old_records)
            ):
                _fail(
                    "EVIDENCE_LOCATOR_MUTATED",
                    f"{experiment_id} changed primary_run_id without appending a new record",
                )
        if new_locator is not None:
            checked.append(experiment_id)
    return {
        "mode": "transition",
        "changed_experiment_count": len(changed_ids),
        "changed_experiment_ids": changed_ids,
        "checked_locator_count": len(checked),
        "checked_locator_ids": checked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if bool(args.base) != bool(args.head):
        parser.error("--base and --head must be supplied together")
    repo = Path(args.repo_root).resolve()
    try:
        if args.base:
            before = _load_registry_bytes(_git_show(repo, args.base, REGISTRY_PATH), "base registry")
            after = _load_registry_bytes(_git_show(repo, args.head, REGISTRY_PATH), "head registry")
            details = {
                **validate_transition(before, after),
                **validate_handoff_materialization_transition(repo, args.base, args.head),
            }
        else:
            details = validate_current(
                _load_registry_bytes((repo / REGISTRY_PATH).read_bytes(), "current registry")
            )
        payload = {
            "status": "PASS",
            "policy_id": "GOV-POSTRUN-EVIDENCE-LOCATOR-01",
            **details,
        }
        if args.json:
            print(json.dumps(payload, sort_keys=True, indent=2))
        else:
            print(
                "Post-run evidence locator: PASS "
                + " ".join(
                    f"{key}={value}" for key, value in details.items() if not isinstance(value, list)
                )
            )
        return 0
    except (OSError, EvidenceLocatorError) as exc:
        code = exc.code if isinstance(exc, EvidenceLocatorError) else "REGISTRY_UNAVAILABLE"
        message = exc.message if isinstance(exc, EvidenceLocatorError) else str(exc)
        payload = {
            "status": "FAIL",
            "policy_id": "GOV-POSTRUN-EVIDENCE-LOCATOR-01",
            "error_code": code,
            "error": message,
        }
        if args.json:
            print(json.dumps(payload, sort_keys=True, indent=2))
        else:
            print(f"Post-run evidence locator: FAIL [{code}] {message}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
