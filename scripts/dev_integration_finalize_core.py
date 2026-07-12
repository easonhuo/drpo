#!/usr/bin/env python3
"""Normalize, gate, and finalize a prepared DRPO dev-integration transaction.

Batch 2B consumes a Batch 2A ``PREPARED`` transaction.  It optionally applies
one separately reviewer-bound registry/handoff registration intent, invokes the
trusted current-main handoff authority, atomically amends the local source
commit, executes the canonical required gates, and records a local ``READY``
commit.  It never pushes, opens a pull request, polls CI, or merges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import yaml

import handoff_delta_shadow as shadow
from dev_integration_write_path import (
    SCHEMA,
    WritePathError,
    full_sha,
    git,
    load_json,
    load_yaml,
    locked,
    now,
    path_value,
    sha256,
    text,
    write_json,
)

VERSION = "0.3.0-batch2b"
INTENT_FILE = "REGISTRATION_INTENT.yaml"
APPROVAL_FILE = "REGISTRATION_APPROVAL.yaml"
NORMALIZATION_REPORT = "NORMALIZATION_REPORT.json"
NORMALIZATION_JOURNAL = "NORMALIZATION_JOURNAL.json"
GATE_REPORT = "GATE_REPORT.json"
READY_COMMIT = "READY_COMMIT.json"
DIAGNOSTIC = "DIAGNOSTIC.json"
REGISTRY_PATH = Path("experiments/registry.yaml")
HANDOFF_PATH = Path("docs/handoff.md")
DELTA_ROOT = Path("docs/handoff_deltas")
GENERATED_PREFIX = "docs/handoff_shadow/stage4/minimal/generated/"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}")
INTENT_KEYS = {
    "schema_version",
    "integration_id",
    "mode",
    "update_id",
    "registry_mutation",
    "handoff_operations",
    "registry_changes",
}
APPROVAL_KEYS = {
    "schema_version",
    "integration_id",
    "intent_sha256",
    "request_sha256",
    "review_decision_sha256",
    "reviewer",
}
MUTATION_KEYS = {
    "kind",
    "experiment_id",
    "expected_before_semantic_sha256",
    "experiment",
}


class FinalizeError(WritePathError):
    """Stable Batch 2B failure."""


def fail(code: str, phase: str, message: str, *recovery: str) -> None:
    raise FinalizeError(code, phase, message, tuple(recovery))


def json_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def exact_keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        fail(
            "REQUEST_INVALID",
            "registration_validation",
            f"{label} contains unknown keys: {unknown}",
        )


def mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(
            "REQUEST_INVALID",
            "registration_validation",
            f"{label} must be a mapping",
        )
    return value


def list_value(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        fail(
            "REQUEST_INVALID",
            "registration_validation",
            f"{label} must be a list",
        )
    return value


def sha256_value(
    value: Any,
    label: str,
    *,
    allow_none: bool = False,
) -> str | None:
    if value is None and allow_none:
        return None
    result = text(value, label)
    if not SHA256_RE.fullmatch(result):
        fail(
            "REQUEST_INVALID",
            "registration_validation",
            f"{label} must be SHA-256",
        )
    return result


def safe_load_yaml(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink():
        fail(
            "UNSAFE_PATH",
            "registration_validation",
            f"{label} may not be a symlink",
        )
    return load_yaml(path, label)


def read_tx_context(
    transaction_dir: Path,
    allowed_states: set[str],
) -> dict[str, Any]:
    tx_path = transaction_dir / "TRANSACTION.json"
    source_path = transaction_dir / "SOURCE_LOCK.json"
    audit_path = transaction_dir / "SCOPE_AUDIT.json"
    prepare_path = transaction_dir / "PREPARE_REPORT.json"
    tx = load_json(tx_path, "transaction")
    source = load_json(source_path, "source lock")
    audit = load_json(audit_path, "scope audit")
    prepare = load_json(prepare_path, "prepare report")
    if any(
        item.get("schema_version") != SCHEMA
        for item in (tx, source, audit, prepare)
    ):
        fail(
            "REQUEST_INVALID",
            "batch2b_preflight",
            "transaction artifact schema mismatch",
        )
    state = text(tx.get("state"), "transaction state")
    if state not in allowed_states:
        fail(
            "REQUEST_INVALID",
            "batch2b_preflight",
            f"invalid transaction state: {state}",
        )
    integration_id = text(tx.get("integration_id"), "integration_id")
    if not ID_RE.fullmatch(integration_id):
        fail(
            "REQUEST_INVALID",
            "batch2b_preflight",
            "invalid integration_id",
        )
    if (
        source.get("integration_id") != integration_id
        or audit.get("integration_id") != integration_id
    ):
        fail(
            "IMMUTABILITY_ERROR",
            "batch2b_preflight",
            "transaction identity mismatch",
        )
    main_sha = full_sha(source.get("main_sha"), "main_sha")
    dev_sha = full_sha(source.get("dev_sha"), "dev_sha")
    source_commit = full_sha(
        prepare.get("source_commit_sha"),
        "source_commit_sha",
    )
    repo = Path(
        text(prepare.get("integration_repo"), "integration_repo")
    ).expanduser().resolve()
    if repo != (transaction_dir / "integration-repo").resolve() or not repo.is_dir():
        fail(
            "IMMUTABILITY_ERROR",
            "batch2b_preflight",
            "integration repository mismatch",
        )
    if prepare.get("main_sha") != main_sha or prepare.get("dev_sha") != dev_sha:
        fail(
            "IMMUTABILITY_ERROR",
            "batch2b_preflight",
            "prepare source SHA mismatch",
        )
    if prepare.get("parent_sha") != main_sha or prepare.get("state") != "PREPARED":
        fail(
            "IMMUTABILITY_ERROR",
            "batch2b_preflight",
            "prepare report contract mismatch",
        )
    request_path = Path(
        text(tx.get("request_path"), "request_path")
    ).expanduser().resolve()
    review_rel = path_value(
        source.get("review_decision_path"),
        "review decision path",
    )
    repo_root = Path(text(tx.get("repo_root"), "repo_root")).expanduser().resolve()
    review_path = (repo_root / review_rel).resolve()
    if sha256(request_path) != source.get("request_sha256"):
        fail(
            "IMMUTABILITY_ERROR",
            "batch2b_preflight",
            "request changed after plan",
        )
    if sha256(review_path) != source.get("review_decision_sha256"):
        fail(
            "IMMUTABILITY_ERROR",
            "batch2b_preflight",
            "review changed after plan",
        )
    request = safe_load_yaml(request_path, "locked request")
    review = safe_load_yaml(review_path, "locked review")
    stored_review = mapping(
        audit.get("review_decision"),
        "stored review decision",
    )
    reviewer = mapping(stored_review.get("reviewer"), "stored reviewer")
    if review.get("reviewer") != reviewer:
        fail(
            "IMMUTABILITY_ERROR",
            "batch2b_preflight",
            "reviewer binding changed",
        )
    if (
        request.get("integration_id") != integration_id
        or review.get("integration_id") != integration_id
    ):
        fail(
            "IMMUTABILITY_ERROR",
            "batch2b_preflight",
            "input identity mismatch",
        )
    head = str(
        git(
            ["rev-parse", "HEAD"],
            cwd=repo,
            phase="batch2b_preflight",
            code="HEAD_DRIFT",
        )
    ).strip()
    interrupted_normalization = False
    if state == "PREPARED" and head != source_commit:
        journal_path = transaction_dir / NORMALIZATION_JOURNAL
        if not journal_path.is_file():
            fail(
                "HEAD_DRIFT",
                "batch2b_preflight",
                "prepared HEAD differs from source commit",
            )
        interrupted_normalization = True
    return {
        "transaction_dir": transaction_dir,
        "transaction_path": tx_path,
        "source_path": source_path,
        "audit_path": audit_path,
        "prepare_path": prepare_path,
        "transaction": tx,
        "source": source,
        "audit": audit,
        "prepare": prepare,
        "request": request,
        "review": review,
        "reviewer": reviewer,
        "integration_id": integration_id,
        "main_sha": main_sha,
        "dev_sha": dev_sha,
        "source_commit_sha": source_commit,
        "repo": repo,
        "remote": text(source.get("remote_location"), "remote_location"),
        "main_ref": text(source.get("main_ref"), "main_ref"),
        "dev_ref": text(source.get("dev_ref"), "dev_ref"),
        "requested_gate_tier": text(
            tx.get("requested_gate_tier", "auto"),
            "requested_gate_tier",
        ),
        "interrupted_normalization": interrupted_normalization,
    }


def restore_source_checkout(context: dict[str, Any], phase: str) -> None:
    git(
        ["reset", "--hard", context["source_commit_sha"]],
        cwd=context["repo"],
        phase=phase,
        code="HEAD_DRIFT",
    )
    git(
        ["clean", "-fd"],
        cwd=context["repo"],
        phase=phase,
        code="WORKTREE_DIRTY",
    )
    ensure_clean(
        context["repo"],
        context["source_commit_sha"],
        phase,
    )


def remote_refs(
    remote: str,
    refs: Sequence[str],
    *,
    phase: str,
) -> dict[str, str]:
    output = str(
        git(
            ["ls-remote", "--refs", remote, *refs],
            phase=phase,
            code="SOURCE_UNRESOLVED",
        )
    )
    found: dict[str, str] = {}
    for line in output.splitlines():
        if "\t" not in line:
            continue
        object_sha, ref = line.split("\t", 1)
        if ref in refs:
            if ref in found:
                fail(
                    "SOURCE_UNRESOLVED",
                    phase,
                    f"duplicate ref: {ref}",
                )
            found[ref] = object_sha
    missing = sorted(set(refs) - set(found))
    if missing:
        fail(
            "SOURCE_UNRESOLVED",
            phase,
            f"missing refs: {missing}",
        )
    return found


def check_freshness(context: dict[str, Any], phase: str) -> None:
    refs = remote_refs(
        context["remote"],
        [context["main_ref"], context["dev_ref"]],
        phase=phase,
    )
    if (
        refs[context["main_ref"]] != context["main_sha"]
        or refs[context["dev_ref"]] != context["dev_sha"]
    ):
        fail(
            "SOURCE_DRIFT",
            phase,
            "main or dev ref moved",
            "create a new plan attempt",
        )


def ensure_clean(repo: Path, expected_head: str, phase: str) -> None:
    head = str(
        git(
            ["rev-parse", "HEAD"],
            cwd=repo,
            phase=phase,
            code="HEAD_DRIFT",
        )
    ).strip()
    if head != expected_head:
        fail(
            "HEAD_DRIFT",
            phase,
            f"HEAD mismatch: expected {expected_head}, found {head}",
        )
    status = str(
        git(
            ["status", "--porcelain=v1"],
            cwd=repo,
            phase=phase,
            code="WORKTREE_DIRTY",
        )
    )
    if status:
        fail(
            "WORKTREE_DIRTY",
            phase,
            "integration repository is dirty",
        )


def ensure_one_parent(
    repo: Path,
    commit_sha: str,
    main_sha: str,
    phase: str,
) -> str:
    row = str(
        git(
            ["rev-list", "--parents", "-n", "1", commit_sha],
            cwd=repo,
            phase=phase,
            code="HEAD_DRIFT",
        )
    ).strip().split()
    if row != [commit_sha, main_sha]:
        fail(
            "HEAD_DRIFT",
            phase,
            "ready commit must have locked main as its unique parent",
        )
    return str(
        git(
            ["rev-parse", f"{commit_sha}^{{tree}}"],
            cwd=repo,
            phase=phase,
            code="HEAD_DRIFT",
        )
    ).strip()


def trusted_main(context: dict[str, Any]) -> Path:
    path = context["transaction_dir"] / "trusted-main"
    if path.exists():
        head = str(
            git(
                ["rev-parse", "HEAD"],
                cwd=path,
                phase="trusted_main",
                code="HEAD_DRIFT",
            )
        ).strip()
        status = str(
            git(
                ["status", "--porcelain=v1"],
                cwd=path,
                phase="trusted_main",
                code="WORKTREE_DIRTY",
            )
        )
        if head != context["main_sha"] or status:
            fail(
                "IMMUTABILITY_ERROR",
                "trusted_main",
                "trusted-main worktree drifted",
            )
        return path
    git(
        ["worktree", "add", "--detach", str(path), context["main_sha"]],
        cwd=context["repo"],
        phase="trusted_main",
        code="INTERNAL_ERROR",
        timeout=300,
    )
    ensure_clean(path, context["main_sha"], "trusted_main")
    return path


def validate_intent(
    transaction_dir: Path,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    intent_path = transaction_dir / INTENT_FILE
    approval_path = transaction_dir / APPROVAL_FILE
    if not intent_path.exists() and not approval_path.exists():
        return None
    if not intent_path.is_file() or not approval_path.is_file():
        fail(
            "REGISTRY_STRUCTURE_ERROR",
            "registration_validation",
            "registration intent and approval must both exist",
        )
    intent = safe_load_yaml(intent_path, "registration intent")
    approval = safe_load_yaml(approval_path, "registration approval")
    exact_keys(intent, INTENT_KEYS, "registration intent")
    exact_keys(approval, APPROVAL_KEYS, "registration approval")
    if intent.get("schema_version") != 1 or approval.get("schema_version") != 1:
        fail(
            "REQUEST_INVALID",
            "registration_validation",
            "registration schema_version must equal 1",
        )
    if (
        intent.get("integration_id") != context["integration_id"]
        or approval.get("integration_id") != context["integration_id"]
    ):
        fail(
            "IMMUTABILITY_ERROR",
            "registration_validation",
            "registration identity mismatch",
        )
    if intent.get("mode") != "authoritative_delta":
        fail(
            "REQUEST_INVALID",
            "registration_validation",
            "registration mode must be authoritative_delta",
        )
    update_id = text(intent.get("update_id"), "update_id")
    if not ID_RE.fullmatch(update_id):
        fail(
            "REQUEST_INVALID",
            "registration_validation",
            "invalid update_id",
        )
    if sha256(intent_path) != sha256_value(
        approval.get("intent_sha256"),
        "intent_sha256",
    ):
        fail(
            "IMMUTABILITY_ERROR",
            "registration_validation",
            "registration intent hash mismatch",
        )
    if approval.get("request_sha256") != context["source"].get("request_sha256"):
        fail(
            "IMMUTABILITY_ERROR",
            "registration_validation",
            "approval request hash mismatch",
        )
    if approval.get("review_decision_sha256") != context["source"].get(
        "review_decision_sha256"
    ):
        fail(
            "IMMUTABILITY_ERROR",
            "registration_validation",
            "approval review hash mismatch",
        )
    approval_reviewer = mapping(
        approval.get("reviewer"),
        "approval reviewer",
    )
    if approval_reviewer != context["reviewer"]:
        fail(
            "REVIEW_DECISION_INVALID",
            "registration_validation",
            "registration reviewer binding mismatch",
        )
    mutation = mapping(
        intent.get("registry_mutation"),
        "registry_mutation",
    )
    exact_keys(mutation, MUTATION_KEYS, "registry_mutation")
    kind = text(mutation.get("kind"), "registry_mutation.kind")
    if kind not in {"add_experiment", "replace_experiment"}:
        fail(
            "REGISTRY_STRUCTURE_ERROR",
            "registration_validation",
            f"unsupported registry mutation: {kind}",
        )
    experiment_id = text(
        mutation.get("experiment_id"),
        "registry_mutation.experiment_id",
    )
    subject = mapping(
        context["request"].get("subject"),
        "locked request subject",
    )
    experiment_ids = list_value(
        subject.get("experiment_ids", []),
        "locked request subject.experiment_ids",
    )
    if experiment_id not in experiment_ids:
        fail(
            "SCOPE_VIOLATION",
            "registration_validation",
            f"registration target is outside the reviewed experiment scope: {experiment_id}",
        )
    experiment = mapping(
        mutation.get("experiment"),
        "registry_mutation.experiment",
    )
    if experiment.get("id") != experiment_id:
        fail(
            "REGISTRY_STRUCTURE_ERROR",
            "registration_validation",
            "experiment id mismatch",
        )
    before_hash = sha256_value(
        mutation.get("expected_before_semantic_sha256"),
        "registry_mutation.expected_before_semantic_sha256",
        allow_none=True,
    )
    if kind == "add_experiment" and before_hash is not None:
        fail(
            "REGISTRY_STRUCTURE_ERROR",
            "registration_validation",
            "add_experiment may not define expected_before_semantic_sha256",
        )
    if kind == "replace_experiment" and before_hash is None:
        fail(
            "REGISTRY_STRUCTURE_ERROR",
            "registration_validation",
            "replace_experiment requires expected_before_semantic_sha256",
        )
    operations = list_value(
        intent.get("handoff_operations"),
        "handoff_operations",
    )
    changes = list_value(
        intent.get("registry_changes"),
        "registry_changes",
    )
    if not changes or not all(isinstance(item, dict) for item in changes):
        fail(
            "REGISTRY_STRUCTURE_ERROR",
            "registration_validation",
            "registry_changes must contain mappings",
        )
    for index, change in enumerate(changes):
        if change.get("entity_id") != experiment_id:
            fail(
                "SCOPE_VIOLATION",
                "registration_validation",
                f"registry_changes[{index}] targets an unreviewed experiment",
            )
    return {
        "path": intent_path,
        "approval_path": approval_path,
        "intent_sha256": sha256(intent_path),
        "approval_sha256": sha256(approval_path),
        "update_id": update_id,
        "mutation": {
            "kind": kind,
            "experiment_id": experiment_id,
            "expected_before_semantic_sha256": before_hash,
            "experiment": experiment,
        },
        "handoff_operations": operations,
        "registry_changes": changes,
    }


def _root_mapping_node(document: yaml.Node) -> yaml.MappingNode:
    if not isinstance(document, yaml.MappingNode):
        fail(
            "REGISTRY_STRUCTURE_ERROR",
            "registry_mutation",
            "registry root must be a mapping",
        )
    return document


def _experiments_node(document: yaml.MappingNode) -> yaml.SequenceNode:
    for key, value in document.value:
        if isinstance(key, yaml.ScalarNode) and key.value == "experiments":
            if not isinstance(value, yaml.SequenceNode):
                fail(
                    "REGISTRY_STRUCTURE_ERROR",
                    "registry_mutation",
                    "experiments must be a sequence",
                )
            return value
    fail(
        "REGISTRY_STRUCTURE_ERROR",
        "registry_mutation",
        "registry lacks experiments sequence",
    )


def _experiment_id_from_node(node: yaml.Node) -> str | None:
    if not isinstance(node, yaml.MappingNode):
        return None
    for key, value in node.value:
        if (
            isinstance(key, yaml.ScalarNode)
            and key.value == "id"
            and isinstance(value, yaml.ScalarNode)
        ):
            return value.value
    return None


def dump_experiment(experiment: dict[str, Any]) -> str:
    rendered = yaml.safe_dump(
        [experiment],
        sort_keys=False,
        allow_unicode=True,
        width=1000,
    )
    if not rendered.endswith("\n"):
        rendered += "\n"
    return rendered


def validate_registry_payload(
    payload: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = mapping(payload, "registry")
    if root.get("schema_version") != 2:
        fail(
            "REGISTRY_STRUCTURE_ERROR",
            "registry_mutation",
            "registry schema_version must equal 2",
        )
    experiments = root.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        fail(
            "REGISTRY_STRUCTURE_ERROR",
            "registry_mutation",
            "registry experiments must be non-empty",
        )
    ids: list[str] = []
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            fail(
                "REGISTRY_STRUCTURE_ERROR",
                "registry_mutation",
                f"experiment {index} must be a mapping",
            )
        ids.append(text(experiment.get("id"), f"experiments[{index}].id"))
    if len(ids) != len(set(ids)):
        fail(
            "REGISTRY_STRUCTURE_ERROR",
            "registry_mutation",
            "registry contains duplicate experiment ids",
        )
    return root, experiments


def apply_registry_mutation(
    registry_text: str,
    mutation: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    try:
        before_payload = yaml.safe_load(registry_text)
        document = yaml.compose(registry_text)
    except yaml.YAMLError as exc:
        fail(
            "REGISTRY_STRUCTURE_ERROR",
            "registry_mutation",
            f"invalid registry YAML: {exc}",
        )
    before_root, before_experiments = validate_registry_payload(before_payload)
    root_node = _root_mapping_node(document)
    sequence = _experiments_node(root_node)
    node_by_id: dict[str, yaml.Node] = {}
    for node in sequence.value:
        experiment_id = _experiment_id_from_node(node)
        if experiment_id is None or experiment_id in node_by_id:
            fail(
                "REGISTRY_STRUCTURE_ERROR",
                "registry_mutation",
                "registry AST experiment identity is invalid",
            )
        node_by_id[experiment_id] = node
    experiment_id = mutation["experiment_id"]
    rendered = dump_experiment(mutation["experiment"])
    if mutation["kind"] == "add_experiment":
        if experiment_id in node_by_id:
            fail(
                "REGISTRY_STRUCTURE_ERROR",
                "registry_mutation",
                f"experiment already exists: {experiment_id}",
            )
        insert_at = sequence.end_mark.index
        prefix = registry_text[:insert_at]
        suffix = registry_text[insert_at:]
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        after_text = prefix + rendered + suffix
    else:
        node = node_by_id.get(experiment_id)
        if node is None:
            fail(
                "REGISTRY_STRUCTURE_ERROR",
                "registry_mutation",
                f"experiment does not exist: {experiment_id}",
            )
        existing = next(
            item
            for item in before_experiments
            if item.get("id") == experiment_id
        )
        if json_hash(existing) != mutation["expected_before_semantic_sha256"]:
            fail(
                "IMMUTABILITY_ERROR",
                "registry_mutation",
                "experiment before-image hash mismatch",
            )
        start = registry_text.rfind("\n", 0, node.start_mark.index) + 1
        end = node.end_mark.index
        while end > start:
            probe = end
            if probe > start and registry_text[probe - 1] == "\n":
                probe -= 1
            line_start = registry_text.rfind("\n", start, probe) + 1
            line = registry_text[line_start:probe]
            if line.strip() and not line.lstrip().startswith("#"):
                break
            end = line_start
        after_text = registry_text[:start] + rendered + registry_text[end:]
    try:
        after_payload = yaml.safe_load(after_text)
    except yaml.YAMLError as exc:
        fail(
            "REGISTRY_STRUCTURE_ERROR",
            "registry_mutation",
            f"mutated registry is invalid: {exc}",
        )
    after_root, after_experiments = validate_registry_payload(after_payload)
    before_other = {
        key: value
        for key, value in before_root.items()
        if key != "experiments"
    }
    after_other = {
        key: value
        for key, value in after_root.items()
        if key != "experiments"
    }
    if before_other != after_other:
        fail(
            "REGISTRY_STRUCTURE_ERROR",
            "registry_mutation",
            "registry mutation changed non-experiment fields",
        )
    before_map = {item["id"]: item for item in before_experiments}
    after_map = {item["id"]: item for item in after_experiments}
    expected_ids = set(before_map)
    if mutation["kind"] == "add_experiment":
        expected_ids.add(experiment_id)
    if set(after_map) != expected_ids:
        fail(
            "REGISTRY_STRUCTURE_ERROR",
            "registry_mutation",
            "registry entity set changed unexpectedly",
        )
    for key, value in before_map.items():
        if key != experiment_id and after_map.get(key) != value:
            fail(
                "REGISTRY_STRUCTURE_ERROR",
                "registry_mutation",
                f"unrelated experiment changed: {key}",
            )
    if after_map.get(experiment_id) != mutation["experiment"]:
        fail(
            "REGISTRY_STRUCTURE_ERROR",
            "registry_mutation",
            "target experiment after-image mismatch",
        )
    return after_text, {
        "kind": mutation["kind"],
        "experiment_id": experiment_id,
        "before_semantic_sha256": (
            None
            if mutation["kind"] == "add_experiment"
            else mutation["expected_before_semantic_sha256"]
        ),
        "after_sha256": json_hash(after_map[experiment_id]),
        "registry_after_sha256": shadow.sha256_text(after_text),
    }


def build_delta(
    *,
    main_sha: str,
    base_handoff: str,
    base_registry: str,
    after_registry: str,
    intent: dict[str, Any],
) -> dict[str, Any]:
    try:
        rendered = shadow.render(
            base_handoff,
            intent["handoff_operations"],
        )
        shadow.verify_history_preservation(
            base_handoff,
            rendered.text,
            intent["handoff_operations"],
        )
    except shadow.HandoffDeltaError as exc:
        fail(
            "DELTA_VALIDATION_ERROR",
            "delta_generation",
            str(exc),
        )
    return {
        "schema_version": 3,
        "update_id": intent["update_id"],
        "mode": "authoritative",
        "base": {
            "commit": main_sha,
            "handoff_sha256": shadow.sha256_text(base_handoff),
            "registry_sha256": shadow.sha256_text(base_registry),
        },
        "renderer_version": 1,
        "operations": intent["handoff_operations"],
        "registry": {
            "mode": "expected_after",
            "exact_base_after_sha256": shadow.sha256_text(after_registry),
            "changes": intent["registry_changes"],
        },
        "expected": {
            "exact_base_candidate_sha256": shadow.sha256_text(rendered.text)
        },
    }


def command(
    args: Sequence[str],
    *,
    cwd: Path,
    phase: str,
    code: str,
    timeout: int = 900,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.time()
    process_env = os.environ.copy()
    process_env.setdefault("GIT_TERMINAL_PROMPT", "0")
    if env:
        process_env.update(env)
    try:
        proc = subprocess.run(
            list(args),
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            env=process_env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        fail(
            code,
            phase,
            f"command failed to start or timed out: {' '.join(args)}: {exc}",
        )
    return {
        "command": list(args),
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "duration_seconds": round(time.time() - started, 6),
    }


def require_command(
    result: dict[str, Any],
    phase: str,
    code: str,
) -> dict[str, Any]:
    if result["returncode"] != 0:
        detail = (result["stderr"] or result["stdout"] or "").strip()[-4000:]
        fail(
            code,
            phase,
            f"command failed ({result['returncode']}): {' '.join(result['command'])}: {detail}",
        )
    return result


def authority_normalize(
    context: dict[str, Any],
    trusted: Path,
    authored_commit: str,
) -> dict[str, Any]:
    result = command(
        [
            sys.executable,
            str(trusted / "scripts/handoff_authority.py"),
            "normalize",
            "--repo-root",
            str(context["repo"]),
            "--trusted-repo-root",
            str(trusted),
            "--current-before",
            context["main_sha"],
            "--source-base",
            context["main_sha"],
            "--source-patch-commit",
            authored_commit,
            "--json",
        ],
        cwd=context["repo"],
        phase="normalization",
        code="NORMALIZATION_ERROR",
        timeout=900,
    )
    require_command(result, "normalization", "NORMALIZATION_ERROR")
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError as exc:
        fail(
            "NORMALIZATION_ERROR",
            "normalization",
            f"authority returned invalid JSON: {exc}",
        )


def authority_verify(
    context: dict[str, Any],
    trusted: Path,
) -> dict[str, Any]:
    result = command(
        [
            sys.executable,
            str(trusted / "scripts/handoff_authority.py"),
            "verify",
            "--repo-root",
            str(context["repo"]),
            "--json",
        ],
        cwd=context["repo"],
        phase="authority_verify",
        code="NORMALIZATION_ERROR",
        timeout=900,
    )
    require_command(result, "authority_verify", "NORMALIZATION_ERROR")
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError as exc:
        fail(
            "NORMALIZATION_ERROR",
            "authority_verify",
            f"authority verify returned invalid JSON: {exc}",
        )


def amend(repo: Path, phase: str) -> str:
    identity = {
        "GIT_AUTHOR_NAME": "DRPO Integration Tool",
        "GIT_AUTHOR_EMAIL": "drpo-integration@local.invalid",
        "GIT_COMMITTER_NAME": "DRPO Integration Tool",
        "GIT_COMMITTER_EMAIL": "drpo-integration@local.invalid",
    }
    git(
        ["diff", "--cached", "--check"],
        cwd=repo,
        phase=phase,
        code="SCOPE_VIOLATION",
    )
    git(
        ["commit", "--amend", "--no-edit"],
        cwd=repo,
        phase=phase,
        code="INTERNAL_ERROR",
        env=identity,
    )
    return str(
        git(
            ["rev-parse", "HEAD"],
            cwd=repo,
            phase=phase,
            code="HEAD_DRIFT",
        )
    ).strip()


def changed_paths(repo: Path, base: str, head: str) -> list[str]:
    output = str(
        git(
            ["diff", "--name-only", f"{base}..{head}"],
            cwd=repo,
            phase="scope_verify",
            code="SCOPE_VIOLATION",
        )
    )
    return [line for line in output.splitlines() if line.strip()]


def normalized_output_allowed(
    path: str,
    intent: dict[str, Any] | None,
) -> bool:
    if intent is None:
        return False
    report_path = (
        f"docs/handoff_deltas/{intent['update_id']}/MATERIALIZATION_REPORT.json"
    )
    return (
        path == HANDOFF_PATH.as_posix()
        or path == report_path
        or path.startswith(GENERATED_PREFIX)
    )


def expected_final_scope(
    context: dict[str, Any],
    intent: dict[str, Any] | None,
    final_paths: Sequence[str],
) -> None:
    original = {
        item["destination_path"] or item["source_path"]
        for item in context["prepare"].get("committed_changes", [])
    }
    actual = set(final_paths)
    if intent is None:
        if actual != original:
            fail(
                "SCOPE_VIOLATION",
                "normalization_verify",
                "code-only final scope drifted",
            )
        return
    delta_path = (
        f"docs/handoff_deltas/{intent['update_id']}/HANDOFF_DELTA.yaml"
    )
    report_path = (
        f"docs/handoff_deltas/{intent['update_id']}/MATERIALIZATION_REPORT.json"
    )
    required = original | {
        REGISTRY_PATH.as_posix(),
        delta_path,
        report_path,
    }
    if intent["handoff_operations"]:
        required.add(HANDOFF_PATH.as_posix())
    missing = sorted(required - actual)
    allowed = required | {HANDOFF_PATH.as_posix()}
    unexpected = sorted(
        path
        for path in actual
        if path not in allowed and not path.startswith(GENERATED_PREFIX)
    )
    if missing or unexpected:
        fail(
            "SCOPE_VIOLATION",
            "normalization_verify",
            f"final scope mismatch (missing={missing}, unexpected={unexpected})",
        )


def preserve_commit_ref(
    context: dict[str, Any],
    label: str,
    commit_sha: str,
) -> str:
    fragment = re.sub(
        r"[^a-z0-9._-]+",
        "-",
        context["integration_id"].lower(),
    ).strip("-")
    ref = (
        f"refs/drpo-integration/{label}/{fragment}/"
        f"{context['transaction_dir'].name}"
    )
    git(
        ["update-ref", ref, commit_sha],
        cwd=context["repo"],
        phase="normalization_refs",
        code="INTERNAL_ERROR",
    )
    return ref


def recover_failed_normalization(
    transaction_dir: Path,
    context: dict[str, Any] | None,
) -> tuple[str, ...]:
    if context is None or (transaction_dir / NORMALIZATION_REPORT).exists():
        return ()
    journal_path = transaction_dir / NORMALIZATION_JOURNAL
    try:
        restore_source_checkout(
            context,
            "normalization_failure_cleanup",
        )
    except WritePathError as cleanup_error:
        if journal_path.is_file():
            journal = load_json(journal_path, "normalization journal")
            journal.update(
                {
                    "status": "CLEANUP_FAILED",
                    "cleanup_error": cleanup_error.message,
                    "failed_at": now(),
                }
            )
            write_json(journal_path, journal)
        return (f"normalization cleanup failed: {cleanup_error.message}",)
    if journal_path.is_file():
        journal = load_json(journal_path, "normalization journal")
        journal.update(
            {
                "status": "FAILED_RESTORED_TO_PREPARED",
                "failed_at": now(),
            }
        )
        write_json(journal_path, journal)
    return (
        "integration repository restored to the immutable PREPARED source commit",
    )


def write_diagnostic(
    transaction_dir: Path,
    context: dict[str, Any] | None,
    error: FinalizeError,
) -> None:
    tx_path = transaction_dir / "TRANSACTION.json"
    try:
        tx = load_json(tx_path, "transaction")
    except WritePathError:
        tx = {"schema_version": SCHEMA}
    state = "STALE" if error.error_code == "SOURCE_DRIFT" else "BLOCKED"
    timestamp = now()
    payload = {
        "schema_version": SCHEMA,
        "tool_version": VERSION,
        "status": "FAIL",
        "state": state,
        "error_code": error.error_code,
        "phase": error.phase,
        "message": error.message,
        "recovery": list(error.recovery),
        "integration_id": (
            None if context is None else context.get("integration_id")
        ),
        "main_sha": None if context is None else context.get("main_sha"),
        "dev_sha": None if context is None else context.get("dev_sha"),
        "transaction_dir": str(transaction_dir),
        "created_at": timestamp,
    }
    write_json(transaction_dir / DIAGNOSTIC, payload)
    tx.update(
        {
            "schema_version": SCHEMA,
            "tool_version": VERSION,
            "state": state,
            "status": "FAIL",
            "error_code": error.error_code,
            "phase": error.phase,
            "updated_at": timestamp,
            "next_action": (
                "create a new plan attempt"
                if state == "STALE"
                else "inspect diagnostic and create a new attempt"
            ),
        }
    )
    write_json(tx_path, tx)


def verify_normalized(
    context: dict[str, Any],
    trusted: Path,
) -> dict[str, Any]:
    report_path = context["transaction_dir"] / NORMALIZATION_REPORT
    report = load_json(report_path, "normalization report")
    recorded_hash = context["transaction"].get("normalization_report_sha256")
    if recorded_hash is not None and recorded_hash != sha256(report_path):
        fail(
            "IMMUTABILITY_ERROR",
            "normalization_idempotence",
            "normalization report hash drifted",
        )
    normalized_commit = full_sha(
        report.get("normalized_commit_sha"),
        "normalized_commit_sha",
    )
    if (
        report.get("integration_id") != context["integration_id"]
        or report.get("main_sha") != context["main_sha"]
    ):
        fail(
            "IMMUTABILITY_ERROR",
            "normalization_idempotence",
            "normalization report identity mismatch",
        )
    ensure_clean(
        context["repo"],
        normalized_commit,
        "normalization_idempotence",
    )
    tree = ensure_one_parent(
        context["repo"],
        normalized_commit,
        context["main_sha"],
        "normalization_idempotence",
    )
    if report.get("tree_sha") != tree:
        fail(
            "HEAD_DRIFT",
            "normalization_idempotence",
            "normalized tree drifted",
        )
    verify = authority_verify(context, trusted)
    if report.get("authority_verify") != verify:
        fail(
            "IMMUTABILITY_ERROR",
            "normalization_idempotence",
            "authority verify payload drifted",
        )
    return {
        "status": "PASS",
        "state": "NORMALIZED",
        "integration_id": context["integration_id"],
        "normalized_commit_sha": normalized_commit,
        "idempotent": True,
    }


def normalize_transaction(transaction_dir: Path) -> dict[str, Any]:
    transaction_dir = transaction_dir.expanduser().resolve()
    context: dict[str, Any] | None = None
    try:
        with locked(transaction_dir):
            context = read_tx_context(
                transaction_dir,
                {"PREPARED", "NORMALIZED"},
            )
            if context["interrupted_normalization"]:
                restore_source_checkout(
                    context,
                    "normalization_recovery",
                )
                recovery_journal = load_json(
                    transaction_dir / NORMALIZATION_JOURNAL,
                    "normalization journal",
                )
                recovery_journal.update(
                    {
                        "status": "RECOVERED_TO_PREPARED",
                        "recovered_at": now(),
                    }
                )
                write_json(
                    transaction_dir / NORMALIZATION_JOURNAL,
                    recovery_journal,
                )
            trusted = trusted_main(context)
            if (
                context["transaction"]["state"] == "NORMALIZED"
                or (transaction_dir / NORMALIZATION_REPORT).exists()
            ):
                result = verify_normalized(context, trusted)
                tx = context["transaction"]
                if tx.get("state") != "NORMALIZED":
                    tx.update(
                        {
                            "state": "NORMALIZED",
                            "status": "PASS",
                            "normalized_commit_sha": result[
                                "normalized_commit_sha"
                            ],
                            "normalization_report_sha256": sha256(
                                transaction_dir / NORMALIZATION_REPORT
                            ),
                            "updated_at": now(),
                            "next_action": "run Batch 2B gate",
                        }
                    )
                    write_json(context["transaction_path"], tx)
                return result
            ensure_clean(
                context["repo"],
                context["source_commit_sha"],
                "normalization_preflight",
            )
            check_freshness(context, "normalization_freshness")
            intent = validate_intent(transaction_dir, context)
            journal_path = transaction_dir / NORMALIZATION_JOURNAL
            journal = {
                "schema_version": SCHEMA,
                "tool_version": VERSION,
                "status": "IN_PROGRESS",
                "integration_id": context["integration_id"],
                "source_commit_sha": context["source_commit_sha"],
                "intent_sha256": (
                    None if intent is None else intent["intent_sha256"]
                ),
                "approval_sha256": (
                    None if intent is None else intent["approval_sha256"]
                ),
                "started_at": now(),
            }
            write_json(journal_path, journal)
            authored_commit = context["source_commit_sha"]
            mutation_report = None
            delta_path: Path | None = None
            if intent is not None:
                registry_file = context["repo"] / REGISTRY_PATH
                base_registry = str(
                    git(
                        [
                            "show",
                            f"{context['main_sha']}:{REGISTRY_PATH.as_posix()}",
                        ],
                        cwd=context["repo"],
                        phase="registry_mutation",
                        code="REGISTRY_STRUCTURE_ERROR",
                    )
                )
                base_handoff = str(
                    git(
                        [
                            "show",
                            f"{context['main_sha']}:{HANDOFF_PATH.as_posix()}",
                        ],
                        cwd=context["repo"],
                        phase="delta_generation",
                        code="DELTA_VALIDATION_ERROR",
                    )
                )
                if registry_file.read_text(encoding="utf-8") != base_registry:
                    fail(
                        "IMMUTABILITY_ERROR",
                        "registry_mutation",
                        "source commit unexpectedly changed registry",
                    )
                after_registry, mutation_report = apply_registry_mutation(
                    base_registry,
                    intent["mutation"],
                )
                delta = build_delta(
                    main_sha=context["main_sha"],
                    base_handoff=base_handoff,
                    base_registry=base_registry,
                    after_registry=after_registry,
                    intent=intent,
                )
                registry_file.write_text(
                    after_registry,
                    encoding="utf-8",
                )
                delta_path = (
                    context["repo"]
                    / DELTA_ROOT
                    / intent["update_id"]
                    / "HANDOFF_DELTA.yaml"
                )
                if delta_path.exists():
                    fail(
                        "IMMUTABILITY_ERROR",
                        "delta_generation",
                        f"delta already exists: {intent['update_id']}",
                    )
                delta_path.parent.mkdir(parents=True)
                delta_path.write_text(
                    yaml.safe_dump(
                        delta,
                        sort_keys=False,
                        allow_unicode=True,
                        width=1000,
                    ),
                    encoding="utf-8",
                )
                git(
                    [
                        "add",
                        "--",
                        REGISTRY_PATH.as_posix(),
                        delta_path.relative_to(context["repo"]).as_posix(),
                    ],
                    cwd=context["repo"],
                    phase="authored_amend",
                    code="INTERNAL_ERROR",
                )
                authored_commit = amend(
                    context["repo"],
                    "authored_amend",
                )
                ensure_one_parent(
                    context["repo"],
                    authored_commit,
                    context["main_sha"],
                    "authored_amend",
                )
            source_ref = preserve_commit_ref(
                context,
                "source",
                context["source_commit_sha"],
            )
            authored_ref = preserve_commit_ref(
                context,
                "authored",
                authored_commit,
            )
            journal.update(
                {
                    "authored_commit_sha": authored_commit,
                    "source_commit_ref": source_ref,
                    "authored_commit_ref": authored_ref,
                    "authored_at": now(),
                }
            )
            write_json(journal_path, journal)
            authority_payload = authority_normalize(
                context,
                trusted,
                authored_commit,
            )
            status_output = str(
                git(
                    ["status", "--porcelain=v1"],
                    cwd=context["repo"],
                    phase="normalization_scope",
                    code="WORKTREE_DIRTY",
                )
            )
            uncommitted = [
                line[3:]
                for line in status_output.splitlines()
                if len(line) >= 4
            ]
            if intent is None and uncommitted:
                fail(
                    "NORMALIZATION_ERROR",
                    "normalization_scope",
                    f"code-only normalization changed files: {uncommitted}",
                )
            if intent is not None:
                unexpected = sorted(
                    path
                    for path in uncommitted
                    if not normalized_output_allowed(path, intent)
                )
                if unexpected:
                    fail(
                        "SCOPE_VIOLATION",
                        "normalization_scope",
                        f"normalizer changed unexpected files: {unexpected}",
                    )
            normalized_commit = authored_commit
            if uncommitted:
                git(
                    ["add", "-A"],
                    cwd=context["repo"],
                    phase="normalization_amend",
                    code="INTERNAL_ERROR",
                )
                normalized_commit = amend(
                    context["repo"],
                    "normalization_amend",
                )
            ensure_clean(
                context["repo"],
                normalized_commit,
                "normalization_verify",
            )
            tree_sha = ensure_one_parent(
                context["repo"],
                normalized_commit,
                context["main_sha"],
                "normalization_verify",
            )
            verify = authority_verify(context, trusted)
            final_paths = changed_paths(
                context["repo"],
                context["main_sha"],
                normalized_commit,
            )
            expected_final_scope(context, intent, final_paths)
            report = {
                "schema_version": SCHEMA,
                "tool_version": VERSION,
                "status": "PASS",
                "state": "NORMALIZED",
                "integration_id": context["integration_id"],
                "main_sha": context["main_sha"],
                "dev_sha": context["dev_sha"],
                "source_commit_sha": context["source_commit_sha"],
                "authored_commit_sha": authored_commit,
                "source_commit_ref": source_ref,
                "authored_commit_ref": authored_ref,
                "normalized_commit_sha": normalized_commit,
                "tree_sha": tree_sha,
                "registration_mode": (
                    "code_only"
                    if intent is None
                    else "authoritative_delta"
                ),
                "registration_intent_sha256": (
                    None if intent is None else intent["intent_sha256"]
                ),
                "registration_approval_sha256": (
                    None if intent is None else intent["approval_sha256"]
                ),
                "update_id": None if intent is None else intent["update_id"],
                "registry_mutation": mutation_report,
                "delta_path": (
                    None
                    if delta_path is None
                    else delta_path.relative_to(context["repo"]).as_posix()
                ),
                "authority_normalize": authority_payload,
                "authority_verify": verify,
                "changed_paths": final_paths,
                "normalized_at": now(),
            }
            write_json(
                transaction_dir / NORMALIZATION_REPORT,
                report,
            )
            journal.update(
                {
                    "status": "COMPLETE",
                    "normalized_commit_sha": normalized_commit,
                    "completed_at": now(),
                }
            )
            write_json(journal_path, journal)
            tx = context["transaction"]
            completed = list(tx.get("completed_states", []))
            if "NORMALIZED" not in completed:
                completed.append("NORMALIZED")
            tx.update(
                {
                    "tool_version": VERSION,
                    "state": "NORMALIZED",
                    "status": "PASS",
                    "completed_states": completed,
                    "normalized_commit_sha": normalized_commit,
                    "normalization_report_sha256": sha256(
                        transaction_dir / NORMALIZATION_REPORT
                    ),
                    "updated_at": now(),
                    "next_action": "run Batch 2B gate",
                }
            )
            write_json(context["transaction_path"], tx)
            return {
                "status": "PASS",
                "state": "NORMALIZED",
                "integration_id": context["integration_id"],
                "normalized_commit_sha": normalized_commit,
                "registration_mode": report["registration_mode"],
                "idempotent": False,
            }
    except FinalizeError as error:
        cleanup = recover_failed_normalization(transaction_dir, context)
        wrapped = FinalizeError(
            error.error_code,
            error.phase,
            error.message,
            tuple(error.recovery) + cleanup,
        )
        write_diagnostic(transaction_dir, context, wrapped)
        raise wrapped from error
    except WritePathError as error:
        cleanup = recover_failed_normalization(transaction_dir, context)
        wrapped = FinalizeError(
            error.error_code,
            error.phase,
            error.message,
            tuple(error.recovery) + cleanup,
        )
        write_diagnostic(transaction_dir, context, wrapped)
        raise wrapped from error
    except Exception as exc:
        cleanup = recover_failed_normalization(transaction_dir, context)
        error = FinalizeError(
            "INTERNAL_ERROR",
            "normalization_internal",
            f"unexpected error: {exc}",
            cleanup,
        )
        write_diagnostic(transaction_dir, context, error)
        raise error from exc


def gate_record(
    label: str,
    args: Sequence[str],
    *,
    cwd: Path,
    log_dir: Path,
    timeout: int = 1800,
) -> dict[str, Any]:
    result = command(
        args,
        cwd=cwd,
        phase="gate",
        code="GATE_FAILURE",
        timeout=timeout,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-._")[:80] or "gate"
    log_path = log_dir / f"{safe}.log"
    log_path.write_text(
        "$ "
        + " ".join(result["command"])
        + "\n"
        + result["stdout"]
        + result["stderr"]
        + f"\n[exit_code] {result['returncode']}\n",
        encoding="utf-8",
    )
    return {
        "label": label,
        "command": result["command"],
        "cwd": result["cwd"],
        "returncode": result["returncode"],
        "passed": result["returncode"] == 0,
        "duration_seconds": result["duration_seconds"],
        "stdout_tail": result["stdout"][-2000:],
        "stderr_tail": result["stderr"][-2000:],
        "log_file": str(log_path),
        "log_sha256": sha256(log_path),
    }


def verify_gate_logs(report: dict[str, Any]) -> None:
    outcomes = report.get("outcomes")
    if not isinstance(outcomes, list):
        fail(
            "IMMUTABILITY_ERROR",
            "gate_report",
            "gate outcomes must be a list",
        )
    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, dict):
            fail(
                "IMMUTABILITY_ERROR",
                "gate_report",
                f"gate outcome {index} is invalid",
            )
        log_value = outcome.get("log_file")
        expected_hash = outcome.get("log_sha256")
        if log_value is None:
            if outcome.get("passed") is True:
                fail(
                    "IMMUTABILITY_ERROR",
                    "gate_report",
                    f"passing gate outcome {index} lacks a log",
                )
            continue
        log_path = Path(text(log_value, f"gate outcome {index} log_file"))
        if not log_path.is_file() or expected_hash != sha256(log_path):
            fail(
                "IMMUTABILITY_ERROR",
                "gate_report",
                f"gate log hash mismatch for outcome {index}",
            )


def gate_transaction(transaction_dir: Path) -> dict[str, Any]:
    transaction_dir = transaction_dir.expanduser().resolve()
    context: dict[str, Any] | None = None
    try:
        with locked(transaction_dir):
            context = read_tx_context(
                transaction_dir,
                {"NORMALIZED", "REQUIRED_GATES_PASSED"},
            )
            normalization_path = transaction_dir / NORMALIZATION_REPORT
            normalization = load_json(
                normalization_path,
                "normalization report",
            )
            if context["transaction"].get(
                "normalization_report_sha256"
            ) != sha256(normalization_path):
                fail(
                    "IMMUTABILITY_ERROR",
                    "gate_preflight",
                    "normalization report hash mismatch",
                )
            normalized_commit = full_sha(
                normalization.get("normalized_commit_sha"),
                "normalized_commit_sha",
            )
            trusted = trusted_main(context)
            if (
                context["transaction"]["state"] == "REQUIRED_GATES_PASSED"
                or (transaction_dir / GATE_REPORT).exists()
            ):
                gate_path = transaction_dir / GATE_REPORT
                report = load_json(gate_path, "gate report")
                if context["transaction"].get(
                    "gate_report_sha256"
                ) != sha256(gate_path):
                    fail(
                        "IMMUTABILITY_ERROR",
                        "gate_idempotence",
                        "gate report hash drifted",
                    )
                verify_gate_logs(report)
                if (
                    report.get("status") != "PASS"
                    or report.get("normalized_commit_sha") != normalized_commit
                ):
                    fail(
                        "IMMUTABILITY_ERROR",
                        "gate_idempotence",
                        "gate report mismatch",
                    )
                ensure_clean(
                    context["repo"],
                    normalized_commit,
                    "gate_idempotence",
                )
                return {
                    "status": "PASS",
                    "state": "REQUIRED_GATES_PASSED",
                    "integration_id": context["integration_id"],
                    "normalized_commit_sha": normalized_commit,
                    "idempotent": True,
                }
            ensure_clean(
                context["repo"],
                normalized_commit,
                "gate_preflight",
            )
            check_freshness(context, "gate_freshness")
            paths = changed_paths(
                context["repo"],
                context["main_sha"],
                normalized_commit,
            )
            changed_python = [
                path
                for path in paths
                if path.endswith(".py")
                and (context["repo"] / path).is_file()
            ]
            log_dir = transaction_dir / "gate-logs"
            outcomes: list[dict[str, Any]] = []
            outcomes.append(
                gate_record(
                    "git-diff-check",
                    [
                        "git",
                        "diff",
                        "--check",
                        f"{context['main_sha']}..{normalized_commit}",
                    ],
                    cwd=context["repo"],
                    log_dir=log_dir,
                )
            )
            if changed_python:
                outcomes.append(
                    gate_record(
                        "python-compile",
                        [
                            sys.executable,
                            "-m",
                            "py_compile",
                            *changed_python,
                        ],
                        cwd=context["repo"],
                        log_dir=log_dir,
                    )
                )
                ruff = shutil.which("ruff")
                if ruff:
                    outcomes.append(
                        gate_record(
                            "ruff-changed",
                            [ruff, "check", *changed_python],
                            cwd=context["repo"],
                            log_dir=log_dir,
                        )
                    )
                else:
                    outcomes.append(
                        {
                            "label": "ruff-changed",
                            "command": [],
                            "cwd": str(context["repo"]),
                            "returncode": 127,
                            "passed": False,
                            "duration_seconds": 0.0,
                            "stdout_tail": "",
                            "stderr_tail": (
                                "required executable 'ruff' was not found on PATH"
                            ),
                            "log_file": None,
                        }
                    )
            outcomes.append(
                gate_record(
                    "handoff-authority",
                    [
                        sys.executable,
                        str(trusted / "scripts/handoff_authority.py"),
                        "verify",
                        "--repo-root",
                        str(context["repo"]),
                        "--json",
                    ],
                    cwd=context["repo"],
                    log_dir=log_dir,
                    timeout=900,
                )
            )
            outcomes.append(
                gate_record(
                    "governance-stage",
                    [
                        sys.executable,
                        str(
                            trusted
                            / "scripts/validate_governance_pipeline_stage_status.py"
                        ),
                        "--repo-root",
                        str(context["repo"]),
                    ],
                    cwd=context["repo"],
                    log_dir=log_dir,
                    timeout=900,
                )
            )
            outcomes.append(
                gate_record(
                    "formal-channel",
                    [
                        sys.executable,
                        str(
                            trusted
                            / "scripts/validate_formal_execution_channel.py"
                        ),
                        "--repo-root",
                        str(context["repo"]),
                    ],
                    cwd=context["repo"],
                    log_dir=log_dir,
                    timeout=900,
                )
            )
            selector_plan_record = gate_record(
                "test-selector-plan",
                [
                    sys.executable,
                    str(trusted / "scripts/select_update_tests.py"),
                    "--repo",
                    str(context["repo"]),
                    "--base",
                    context["main_sha"],
                    "--head",
                    normalized_commit,
                    "--mode",
                    context["requested_gate_tier"],
                    "--map",
                    str(
                        trusted
                        / "tools/drpo-update/test_impact_map.json"
                    ),
                    "--json",
                ],
                cwd=context["repo"],
                log_dir=log_dir,
                timeout=900,
            )
            outcomes.append(selector_plan_record)
            selector_plan = None
            if selector_plan_record["passed"]:
                try:
                    selector_plan = json.loads(
                        selector_plan_record["stdout_tail"]
                    )
                except json.JSONDecodeError:
                    raw = Path(
                        selector_plan_record["log_file"]
                    ).read_text(encoding="utf-8")
                    start = raw.find("{")
                    end = raw.rfind("}")
                    if start >= 0 and end > start:
                        selector_plan = json.loads(raw[start : end + 1])
                    else:
                        outcomes.append(
                            {
                                "label": "test-selector-plan-json",
                                "command": [],
                                "cwd": str(context["repo"]),
                                "returncode": 2,
                                "passed": False,
                                "duration_seconds": 0.0,
                                "stdout_tail": "",
                                "stderr_tail": (
                                    "selector plan did not emit JSON"
                                ),
                                "log_file": None,
                            }
                        )
            if selector_plan is not None:
                outcomes.append(
                    gate_record(
                        "test-selector-execute",
                        [
                            sys.executable,
                            str(trusted / "scripts/select_update_tests.py"),
                            "--repo",
                            str(context["repo"]),
                            "--base",
                            context["main_sha"],
                            "--head",
                            normalized_commit,
                            "--mode",
                            context["requested_gate_tier"],
                            "--map",
                            str(
                                trusted
                                / "tools/drpo-update/test_impact_map.json"
                            ),
                            "--execute",
                        ],
                        cwd=context["repo"],
                        log_dir=log_dir,
                        timeout=3600,
                    )
                )
            failures = [
                item
                for item in outcomes
                if not item["passed"]
            ]
            report = {
                "schema_version": SCHEMA,
                "tool_version": VERSION,
                "status": "PASS" if not failures else "FAIL",
                "state": (
                    "REQUIRED_GATES_PASSED"
                    if not failures
                    else "BLOCKED"
                ),
                "integration_id": context["integration_id"],
                "main_sha": context["main_sha"],
                "normalized_commit_sha": normalized_commit,
                "requested_gate_tier": context["requested_gate_tier"],
                "changed_paths": paths,
                "selector_plan": selector_plan,
                "outcomes": outcomes,
                "first_blocker": (
                    None
                    if not failures
                    else failures[0]["label"]
                ),
                "passed_count": len(outcomes) - len(failures),
                "failed_count": len(failures),
                "completed_at": now(),
            }
            write_json(transaction_dir / GATE_REPORT, report)
            if failures:
                fail(
                    "GATE_FAILURE",
                    "gate",
                    f"{len(failures)} required gate commands failed: "
                    + ", ".join(item["label"] for item in failures),
                )
            ensure_clean(
                context["repo"],
                normalized_commit,
                "gate_postflight",
            )
            tx = context["transaction"]
            completed = list(tx.get("completed_states", []))
            if "REQUIRED_GATES_PASSED" not in completed:
                completed.append("REQUIRED_GATES_PASSED")
            tx.update(
                {
                    "tool_version": VERSION,
                    "state": "REQUIRED_GATES_PASSED",
                    "status": "PASS",
                    "completed_states": completed,
                    "gate_report_sha256": sha256(
                        transaction_dir / GATE_REPORT
                    ),
                    "updated_at": now(),
                    "next_action": "run Batch 2B finalize",
                }
            )
            write_json(context["transaction_path"], tx)
            return {
                "status": "PASS",
                "state": "REQUIRED_GATES_PASSED",
                "integration_id": context["integration_id"],
                "normalized_commit_sha": normalized_commit,
                "selected_mode": (
                    None
                    if selector_plan is None
                    else selector_plan.get("selected_mode")
                ),
                "idempotent": False,
            }
    except FinalizeError as error:
        write_diagnostic(transaction_dir, context, error)
        raise
    except WritePathError as error:
        wrapped = FinalizeError(
            error.error_code,
            error.phase,
            error.message,
            error.recovery,
        )
        write_diagnostic(transaction_dir, context, wrapped)
        raise wrapped from error
    except Exception as exc:
        error = FinalizeError(
            "INTERNAL_ERROR",
            "gate_internal",
            f"unexpected error: {exc}",
        )
        write_diagnostic(transaction_dir, context, error)
        raise error from exc


def finalize_transaction(transaction_dir: Path) -> dict[str, Any]:
    transaction_dir = transaction_dir.expanduser().resolve()
    context: dict[str, Any] | None = None
    try:
        with locked(transaction_dir):
            context = read_tx_context(
                transaction_dir,
                {"REQUIRED_GATES_PASSED", "READY"},
            )
            normalization_path = transaction_dir / NORMALIZATION_REPORT
            gate_path = transaction_dir / GATE_REPORT
            normalization = load_json(
                normalization_path,
                "normalization report",
            )
            gate_report = load_json(gate_path, "gate report")
            if context["transaction"].get(
                "normalization_report_sha256"
            ) != sha256(normalization_path):
                fail(
                    "IMMUTABILITY_ERROR",
                    "finalize_preflight",
                    "normalization report hash mismatch",
                )
            if context["transaction"].get(
                "gate_report_sha256"
            ) != sha256(gate_path):
                fail(
                    "IMMUTABILITY_ERROR",
                    "finalize_preflight",
                    "gate report hash mismatch",
                )
            verify_gate_logs(gate_report)
            normalized_commit = full_sha(
                normalization.get("normalized_commit_sha"),
                "normalized_commit_sha",
            )
            if (
                gate_report.get("status") != "PASS"
                or gate_report.get("normalized_commit_sha") != normalized_commit
            ):
                fail(
                    "GATE_FAILURE",
                    "finalize_preflight",
                    "passing gate report is missing or mismatched",
                )
            if (
                context["transaction"]["state"] == "READY"
                or (transaction_dir / READY_COMMIT).exists()
            ):
                ready_path = transaction_dir / READY_COMMIT
                ready = load_json(ready_path, "ready commit")
                if context["transaction"].get(
                    "ready_commit_record_sha256"
                ) != sha256(ready_path):
                    fail(
                        "IMMUTABILITY_ERROR",
                        "finalize_idempotence",
                        "ready record hash drifted",
                    )
                if (
                    ready.get("ready_commit_sha") != normalized_commit
                    or ready.get("gate_report_sha256")
                    != sha256(transaction_dir / GATE_REPORT)
                ):
                    fail(
                        "IMMUTABILITY_ERROR",
                        "finalize_idempotence",
                        "ready record drifted",
                    )
                ensure_clean(
                    context["repo"],
                    normalized_commit,
                    "finalize_idempotence",
                )
                return {
                    "status": "PASS",
                    "state": "READY",
                    "integration_id": context["integration_id"],
                    "ready_commit_sha": normalized_commit,
                    "idempotent": True,
                }
            check_freshness(context, "finalize_freshness")
            ensure_clean(
                context["repo"],
                normalized_commit,
                "finalize_preflight",
            )
            tree_sha = ensure_one_parent(
                context["repo"],
                normalized_commit,
                context["main_sha"],
                "finalize_preflight",
            )
            trusted = trusted_main(context)
            verify = authority_verify(context, trusted)
            ready = {
                "schema_version": SCHEMA,
                "tool_version": VERSION,
                "status": "PASS",
                "state": "READY",
                "integration_id": context["integration_id"],
                "main_sha": context["main_sha"],
                "dev_sha": context["dev_sha"],
                "source_commit_sha": context["source_commit_sha"],
                "ready_commit_sha": normalized_commit,
                "parent_sha": context["main_sha"],
                "tree_sha": tree_sha,
                "normalization_report_sha256": sha256(
                    transaction_dir / NORMALIZATION_REPORT
                ),
                "gate_report_sha256": sha256(
                    transaction_dir / GATE_REPORT
                ),
                "authority_verify": verify,
                "changed_paths": changed_paths(
                    context["repo"],
                    context["main_sha"],
                    normalized_commit,
                ),
                "ready_at": now(),
                "publish_automation": False,
            }
            ready_path = transaction_dir / READY_COMMIT
            write_json(ready_path, ready)
            tx = context["transaction"]
            completed = list(tx.get("completed_states", []))
            if "READY" not in completed:
                completed.append("READY")
            tx.update(
                {
                    "tool_version": VERSION,
                    "state": "READY",
                    "status": "PASS",
                    "completed_states": completed,
                    "ready_commit_sha": normalized_commit,
                    "ready_commit_record_sha256": sha256(ready_path),
                    "updated_at": now(),
                    "next_action": "manual review/publish outside V1",
                }
            )
            write_json(context["transaction_path"], tx)
            return {
                "status": "PASS",
                "state": "READY",
                "integration_id": context["integration_id"],
                "ready_commit_sha": normalized_commit,
                "idempotent": False,
            }
    except FinalizeError as error:
        write_diagnostic(transaction_dir, context, error)
        raise
    except WritePathError as error:
        wrapped = FinalizeError(
            error.error_code,
            error.phase,
            error.message,
            error.recovery,
        )
        write_diagnostic(transaction_dir, context, wrapped)
        raise wrapped from error
    except Exception as exc:
        error = FinalizeError(
            "INTERNAL_ERROR",
            "finalize_internal",
            f"unexpected error: {exc}",
        )
        write_diagnostic(transaction_dir, context, error)
        raise error from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(
        dest="command_name",
        required=True,
    )
    for name in ("normalize", "gate", "finalize"):
        child = commands.add_parser(name)
        child.add_argument("--transaction-dir", required=True)
        child.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    function = {
        "normalize": normalize_transaction,
        "gate": gate_transaction,
        "finalize": finalize_transaction,
    }[args.command_name]
    try:
        result = function(Path(args.transaction_dir))
        print(
            json.dumps(result, sort_keys=True)
            if args.json
            else f"PASS {result['integration_id']}: {result['state']}"
        )
        return 0
    except FinalizeError as error:
        payload = {
            "status": "FAIL",
            "error_code": error.error_code,
            "phase": error.phase,
            "message": error.message,
        }
        print(
            json.dumps(payload, sort_keys=True)
            if args.json
            else f"FAIL {error}",
            file=sys.stdout if args.json else sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
