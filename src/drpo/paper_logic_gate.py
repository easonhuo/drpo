from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .paper_logic_artifacts import (
    validate_candidate,
    validate_mapping,
    validate_paragraphs,
    validate_section,
)
from .paper_logic_common import (
    INITIATIVE,
    LEVELS,
    GateError,
    file_ref,
    inside,
    load_policy,
    read_yaml,
    strings,
    text,
)

PaperLogicGateError = GateError


def invalidation(level: str, section_id: str, paragraph_ids: list[str]) -> list[str]:
    if level == "wording":
        return [f"prose:{item}" for item in paragraph_ids]
    if level == "paragraph":
        return [
            *[f"source_mapping:{item}" for item in paragraph_ids],
            *[f"prose:{item}" for item in paragraph_ids],
            f"adjacent_transitions:{section_id}",
        ]
    return [
        f"paragraph_logic:{section_id}",
        f"source_mapping:{section_id}",
        f"prose:{section_id}",
    ]


def validate_manifest(
    *, repo: Path, manifest_path: Path, policy_path: Path
) -> dict[str, Any]:
    repo = repo.resolve()
    policy = load_policy(repo, policy_path)
    manifest = read_yaml(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("initiative") != INITIATIVE:
        raise GateError("authoring manifest identity mismatch")
    level = manifest.get("edit_level")
    if level not in LEVELS:
        raise GateError(f"unknown edit_level: {level}")
    section_id = text(manifest, "section_id", "manifest")
    targets = strings(manifest, "target_paragraph_ids", "manifest")
    source_ref = manifest.get("source")
    source_path = file_ref(repo, source_ref, "source")
    source_sha = str(source_ref["sha256"])
    source_text = source_path.read_text(encoding="utf-8")
    contract = policy["levels"][level]
    artifacts: dict[str, tuple[Path, dict[str, Any]]] = {}
    for name in contract["required_artifacts"]:
        path = file_ref(repo, manifest.get(name), name)
        artifacts[name] = (path, read_yaml(path))

    section_paragraphs: list[str] | None = None
    if "section_logic" in artifacts:
        actual_section, section_paragraphs = validate_section(
            artifacts["section_logic"][1], source_sha
        )
        if actual_section != section_id:
            raise GateError("section logic map section mismatch")
    paragraph_ids, owners, paragraph_order = validate_paragraphs(
        artifacts["paragraph_logic"][1], source_sha, section_id
    )
    if section_paragraphs is not None and section_paragraphs != paragraph_ids:
        raise GateError("section and paragraph maps disagree on IDs or order")
    if set(targets) - set(paragraph_ids):
        raise GateError("manifest targets unknown paragraph IDs")
    expected_paragraphs = [item for item in paragraph_ids if item in targets]
    if targets != expected_paragraphs:
        raise GateError("manifest targets must follow approved paragraph order")

    mapping = validate_mapping(
        artifacts["source_mapping"][1], source_sha, source_text
    )
    candidate_status, candidate_paragraphs, candidate_nodes, candidate_order = (
        validate_candidate(artifacts["candidate"][1], source_sha, section_id)
    )
    if candidate_paragraphs != expected_paragraphs:
        raise GateError("candidate paragraphs must match authorized target order")
    if set(mapping) != set(candidate_nodes):
        unauthorized = sorted(set(candidate_nodes) - set(mapping))
        missing = sorted(set(mapping) - set(candidate_nodes))
        raise GateError(
            f"candidate mapping mismatch: unauthorized={unauthorized} missing={missing}"
        )
    expected_nodes = {
        node
        for paragraph_id in expected_paragraphs
        for node in paragraph_order[paragraph_id]
    }
    if set(mapping) != expected_nodes:
        omitted = sorted(expected_nodes - set(mapping))
        extra = sorted(set(mapping) - expected_nodes)
        raise GateError(
            "mapping must cover every approved node in target paragraphs: "
            f"omitted={omitted} extra={extra}"
        )

    authorization = manifest.get("authorization")
    if not isinstance(authorization, dict):
        raise GateError("manifest requires authorization")
    approved_by = text(authorization, "approved_by", "manifest.authorization")
    text(authorization, "approved_at", "manifest.authorization")
    allow_strengthening = authorization.get("allow_claim_strengthening") is True

    for node_id, operation in mapping.items():
        owner = owners.get(node_id)
        if owner is None or operation["paragraph_id"] != owner:
            raise GateError(f"mapping node {node_id} disagrees with paragraph logic")
        candidate_owner, candidate_text = candidate_nodes[node_id]
        if candidate_owner != owner or owner not in targets:
            raise GateError(f"candidate node {node_id} expands target scope")
        action = operation["action"]
        source_sentence = operation.get("source_text")
        if action in {"KEEP", "MOVE"} and candidate_text != source_sentence:
            raise GateError(f"frozen sentence changed for node {node_id} ({action})")
        if action in {"TRIM", "REVISE"} and candidate_text == source_sentence:
            raise GateError(f"{action} node {node_id} is unchanged")
        if operation["claim_impact"] == "strengthen" and not allow_strengthening:
            raise GateError(f"claim strengthening is not authorized for node {node_id}")
    for paragraph_id in expected_paragraphs:
        approved_order = paragraph_order[paragraph_id]
        actual_order = candidate_order[paragraph_id]
        moved_nodes = {node for node in approved_order if mapping[node]["action"] == "MOVE"}
        if actual_order != approved_order:
            if not moved_nodes:
                raise GateError(f"sentence order changed without MOVE: {paragraph_id}")
            approved_fixed = [node for node in approved_order if node not in moved_nodes]
            actual_fixed = [node for node in actual_order if node not in moved_nodes]
            if actual_fixed != approved_fixed:
                raise GateError(
                    f"non-MOVE sentence order changed in paragraph {paragraph_id}"
                )

    return {
        "status": "PASS",
        "initiative": INITIATIVE,
        "edit_level": level,
        "section_id": section_id,
        "target_paragraph_ids": targets,
        "source": {
            "path": source_path.relative_to(repo).as_posix(),
            "sha256": source_sha,
        },
        "selected_guidance_rules": contract["guidance_rules"],
        "selected_playbook_modules": contract["playbook_modules"],
        "required_artifacts": contract["required_artifacts"],
        "policy_source_binding": policy["_source_binding"],
        "writing_contract": policy["_resolved_levels"][level],
        "validated_node_count": len(mapping),
        "candidate_status": candidate_status,
        "authorized_by": approved_by,
        "invalidation_scope": invalidation(str(level), section_id, targets),
    }


def plan(
    level: str, *, section_id: str, paragraph_ids: list[str], policy: dict[str, Any]
) -> dict[str, Any]:
    contract = policy["levels"][level]
    return {
        "status": "PASS",
        "initiative": INITIATIVE,
        "edit_level": level,
        "section_id": section_id,
        "target_paragraph_ids": paragraph_ids,
        "selected_guidance_rules": contract["guidance_rules"],
        "selected_playbook_modules": contract["playbook_modules"],
        "required_artifacts": contract["required_artifacts"],
        "policy_source_binding": policy["_source_binding"],
        "writing_contract": policy["_resolved_levels"][level],
        "invalidation_scope": invalidation(level, section_id, paragraph_ids),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--policy", default="docs/manuscript/paper_logic_gate_policy.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--manifest", required=True)
    validate_parser.add_argument("--report")
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--edit-level", choices=sorted(LEVELS), required=True)
    plan_parser.add_argument("--section-id", required=True)
    plan_parser.add_argument("--paragraph-id", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path(args.repo_root).resolve()
    try:
        policy_path = inside(repo, args.policy)
        policy = load_policy(repo, policy_path)
        if args.command == "validate":
            result = validate_manifest(
                repo=repo,
                manifest_path=inside(repo, args.manifest),
                policy_path=policy_path,
            )
            if args.report:
                report = inside(repo, args.report)
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        else:
            paragraphs = args.paragraph_id or [f"{args.section_id}-P01"]
            result = plan(
                args.edit_level,
                section_id=args.section_id,
                paragraph_ids=paragraphs,
                policy=policy,
            )
    except (GateError, OSError, UnicodeError, yaml.YAMLError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
