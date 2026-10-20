from __future__ import annotations

from typing import Any

from .paper_logic_common import (
    ACTIONS,
    CLAIM_IMPACTS,
    GateError,
    approval,
    strings,
    text,
)


def validate_section(row: dict[str, Any], source_sha: str) -> tuple[str, list[str]]:
    if row.get("schema_version") != 1 or row.get("artifact_type") != "section_logic_map":
        raise GateError("invalid section logic map identity")
    if row.get("status") != "approved":
        raise GateError("section logic map must be approved")
    if row.get("source_sha256") != source_sha:
        raise GateError("section logic map is stale relative to source")
    section_id = text(row, "section_id", "section logic map")
    for key in ("central_question", "entry_point", "exit_point"):
        text(row, key, "section logic map")
    if len(strings(row, "chain", "section logic map")) < 2:
        raise GateError("section logic map chain needs at least two steps")
    paragraph_ids = strings(row, "paragraph_ids", "section logic map")
    approval(row, "section logic map")
    return section_id, paragraph_ids


def validate_paragraphs(
    row: dict[str, Any], source_sha: str, section_id: str
) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    if row.get("schema_version") != 1 or row.get("artifact_type") != "paragraph_logic_map":
        raise GateError("invalid paragraph logic map identity")
    if row.get("status") != "approved":
        raise GateError("paragraph logic map must be approved")
    if row.get("source_sha256") != source_sha:
        raise GateError("paragraph logic map is stale relative to source")
    if text(row, "section_id", "paragraph logic map") != section_id:
        raise GateError("paragraph logic map section mismatch")
    paragraphs = row.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise GateError("paragraph logic map requires paragraphs")
    paragraph_ids: list[str] = []
    owners: dict[str, str] = {}
    order: dict[str, list[str]] = {}
    for paragraph in paragraphs:
        if not isinstance(paragraph, dict):
            raise GateError("invalid paragraph logic record")
        paragraph_id = text(paragraph, "id", "paragraph logic record")
        if paragraph_id in paragraph_ids:
            raise GateError(f"duplicate paragraph logic id: {paragraph_id}")
        paragraph_ids.append(paragraph_id)
        text(paragraph, "responsibility", paragraph_id)
        text(paragraph, "topic_claim", paragraph_id)
        nodes = paragraph.get("sentence_nodes")
        if not isinstance(nodes, list) or not nodes:
            raise GateError(f"paragraph logic {paragraph_id} requires sentence_nodes")
        order[paragraph_id] = []
        for node in nodes:
            if not isinstance(node, dict):
                raise GateError(f"invalid sentence node in {paragraph_id}")
            node_id = text(node, "id", f"sentence node in {paragraph_id}")
            if node_id in owners:
                raise GateError(f"duplicate sentence node id: {node_id}")
            text(node, "role", node_id)
            text(node, "instruction", node_id)
            owners[node_id] = paragraph_id
            order[paragraph_id].append(node_id)
    approval(row, "paragraph logic map")
    return paragraph_ids, owners, order


def validate_mapping(
    row: dict[str, Any], source_sha: str, source_text: str
) -> dict[str, dict[str, Any]]:
    if row.get("schema_version") != 1 or row.get("artifact_type") != "source_mapping":
        raise GateError("invalid source mapping identity")
    if row.get("status") != "complete" or row.get("source_sha256") != source_sha:
        raise GateError("source mapping is incomplete or stale relative to source")
    operations = row.get("operations")
    if not isinstance(operations, list) or not operations:
        raise GateError("source mapping requires operations")
    result: dict[str, dict[str, Any]] = {}
    claimed_sources: set[tuple[str, int]] = set()
    for operation in operations:
        if not isinstance(operation, dict):
            raise GateError("invalid source mapping operation")
        node_id = text(operation, "node_id", "source mapping operation")
        if node_id in result:
            raise GateError(f"duplicate source mapping node: {node_id}")
        paragraph_id = text(operation, "paragraph_id", node_id)
        action = operation.get("action")
        impact = operation.get("claim_impact")
        if action not in ACTIONS or impact not in CLAIM_IMPACTS:
            raise GateError(f"invalid action or claim impact for {node_id}")
        source_sentence = operation.get("source_text")
        reason = str(operation.get("reason", "")).strip()
        if action == "ADD":
            if source_sentence not in (None, "") or not reason:
                raise GateError(f"ADD {node_id} requires reason and no source_text")
        else:
            if not isinstance(source_sentence, str) or not source_sentence:
                raise GateError(f"{action} {node_id} requires source_text")
            count = source_text.count(source_sentence)
            occurrence = operation.get("source_occurrence")
            if count == 0:
                raise GateError(f"source_text for {node_id} is absent from source")
            if count > 1 and not isinstance(occurrence, int):
                raise GateError(f"source_text for {node_id} is ambiguous")
            if occurrence is not None and (
                not isinstance(occurrence, int)
                or isinstance(occurrence, bool)
                or not 1 <= occurrence <= count
            ):
                raise GateError(f"invalid source_occurrence for {node_id}")
            resolved_occurrence = 1 if occurrence is None else occurrence
            source_key = (source_sentence, resolved_occurrence)
            if source_key in claimed_sources:
                raise GateError(f"source span is mapped more than once: {node_id}")
            claimed_sources.add(source_key)
            if action in {"TRIM", "REVISE"} and not reason:
                raise GateError(f"{action} {node_id} requires reason")
        result[node_id] = {**operation, "paragraph_id": paragraph_id}
    return result


def validate_candidate(
    row: dict[str, Any], source_sha: str, section_id: str
) -> tuple[str, list[str], dict[str, tuple[str, str]], dict[str, list[str]]]:
    if row.get("schema_version") != 1 or row.get("artifact_type") != "prose_candidate":
        raise GateError("invalid prose candidate identity")
    status = row.get("status")
    if status not in {"draft", "audited"} or row.get("source_sha256") != source_sha:
        raise GateError("candidate is invalid or stale relative to source")
    if text(row, "section_id", "candidate") != section_id:
        raise GateError("candidate section mismatch")
    paragraphs = row.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise GateError("candidate requires paragraphs")
    paragraph_ids: list[str] = []
    nodes: dict[str, tuple[str, str]] = {}
    order: dict[str, list[str]] = {}
    for paragraph in paragraphs:
        if not isinstance(paragraph, dict):
            raise GateError("invalid candidate paragraph")
        paragraph_id = text(paragraph, "id", "candidate paragraph")
        if paragraph_id in paragraph_ids:
            raise GateError(f"duplicate candidate paragraph: {paragraph_id}")
        paragraph_ids.append(paragraph_id)
        order[paragraph_id] = []
        sentences = paragraph.get("sentences")
        if not isinstance(sentences, list) or not sentences:
            raise GateError(f"candidate paragraph {paragraph_id} requires sentences")
        for sentence in sentences:
            if not isinstance(sentence, dict):
                raise GateError(f"invalid candidate sentence in {paragraph_id}")
            node_id = text(sentence, "node_id", paragraph_id)
            if node_id in nodes:
                raise GateError(f"duplicate candidate sentence node: {node_id}")
            nodes[node_id] = (paragraph_id, text(sentence, "text", node_id))
            order[paragraph_id].append(node_id)
    return str(status), paragraph_ids, nodes, order
