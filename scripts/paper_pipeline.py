#!/usr/bin/env python3
"""Bidirectional domain-agnostic manuscript pipeline.

The pipeline treats a stable-ID manuscript graph as the reconciliation layer among
outline, paragraph blueprint, prose, LaTeX, appendix, figures, and the Overleaf
package.  A user may edit any one registered textual layer and run ``sync``;
block-level changes are imported into the graph, reconciled, and propagated to
all other layers.  Conflicting edits to the same node fail closed.

For high-quality future regeneration, ``--generator-cmd`` may point to a trusted
local command that reads a JSON node on stdin and returns updated ``blueprint``
and ``prose`` fields as JSON.  Without that command, the deterministic backend
produces a conservative compilable scaffold and never invents experimental
results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

BEGIN_RE = re.compile(r"^<!--\s*MANUSCRIPT:BEGIN\s+([A-Z][A-Z0-9-]*)\s*-->\s*$")
END_RE = re.compile(r"^<!--\s*MANUSCRIPT:END\s+([A-Z][A-Z0-9-]*)\s*-->\s*$")
HEADING_RE = re.compile(r"^##\s+\[([A-Z][A-Z0-9-]*)\]\s+(.+?)\s*$")
PARENT_OUTLINE_RE = re.compile(r"^Parent-Outline-SHA256:\s*`?([0-9a-f]{64})`?\s*$")
PARENT_BLUEPRINT_RE = re.compile(r"^Parent-Blueprint-SHA256:\s*`?([0-9a-f]{64})`?\s*$")
FIELD_RE = re.compile(r"^\*\*([^*]+):\*\*\s*(.*)$")
BODY_MARKER = "**Body:**"


class PipelineError(RuntimeError):
    pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineError(f"missing manuscript graph: {path}") from exc
    if not isinstance(data, dict):
        raise PipelineError("manuscript graph root must be a mapping")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=1000),
        encoding="utf-8",
    )


def load_project_profile(graph: dict[str, Any], root: Path) -> dict[str, Any]:
    artifacts = graph.get("artifacts", {})
    configured = artifacts.get("project_profile")
    if not configured:
        contract_ref = artifacts.get("publication_quality_contract")
        if contract_ref:
            contract_path = root / str(contract_ref)
            if not contract_path.is_file():
                raise PipelineError(f"missing publication-quality contract: {contract_path}")
            contract = read_yaml(contract_path)
            configured = contract.get("project_profile")
    if not configured:
        return {}
    path = root / str(configured)
    if not path.is_file():
        raise PipelineError(f"missing manuscript project profile: {path}")
    return read_yaml(path)


def project_validation_errors(node: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    rules = profile.get("validation_rules", {})
    if not isinstance(rules, dict):
        raise PipelineError("project profile validation_rules must be a mapping")
    node_id = str(node.get("id", ""))
    text = " ".join(
        str(node.get(field, "")) for field in ("title", "claim", "role", "prose")
    ).lower()
    errors: list[str] = []
    for phrase in rules.get("forbidden_phrases", []):
        token = str(phrase).strip().lower()
        if token and token in text:
            errors.append(f"node {node_id} contains project-forbidden framing: {phrase}")
    for rule in rules.get("conditional_forbidden", []):
        if not isinstance(rule, dict):
            raise PipelineError("conditional_forbidden rules must be mappings")
        trigger = str(rule.get("trigger", "")).strip().lower()
        forbidden = str(rule.get("forbidden", "")).strip().lower()
        exemptions = [str(item).strip().lower() for item in rule.get("exemptions", [])]
        if trigger and forbidden and trigger in text and forbidden in text:
            if not any(item and item in text for item in exemptions):
                errors.append(
                    f"node {node_id} violates project terminology rule: {rule.get('forbidden')}"
                )
    return errors


def normalize_payload(lines: list[str]) -> str:
    values = [line.rstrip() for line in lines]
    while values and not values[0]:
        values.pop(0)
    while values and not values[-1]:
        values.pop()
    return "\n".join(values) + "\n"


@dataclass
class ParsedBlock:
    node_id: str
    title: str
    payload: str

    @property
    def digest(self) -> str:
        return sha256_text(self.payload)


def parse_blocks(path: Path) -> dict[str, ParsedBlock]:
    text = path.read_text(encoding="utf-8")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    result: dict[str, ParsedBlock] = {}
    i = 0
    while i < len(lines):
        begin = BEGIN_RE.match(lines[i])
        if not begin:
            i += 1
            continue
        node_id = begin.group(1)
        j = i + 1
        while j < len(lines) and not END_RE.match(lines[j]):
            if BEGIN_RE.match(lines[j]):
                raise PipelineError(f"nested block in {path}: {node_id}")
            j += 1
        if j >= len(lines):
            raise PipelineError(f"unterminated block in {path}: {node_id}")
        end = END_RE.match(lines[j])
        assert end
        if end.group(1) != node_id:
            raise PipelineError(f"mismatched block end in {path}: {node_id}")
        payload_lines = lines[i + 1 : j]
        first = next((k for k, line in enumerate(payload_lines) if line.strip()), None)
        if first is None:
            raise PipelineError(f"empty block in {path}: {node_id}")
        heading = HEADING_RE.match(payload_lines[first])
        if not heading or heading.group(1) != node_id:
            raise PipelineError(f"invalid block heading in {path}: {node_id}")
        payload = normalize_payload(payload_lines)
        result[node_id] = ParsedBlock(node_id, heading.group(2).strip(), payload)
        i = j + 1
    if not result:
        raise PipelineError(f"no manuscript blocks found in {path}")
    return result


def _parse_bullets(lines: list[str], start: int) -> tuple[list[str], int]:
    out: list[str] = []
    i = start
    while i < len(lines):
        line = lines[i]
        if line.startswith("- "):
            out.append(line[2:].strip())
            i += 1
            continue
        if not line.strip():
            i += 1
            continue
        break
    return out, i


def _json_bullets(values: list[Any]) -> list[str]:
    return [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in values]


def _decode_json_bullets(values: Any, *, field: str) -> list[Any]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise PipelineError(f"{field} must be a bullet list")
    decoded: list[Any] = []
    for item in values:
        if not isinstance(item, str):
            raise PipelineError(f"{field} contains a non-string bullet")
        try:
            decoded.append(json.loads(item))
        except json.JSONDecodeError as exc:
            raise PipelineError(f"{field} contains invalid JSON: {item}") from exc
    return decoded


def _parse_word_budget(value: Any) -> list[int]:
    if isinstance(value, list) and len(value) == 2 and all(isinstance(x, int) for x in value):
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*(?:--|-|to)\s*(\d+)\s*", value)
        if match:
            return [int(match.group(1)), int(match.group(2))]
    raise PipelineError(f"invalid word budget: {value!r}")


def _render_list_field(
    lines: list[str], label: str, values: Any, *, json_items: bool = False
) -> None:
    lines += [f"**{label}:**"]
    items = values or []
    if json_items:
        items = _json_bullets(list(items))
    for item in items:
        lines.append(f"- {item}")
    lines.append("")


def parse_outline_block(block: ParsedBlock) -> dict[str, Any]:
    lines = block.payload.splitlines()[1:]
    fields: dict[str, Any] = {"title": block.title}
    i = 0
    while i < len(lines):
        match = FIELD_RE.match(lines[i])
        if not match:
            i += 1
            continue
        key = match.group(1).strip().lower().replace(" ", "_")
        value = match.group(2).strip()
        if value:
            fields[key] = value
            i += 1
        else:
            bullets, i = _parse_bullets(lines, i + 1)
            fields[key] = bullets
    return fields


def parse_blueprint_block(block: ParsedBlock) -> dict[str, Any]:
    lines = block.payload.splitlines()[1:]
    if not lines or not PARENT_OUTLINE_RE.match(lines[0]):
        raise PipelineError(f"blueprint block {block.node_id} lacks parent outline hash")
    fields: dict[str, Any] = {"title": block.title}
    i = 1
    while i < len(lines):
        match = FIELD_RE.match(lines[i])
        if not match:
            i += 1
            continue
        key = match.group(1).strip().lower().replace(" ", "_")
        value = match.group(2).strip()
        if value:
            fields[key] = value
            i += 1
        else:
            bullets, i = _parse_bullets(lines, i + 1)
            fields[key] = bullets
    return fields


def parse_prose_block(block: ParsedBlock) -> dict[str, Any]:
    lines = block.payload.splitlines()[1:]
    if not lines or not PARENT_BLUEPRINT_RE.match(lines[0]):
        raise PipelineError(f"prose block {block.node_id} lacks parent blueprint hash")
    fields: dict[str, Any] = {"title": block.title}
    i = 1
    body_start: int | None = None
    while i < len(lines):
        if lines[i].strip() == BODY_MARKER:
            body_start = i + 1
            break
        match = FIELD_RE.match(lines[i])
        if not match:
            i += 1
            continue
        key = match.group(1).strip().lower().replace(" ", "_")
        value = match.group(2).strip()
        if value:
            fields[key] = value
            i += 1
        else:
            bullets, i = _parse_bullets(lines, i + 1)
            fields[key] = bullets
    if body_start is None:
        raise PipelineError(f"prose block {block.node_id} lacks {BODY_MARKER}")
    fields["body"] = "\n".join(lines[body_start:]).strip()
    return fields


def section_map(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sections = graph.get("sections")
    if not isinstance(sections, list):
        raise PipelineError("graph sections must be a list")
    return {str(row["id"]): row for row in sections}


def nodes_sorted(graph: dict[str, Any]) -> list[dict[str, Any]]:
    sections = section_map(graph)
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise PipelineError("graph nodes must be a list")
    return sorted(
        nodes,
        key=lambda node: (
            int(sections[str(node["section"])].get("order", 0)),
            int(node.get("order", 0)),
        ),
    )


def node_map(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node["id"]): node for node in nodes_sorted(graph)}


def outline_payload(node: dict[str, Any]) -> str:
    lines = [f"## [{node['id']}] {node['title']}", ""]
    lines += [f"**Claim:** {node.get('claim', '')}", ""]
    lines += [f"**Reader question:** {node.get('reader_question', '')}", ""]
    lines += [f"**Role:** {node.get('role', '')}", ""]
    for label, key in (
        ("Required evidence", "evidence"),
        ("Must include", "must_include"),
        ("Must avoid", "must_avoid"),
    ):
        lines.append(f"**{label}:**")
        for item in node.get(key, []) or []:
            lines.append(f"- {item}")
        lines.append("")
    return normalize_payload(lines)


def blueprint_payload(node: dict[str, Any], parent_hash: str) -> str:
    bp = node.get("blueprint", {}) or {}
    budget = bp.get("word_budget", [])
    budget_text = (
        f"{budget[0]}--{budget[1]}" if isinstance(budget, list) and len(budget) == 2 else ""
    )
    lines = [
        f"## [{node['id']}] {node['title']}",
        f"Parent-Outline-SHA256: `{parent_hash}`",
        "",
        f"**Claim:** {node.get('claim', '')}",
        "",
        f"**Reader question:** {node.get('reader_question', '')}",
        "",
        f"**Role:** {node.get('role', '')}",
        "",
        f"**Topic sentence:** {bp.get('topic_sentence', node.get('claim', ''))}",
        "",
    ]
    _render_list_field(lines, "Logical moves", bp.get("moves", node.get("must_include", [])))
    _render_list_field(lines, "Sentence plan", bp.get("sentence_plan", []), json_items=True)
    _render_list_field(lines, "Evidence use", bp.get("evidence_use", node.get("evidence", [])))
    _render_list_field(lines, "Citation refs", bp.get("citation_refs", []))
    _render_list_field(lines, "Theorem or equation refs", bp.get("theorem_or_equation_refs", []))
    _render_list_field(lines, "Appendix bindings", bp.get("appendix_bindings", []))
    _render_list_field(lines, "Allowed conclusions", bp.get("allowed_conclusions", []))
    _render_list_field(lines, "Forbidden conclusions", bp.get("forbidden_conclusions", []))
    lines += [
        f"**Reviewer objection:** {bp.get('reviewer_objection', '')}",
        "",
        f"**Objection response:** {bp.get('objection_response', '')}",
        "",
        f"**Word budget:** {budget_text}",
        "",
        f"**Transition:** {bp.get('transition', '')}",
        "",
    ]
    return normalize_payload(lines)


def prose_payload(node: dict[str, Any], parent_hash: str) -> str:
    body = str(node.get("prose", "")).strip()
    bp = node.get("blueprint", {}) or {}
    budget = bp.get("word_budget", [])
    budget_text = (
        f"{budget[0]}--{budget[1]}" if isinstance(budget, list) and len(budget) == 2 else ""
    )
    lines = [
        f"## [{node['id']}] {node['title']}",
        f"Parent-Blueprint-SHA256: `{parent_hash}`",
        "",
        f"**Claim:** {node.get('claim', '')}",
        "",
        f"**Reader question:** {node.get('reader_question', '')}",
        "",
        f"**Role:** {node.get('role', '')}",
        "",
        f"**Topic sentence:** {bp.get('topic_sentence', '')}",
        "",
    ]
    _render_list_field(lines, "Logical moves", bp.get("moves", node.get("must_include", [])))
    _render_list_field(lines, "Sentence plan", bp.get("sentence_plan", []), json_items=True)
    _render_list_field(lines, "Evidence use", bp.get("evidence_use", node.get("evidence", [])))
    _render_list_field(lines, "Citation refs", bp.get("citation_refs", []))
    _render_list_field(lines, "Theorem or equation refs", bp.get("theorem_or_equation_refs", []))
    _render_list_field(lines, "Appendix bindings", bp.get("appendix_bindings", []))
    lines += [
        f"**Word budget:** {budget_text}",
        "",
        BODY_MARKER,
        "",
        body,
        "",
    ]
    return normalize_payload(lines)


def render_markdown(graph: dict[str, Any], layer: str) -> str:
    sections = section_map(graph)
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in sections}
    for node in nodes_sorted(graph):
        grouped[str(node["section"])].append(node)
    header = graph["artifacts"].get(f"{layer}_header", "")
    chunks = [header.rstrip(), ""] if header else []
    for section in sorted(sections.values(), key=lambda x: int(x.get("order", 0))):
        sid = str(section["id"])
        chunks += [f"# {section['title']}", ""]
        for node in grouped.get(sid, []):
            op = outline_payload(node)
            if layer == "outline":
                payload = op
            else:
                oh = sha256_text(op)
                bp = blueprint_payload(node, oh)
                payload = bp if layer == "blueprint" else prose_payload(node, sha256_text(bp))
            chunks += [
                f"<!-- MANUSCRIPT:BEGIN {node['id']} -->",
                payload.rstrip(),
                f"<!-- MANUSCRIPT:END {node['id']} -->",
                "",
            ]
    return "\n".join(chunks).rstrip() + "\n"


def tex_escape_title(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def render_tex_sections(graph: dict[str, Any], root: Path) -> None:
    overleaf = root / graph["artifacts"]["overleaf_root"]
    sections = section_map(graph)
    nodes = nodes_sorted(graph)
    for folder in (overleaf / "sections", overleaf / "appendix"):
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.tex"):
            old.unlink()
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in sections}
    for node in nodes:
        grouped[str(node["section"])].append(node)
    main_inputs: list[str] = []
    appendix_inputs: list[str] = []
    for section in sorted(sections.values(), key=lambda x: int(x.get("order", 0))):
        sid = str(section["id"])
        folder_name = "appendix" if section.get("appendix") else "sections"
        filename = str(section.get("tex_file", f"{sid}.tex"))
        path = overleaf / folder_name / filename
        body: list[str] = []
        if sid == "abstract":
            body.append("\\begin{abstract}")
        else:
            command = "section" if not section.get("subsection") else "subsection"
            body.append(f"\\{command}{{{tex_escape_title(str(section['title']))}}}")
            if section.get("label"):
                body.append(f"\\label{{{section['label']}}}")
        for node in grouped.get(sid, []):
            body += [
                "",
                f"% MANUSCRIPT-NODE: {node['id']}",
                str(node.get("prose", "")).strip(),
                f"% END-MANUSCRIPT-NODE: {node['id']}",
            ]
        if sid == "abstract":
            body.append("\\end{abstract}")
        path.write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")
        rel = f"{folder_name}/{filename}"
        if section.get("appendix"):
            appendix_inputs.append(rel)
        else:
            main_inputs.append(rel)

    metadata = graph.get("metadata", {})
    authors = metadata.get("authors", [])
    author_rows = "\n".join(
        f"    \\icmlauthor{{{row['name']}}}{{{row.get('affiliation', 'comp')}}}" for row in authors
    )
    main_input_lines = "\n".join(f"\\input{{{path}}}" for path in main_inputs)
    appendix_input_lines = "\n".join(f"\\input{{{path}}}" for path in appendix_inputs)
    main_tex = rf"""\documentclass{{article}}
\usepackage{{microtype}}
\usepackage{{graphicx}}
\usepackage{{subcaption}}
\usepackage{{booktabs}}
\usepackage{{multirow}}
\usepackage{{amsmath,amssymb,mathtools,amsthm,bm}}
\usepackage{{algorithm,algorithmic}}
\usepackage{{hyperref}}
\usepackage[capitalize,noabbrev]{{cleveref}}
\usepackage[preprint]{{icml2026}}
\newtheorem{{theorem}}{{Theorem}}[section]
\newtheorem{{proposition}}[theorem]{{Proposition}}
\newtheorem{{lemma}}[theorem]{{Lemma}}
\newtheorem{{corollary}}[theorem]{{Corollary}}
\theoremstyle{{definition}}
\newtheorem{{definition}}[theorem]{{Definition}}
\newtheorem{{assumption}}[theorem]{{Assumption}}
\theoremstyle{{remark}}
\newtheorem{{remark}}[theorem]{{Remark}}
\icmltitlerunning{{{metadata.get("running_title", metadata.get("title", "Manuscript"))}}}
\begin{{document}}
\twocolumn[
\icmltitle{{{metadata["title"]}}}
\icmlsetsymbol{{equal}}{{*}}
\begin{{icmlauthorlist}}
{author_rows}
\end{{icmlauthorlist}}
\icmlaffiliation{{comp}}{{{metadata.get("affiliation", "Anonymous Institution")}}}
\icmlcorrespondingauthor{{{metadata.get("corresponding_name", "Corresponding Author")}}}{{{metadata.get("corresponding_email", "corresponding@example.com")}}}
\icmlkeywords{{{metadata.get("keywords", "Machine Learning")}}}
\vskip 0.3in
]
\printAffiliationsAndNotice{{\icmlEqualContribution}}

% This source is generated from docs/manuscript/paper_graph.yaml.
% Edit a registered layer and run scripts/paper_pipeline.py sync; do not hand-edit
% generated TeX without importing the corresponding node change.
{main_input_lines}

\bibliography{{references}}
\bibliographystyle{{icml2026}}

\clearpage
\appendix
{appendix_input_lines}
\end{{document}}
"""
    (overleaf / "main.tex").write_text(main_tex, encoding="utf-8")


def deterministic_regenerate(node: dict[str, Any], source_layer: str) -> None:
    claim = str(node.get("claim", "")).strip()
    include = [str(x) for x in node.get("must_include", []) or []]
    evidence = [str(x) for x in node.get("evidence", []) or []]
    blueprint = node.setdefault("blueprint", {})
    blueprint.setdefault("topic_sentence", claim)
    blueprint.setdefault("moves", include)
    blueprint.setdefault("evidence_use", evidence)
    blueprint.setdefault("sentence_plan", [])
    if not blueprint.get("transition"):
        blueprint["transition"] = "The next block develops the consequence of this claim."
    if source_layer != "prose":
        plan = blueprint.get("sentence_plan") or []
        drafted = [
            str(step.get("draft", "")).strip()
            for step in plan
            if isinstance(step, dict) and str(step.get("draft", "")).strip()
        ]
        if drafted:
            node["prose"] = " ".join(sentence.rstrip(".") + "." for sentence in drafted)
        else:
            sentences = [claim.rstrip(".") + "."] if claim else []
            sentences.extend(item.rstrip(".") + "." for item in include)
            if evidence:
                sentences.append(
                    "The corresponding evidence is reported only where the registered result status permits it."
                )
            node["prose"] = " ".join(sentences)


def command_regenerate(node: dict[str, Any], source_layer: str, command: str) -> None:
    proc = subprocess.run(
        command,
        input=json.dumps({"source_layer": source_layer, "node": node}, ensure_ascii=False),
        text=True,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise PipelineError(f"generator command failed for {node['id']}: {proc.stderr.strip()}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise PipelineError(f"generator returned invalid JSON for {node['id']}") from exc
    if not isinstance(payload, dict):
        raise PipelineError("generator response must be a mapping")
    if isinstance(payload.get("blueprint"), dict):
        node["blueprint"] = payload["blueprint"]
    if isinstance(payload.get("prose"), str):
        node["prose"] = payload["prose"]


def apply_outline_import(node: dict[str, Any], fields: dict[str, Any]) -> None:
    mapping = {
        "title": "title",
        "claim": "claim",
        "reader_question": "reader_question",
        "role": "role",
        "required_evidence": "evidence",
        "must_include": "must_include",
        "must_avoid": "must_avoid",
    }
    for src, dst in mapping.items():
        if src in fields:
            node[dst] = fields[src]


def apply_blueprint_import(node: dict[str, Any], fields: dict[str, Any]) -> None:
    for src, dst in (
        ("title", "title"),
        ("claim", "claim"),
        ("reader_question", "reader_question"),
        ("role", "role"),
    ):
        if fields.get(src):
            node[dst] = fields[src]
    bp = node.setdefault("blueprint", {})
    simple = {
        "topic_sentence": "topic_sentence",
        "logical_moves": "moves",
        "evidence_use": "evidence_use",
        "citation_refs": "citation_refs",
        "theorem_or_equation_refs": "theorem_or_equation_refs",
        "appendix_bindings": "appendix_bindings",
        "allowed_conclusions": "allowed_conclusions",
        "forbidden_conclusions": "forbidden_conclusions",
        "reviewer_objection": "reviewer_objection",
        "objection_response": "objection_response",
        "transition": "transition",
    }
    for src, dst in simple.items():
        if src in fields:
            bp[dst] = fields[src]
    if "sentence_plan" in fields:
        decoded = _decode_json_bullets(fields["sentence_plan"], field="sentence_plan")
        if not all(isinstance(item, dict) for item in decoded):
            raise PipelineError("sentence_plan bullets must decode to mappings")
        bp["sentence_plan"] = decoded
    if "word_budget" in fields:
        bp["word_budget"] = _parse_word_budget(fields["word_budget"])
    node["must_include"] = list(bp.get("moves", []))
    node["evidence"] = list(bp.get("evidence_use", []))


def apply_prose_import(node: dict[str, Any], fields: dict[str, Any]) -> None:
    for src, dst in (
        ("title", "title"),
        ("claim", "claim"),
        ("reader_question", "reader_question"),
        ("role", "role"),
    ):
        if fields.get(src):
            node[dst] = fields[src]
    bp = node.setdefault("blueprint", {})
    if fields.get("topic_sentence"):
        bp["topic_sentence"] = fields["topic_sentence"]
    for src, dst in (
        ("logical_moves", "moves"),
        ("evidence_use", "evidence_use"),
        ("citation_refs", "citation_refs"),
        ("theorem_or_equation_refs", "theorem_or_equation_refs"),
        ("appendix_bindings", "appendix_bindings"),
    ):
        if src in fields:
            bp[dst] = list(fields[src])
    if "sentence_plan" in fields:
        decoded = _decode_json_bullets(fields["sentence_plan"], field="sentence_plan")
        if not all(isinstance(item, dict) for item in decoded):
            raise PipelineError("sentence_plan bullets must decode to mappings")
        bp["sentence_plan"] = decoded
    if "word_budget" in fields:
        bp["word_budget"] = _parse_word_budget(fields["word_budget"])
    node["must_include"] = list(bp.get("moves", []))
    node["evidence"] = list(bp.get("evidence_use", []))
    node["prose"] = fields.get("body", "")


def artifact_paths(graph: dict[str, Any], root: Path) -> dict[str, Path]:
    art = graph.get("artifacts", {})
    return {
        "outline": root / art["outline"],
        "blueprint": root / art["blueprint"],
        "prose": root / art["prose"],
        "state": root / art["state"],
    }


def render(graph: dict[str, Any], root: Path) -> dict[str, Any]:
    paths = artifact_paths(graph, root)
    for layer in ("outline", "blueprint", "prose"):
        paths[layer].parent.mkdir(parents=True, exist_ok=True)
        paths[layer].write_text(render_markdown(graph, layer), encoding="utf-8")
    render_tex_sections(graph, root)
    state = {
        "schema_version": 1,
        "graph_sha256": sha256_file(root / graph["graph_path"]),
        "layers": {},
    }
    for layer in ("outline", "blueprint", "prose"):
        blocks = parse_blocks(paths[layer])
        state["layers"][layer] = {
            "path": str(paths[layer].relative_to(root)),
            "file_sha256": sha256_file(paths[layer]),
            "blocks": {node_id: block.digest for node_id, block in blocks.items()},
        }
    paths["state"].parent.mkdir(parents=True, exist_ok=True)
    paths["state"].write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return state


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"layers": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def sync(
    graph: dict[str, Any], root: Path, generator_cmd: str | None, prefer: str | None
) -> list[str]:
    paths = artifact_paths(graph, root)
    state = load_state(paths["state"])
    current: dict[str, dict[str, ParsedBlock]] = {}
    changed: dict[str, list[str]] = {}
    for layer in ("outline", "blueprint", "prose"):
        current[layer] = parse_blocks(paths[layer])
        previous = state.get("layers", {}).get(layer, {}).get("blocks", {})
        for node_id, block in current[layer].items():
            if previous.get(node_id) != block.digest:
                changed.setdefault(node_id, []).append(layer)
    if not changed:
        return []
    nodes = node_map(graph)
    imported: list[str] = []
    for node_id, layers in sorted(changed.items()):
        if node_id not in nodes:
            raise PipelineError(f"edited artifact contains unknown node: {node_id}")
        if len(layers) > 1:
            if prefer not in layers:
                raise PipelineError(
                    f"conflicting edits for {node_id}: {layers}; rerun with --prefer one of them"
                )
            source = prefer
        else:
            source = layers[0]
        node = nodes[node_id]
        if source == "outline":
            apply_outline_import(node, parse_outline_block(current[source][node_id]))
        elif source == "blueprint":
            apply_blueprint_import(node, parse_blueprint_block(current[source][node_id]))
        else:
            apply_prose_import(node, parse_prose_block(current[source][node_id]))
        if generator_cmd:
            command_regenerate(node, source, generator_cmd)
        else:
            deterministic_regenerate(node, source)
        imported.append(f"{node_id}:{source}")
    graph_path = root / graph["graph_path"]
    persisted = dict(graph)
    persisted.pop("graph_path", None)
    write_yaml(graph_path, persisted)
    render(graph, root)
    return imported


def apply_delta(
    graph: dict[str, Any], root: Path, delta_path: Path, generator_cmd: str | None
) -> list[str]:
    payload = read_yaml(delta_path)
    changes = payload.get("changes")
    if not isinstance(changes, list) or not changes:
        raise PipelineError("delta file must contain a non-empty changes list")
    nodes = node_map(graph)
    applied: list[str] = []
    allowed = {
        "title",
        "claim",
        "reader_question",
        "role",
        "evidence",
        "must_include",
        "must_avoid",
        "evidence_status",
        "blueprint",
        "prose",
    }
    for row in changes:
        if not isinstance(row, dict) or not row.get("id"):
            raise PipelineError("each delta change must be a mapping with id")
        node_id = str(row["id"])
        if node_id not in nodes:
            raise PipelineError(f"delta references unknown node: {node_id}")
        unknown = set(row) - allowed - {"id", "regenerate"}
        if unknown:
            raise PipelineError(f"delta for {node_id} has unknown fields: {sorted(unknown)}")
        node = nodes[node_id]
        for key in allowed:
            if key in row:
                node[key] = row[key]
        if row.get("regenerate", True):
            if generator_cmd:
                command_regenerate(node, "delta", generator_cmd)
            elif "prose" not in row:
                deterministic_regenerate(node, "delta")
        applied.append(node_id)
    graph_path = root / graph["graph_path"]
    persisted = dict(graph)
    persisted.pop("graph_path", None)
    write_yaml(graph_path, persisted)
    render(graph, root)
    return applied


def validate(graph: dict[str, Any], root: Path) -> dict[str, Any]:
    errors: list[str] = []
    project_profile = load_project_profile(graph, root)
    sections = section_map(graph)
    nodes = nodes_sorted(graph)
    ids: set[str] = set()
    for node in nodes:
        node_id = str(node.get("id", ""))
        if not re.fullmatch(r"[A-Z][A-Z0-9-]*", node_id):
            errors.append(f"invalid node id: {node_id}")
        if node_id in ids:
            errors.append(f"duplicate node id: {node_id}")
        ids.add(node_id)
        if str(node.get("section")) not in sections:
            errors.append(f"node {node_id} references unknown section")
        for field in ("title", "claim", "reader_question", "role", "prose"):
            if not str(node.get(field, "")).strip():
                errors.append(f"node {node_id} missing {field}")
        errors.extend(project_validation_errors(node, project_profile))
        if node.get("evidence_status") in {"tbd", "not_run"}:
            numeric = re.search(r"\b\d+(?:\.\d+)?(?:%|x|×)\b", str(node.get("prose", "")), re.I)
            if numeric and "TBD" not in str(node.get("prose", "")):
                errors.append(f"node {node_id} has numeric claim despite TBD evidence")
    paths = artifact_paths(graph, root)
    expected_ids = [str(node["id"]) for node in nodes]
    parsed: dict[str, dict[str, ParsedBlock]] = {}
    for layer in ("outline", "blueprint", "prose"):
        if not paths[layer].exists():
            errors.append(f"missing generated {layer}: {paths[layer]}")
            continue
        try:
            parsed[layer] = parse_blocks(paths[layer])
        except PipelineError as exc:
            errors.append(str(exc))
            continue
        if list(parsed[layer]) != expected_ids:
            errors.append(f"{layer} node order/coverage differs from graph")
    if all(layer in parsed for layer in ("outline", "blueprint", "prose")):
        for node_id in expected_ids:
            oh = parsed["outline"][node_id].digest
            bp_lines = parsed["blueprint"][node_id].payload.splitlines()[1:]
            bp_parent = PARENT_OUTLINE_RE.match(bp_lines[0]) if bp_lines else None
            if not bp_parent or bp_parent.group(1) != oh:
                errors.append(f"blueprint parent hash mismatch for {node_id}")
            bh = parsed["blueprint"][node_id].digest
            prose_lines = parsed["prose"][node_id].payload.splitlines()[1:]
            prose_parent = PARENT_BLUEPRINT_RE.match(prose_lines[0]) if prose_lines else None
            if not prose_parent or prose_parent.group(1) != bh:
                errors.append(f"prose parent hash mismatch for {node_id}")
    overleaf = root / graph["artifacts"]["overleaf_root"]
    main_tex = overleaf / "main.tex"
    if not main_tex.exists():
        errors.append("missing generated Overleaf main.tex")
    else:
        tex_text = (
            "\n".join(
                path.read_text(encoding="utf-8")
                for path in sorted((overleaf / "sections").glob("*.tex"))
            )
            + "\n"
            + "\n".join(
                path.read_text(encoding="utf-8")
                for path in sorted((overleaf / "appendix").glob("*.tex"))
            )
        )
        for node_id in expected_ids:
            if tex_text.count(f"% MANUSCRIPT-NODE: {node_id}") != 1:
                errors.append(f"TeX projection missing or duplicates node {node_id}")
    if errors:
        raise PipelineError("validation failed:\n- " + "\n- ".join(errors))
    return {"status": "PASS", "nodes": len(nodes), "sections": len(sections)}


def run_checked(command: list[str], cwd: Path) -> None:
    proc = subprocess.run(
        command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if proc.returncode != 0:
        raise PipelineError(f"command failed ({' '.join(command)}):\n{proc.stdout}")
    print(proc.stdout, end="")


def compile_pdf(graph: dict[str, Any], root: Path) -> Path:
    overleaf = root / graph["artifacts"]["overleaf_root"]
    subprocess.run(
        ["latexmk", "-C", "main.tex"],
        cwd=overleaf,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    bibtex_rule = '$bibtex="/usr/bin/bibtex.original %O %B"'
    command = [
        "latexmk",
        "-e",
        bibtex_rule,
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "main.tex",
    ]
    run_checked(command, overleaf)
    run_checked(command, overleaf)
    pdf = overleaf / "main.pdf"
    if not pdf.exists():
        raise PipelineError("latexmk completed without main.pdf")
    log_path = overleaf / "main.log"
    if log_path.exists():
        log = log_path.read_text(encoding="utf-8", errors="replace")
        fatal_patterns = (
            "There were undefined references",
            "There were undefined citations",
            "Overfull \\hbox",
            "Overfull \\vbox",
        )
        found = [item for item in fatal_patterns if item in log]
        if found:
            raise PipelineError("LaTeX quality audit failed: " + ", ".join(found))
    return pdf


ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def package_overleaf(graph: dict[str, Any], root: Path, output: Path) -> Path:
    """Create a byte-reproducible Overleaf archive.

    ``zipfile.write`` records checkout mtimes, so the previously tracked release
    ZIP became dirty every time the package gate ran in a fresh worktree.  Write
    members explicitly with a fixed timestamp and stable POSIX modes instead.
    """
    overleaf = root / graph["artifacts"]["overleaf_root"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(overleaf.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(overleaf)
            if rel.parts and rel.parts[0] in {"legacy_source", "build", "releases"}:
                continue
            if path.suffix in {
                ".aux",
                ".bbl",
                ".blg",
                ".fdb_latexmk",
                ".fls",
                ".log",
                ".out",
                ".synctex.gz",
            }:
                continue
            info = zipfile.ZipInfo(rel.as_posix(), date_time=ZIP_EPOCH)
            info.create_system = 3
            mode = stat.S_IMODE(path.stat().st_mode)
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(
                info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("render", "sync", "apply-delta", "validate", "compile", "package-overleaf", "all"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--graph", type=Path, default=Path("docs/manuscript/paper_graph.yaml"))
    parser.add_argument("--generator-cmd")
    parser.add_argument("--prefer", choices=("outline", "blueprint", "prose"))
    parser.add_argument("--delta", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("paper/releases/manuscript_overleaf.zip")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    graph_path = (root / args.graph).resolve()
    graph = read_yaml(graph_path)
    graph["graph_path"] = graph_path.relative_to(root).as_posix()
    try:
        if args.command == "render":
            render(graph, root)
        elif args.command == "sync":
            imported = sync(graph, root, args.generator_cmd, args.prefer)
            print("Imported: " + (", ".join(imported) if imported else "none"))
        elif args.command == "apply-delta":
            if args.delta is None:
                raise PipelineError("apply-delta requires --delta")
            applied = apply_delta(graph, root, (root / args.delta).resolve(), args.generator_cmd)
            print("Applied delta: " + ", ".join(applied))
        elif args.command == "validate":
            print(json.dumps(validate(graph, root), indent=2))
        elif args.command == "compile":
            print(compile_pdf(graph, root))
        elif args.command == "package-overleaf":
            print(package_overleaf(graph, root, (root / args.output).resolve()))
        else:
            render(graph, root)
            print(json.dumps(validate(graph, root), indent=2))
            print(compile_pdf(graph, root))
            print(package_overleaf(graph, root, (root / args.output).resolve()))
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
