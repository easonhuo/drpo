#!/usr/bin/env python3
"""Publication-quality outline -> blueprint -> prose compiler and gate.

This layer does not replace the manuscript graph.  It turns the graph's rich
outline and paragraph-blueprint contracts into auditable generation packets,
then verifies that the prose realizes those packets before the release build.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


class PublicationQualityError(RuntimeError):
    pass


@dataclass(frozen=True)
class Paths:
    root: Path
    graph: Path
    contract: Path
    output: Path


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PublicationQualityError(f"cannot read YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PublicationQualityError(f"YAML root must be a mapping: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def strip_latex(text: str) -> str:
    value = re.sub(r"%.*", " ", text)
    value = re.sub(r"\\(?:begin|end)\{[^}]+\}", " ", value)
    value = re.sub(r"\\(?:cite\w*|ref|cref|Cref|eqref)\{[^}]+\}", " ", value)
    value = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?", " ", value)
    value = value.replace("{", " ").replace("}", " ").replace("$", " ")
    return re.sub(r"\s+", " ", value).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", strip_latex(text)))


def sentence_count(text: str) -> int:
    plain = strip_latex(text)
    return len([part for part in re.split(r"(?<=[.!?])\s+", plain) if part.strip()])


def citation_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for group in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", text):
        keys.update(item.strip() for item in group.split(",") if item.strip())
    return keys


def bibliography_keys(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"@[A-Za-z]+\s*\{\s*([^,\s]+)\s*,", text))


def require_string_list(value: Any, *, label: str, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise PublicationQualityError(
            f"{label} must be a {'possibly empty ' if allow_empty else ''}list"
        )
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise PublicationQualityError(f"{label} contains an empty or non-string item")
        result.append(item.strip())
    return result


def graph_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise PublicationQualityError("paper graph nodes must be a list")
    checked: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict) or not str(node.get("id", "")).strip():
            raise PublicationQualityError("paper graph contains an invalid node")
        checked.append(node)
    return checked


def selected_nodes(graph: dict[str, Any], contract: dict[str, Any]) -> list[dict[str, Any]]:
    scope = contract.get("scope", {})
    ids = require_string_list(scope.get("node_ids"), label="quality scope node_ids")
    by_id = {str(node["id"]): node for node in graph_nodes(graph)}
    missing = [node_id for node_id in ids if node_id not in by_id]
    if missing:
        raise PublicationQualityError(f"quality contract references missing graph nodes: {missing}")
    return [by_id[node_id] for node_id in ids]


def validate_outline(node: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    node_id = str(node["id"])
    errors: list[str] = []
    for field in profile.get("required_outline_fields", []):
        value = node.get(field)
        if not value:
            errors.append(f"{node_id}: outline lacks {field}")
    must_include = node.get("must_include")
    if not isinstance(must_include, list) or len(must_include) < int(
        profile.get("min_outline_obligations", 1)
    ):
        errors.append(f"{node_id}: outline has too few content obligations")
    if node.get("claim") == node.get("reader_question"):
        errors.append(f"{node_id}: claim and reader question are identical")
    return errors


def validate_blueprint(
    node: dict[str, Any], profile: dict[str, Any], required_roles: list[str]
) -> list[str]:
    node_id = str(node["id"])
    bp = node.get("blueprint")
    if not isinstance(bp, dict):
        return [f"{node_id}: blueprint is missing"]
    errors: list[str] = []
    for field in profile.get("required_blueprint_fields", []):
        if field not in bp or bp.get(field) is None:
            errors.append(f"{node_id}: blueprint lacks {field}")
        elif isinstance(bp.get(field), str) and not str(bp.get(field)).strip():
            errors.append(f"{node_id}: blueprint has empty {field}")
    plan = bp.get("sentence_plan")
    if not isinstance(plan, list):
        return errors + [f"{node_id}: blueprint sentence_plan must be a list"]
    if len(plan) < int(profile.get("min_sentence_units", 1)):
        errors.append(f"{node_id}: blueprint has too few sentence units")
    roles: list[str] = []
    for index, step in enumerate(plan, start=1):
        if not isinstance(step, dict):
            errors.append(f"{node_id}: sentence unit {index} is not a mapping")
            continue
        role = str(step.get("role", "")).strip()
        instruction = str(step.get("instruction", "")).strip()
        anchors = step.get("anchors")
        if not role or not instruction:
            errors.append(f"{node_id}: sentence unit {index} lacks role or instruction")
        if not isinstance(anchors, list) or not anchors:
            errors.append(f"{node_id}: sentence unit {index} lacks coverage anchors")
        roles.append(role)
    if len(set(roles)) != len(roles):
        errors.append(f"{node_id}: sentence roles are not unique")
    missing_roles = [role for role in required_roles if role not in roles]
    if missing_roles:
        errors.append(f"{node_id}: missing required sentence roles {missing_roles}")
    budget = bp.get("word_budget")
    if not (
        isinstance(budget, list)
        and len(budget) == 2
        and all(isinstance(x, int) and x > 0 for x in budget)
    ):
        errors.append(f"{node_id}: word_budget must be [min,max] positive integers")
    elif budget[0] > budget[1]:
        errors.append(f"{node_id}: invalid word_budget ordering")
    topic = str(bp.get("topic_sentence", "")).strip()
    claim = str(node.get("claim", "")).strip()
    if topic and re.sub(r"\W+", "", topic.lower()) == re.sub(r"\W+", "", claim.lower()):
        errors.append(f"{node_id}: topic sentence merely copies the outline claim")
    return errors


def realization_trace(node: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    node_id = str(node["id"])
    prose = str(node.get("prose", ""))
    lower = strip_latex(prose).lower()
    trace: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, step in enumerate(node["blueprint"]["sentence_plan"], start=1):
        anchors = [str(item).strip() for item in step.get("anchors", []) if str(item).strip()]
        matched = [anchor for anchor in anchors if strip_latex(anchor).lower() in lower]
        sid = f"{node_id}-S{index:02d}"
        trace.append({"sid": sid, "role": step.get("role"), "anchors": anchors, "matched": matched})
        if not matched:
            errors.append(f"{node_id}: prose does not realize {sid} ({step.get('role')})")
    return trace, errors


def validate_prose(
    node: dict[str, Any],
    profile: dict[str, Any],
    bib_keys: set[str],
    all_tex: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    node_id = str(node["id"])
    prose = str(node.get("prose", "")).strip()
    errors: list[str] = []
    bp = node["blueprint"]
    budget = bp["word_budget"]
    count = word_count(prose)
    if count < budget[0] or count > budget[1]:
        errors.append(f"{node_id}: prose word count {count} outside budget {budget}")
    if sentence_count(prose) < int(profile.get("min_prose_sentences", 1)):
        errors.append(f"{node_id}: prose has too few complete explanatory sentences")
    required_citations = set(
        require_string_list(
            bp.get("citation_refs", []), label=f"{node_id} citation_refs", allow_empty=True
        )
    )
    unknown = required_citations - bib_keys
    if unknown:
        errors.append(f"{node_id}: blueprint cites missing bibliography keys {sorted(unknown)}")
    missing_in_prose = required_citations - citation_keys(prose)
    if missing_in_prose:
        errors.append(f"{node_id}: prose omits required citations {sorted(missing_in_prose)}")
    for token in require_string_list(
        bp.get("theorem_or_equation_refs", []),
        label=f"{node_id} theorem_or_equation_refs",
        allow_empty=True,
    ):
        if (
            token not in prose
            and f"\\label{{{token}}}" not in prose
            and f"\\ref{{{token}}}" not in prose
            and f"\\eqref{{{token}}}" not in prose
        ):
            errors.append(f"{node_id}: prose omits theorem/equation binding {token}")
    for label in require_string_list(
        bp.get("appendix_bindings", []), label=f"{node_id} appendix_bindings", allow_empty=True
    ):
        if f"\\label{{{label}}}" not in all_tex:
            errors.append(f"{node_id}: appendix binding has no label in TeX: {label}")
    for forbidden in require_string_list(
        bp.get("forbidden_conclusions", []),
        label=f"{node_id} forbidden_conclusions",
        allow_empty=True,
    ):
        if strip_latex(forbidden).lower() in strip_latex(prose).lower():
            errors.append(f"{node_id}: prose contains forbidden conclusion: {forbidden}")
    trace, trace_errors = realization_trace(node)
    errors.extend(trace_errors)
    return errors, trace


def prompt_packet(node: dict[str, Any]) -> dict[str, Any]:
    bp = node["blueprint"]
    units: list[dict[str, Any]] = []
    for index, step in enumerate(bp["sentence_plan"], start=1):
        units.append(
            {
                "sid": f"{node['id']}-S{index:02d}",
                "role": step["role"],
                "instruction": step["instruction"],
                "anchors": step["anchors"],
            }
        )
    prompt = "\n".join(
        [
            f"Write the publication-ready paragraph(s) for {node['id']} — {node['title']}.",
            f"Reader question: {node['reader_question']}",
            f"Outline claim: {node['claim']}",
            f"Word budget: {bp['word_budget'][0]}-{bp['word_budget'][1]}",
            "Realize every sentence unit in order. Do not collapse definitions, results, interpretation, boundary, and transition into one sentence.",
            *[
                f"- {unit['sid']} [{unit['role']}]: {unit['instruction']} Anchors={unit['anchors']}"
                for unit in units
            ],
            f"Citations: {bp.get('citation_refs', [])}",
            f"Theorem/equation bindings: {bp.get('theorem_or_equation_refs', [])}",
            f"Appendix bindings: {bp.get('appendix_bindings', [])}",
            f"Allowed conclusions: {bp.get('allowed_conclusions', [])}",
            f"Forbidden conclusions: {bp.get('forbidden_conclusions', [])}",
            f"Reviewer objection: {bp.get('reviewer_objection', '')}",
            f"Required response: {bp.get('objection_response', '')}",
            f"Transition: {bp.get('transition', '')}",
            "Use exact registered terminology. Do not invent evidence, citations, results, or stronger universality claims.",
        ]
    )
    return {
        "id": node["id"],
        "title": node["title"],
        "outline_claim": node["claim"],
        "reader_question": node["reader_question"],
        "word_budget": bp["word_budget"],
        "sentence_units": units,
        "citation_refs": bp.get("citation_refs", []),
        "theorem_or_equation_refs": bp.get("theorem_or_equation_refs", []),
        "appendix_bindings": bp.get("appendix_bindings", []),
        "prompt": prompt,
    }


def invoke_generator(command: str, packet: dict[str, Any]) -> str:
    proc = subprocess.run(
        command,
        input=json.dumps(packet, ensure_ascii=False),
        text=True,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise PublicationQualityError(f"generator failed for {packet['id']}: {proc.stderr.strip()}")
    try:
        response = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise PublicationQualityError(
            f"generator returned invalid JSON for {packet['id']}"
        ) from exc
    prose = response.get("prose") if isinstance(response, dict) else None
    if not isinstance(prose, str) or not prose.strip():
        raise PublicationQualityError(f"generator returned no prose for {packet['id']}")
    return prose.strip()


def render_blueprint(nodes: Iterable[dict[str, Any]], path: Path) -> None:
    lines = ["# Publication-quality paragraph blueprint", ""]
    for node in nodes:
        bp = node["blueprint"]
        lines.extend(
            [
                f"## [{node['id']}] {node['title']}",
                "",
                f"- Reader question: {node['reader_question']}",
                f"- Paragraph claim: {node['claim']}",
                f"- Word budget: {bp['word_budget'][0]}-{bp['word_budget'][1]}",
                "- Sentence units:",
            ]
        )
        for index, step in enumerate(bp["sentence_plan"], start=1):
            lines.append(
                f"  {index}. `{node['id']}-S{index:02d}` **{step['role']}** — {step['instruction']} "
                f"(anchors: {', '.join(step['anchors'])})"
            )
        lines.extend(
            [
                f"- Citations: {', '.join(bp.get('citation_refs', [])) or 'none'}",
                f"- Theorem/equation refs: {', '.join(bp.get('theorem_or_equation_refs', [])) or 'none'}",
                f"- Appendix bindings: {', '.join(bp.get('appendix_bindings', [])) or 'none'}",
                f"- Reviewer objection: {bp['reviewer_objection']}",
                f"- Response: {bp['objection_response']}",
                f"- Transition: {bp['transition']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def render_prose(
    nodes: Iterable[dict[str, Any]], path: Path, generated: dict[str, str] | None = None
) -> None:
    lines = ["# Publication-quality prose candidate", ""]
    generated = generated or {}
    for node in nodes:
        text = generated.get(str(node["id"]), str(node.get("prose", "")).strip())
        lines.extend([f"## [{node['id']}] {node['title']}", "", text, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build(paths: Paths, generator_cmd: str | None = None) -> dict[str, Any]:
    graph = read_yaml(paths.graph)
    contract = read_yaml(paths.contract)
    nodes = selected_nodes(graph, contract)
    packets = [prompt_packet(node) for node in nodes]
    generated: dict[str, str] = {}
    if generator_cmd:
        for packet in packets:
            generated[str(packet["id"])] = invoke_generator(generator_cmd, packet)
    paths.output.mkdir(parents=True, exist_ok=True)
    render_blueprint(nodes, paths.output / "blueprint.md")
    render_prose(nodes, paths.output / "prose_candidate.md", generated)
    write_json(paths.output / "prose_packets.json", {"schema_version": 1, "packets": packets})
    return {"status": "BUILT", "node_count": len(nodes), "generated_count": len(generated)}


def validate(paths: Paths) -> dict[str, Any]:
    graph = read_yaml(paths.graph)
    contract = read_yaml(paths.contract)
    nodes = selected_nodes(graph, contract)
    profiles = contract.get("section_profiles")
    if not isinstance(profiles, dict):
        raise PublicationQualityError("quality contract section_profiles must be a mapping")
    roles = contract.get("required_roles")
    if not isinstance(roles, dict):
        raise PublicationQualityError("quality contract required_roles must be a mapping")
    bib_path = paths.root / contract["bibliography"]
    bib = bibliography_keys(bib_path)
    all_tex = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((paths.root / "paper/overleaf").rglob("*.tex"))
    )
    errors: list[str] = []
    node_reports: list[dict[str, Any]] = []
    for node in nodes:
        section = str(node["section"])
        profile = profiles.get(section)
        if not isinstance(profile, dict):
            errors.append(f"{node['id']}: no section profile for {section}")
            continue
        required = roles.get(str(node["id"]), [])
        if not isinstance(required, list):
            errors.append(f"{node['id']}: required_roles entry must be a list")
            required = []
        node_errors = validate_outline(node, profile)
        node_errors.extend(validate_blueprint(node, profile, [str(x) for x in required]))
        trace: list[dict[str, Any]] = []
        if not node_errors:
            prose_errors, trace = validate_prose(node, profile, bib, all_tex)
            node_errors.extend(prose_errors)
        errors.extend(node_errors)
        node_reports.append(
            {
                "id": node["id"],
                "section": section,
                "word_count": word_count(str(node.get("prose", ""))),
                "sentence_count": sentence_count(str(node.get("prose", ""))),
                "trace": trace,
                "status": "PASS" if not node_errors else "FAIL",
                "errors": node_errors,
            }
        )
    report = {
        "schema_version": 1,
        "profile": contract.get("profile"),
        "node_count": len(nodes),
        "status": "PASS" if not errors else "FAIL",
        "nodes": node_reports,
        "errors": errors,
    }
    paths.output.mkdir(parents=True, exist_ok=True)
    write_json(paths.output / "quality_report.json", report)
    if errors:
        raise PublicationQualityError(
            "publication-quality validation failed:\n- " + "\n- ".join(errors)
        )
    return report


def make_paths(args: argparse.Namespace) -> Paths:
    root = args.repo_root.resolve()
    return Paths(
        root=root,
        graph=(root / args.graph).resolve(),
        contract=(root / args.contract).resolve(),
        output=(root / args.output).resolve(),
    )


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("build", "validate", "all"))
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--graph", type=Path, default=Path("docs/manuscript/paper_graph.yaml"))
    ap.add_argument(
        "--contract",
        type=Path,
        default=Path("docs/manuscript/publication_quality_contract.yaml"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("paper/publication_quality_v1"),
    )
    ap.add_argument("--generator-cmd")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    paths = make_paths(args)
    try:
        result: dict[str, Any] = {}
        if args.command in {"build", "all"}:
            result["build"] = build(paths, args.generator_cmd)
        if args.command in {"validate", "all"}:
            result["validate"] = validate(paths)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except PublicationQualityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
