#!/usr/bin/env python3
"""Compile an executable manuscript blueprint into prose and visual products.

This is a small, generic downstream graph transformer.  It deliberately reuses
exactly the same structural contract as the approved-outline -> blueprint stage:
stable IDs, exact order, explicit disabled nodes, parent hashes, deterministic
outputs, and fail-closed validation.  Node-specific work is isolated in a YAML
adapter contract rather than a second manuscript pipeline.

The transformer produces:

* ``product_graph.json``: one-to-one blueprint lineage plus sentence units;
* ``prose_packets.json``: evidence-bound paragraph packets for LLM refinement;
* ``prose_draft.md``: deterministic first-pass prose rendered from templates;
* ``figure_specs.json``: claim/metric/panel bindings;
* empirical figures declared by the contract; and
* ``validation_report.json``.

It does not change any scientific result or experiment state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class TransformError(RuntimeError):
    """Expected fail-closed downstream-transform error."""


@dataclass(frozen=True)
class TransformPaths:
    repo: Path
    contract: Path
    blueprint: Path
    snapshot: Path
    output: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TransformError(f"expected JSON mapping: {path}")
    return payload


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TransformError(f"expected YAML mapping: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_repo_path(repo: Path, value: str) -> Path:
    candidate = (repo / value).resolve()
    try:
        candidate.relative_to(repo.resolve())
    except ValueError as exc:
        raise TransformError(f"path escapes repository: {value}") from exc
    return candidate


def resolve_value(root: Any, dotted_path: str) -> Any:
    """Resolve mapping keys and numeric list indices in a dotted path."""
    current = root
    for component in dotted_path.split("."):
        if isinstance(current, Mapping):
            if component not in current:
                raise TransformError(f"metric path does not resolve: {dotted_path}")
            current = current[component]
        elif isinstance(current, list) and component.isdigit():
            index = int(component)
            if index >= len(current):
                raise TransformError(f"metric list index is out of range: {dotted_path}")
            current = current[index]
        else:
            raise TransformError(f"metric path does not resolve: {dotted_path}")
    if current is None:
        raise TransformError(f"metric path resolves to null: {dotted_path}")
    if isinstance(current, float) and not math.isfinite(current):
        raise TransformError(f"metric path resolves to a non-finite value: {dotted_path}")
    return current


def require_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TransformError(f"{label} must be a non-empty string")
    return value.strip()


def require_string_list(value: Any, *, label: str, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise TransformError(
            f"{label} must be a {'possibly empty' if allow_empty else 'non-empty'} list"
        )
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise TransformError(f"{label} must contain non-empty strings")
    return [str(item).strip() for item in value]


def blueprint_nodes(blueprint: Mapping[str, Any]) -> list[dict[str, Any]]:
    nodes = blueprint.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise TransformError("blueprint nodes must be a non-empty list")
    checked: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            raise TransformError("blueprint contains a non-mapping node")
        require_string(node.get("id"), label="blueprint node id")
        checked.append(node)
    ids = [str(node["id"]) for node in checked]
    if len(ids) != len(set(ids)):
        raise TransformError("blueprint contains duplicate node IDs")
    return checked


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def role_map(node: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    plan = node.get("sentence_plan")
    if not isinstance(plan, list) or not plan:
        raise TransformError(f"enabled node {node.get('id')} has no sentence plan")
    result: dict[str, dict[str, Any]] = {}
    for step in plan:
        if not isinstance(step, dict):
            raise TransformError(f"node {node.get('id')} has an invalid sentence-plan step")
        role = require_string(step.get("role"), label=f"{node.get('id')} sentence role")
        require_string(step.get("instruction"), label=f"{node.get('id')} instruction")
        if role in result:
            raise TransformError(f"node {node.get('id')} repeats sentence role {role}")
        result[role] = step
    return result


def build_prompt(node: Mapping[str, Any], units: list[Mapping[str, Any]]) -> str:
    lines = [
        f"Write exactly one manuscript paragraph for {node['id']}: {node['title']}.",
        f"Reader question: {node.get('reader_question', '')}",
        f"Paragraph claim: {node.get('paragraph_claim', '')}",
        "Realize the following sentence units in order without combining or omitting them:",
    ]
    for unit in units:
        lines.append(
            f"- {unit['sid']} [{unit['role']}]: {unit['instruction']} "
            f"Evidence={unit['evidence_refs']} Metrics={unit['metric_paths']}"
        )
    lines.extend(
        [
            f"Allowed conclusions: {node.get('allowed_conclusions', [])}",
            f"Forbidden conclusions: {node.get('forbidden_conclusions', [])}",
            f"Reviewer objection to pre-empt: {node.get('reviewer_objection', '')}",
            f"Required response: {node.get('objection_response', '')}",
            f"Transition: {node.get('transition_to_next', '')}",
            "Do not introduce unregistered empirical claims, method rankings, or OOD wording.",
        ]
    )
    return "\n".join(lines)


_TEMPLATE_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?::([^{}]+))?\}")


def render_template(template: str, values: Mapping[str, Any], *, label: str) -> str:
    """Render a deliberately restricted format-string dialect."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        fmt = match.group(2)
        if name not in values:
            raise TransformError(f"{label} references unknown metric alias {name}")
        value = values[name]
        if fmt:
            try:
                return format(value, fmt)
            except (TypeError, ValueError) as exc:
                raise TransformError(f"cannot format {name} with {fmt!r} in {label}") from exc
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    rendered = _TEMPLATE_RE.sub(replace, template).strip()
    if "{" in rendered or "}" in rendered:
        raise TransformError(f"unresolved template token in {label}: {rendered}")
    return rendered


def compile_products(
    *,
    blueprint: dict[str, Any],
    snapshot: dict[str, Any],
    contract: dict[str, Any],
    blueprint_sha: str,
    snapshot_sha: str,
) -> dict[str, Any]:
    if contract.get("schema_version") != 1:
        raise TransformError("downstream contract schema_version must be 1")
    node_contracts = contract.get("nodes")
    if not isinstance(node_contracts, dict):
        raise TransformError("downstream contract nodes must be a mapping")

    source_nodes = blueprint_nodes(blueprint)
    enabled_ids = [str(node["id"]) for node in source_nodes if node.get("status") == "enabled"]
    if set(node_contracts) != set(enabled_ids):
        raise TransformError(
            "downstream contract must cover exactly the enabled blueprint nodes; "
            f"enabled={enabled_ids}, configured={sorted(node_contracts)}"
        )

    product_nodes: list[dict[str, Any]] = []
    prose_packets: list[dict[str, Any]] = []
    figure_specs: list[dict[str, Any]] = []
    prose_paragraphs: list[dict[str, str]] = []

    for source in source_nodes:
        node_id = str(source["id"])
        common = {
            "id": node_id,
            "section": source.get("section"),
            "order": source.get("order"),
            "title": source.get("title"),
            "outline_block_sha256": source.get("outline_block_sha256"),
            "source_blueprint_sha256": blueprint_sha,
            "status": source.get("status"),
        }
        if source.get("status") == "disabled_with_reason":
            product_nodes.append({**common, "disabled_reason": source.get("disabled_reason")})
            continue
        if source.get("status") != "enabled":
            raise TransformError(f"unknown blueprint status for {node_id}: {source.get('status')}")

        source_roles = role_map(source)
        adapter = node_contracts[node_id]
        if not isinstance(adapter, dict):
            raise TransformError(f"contract node {node_id} must be a mapping")
        bindings = adapter.get("sentence_bindings")
        if not isinstance(bindings, dict) or set(bindings) != set(source_roles):
            raise TransformError(
                f"sentence bindings for {node_id} must match blueprint roles exactly; "
                f"blueprint={list(source_roles)}, configured={list(bindings or {})}"
            )

        units: list[dict[str, Any]] = []
        paragraphs: list[str] = []
        assigned_metrics: set[str] = set()
        assigned_figures: set[str] = set()
        assigned_tables: set[str] = set()

        for index, (role, source_step) in enumerate(source_roles.items(), start=1):
            binding = bindings[role]
            if not isinstance(binding, dict):
                raise TransformError(f"sentence binding {node_id}:{role} must be a mapping")
            metric_aliases = binding.get("metrics", {})
            if not isinstance(metric_aliases, dict):
                raise TransformError(f"metrics for {node_id}:{role} must be a mapping")
            values: dict[str, Any] = {}
            metric_paths: list[str] = []
            for alias, path in metric_aliases.items():
                require_string(alias, label=f"metric alias for {node_id}:{role}")
                path = require_string(path, label=f"metric path for {node_id}:{role}:{alias}")
                values[str(alias)] = resolve_value(snapshot, path)
                metric_paths.append(path)
                assigned_metrics.add(path)
            evidence = binding.get("evidence_refs", source.get("evidence_refs", []))
            evidence_refs = require_string_list(
                evidence, label=f"evidence refs for {node_id}:{role}", allow_empty=False
            )
            figure_bindings = require_string_list(
                binding.get("figure_bindings", []),
                label=f"figure bindings for {node_id}:{role}",
                allow_empty=True,
            )
            table_bindings = require_string_list(
                binding.get("table_bindings", []),
                label=f"table bindings for {node_id}:{role}",
                allow_empty=True,
            )
            assigned_figures.update(figure_bindings)
            assigned_tables.update(table_bindings)
            template = require_string(
                binding.get("template"), label=f"template for {node_id}:{role}"
            )
            rendered = render_template(template, values, label=f"{node_id}:{role}")
            sid = f"{node_id}-S{index:02d}"
            unit = {
                "sid": sid,
                "role": role,
                "instruction": source_step["instruction"],
                "evidence_refs": evidence_refs,
                "metric_paths": metric_paths,
                "metric_values": values,
                "figure_bindings": figure_bindings,
                "table_bindings": table_bindings,
                "draft_sentence": rendered,
            }
            units.append(unit)
            paragraphs.append(rendered)

        source_metric_paths = set(source.get("metric_paths", []))
        allowed_unassigned = set(adapter.get("allowed_unassigned_metric_paths", []))
        covered_source_metrics = {
            source_path
            for source_path in source_metric_paths
            if any(
                assigned_path == source_path or assigned_path.startswith(source_path + ".")
                for assigned_path in assigned_metrics
            )
        }
        unknown_metrics = {
            assigned_path
            for assigned_path in assigned_metrics
            if not any(
                assigned_path == source_path or assigned_path.startswith(source_path + ".")
                for source_path in source_metric_paths
            )
        }
        unresolved_metrics = source_metric_paths - covered_source_metrics - allowed_unassigned
        if unresolved_metrics:
            raise TransformError(
                f"node {node_id} leaves blueprint metrics unassigned: {sorted(unresolved_metrics)}"
            )
        if unknown_metrics:
            raise TransformError(
                f"node {node_id} assigns metrics absent from blueprint: {sorted(unknown_metrics)}"
            )

        visuals = adapter.get("visual_products", [])
        if not isinstance(visuals, list):
            raise TransformError(f"visual_products for {node_id} must be a list")
        configured_figure_ids: set[str] = set()
        for visual in visuals:
            if not isinstance(visual, dict):
                raise TransformError(f"visual product for {node_id} must be a mapping")
            figure_id = require_string(visual.get("figure_id"), label=f"figure ID for {node_id}")
            if figure_id in configured_figure_ids:
                raise TransformError(f"duplicate figure ID in {node_id}: {figure_id}")
            configured_figure_ids.add(figure_id)
            panels = visual.get("panels")
            if not isinstance(panels, list) or not panels:
                raise TransformError(f"figure {figure_id} requires panels")
            panel_specs: list[dict[str, Any]] = []
            visual_metric_paths: set[str] = set()
            for panel_index, panel in enumerate(panels):
                if not isinstance(panel, dict):
                    raise TransformError(f"figure {figure_id} panel must be a mapping")
                panel_id = require_string(panel.get("panel_id"), label=f"panel ID for {figure_id}")
                series = panel.get("series")
                if not isinstance(series, list) or not series:
                    raise TransformError(f"figure {figure_id}:{panel_id} requires series")
                checked_series: list[dict[str, Any]] = []
                for series_row in series:
                    if not isinstance(series_row, dict):
                        raise TransformError(f"series in {figure_id}:{panel_id} must be a mapping")
                    value_path = require_string(
                        series_row.get("value_path"),
                        label=f"value path in {figure_id}:{panel_id}",
                    )
                    value = resolve_value(snapshot, value_path)
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        raise TransformError(f"bar value must be numeric: {value_path}")
                    visual_metric_paths.add(value_path)
                    ci_path = series_row.get("ci_path")
                    ci_value = None
                    if ci_path is not None:
                        ci_path = require_string(
                            ci_path, label=f"CI path in {figure_id}:{panel_id}"
                        )
                        ci_value = resolve_value(snapshot, ci_path)
                        if (
                            not isinstance(ci_value, list)
                            or len(ci_value) != 2
                            or not all(isinstance(item, (int, float)) for item in ci_value)
                        ):
                            raise TransformError(f"CI must resolve to [low, high]: {ci_path}")
                        visual_metric_paths.add(ci_path)
                    checked_series.append(
                        {
                            "label": require_string(series_row.get("label"), label="series label"),
                            "value_path": value_path,
                            "value": float(value),
                            "ci_path": ci_path,
                            "ci": ci_value,
                        }
                    )
                panel_specs.append(
                    {
                        "panel_id": panel_id,
                        "question": require_string(
                            panel.get("question"),
                            label=f"question for {figure_id}:{panel_id}",
                        ),
                        "title": require_string(
                            panel.get("title"),
                            label=f"title for {figure_id}:{panel_id}",
                        ),
                        "ylabel": require_string(
                            panel.get("ylabel"),
                            label=f"ylabel for {figure_id}:{panel_id}",
                        ),
                        "series": checked_series,
                        "order": panel_index + 1,
                    }
                )
            supports_roles = require_string_list(
                visual.get("supports_roles"), label=f"supports_roles for {figure_id}"
            )
            unknown_roles = set(supports_roles) - set(source_roles)
            if unknown_roles:
                raise TransformError(
                    f"figure {figure_id} supports unknown roles: {sorted(unknown_roles)}"
                )
            sentence_ids = [unit["sid"] for unit in units if unit["role"] in supports_roles]
            figure_specs.append(
                {
                    "figure_id": figure_id,
                    "owner_node_id": node_id,
                    "kind": require_string(
                        visual.get("kind", "empirical"), label=f"kind for {figure_id}"
                    ),
                    "renderer": require_string(
                        visual.get("renderer", "bar_panels"),
                        label=f"renderer for {figure_id}",
                    ),
                    "output_files": require_string_list(
                        visual.get("output_files"),
                        label=f"output_files for {figure_id}",
                    ),
                    "supports_sentence_ids": sentence_ids,
                    "source_figure_refs": source.get("figure_refs", []),
                    "metric_paths": sorted(visual_metric_paths),
                    "panels": panel_specs,
                    "caption": require_string(
                        visual.get("caption"), label=f"caption for {figure_id}"
                    ),
                    "allowed_conclusions": source.get("allowed_conclusions", []),
                    "forbidden_conclusions": source.get("forbidden_conclusions", []),
                }
            )

        if set(assigned_figures) != configured_figure_ids:
            raise TransformError(
                f"node {node_id} sentence/figure bindings disagree: "
                f"sentences={sorted(assigned_figures)}, figures={sorted(configured_figure_ids)}"
            )
        source_table_refs = set(source.get("table_refs", []))
        if assigned_tables != source_table_refs:
            raise TransformError(
                f"node {node_id} table bindings must cover source table refs exactly: "
                f"assigned={sorted(assigned_tables)}, source={sorted(source_table_refs)}"
            )

        draft = " ".join(paragraphs)
        packet = {
            "paragraph_id": node_id,
            "title": source.get("title"),
            "paragraph_claim": source.get("paragraph_claim"),
            "sentence_units": units,
            "allowed_conclusions": source.get("allowed_conclusions", []),
            "forbidden_conclusions": source.get("forbidden_conclusions", []),
            "reviewer_objection": source.get("reviewer_objection"),
            "objection_response": source.get("objection_response"),
            "transition_to_next": source.get("transition_to_next"),
            "generation_prompt": build_prompt(source, units),
            "deterministic_draft": draft,
        }
        prose_packets.append(packet)
        prose_paragraphs.append({"id": node_id, "title": str(source.get("title")), "text": draft})
        product_nodes.append(
            {
                **common,
                "sentence_units": units,
                "prose_product_id": node_id,
                "figure_product_ids": sorted(configured_figure_ids),
                "table_product_ids": sorted(assigned_tables),
            }
        )

    return {
        "product_graph": {
            "schema_version": 1,
            "task_id": contract.get("task_id"),
            "source_blueprint_sha256": blueprint_sha,
            "source_snapshot_sha256": snapshot_sha,
            "node_count": len(product_nodes),
            "enabled_node_ids": enabled_ids,
            "nodes": product_nodes,
        },
        "prose_packets": {
            "schema_version": 1,
            "task_id": contract.get("task_id"),
            "source_blueprint_sha256": blueprint_sha,
            "paragraphs": prose_packets,
        },
        "prose_paragraphs": prose_paragraphs,
        "figure_specs": {
            "schema_version": 1,
            "task_id": contract.get("task_id"),
            "source_snapshot_sha256": snapshot_sha,
            "figures": figure_specs,
        },
    }


def render_prose_markdown(paragraphs: Iterable[Mapping[str, str]], path: Path) -> None:
    lines = ["# Blueprint-derived prose draft", ""]
    for paragraph in paragraphs:
        lines.extend([f"## {paragraph['id']} — {paragraph['title']}", "", paragraph["text"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def render_figure(spec: Mapping[str, Any], output_root: Path) -> list[str]:
    if spec.get("renderer") != "bar_panels":
        raise TransformError(f"unsupported renderer: {spec.get('renderer')}")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise TransformError("matplotlib is required to render empirical figures") from exc

    panels = spec["panels"]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 4.2), squeeze=False)
    for axis, panel in zip(axes[0], panels):
        labels = [row["label"] for row in panel["series"]]
        values = [row["value"] for row in panel["series"]]
        positions = list(range(len(labels)))
        lower: list[float] = []
        upper: list[float] = []
        has_ci = False
        for row in panel["series"]:
            if row["ci"] is None:
                lower.append(0.0)
                upper.append(0.0)
            else:
                has_ci = True
                low, high = row["ci"]
                lower.append(max(0.0, row["value"] - float(low)))
                upper.append(max(0.0, float(high) - row["value"]))
        yerr = [lower, upper] if has_ci else None
        axis.bar(positions, values, yerr=yerr, capsize=4 if has_ci else 0)
        axis.set_xticks(positions, labels, rotation=25, ha="right")
        axis.set_ylabel(panel["ylabel"])
        axis.set_title(f"({panel['panel_id']}) {panel['title']}")
        axis.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    written: list[str] = []
    for relative in spec["output_files"]:
        path = output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        written.append(path.as_posix())
    plt.close(fig)
    return written


def validate_products(
    *,
    blueprint: dict[str, Any],
    snapshot: dict[str, Any],
    contract: dict[str, Any],
    products: dict[str, Any],
    rendered_files: Iterable[Path] = (),
) -> dict[str, Any]:
    graph = products["product_graph"]
    source_nodes = blueprint_nodes(blueprint)
    target_nodes = graph.get("nodes")
    if not isinstance(target_nodes, list):
        raise TransformError("product graph nodes must be a list")
    source_ids = [str(node["id"]) for node in source_nodes]
    target_ids = [str(node["id"]) for node in target_nodes]
    if source_ids != target_ids:
        raise TransformError("downstream graph must preserve blueprint IDs and order exactly")
    for source, target in zip(source_nodes, target_nodes):
        for field in (
            "id",
            "section",
            "order",
            "title",
            "outline_block_sha256",
            "status",
        ):
            if source.get(field) != target.get(field):
                raise TransformError(f"downstream graph changed {field} for {source.get('id')}")
        if source.get("status") != "enabled":
            continue
        units = target.get("sentence_units")
        if not isinstance(units, list) or len(units) != len(source.get("sentence_plan", [])):
            raise TransformError(f"sentence-unit count mismatch for {source.get('id')}")
        expected_sids = [f"{source['id']}-S{index:02d}" for index in range(1, len(units) + 1)]
        if [unit.get("sid") for unit in units] != expected_sids:
            raise TransformError(f"unstable sentence IDs for {source.get('id')}")

    prose = products["prose_packets"].get("paragraphs")
    if not isinstance(prose, list):
        raise TransformError("prose packets must contain paragraphs")
    for packet in prose:
        text = str(packet.get("deterministic_draft", ""))
        normalized = normalize_text(text)
        if not text:
            raise TransformError(f"empty deterministic draft for {packet.get('paragraph_id')}")
        if "tbd" in normalized:
            raise TransformError(f"TBD leaked into draft for {packet.get('paragraph_id')}")
        if str(packet.get("paragraph_id", "")).startswith("EXP-") and "c-u1" in normalized:
            if re.search(r"\bood\b|out-of-distribution", normalized):
                raise TransformError("C-U1 prose may not use OOD terminology")
        for forbidden in packet.get("forbidden_conclusions", []):
            if normalize_text(str(forbidden)) in normalized:
                raise TransformError(
                    f"forbidden conclusion leaked into {packet.get('paragraph_id')}: {forbidden}"
                )

    figure_ids: set[str] = set()
    valid_sentence_ids = {
        unit["sid"]
        for node in target_nodes
        if node.get("status") == "enabled"
        for unit in node.get("sentence_units", [])
    }
    for figure in products["figure_specs"].get("figures", []):
        figure_id = str(figure.get("figure_id"))
        if figure_id in figure_ids:
            raise TransformError(f"duplicate figure ID: {figure_id}")
        figure_ids.add(figure_id)
        if not set(figure.get("supports_sentence_ids", [])) <= valid_sentence_ids:
            raise TransformError(f"figure {figure_id} refers to unknown sentence IDs")
        if figure.get("kind") == "empirical" and not figure.get("metric_paths"):
            raise TransformError(f"empirical figure {figure_id} has no metric paths")
        for metric_path in figure.get("metric_paths", []):
            resolve_value(snapshot, metric_path)
        caption = normalize_text(str(figure.get("caption", "")))
        if "c-u1" in caption and re.search(r"\bood\b|out-of-distribution", caption):
            raise TransformError(f"C-U1 figure {figure_id} uses OOD terminology")

    missing_files = [
        str(path) for path in rendered_files if not path.is_file() or path.stat().st_size == 0
    ]
    if missing_files:
        raise TransformError(f"rendered figure files are missing or empty: {missing_files}")
    return {
        "status": "PASS",
        "node_count": len(target_nodes),
        "enabled_node_count": len(graph.get("enabled_node_ids", [])),
        "sentence_unit_count": sum(
            len(node.get("sentence_units", []))
            for node in target_nodes
            if node.get("status") == "enabled"
        ),
        "figure_count": len(figure_ids),
        "rendered_file_count": len(list(rendered_files)),
    }


def load_paths(args: argparse.Namespace) -> TransformPaths:
    repo = Path(args.repo_root).resolve()
    contract_path = resolve_repo_path(repo, args.contract)
    contract = read_yaml(contract_path)
    source = contract.get("source")
    if not isinstance(source, dict):
        raise TransformError("contract source must be a mapping")
    blueprint = resolve_repo_path(
        repo,
        args.blueprint or require_string(source.get("blueprint"), label="source blueprint"),
    )
    snapshot = resolve_repo_path(
        repo,
        args.snapshot or require_string(source.get("snapshot"), label="source snapshot"),
    )
    output = resolve_repo_path(
        repo,
        args.output or require_string(contract.get("output_root"), label="output_root"),
    )
    return TransformPaths(
        repo=repo,
        contract=contract_path,
        blueprint=blueprint,
        snapshot=snapshot,
        output=output,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = load_paths(args)
    contract = read_yaml(paths.contract)
    blueprint = read_json(paths.blueprint)
    snapshot = read_json(paths.snapshot)
    products = compile_products(
        blueprint=blueprint,
        snapshot=snapshot,
        contract=contract,
        blueprint_sha=sha256_file(paths.blueprint),
        snapshot_sha=sha256_file(paths.snapshot),
    )
    paths.output.mkdir(parents=True, exist_ok=True)
    write_json(paths.output / "product_graph.json", products["product_graph"])
    write_json(paths.output / "prose_packets.json", products["prose_packets"])
    write_json(paths.output / "figure_specs.json", products["figure_specs"])
    render_prose_markdown(products["prose_paragraphs"], paths.output / "prose_draft.md")

    rendered_paths: list[Path] = []
    if not args.skip_figures:
        for figure in products["figure_specs"]["figures"]:
            render_figure(figure, paths.output)
            rendered_paths.extend(paths.output / relative for relative in figure["output_files"])
    report = validate_products(
        blueprint=blueprint,
        snapshot=snapshot,
        contract=contract,
        products=products,
        rendered_files=rendered_paths,
    )
    report.update(
        {
            "task_id": contract.get("task_id"),
            "blueprint": paths.blueprint.relative_to(paths.repo).as_posix(),
            "blueprint_sha256": sha256_file(paths.blueprint),
            "snapshot": paths.snapshot.relative_to(paths.repo).as_posix(),
            "snapshot_sha256": sha256_file(paths.snapshot),
            "contract": paths.contract.relative_to(paths.repo).as_posix(),
            "contract_sha256": sha256_file(paths.contract),
        }
    )
    write_json(paths.output / "validation_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--contract",
        default="docs/manuscript/manuscript_downstream_contract.yaml",
    )
    parser.add_argument("--blueprint", default=None)
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="compile and validate graph/prose/figure specs without invoking matplotlib",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        yaml.YAMLError,
        TransformError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
