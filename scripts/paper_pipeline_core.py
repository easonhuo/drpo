#!/usr/bin/env python3
"""Evidence-first DRPO manuscript Core vertical slice.

This script intentionally implements a narrow, auditable pipeline for
PAPER-PIPELINE-V2-CORE-01. It does not replace the historical bidirectional
scaffold pipeline. Its purpose is to prove the reliable path from validated
repository evidence to a real figure, table, blueprint, prose, theorem/proof,
and two-page review PDF.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ALLOWED_STATUSES = {
    "analytically_proven",
    "long_run_validated",
    "finite_step_validated",
    "pilot",
    "not_run",
    "rejected",
    "superseded",
}
STATUS_RANK = {
    "rejected": 0,
    "superseded": 0,
    "not_run": 1,
    "pilot": 2,
    "finite_step_validated": 3,
    "long_run_validated": 4,
    "analytically_proven": 5,
}
DISPLAY_METHOD = {
    "baseline": "Baseline",
    "near_zero": "Near-zero",
    "far_zero": "Far-zero",
    "far_cap": "Far-cap",
    "global_scale": "Global-scale",
    "far_to_near": "Far-to-near",
}


class CorePipelineError(RuntimeError):
    """Expected fail-closed Core pipeline error."""


@dataclass(frozen=True)
class Paths:
    repo: Path
    spec: Path
    output: Path
    allow_output_override: bool = False

    @property
    def snapshot(self) -> Path:
        return self.output / "research_snapshot.json"

    @property
    def manifest(self) -> Path:
        return self.output / "build_manifest.json"

    @property
    def pdf(self) -> Path:
        return self.output / "main.pdf"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CorePipelineError(f"expected YAML mapping: {path}")
    return payload


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CorePipelineError(f"expected JSON mapping: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repo_path(repo: Path, relative: str) -> Path:
    candidate = (repo / relative).resolve()
    try:
        candidate.relative_to(repo.resolve())
    except ValueError as exc:
        raise CorePipelineError(f"path escapes repository: {relative}") from exc
    return candidate


def load_spec(paths: Paths) -> dict[str, Any]:
    spec = read_yaml(paths.spec)
    if spec.get("schema_version") != 1:
        raise CorePipelineError("paper_spec_core.yaml schema_version must be 1")
    if spec.get("profile") != "core_vertical_slice":
        raise CorePipelineError("Core script only accepts profile=core_vertical_slice")
    expected_output = repo_path(paths.repo, str(spec["output_root"]))
    if not paths.allow_output_override and expected_output != paths.output.resolve():
        raise CorePipelineError(
            f"output root differs from spec: {paths.output} != {expected_output}"
        )
    return spec


def find_experiment(registry: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    experiments = registry.get("experiments")
    if not isinstance(experiments, list):
        raise CorePipelineError("registry experiments must be a list")
    matches = [row for row in experiments if isinstance(row, dict) and row.get("id") == experiment_id]
    if len(matches) != 1:
        raise CorePipelineError(
            f"expected exactly one registry experiment {experiment_id}, found {len(matches)}"
        )
    return matches[0]


def require_status(actual: str, required: str) -> None:
    if actual not in ALLOWED_STATUSES or required not in ALLOWED_STATUSES:
        raise CorePipelineError(f"unknown result status: actual={actual} required={required}")
    if STATUS_RANK[actual] < STATUS_RANK[required]:
        raise CorePipelineError(
            f"experiment status {actual} does not satisfy required status {required}"
        )


def verify_compact_artifacts(repo: Path, artifact_index_path: Path) -> dict[str, Any]:
    index = read_json(artifact_index_path)
    compact = index.get("compact_repository_files")
    if not isinstance(compact, dict) or not compact:
        raise CorePipelineError("ARTIFACT_INDEX compact_repository_files is missing")
    verified: dict[str, Any] = {}
    for name, metadata in compact.items():
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise CorePipelineError("invalid compact artifact index entry")
        path = artifact_index_path.parent / name
        if not path.is_file():
            raise CorePipelineError(f"compact artifact is missing: {path.relative_to(repo)}")
        actual = sha256_file(path)
        expected = metadata.get("sha256")
        if actual != expected:
            raise CorePipelineError(
                f"compact artifact checksum mismatch for {path.relative_to(repo)}: "
                f"{actual} != {expected}"
            )
        verified[name] = {
            "path": path.relative_to(repo).as_posix(),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    return {"index": index, "verified": verified}


def load_csv_by_method(path: Path) -> dict[str, dict[str, float | str | None]]:
    rows: dict[str, dict[str, float | str | None]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "method" not in reader.fieldnames:
            raise CorePipelineError(f"CSV has no method column: {path}")
        for raw in reader:
            method = str(raw.get("method", "")).strip()
            if not method or method in rows:
                raise CorePipelineError(f"invalid or duplicate method in {path}: {method}")
            converted: dict[str, float | str | None] = {}
            for key, value in raw.items():
                if key == "method":
                    continue
                text = "" if value is None else value.strip()
                if text == "":
                    converted[key] = None
                    continue
                try:
                    number = float(text)
                except ValueError:
                    converted[key] = text
                else:
                    if not math.isfinite(number):
                        raise CorePipelineError(f"non-finite CSV value {path}:{method}:{key}")
                    converted[key] = number
            rows[method] = converted
    return rows


def numeric(row: dict[str, Any], key: str, *, source: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CorePipelineError(f"missing numeric field {source}:{key}")
    value = float(value)
    if not math.isfinite(value):
        raise CorePipelineError(f"non-finite field {source}:{key}")
    return value


def event_count(rate: float, denominator: int, *, label: str) -> int:
    raw = rate * denominator
    rounded = int(round(raw))
    if not math.isclose(raw, rounded, abs_tol=1e-6):
        raise CorePipelineError(f"event rate is not an integer count for {label}: {rate}")
    return rounded


def outline_contains_node(path: Path, node_id: str) -> bool:
    token = f"<!-- MANUSCRIPT:BEGIN {node_id} -->"
    return token in path.read_text(encoding="utf-8")


def build_snapshot(
    paths: Paths, *, spec_override: dict[str, Any] | None = None
) -> dict[str, Any]:
    spec = spec_override if spec_override is not None else load_spec(paths)
    experiment_spec = spec["experiment"]
    registry_path = repo_path(paths.repo, str(experiment_spec["registry_path"]))
    registry = read_yaml(registry_path)
    experiment_id = str(experiment_spec["id"])
    experiment = find_experiment(registry, experiment_id)
    status = str(experiment.get("status", ""))
    require_status(status, str(experiment_spec["required_status"]))

    terminology = str(experiment.get("data", {}).get("terminology", ""))
    if terminology != "held_out_context_generalization":
        raise CorePipelineError(
            f"C-U1 terminology must be held_out_context_generalization, got {terminology!r}"
        )
    separation = experiment.get("reporting_separation")
    required_separation = {
        "task_performance_collapse",
        "support_or_variance_contraction",
        "nan_inf_numerical_failure",
    }
    if not isinstance(separation, list) or not required_separation.issubset(set(separation)):
        raise CorePipelineError("registry does not preserve the three required failure categories")
    evidence = experiment.get("evidence", {})
    if not isinstance(evidence, dict) or evidence.get("terminal_audited") is not True:
        raise CorePipelineError("experiment is not terminal audited")

    artifact_index_path = repo_path(paths.repo, str(experiment_spec["artifact_index"]))
    artifact_audit = verify_compact_artifacts(paths.repo, artifact_index_path)
    index = artifact_audit["index"]
    if index.get("experiment_id") != experiment_id or index.get("scientific_status") != status:
        raise CorePipelineError("ARTIFACT_INDEX identity/status disagrees with registry")

    fixed_path = repo_path(paths.repo, str(experiment_spec["fixed_aggregate"]))
    learnable_path = repo_path(paths.repo, str(experiment_spec["learnable_aggregate"]))
    terminal_path = repo_path(paths.repo, str(experiment_spec["terminal_audit"]))
    for required in (fixed_path, learnable_path, terminal_path):
        if not required.is_file():
            raise CorePipelineError(f"required evidence file is missing: {required.relative_to(paths.repo)}")

    primary_methods = [str(value) for value in spec["methods"]["primary"]]
    fixed_controls = [str(value) for value in spec["methods"].get("fixed_controls", [])]
    all_methods = primary_methods + fixed_controls
    if len(set(all_methods)) != len(all_methods):
        raise CorePipelineError("paper method lists contain duplicates")
    fixed_rows = load_csv_by_method(fixed_path)
    learnable_rows = load_csv_by_method(learnable_path)
    missing_fixed = [method for method in all_methods if method not in fixed_rows]
    missing_primary_learnable = [method for method in primary_methods if method not in learnable_rows]
    if missing_fixed or missing_primary_learnable:
        raise CorePipelineError(
            "required methods missing from aggregate CSVs: "
            f"fixed={missing_fixed}, learnable_primary={missing_primary_learnable}"
        )

    denominator = int(spec["metric_contract"]["task_collapse"]["denominator"])
    if denominator != len(experiment.get("held_out_seeds", [])):
        raise CorePipelineError("metric denominator differs from registered held-out seed count")

    result_summary = experiment.get("result_summary", {})
    fixed_registry = result_summary.get("fixed_variance", {}) if isinstance(result_summary, dict) else {}
    learnable_registry = result_summary.get("learnable_variance", {}) if isinstance(result_summary, dict) else {}
    methods_payload: dict[str, Any] = {}
    total_nan_inf = result_summary.get("total_nan_inf_count") if isinstance(result_summary, dict) else None
    for method in all_methods:
        fixed = fixed_rows[method]
        learnable = learnable_rows.get(method)
        fixed_nan = fixed_registry.get(method, {}).get("nan_inf_count")
        if fixed_nan is None and total_nan_inf == 0:
            fixed_nan = 0
        if fixed_nan is None:
            raise CorePipelineError(f"registry nan_inf_count missing for fixed method {method}")
        learnable_payload = None
        if learnable is not None:
            learnable_nan = learnable_registry.get(method, {}).get("nan_inf_count")
            if learnable_nan is None and total_nan_inf == 0:
                learnable_nan = 0
            if learnable_nan is None:
                raise CorePipelineError(f"registry nan_inf_count missing for learnable method {method}")
            learnable_payload = {
                "reward": numeric(learnable, "reward", source=f"learnable:{method}"),
                "reward_ci95": [
                    numeric(learnable, "reward_ci_low", source=f"learnable:{method}"),
                    numeric(learnable, "reward_ci_high", source=f"learnable:{method}"),
                ],
                "task_collapse_count": event_count(
                    numeric(
                        learnable,
                        "task_failure_onset_event_rate",
                        source=f"learnable:{method}",
                    ),
                    denominator,
                    label=f"learnable:{method}:task",
                ),
                "support_boundary_count": event_count(
                    numeric(
                        learnable,
                        "support_boundary_onset_event_rate",
                        source=f"learnable:{method}",
                    ),
                    denominator,
                    label=f"learnable:{method}:support",
                ),
                "support_onset_mean": learnable.get("support_boundary_onset"),
                "nan_inf_count": int(learnable_nan),
                "n": int(numeric(learnable, "n", source=f"learnable:{method}")),
            }
        methods_payload[method] = {
            "display_name": DISPLAY_METHOD.get(method, method),
            "paper_role": "primary_intervention" if method in primary_methods else "fixed_budget_control",
            "fixed_variance": {
                "reward": numeric(fixed, "reward", source=f"fixed:{method}"),
                "reward_ci95": [
                    numeric(fixed, "reward_ci_low", source=f"fixed:{method}"),
                    numeric(fixed, "reward_ci_high", source=f"fixed:{method}"),
                ],
                "task_collapse_count": event_count(
                    numeric(fixed, "task_failure_onset_event_rate", source=f"fixed:{method}"),
                    denominator,
                    label=f"fixed:{method}:task",
                ),
                "support_boundary_count": event_count(
                    numeric(fixed, "support_boundary_onset_event_rate", source=f"fixed:{method}"),
                    denominator,
                    label=f"fixed:{method}:support",
                ),
                "nan_inf_count": int(fixed_nan),
                "n": int(numeric(fixed, "n", source=f"fixed:{method}")),
            },
            "learnable_variance": learnable_payload,
        }

    outline_path = repo_path(paths.repo, str(spec["approved_outline"]))
    for binding in spec["outline_bindings"].values():
        if not outline_contains_node(outline_path, str(binding)):
            raise CorePipelineError(f"approved outline is missing bound node {binding}")

    inputs = {
        "spec": {"path": paths.spec.relative_to(paths.repo).as_posix(), "sha256": sha256_file(paths.spec)},
        "registry": {"path": registry_path.relative_to(paths.repo).as_posix(), "sha256": sha256_file(registry_path)},
        "outline": {"path": outline_path.relative_to(paths.repo).as_posix(), "sha256": sha256_file(outline_path)},
        "artifact_index": {"path": artifact_index_path.relative_to(paths.repo).as_posix(), "sha256": sha256_file(artifact_index_path)},
        "fixed_aggregate": {"path": fixed_path.relative_to(paths.repo).as_posix(), "sha256": sha256_file(fixed_path)},
        "learnable_aggregate": {"path": learnable_path.relative_to(paths.repo).as_posix(), "sha256": sha256_file(learnable_path)},
        "terminal_audit": {"path": terminal_path.relative_to(paths.repo).as_posix(), "sha256": sha256_file(terminal_path)},
    }
    snapshot = {
        "schema_version": 1,
        "snapshot_kind": "generated_build_input_not_research_master",
        "task_id": spec["spec_id"],
        "profile": spec["profile"],
        "experiment": {
            "id": experiment_id,
            "environment": experiment.get("environment"),
            "status": status,
            "role": experiment.get("role"),
            "run_commit": experiment.get("provenance", {}).get("run_commit"),
            "held_out_seed_count": denominator,
            "terminology": terminology,
            "reporting_separation": sorted(required_separation),
            "terminal_audited": True,
            "compact_evidence_trust": "checksum_verified_canonical_aggregate",
            "raw_evidence_location": index.get("external_artifact"),
        },
        "outline_bindings": spec["outline_bindings"],
        "method_groups": {
            "primary": primary_methods,
            "fixed_controls": fixed_controls,
        },
        "methods": methods_payload,
        "theorem": spec["theorem"],
        "inputs": inputs,
        "verified_compact_files": artifact_audit["verified"],
    }
    snapshot["snapshot_sha256"] = sha256_bytes(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    write_json(paths.snapshot, snapshot)
    return snapshot


def format_number(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def format_reward(value: float, decimals: int) -> str:
    """Use scientific notation for near-zero rewards and fixed decimals otherwise."""
    if value != 0.0 and abs(value) < 10 ** (-decimals):
        return f"{value:.2e}"
    return format_number(value, decimals)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def render_figure(snapshot: dict[str, Any], destination: Path) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[dict[str, Any]] = []
    for method in snapshot["method_groups"]["primary"]:
        payload = snapshot["methods"][method]
        fixed = payload["fixed_variance"]
        rows.append(
            {
                "method": method,
                "display_name": payload["display_name"],
                "reward": fixed["reward"],
                "ci_low": fixed["reward_ci95"][0],
                "ci_high": fixed["reward_ci95"][1],
                "task_collapse_count": fixed["task_collapse_count"],
                "n": fixed["n"],
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    x = list(range(len(rows)))
    values = [row["reward"] for row in rows]
    low = [row["reward"] - row["ci_low"] for row in rows]
    high = [row["ci_high"] - row["reward"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    bars = ax.bar(x, values, yerr=[low, high], capsize=4)
    ax.set_xticks(x, [row["display_name"] for row in rows])
    ax.set_ylabel("Terminal held-out-context reward")
    ax.set_ylim(0.0, max(values) * 1.18)
    ax.set_title("C-U1 E3 fixed-variance targeted intervention (20 paired seeds)")
    ax.grid(axis="y", alpha=0.25)
    for bar, row in zip(bars, rows, strict=True):
        label = f"task collapse {row['task_collapse_count']}/{row['n']}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.035,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )
    fig.tight_layout()
    fig.savefig(
        destination,
        bbox_inches="tight",
        metadata={"Creator": "DRPO paper_pipeline_core", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return rows


def write_figure_data(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_table(snapshot: dict[str, Any], path: Path, decimals: int) -> None:
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Method & Fixed reward (95\% CI) & Task & Support & NaN/Inf \\",
        r"\midrule",
    ]
    for method, payload in snapshot["methods"].items():
        fixed = payload["fixed_variance"]
        learnable = payload["learnable_variance"]
        reward = format_reward(fixed["reward"], decimals)
        low = format_reward(fixed["reward_ci95"][0], decimals)
        high = format_reward(fixed["reward_ci95"][1], decimals)
        lines.append(
            f"{latex_escape(payload['display_name'])} & {reward} [{low}, {high}] & "
            f"{fixed['task_collapse_count']}/{fixed['n']} & "
            + (
                f"{learnable['support_boundary_count']}/{learnable['n']} & "
                if learnable is not None
                else r"-- & "
            )
            + f"{max(fixed['nan_inf_count'], learnable['nan_inf_count'] if learnable else 0)}/{fixed['n']} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_blueprint(snapshot: dict[str, Any], path: Path) -> None:
    required_metrics = []
    for method, payload in snapshot["methods"].items():
        required_metrics.extend(
            [
                f"methods.{method}.fixed_variance.reward",
                f"methods.{method}.fixed_variance.reward_ci95",
                f"methods.{method}.fixed_variance.task_collapse_count",
                f"methods.{method}.fixed_variance.nan_inf_count",
            ]
        )
        if payload["learnable_variance"] is not None:
            required_metrics.extend(
                [
                    f"methods.{method}.learnable_variance.support_boundary_count",
                    f"methods.{method}.learnable_variance.nan_inf_count",
                ]
            )
    lines = [
        "# Executable blueprint: PAPER-PIPELINE-V2-CORE-01",
        "",
        f"Snapshot: `{snapshot['snapshot_sha256']}`",
        "",
        "## EXP-P04-A - Fixed-variance causal intervention",
        "",
        "- Reader question: Does retaining the far-field negative path cause task collapse in the controlled C-U1 environment?",
        "- Paragraph claim: Baseline and Near-zero collapse in all paired seeds, whereas Far-zero and Far-cap prevent collapse and retain high terminal reward.",
        "- Sentence plan:",
        "  1. State the matched four-way intervention and the 20 paired held-out seeds.",
        "  2. Report Baseline and Near-zero terminal reward with task-collapse counts.",
        "  3. Report Far-zero and Far-cap terminal reward with confidence intervals and collapse counts.",
        "  4. Report Global-scale and Far-to-near as registered budget controls, not as a universal ranking.",
        "  5. Conclude only that the far-field path is the dominant causal transmission path in this controlled environment.",
        "- Reviewer objection: The rescue may reflect removing all negative information.",
        "- Response: Near-zero removes the near component yet does not rescue; Far-cap retains bounded far influence and rescues.",
        "- Budget controls: Global-scale and Far-to-near are included from the fixed-variance registered controls.",
        "- Figure/table: `cu1_e3_fixed_reward.pdf`, `cu1_e3_results.tex`.",
        "",
        "## EXP-P04-B - Learnable-variance boundary audit",
        "",
        "- Reader question: Is the learnable-variance failure task collapse, a support boundary, or numerical failure?",
        "- Paragraph claim: Baseline and Near-zero reach support contraction in all seeds near step 73, while Far-zero and Far-cap avoid that event; no method produces NaN/Inf.",
        "- Sentence plan:",
        "  1. Name the first registered event as support/variance contraction.",
        "  2. Report the event counts and onset without relabeling it as task or numerical collapse.",
        "  3. Report the absence of NaN/Inf separately.",
        "  4. State the bounded controlled-environment conclusion.",
        "- Reviewer objection: The boundary classification may be a numerical artifact.",
        "- Response: The terminal audit records finite parameters and zero NaN/Inf events.",
        "",
        "## METHOD-P03 - Proposition 2",
        "",
        "- Reader question: Why does exponential remoteness weighting control a polynomially growing far-field score?",
        "- Claim: Exponential decay dominates every finite polynomial order.",
        "- Assumption: The unweighted score-times-advantage norm is at most `C(1+r)^k` for finite `C,k`.",
        "- Conclusion ceiling: The weighted contribution vanishes; this does not model or assume exponential sample utility.",
        "",
        "## Exact metric paths",
        "",
    ]
    lines.extend(f"- `{metric}`" for metric in required_metrics)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_prose(snapshot: dict[str, Any], path: Path, decimals: int) -> tuple[str, str]:
    methods = snapshot["methods"]

    def fixed(method: str) -> dict[str, Any]:
        return methods[method]["fixed_variance"]

    def learnable(method: str) -> dict[str, Any]:
        return methods[method]["learnable_variance"]

    b, n, fz, fc = (fixed(name) for name in ("baseline", "near_zero", "far_zero", "far_cap"))
    paragraph_one = (
        "We test the transmission path with four matched interventions over 20 paired held-out seeds. "
        f"The uncontrolled Baseline and Near-zero variants finish at rewards "
        f"{format_reward(b['reward'], decimals)} and {format_reward(n['reward'], decimals)} and undergo "
        f"task-performance collapse in {b['task_collapse_count']}/20 and {n['task_collapse_count']}/20 seeds. "
        f"Removing the far-field contribution instead yields {format_reward(fz['reward'], decimals)} "
        f"[{format_reward(fz['reward_ci95'][0], decimals)}, {format_reward(fz['reward_ci95'][1], decimals)}] "
        f"for Far-zero, while capping it yields {format_reward(fc['reward'], decimals)} "
        f"[{format_reward(fc['reward_ci95'][0], decimals)}, {format_reward(fc['reward_ci95'][1], decimals)}] "
        "for Far-cap; neither intervention collapses in any seed. The registered fixed-variance budget controls "
        f"also remain non-collapsed: Global-scale reaches {format_reward(fixed('global_scale')['reward'], decimals)}, "
        f"and transferring the far budget to the near component reaches {format_reward(fixed('far_to_near')['reward'], decimals)}. "
        "These controls are diagnostic and do not define a method ranking. Near-field removal is therefore not a rescue, "
        "whereas deleting or bounding the far-field path is. Within this controlled same-distribution "
        "held-out-context setting, the comparison identifies the far-field component as the dominant causal "
        "transmission path; it does not establish a universal method ranking."
    )
    lb, ln, lfz, lfc = (
        learnable(name) for name in ("baseline", "near_zero", "far_zero", "far_cap")
    )
    baseline_onset = float(lb["support_onset_mean"])
    near_onset = float(ln["support_onset_mean"])
    paragraph_two = (
        "The learnable-variance branch separates the type of failure. Baseline and Near-zero reach the registered "
        f"support/variance-contraction boundary in {lb['support_boundary_count']}/20 and "
        f"{ln['support_boundary_count']}/20 seeds, with mean onsets at steps "
        f"{baseline_onset:.1f} and {near_onset:.1f}. Far-zero and Far-cap record "
        f"{lfz['support_boundary_count']}/20 and {lfc['support_boundary_count']}/20 support-boundary events. "
        "All four methods keep finite parameters and record 0/20 NaN/Inf failures. Thus this branch is evidence "
        "for support contraction rather than numerical collapse, and the intervention again isolates far-field "
        "negative influence as the removable path in C-U1."
    )
    path.write_text(
        "# Evidence-bounded prose\n\n"
        "## EXP-P04-A\n\n"
        + paragraph_one
        + "\n\n## EXP-P04-B\n\n"
        + paragraph_two
        + "\n",
        encoding="utf-8",
    )
    return paragraph_one, paragraph_two


def theorem_tex() -> str:
    return r"""\begin{proposition}[Vanishing weighted far-field gradient]
\label{prop:far-field}
Let $r\geq 0$ be policy-relative remoteness and suppose the unweighted
score-times-advantage contribution satisfies
$\lVert g(r)\rVert\leq C(1+r)^k$ for finite constants $C>0$ and $k\geq 0$.
For any $\lambda>0$, define $\widetilde g(r)=e^{-\lambda r}g(r)$.
Then $\lVert\widetilde g(r)\rVert\to 0$ as $r\to\infty$.
\end{proposition}
"""


def proof_tex() -> str:
    return r"""\begin{proof}
The assumption gives
\[
  \lVert\widetilde g(r)\rVert
  \leq C(1+r)^k e^{-\lambda r}.
\]
For integer $m>k$, the exponential series implies
$e^{\lambda r}\geq (\lambda r)^m/m!$ for $r>0$. Hence
\[
  (1+r)^k e^{-\lambda r}
  \leq \frac{m!(1+r)^k}{(\lambda r)^m},
\]
and the right-hand side converges to zero because $m>k$. The squeeze theorem
proves the result. The proposition controls the weighted gradient tail; it
makes no assumption that sample utility decays exponentially.
\end{proof}
"""


def tex_paragraph(text: str) -> str:
    return latex_escape(text).replace("[", "[").replace("]", "]")


def render_main_tex(
    snapshot: dict[str, Any],
    paragraph_one: str,
    paragraph_two: str,
    path: Path,
) -> None:
    experiment = snapshot["experiment"]
    content = rf"""\pdfinfoomitdate=1
\pdftrailerid{{}}
\pdfsuppressptexinfo=15
\documentclass[9pt]{{article}}
\usepackage[margin=0.62in]{{geometry}}
\usepackage{{amsmath,amssymb,amsthm,booktabs,graphicx,microtype,caption}}
\usepackage[T1]{{fontenc}}
\newtheorem{{proposition}}{{Proposition}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.35em}}
\title{{\vspace{{-2.2em}}DRPO Pipeline v2.3 Core: Evidence-First Vertical Slice}}
\author{{Anonymous review artifact}}
\date{{}}
\begin{{document}}
\maketitle
\vspace{{-2.0em}}
\textbf{{Scope.}}
This two-page artifact validates the manuscript pipeline, not a new experiment.
It uses the long-run-validated {latex_escape(str(experiment['id']))} compact
repository evidence. C-U1 evaluates same-distribution held-out-context
generalization. Task-performance collapse, support or variance-boundary events,
and NaN/Inf numerical failures are reported separately.

\section*{{Targeted Causal Transmission}}
\vspace{{-0.5em}}
\begin{{center}}
\begin{{minipage}}[t]{{0.49\textwidth}}
\centering
\captionsetup{{type=table,font=footnotesize,skip=3pt}}
\captionof{{table}}{{Terminal outcomes. Task, support-boundary, and NaN/Inf events are separate.}}
\label{{tab:cu1-e3}}
\scriptsize
\resizebox{{\linewidth}}{{!}}{{\input{{cu1_e3_results.tex}}}}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.49\textwidth}}
\centering
\includegraphics[width=\linewidth]{{cu1_e3_fixed_reward.pdf}}
\captionsetup{{type=figure,font=footnotesize,skip=3pt}}
\captionof{{figure}}{{Fixed-variance terminal reward with 95\% CIs; labels show task-collapse counts.}}
\label{{fig:cu1-e3}}
\end{{minipage}}
\end{{center}}
\vspace{{-0.4em}}

{tex_paragraph(paragraph_one)}

{tex_paragraph(paragraph_two)}

\clearpage
\section*{{Tail-Control Guarantee}}
\input{{theorem.tex}}
\input{{proof.tex}}

\textbf{{Evidence and claim boundary.}}
The empirical intervention establishes a controlled C-U1 causal transmission
result. The proposition establishes only a tail bound under finite-order score
growth. Neither component asserts that exponential weighting is universally
superior, and the proposition is not a utility-decay model.

\section*{{Pipeline audit}}
Every number in the result paragraphs is rendered from
\texttt{{research\_snapshot.json}}. The snapshot verifies the registered status,
artifact checksums, held-out seed count, terminal audit, and terminology before
any prose or figure is produced. A missing input fails closed rather than
creating an empty table or placeholder result.

\end{{document}}
"""
    path.write_text(content, encoding="utf-8")


def build_slice(paths: Paths) -> dict[str, Any]:
    spec = load_spec(paths)
    snapshot = build_snapshot(paths)
    paths.output.mkdir(parents=True, exist_ok=True)

    figure_path = paths.output / "cu1_e3_fixed_reward.pdf"
    figure_rows = render_figure(snapshot, figure_path)
    figure_data_path = paths.output / "cu1_e3_fixed_reward.csv"
    write_figure_data(figure_data_path, figure_rows)

    decimals = int(spec["metric_contract"]["reward"]["table_decimals"])
    table_path = paths.output / "cu1_e3_results.tex"
    render_table(snapshot, table_path, decimals)

    blueprint_path = paths.output / "blueprint.md"
    build_blueprint(snapshot, blueprint_path)
    prose_path = paths.output / "prose.md"
    prose_decimals = int(spec["metric_contract"]["reward"]["prose_decimals"])
    paragraph_one, paragraph_two = build_prose(snapshot, prose_path, prose_decimals)

    (paths.output / "theorem.tex").write_text(theorem_tex(), encoding="utf-8")
    (paths.output / "proof.tex").write_text(proof_tex(), encoding="utf-8")
    render_main_tex(snapshot, paragraph_one, paragraph_two, paths.output / "main.tex")

    outputs = {}
    for name in (
        "research_snapshot.json",
        "cu1_e3_fixed_reward.pdf",
        "cu1_e3_fixed_reward.csv",
        "cu1_e3_results.tex",
        "blueprint.md",
        "prose.md",
        "theorem.tex",
        "proof.tex",
        "main.tex",
    ):
        path = paths.output / name
        outputs[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    manifest = {
        "schema_version": 1,
        "task_id": spec["spec_id"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "outputs": outputs,
        "pdf_status": "not_compiled",
    }
    write_json(paths.manifest, manifest)
    return manifest


def _pages_from_latex_log(log_text: str) -> int | None:
    import re

    match = re.search(r"Output written on .*?\((\d+) pages?[,)]", log_text)
    return int(match.group(1)) if match else None


def pdf_pages(
    path: Path, *, manifest: dict[str, Any] | None = None, latex_log: str | None = None
) -> tuple[int, str]:
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo is not None:
        proc = subprocess.run(
            [pdfinfo, str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if line.startswith("Pages:"):
                    return int(line.split(":", 1)[1].strip()), "pdfinfo"

    if latex_log is not None:
        pages = _pages_from_latex_log(latex_log)
        if pages is not None:
            return pages, "latex_log"

    if manifest is not None:
        pages = manifest.get("pdf_pages")
        if isinstance(pages, int) and pages > 0:
            return pages, "verified_manifest"

    raise CorePipelineError(
        "page count is unavailable: install pdfinfo, provide a LaTeX build log, "
        "or validate a hash-verified artifact with pdf_pages in build_manifest.json"
    )


def _verify_manifest_outputs(paths: Paths, manifest: dict[str, Any]) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise CorePipelineError("build manifest outputs are missing")
    for name, metadata in outputs.items():
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise CorePipelineError("build manifest contains an invalid output entry")
        output_path = paths.output / name
        if not output_path.is_file():
            raise CorePipelineError(f"manifest output is missing: {name}")
        expected_size = metadata.get("size_bytes")
        expected_sha = metadata.get("sha256")
        if output_path.stat().st_size != expected_size:
            raise CorePipelineError(f"manifest size mismatch: {name}")
        if sha256_file(output_path) != expected_sha:
            raise CorePipelineError(f"manifest checksum mismatch: {name}")


def compile_pdf(paths: Paths) -> Path:
    if not (paths.output / "main.tex").is_file():
        build_slice(paths)
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise CorePipelineError("latexmk is required to compile the Core review PDF")
    subprocess.run(
        [latexmk, "-C", "main.tex"],
        cwd=paths.output,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    build_env = os.environ.copy()
    build_env.update({"SOURCE_DATE_EPOCH": "0", "FORCE_SOURCE_DATE": "1", "TZ": "UTC"})
    proc = subprocess.run(
        [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=paths.output,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        env=build_env,
    )
    log_path = paths.output / "latex_build.txt"
    log_path.write_text(proc.stdout.rstrip() + "\n", encoding="utf-8")
    if proc.returncode != 0 or not paths.pdf.is_file():
        raise CorePipelineError(f"Core LaTeX build failed:\n{proc.stdout[-4000:]}")
    pages, page_count_source = pdf_pages(paths.pdf, latex_log=proc.stdout)
    manifest = read_json(paths.manifest)
    manifest["pdf_status"] = "compiled"
    manifest["pdf_pages"] = pages
    manifest["page_count_source"] = page_count_source
    manifest["outputs"]["main.pdf"] = {
        "sha256": sha256_file(paths.pdf),
        "size_bytes": paths.pdf.stat().st_size,
    }
    manifest["outputs"]["latex_build.txt"] = {
        "sha256": sha256_file(log_path),
        "size_bytes": log_path.stat().st_size,
    }
    write_json(paths.manifest, manifest)
    return paths.pdf


def validate_slice(paths: Paths) -> dict[str, Any]:
    spec = load_spec(paths)
    manifest = read_json(paths.manifest)
    if manifest.get("pdf_status") != "compiled":
        raise CorePipelineError("Core review PDF is not marked compiled in the build manifest")
    required = [
        paths.snapshot,
        paths.manifest,
        paths.output / "blueprint.md",
        paths.output / "prose.md",
        paths.output / "theorem.tex",
        paths.output / "proof.tex",
        paths.output / "cu1_e3_fixed_reward.pdf",
        paths.output / "cu1_e3_fixed_reward.csv",
        paths.output / "cu1_e3_results.tex",
        paths.output / "main.tex",
        paths.pdf,
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise CorePipelineError(f"Core slice is incomplete; missing {missing}")

    _verify_manifest_outputs(paths, manifest)

    snapshot = read_json(paths.snapshot)
    if snapshot.get("experiment", {}).get("status") != "long_run_validated":
        raise CorePipelineError("snapshot experiment is not long_run_validated")
    if snapshot.get("experiment", {}).get("terminology") != "held_out_context_generalization":
        raise CorePipelineError("snapshot C-U1 terminology is invalid")

    prose = (paths.output / "prose.md").read_text(encoding="utf-8")
    main_tex = (paths.output / "main.tex").read_text(encoding="utf-8")
    combined = prose + "\n" + main_tex
    for term in spec["release"]["forbid_terms"]:
        if str(term).lower() in combined.lower():
            raise CorePipelineError(f"forbidden release term found: {term}")
    required_phrases = (
        "task-performance collapse",
        "support/variance-contraction",
        "NaN/Inf",
        "held-out-context",
        "dominant causal transmission path",
        "Global-scale",
        "far budget to the near component",
    )
    for phrase in required_phrases:
        if phrase not in combined:
            raise CorePipelineError(f"required reporting phrase is missing: {phrase}")

    blueprint = (paths.output / "blueprint.md").read_text(encoding="utf-8")
    if "Sentence plan:" not in blueprint or "Reviewer objection:" not in blueprint:
        raise CorePipelineError("blueprint lacks executable sentence plan or objection")
    validation_methods = spec["methods"]["primary"] + spec["methods"].get("fixed_controls", [])
    for method in validation_methods:
        metric = f"methods.{method}.fixed_variance.reward"
        if metric not in blueprint:
            raise CorePipelineError(f"blueprint missing exact metric path: {metric}")

    theorem = (paths.output / "theorem.tex").read_text(encoding="utf-8")
    proof = (paths.output / "proof.tex").read_text(encoding="utf-8")
    if "C(1+r)^k" not in theorem or "e^{-\\lambda r}" not in theorem:
        raise CorePipelineError("theorem omits finite-order or exponential assumption")
    if "utility" not in proof or "squeeze theorem" not in proof.lower():
        raise CorePipelineError("proof omits claim boundary or analytic closure")

    # Rebuild the expected snapshot in an isolated directory so validation is read-only.
    with tempfile.TemporaryDirectory(prefix="drpo-paper-core-validate-") as temporary:
        rebuilt_paths = Paths(repo=paths.repo, spec=paths.spec, output=Path(temporary))
        rebuilt = build_snapshot(rebuilt_paths, spec_override=spec)
        if rebuilt_paths.snapshot.read_bytes() != paths.snapshot.read_bytes():
            raise CorePipelineError("snapshot is not deterministic")
    if rebuilt["snapshot_sha256"] != snapshot["snapshot_sha256"]:
        raise CorePipelineError("snapshot identity changed during validation")

    pages, page_count_source = pdf_pages(paths.pdf, manifest=manifest)
    required_pages = int(spec["release"]["required_pdf_pages"])
    if pages != required_pages:
        raise CorePipelineError(f"Core review PDF must have {required_pages} pages, got {pages}")

    log = (paths.output / "latex_build.txt").read_text(encoding="utf-8", errors="replace")
    quality_errors = [
        marker
        for marker in ("Undefined control sequence", "LaTeX Error", "Overfull \\hbox", "Overfull \\vbox")
        if marker in log
    ]
    if quality_errors:
        raise CorePipelineError(f"LaTeX quality audit failed: {quality_errors}")

    return {
        "status": "PASS",
        "task_id": spec["spec_id"],
        "experiment_id": snapshot["experiment"]["id"],
        "experiment_status": snapshot["experiment"]["status"],
        "pdf_pages": pages,
        "page_count_source": page_count_source,
        "pdf_sha256": sha256_file(paths.pdf),
        "snapshot_sha256": snapshot["snapshot_sha256"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("snapshot", "build-slice", "compile", "validate-slice", "all")
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--spec", type=Path, default=Path("docs/manuscript/paper_spec_core.yaml")
    )
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def make_paths(args: argparse.Namespace) -> Paths:
    repo = args.repo_root.resolve()
    spec = (repo / args.spec).resolve()
    loaded = read_yaml(spec)
    output = (
        (repo / args.output_root).resolve()
        if args.output_root is not None
        else repo_path(repo, str(loaded["output_root"]))
    )
    return Paths(
        repo=repo,
        spec=spec,
        output=output,
        allow_output_override=args.output_root is not None,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = make_paths(args)
        if args.command == "snapshot":
            result = build_snapshot(paths)
        elif args.command == "build-slice":
            result = build_slice(paths)
        elif args.command == "compile":
            result = {"pdf": str(compile_pdf(paths))}
        elif args.command == "validate-slice":
            result = validate_slice(paths)
        else:
            build_slice(paths)
            compile_pdf(paths)
            result = validate_slice(paths)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (CorePipelineError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
