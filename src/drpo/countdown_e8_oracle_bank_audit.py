#!/usr/bin/env python3
"""Standard audit artifacts for Countdown E8 oracle-offline bank v2.

The audit is intentionally model-free: it validates the canonical corpus itself
before any downstream model-specific scoring, calibration, or training.  It
produces machine-readable summaries, CSV tables, plots, suspicious-row samples,
and a Markdown report so a 6000-row corpus can be accepted or rejected without
manual ad-hoc inspection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

NEGATIVE_BINS = ("detail_wrong", "near_value_wrong", "mid_value_wrong", "far_value_wrong")
DIFFICULTY_BINS = ("easy", "medium", "hard")
DEFAULT_EXPECTED_ROWS = 6000
DEFAULT_MIN_NEGATIVES_PER_PROMPT = 8
DEFAULT_MIN_BINS_PER_PROMPT = 3
DEFAULT_MAX_DUPLICATE_NEGATIVE_RATE = 0.05
DEFAULT_MAX_INCOMPLETE_ROW_RATE = 0.02


@dataclass(frozen=True)
class AuditThresholds:
    expected_rows: int | None = DEFAULT_EXPECTED_ROWS
    min_negatives_per_prompt: int = DEFAULT_MIN_NEGATIVES_PER_PROMPT
    min_bins_per_prompt: int = DEFAULT_MIN_BINS_PER_PROMPT
    max_duplicate_negative_rate: float = DEFAULT_MAX_DUPLICATE_NEGATIVE_RATE
    max_incomplete_row_rate: float = DEFAULT_MAX_INCOMPLETE_ROW_RATE
    spotcheck_per_difficulty: int = 10
    spotcheck_per_negative_bin: int = 10
    random_seed: int = 2026070902


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _describe(values: Sequence[float]) -> dict[str, float | int | None]:
    clean = [float(x) for x in values if math.isfinite(float(x))]
    if not clean:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    clean.sort()

    def percentile(q: float) -> float:
        if len(clean) == 1:
            return clean[0]
        pos = q * (len(clean) - 1)
        low = int(math.floor(pos))
        high = int(math.ceil(pos))
        if low == high:
            return clean[low]
        return clean[low] * (high - pos) + clean[high] * (pos - low)

    return {
        "count": len(clean),
        "min": clean[0],
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "max": clean[-1],
        "mean": sum(clean) / len(clean),
    }


def _difficulty(row: Mapping[str, Any]) -> str:
    payload = row.get("difficulty", {})
    if isinstance(payload, Mapping):
        value = payload.get("difficulty_bin", "unknown")
    else:
        value = "unknown"
    return str(value or "unknown")


def _negative_expression(negative: Mapping[str, Any]) -> str:
    return str(negative.get("expression", ""))


def _negative_bin(negative: Mapping[str, Any]) -> str:
    return str(negative.get("negative_bin", "unknown"))


def _negative_source(negative: Mapping[str, Any]) -> str:
    return str(negative.get("source", "unknown"))


def _is_positive_ok(row: Mapping[str, Any]) -> bool:
    positives = row.get("positives")
    if isinstance(positives, list) and positives:
        first = positives[0]
        if isinstance(first, Mapping):
            return bool(first.get("correct") and first.get("valid_format") and first.get("uses_numbers"))
    return bool(row.get("oracle_positive"))


def _row_id(row: Mapping[str, Any], index: int) -> str:
    return str(row.get("row_id") or row.get("source_prompt_id") or row.get("id") or f"row_{index:07d}")


def _collect_metrics(rows: Sequence[dict[str, Any]], thresholds: AuditThresholds) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    difficulty_counts: Counter[str] = Counter()
    negative_by_difficulty: dict[str, Counter[str]] = defaultdict(Counter)
    source_by_difficulty: dict[str, Counter[str]] = defaultdict(Counter)
    negatives_per_prompt: dict[str, list[int]] = defaultdict(list)
    value_errors: dict[str, list[float]] = defaultdict(list)
    structure_distances: dict[str, list[float]] = defaultdict(list)
    row_records: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    incomplete_rows: list[dict[str, Any]] = []
    accidental_correct_rows: list[dict[str, Any]] = []
    suspicious_rows: list[dict[str, Any]] = []
    positive_failures = 0
    total_negatives = 0
    duplicate_negative_count = 0
    prompt_ids: Counter[str] = Counter()
    all_value_errors: list[float] = []
    all_structure_distances: list[float] = []

    for index, row in enumerate(rows):
        rid = _row_id(row, index)
        source_prompt_id = str(row.get("source_prompt_id", rid))
        prompt_ids[source_prompt_id] += 1
        difficulty = _difficulty(row)
        difficulty_counts[difficulty] += 1
        negatives = row.get("negatives") if isinstance(row.get("negatives"), list) else []
        total_negatives += len(negatives)
        negatives_per_prompt[difficulty].append(len(negatives))
        if not _is_positive_ok(row):
            positive_failures += 1
            suspicious_rows.append({"row_id": rid, "reason": "positive_missing_or_not_marked_correct"})

        bins_for_row: Counter[str] = Counter()
        seen_negative_expr: set[str] = set()
        duplicates_in_row = 0
        accidental_correct = 0
        for negative in negatives:
            if not isinstance(negative, Mapping):
                suspicious_rows.append({"row_id": rid, "reason": "negative_not_mapping"})
                continue
            expression = _negative_expression(negative)
            if expression in seen_negative_expr:
                duplicates_in_row += 1
                duplicate_negative_count += 1
            seen_negative_expr.add(expression)
            if bool(negative.get("correct")):
                accidental_correct += 1
                accidental_correct_rows.append({"row_id": rid, "expression": expression, "negative_bin": _negative_bin(negative)})
            bin_name = _negative_bin(negative)
            source = _negative_source(negative)
            bins_for_row[bin_name] += 1
            negative_by_difficulty[difficulty][bin_name] += 1
            source_by_difficulty[difficulty][source] += 1
            if "value_error" in negative:
                value = float(negative["value_error"])
                value_errors[bin_name].append(value)
                all_value_errors.append(value)
            if "structure_distance" in negative:
                value = float(negative["structure_distance"])
                structure_distances[bin_name].append(value)
                all_structure_distances.append(value)
        missing_bins = [name for name in NEGATIVE_BINS if bins_for_row.get(name, 0) <= 0]
        incomplete_reasons: list[str] = []
        if len(negatives) < thresholds.min_negatives_per_prompt:
            incomplete_reasons.append("too_few_negatives")
        if sum(1 for name in NEGATIVE_BINS if bins_for_row.get(name, 0) > 0) < thresholds.min_bins_per_prompt:
            incomplete_reasons.append("too_few_negative_bins")
        if accidental_correct:
            incomplete_reasons.append("negative_marked_correct")
        if duplicates_in_row:
            duplicate_rows.append({"row_id": rid, "duplicate_negative_count": duplicates_in_row})
        if incomplete_reasons:
            incomplete = {
                "row_id": rid,
                "difficulty": difficulty,
                "negative_count": len(negatives),
                "covered_bins": sorted(k for k, v in bins_for_row.items() if v > 0),
                "missing_bins": missing_bins,
                "reasons": incomplete_reasons,
            }
            incomplete_rows.append(incomplete)
            suspicious_rows.append(incomplete)
        row_records.append({
            "row_id": rid,
            "source_prompt_id": source_prompt_id,
            "difficulty": difficulty,
            "negative_count": len(negatives),
            "covered_negative_bins": ";".join(sorted(k for k, v in bins_for_row.items() if v > 0)),
            "duplicate_negative_count": duplicates_in_row,
            "accidental_correct_negative_count": accidental_correct,
        })

    duplicate_prompt_ids = [{"source_prompt_id": key, "count": value} for key, value in sorted(prompt_ids.items()) if value > 1]
    duplicate_rate = _safe_div(duplicate_negative_count, total_negatives)
    incomplete_rate = _safe_div(len(incomplete_rows), len(rows))
    accidental_correct_rate = _safe_div(len(accidental_correct_rows), total_negatives)
    positive_correct_rate = _safe_div(len(rows) - positive_failures, len(rows))
    hard_checks = {
        "row_count": len(rows),
        "expected_rows": thresholds.expected_rows,
        "row_count_ok": thresholds.expected_rows is None or len(rows) == thresholds.expected_rows,
        "oracle_positive_correct_rate": positive_correct_rate,
        "oracle_positive_correct_ok": positive_failures == 0,
        "negative_accidental_correct_rate": accidental_correct_rate,
        "negative_accidental_correct_ok": len(accidental_correct_rows) == 0,
        "duplicate_prompt_ids": len(duplicate_prompt_ids),
        "duplicate_prompt_ids_ok": len(duplicate_prompt_ids) == 0,
        "duplicate_negative_rate": duplicate_rate,
        "duplicate_negative_rate_ok": duplicate_rate <= thresholds.max_duplicate_negative_rate,
        "incomplete_row_rate": incomplete_rate,
        "incomplete_row_rate_ok": incomplete_rate <= thresholds.max_incomplete_row_rate,
    }
    difficulty_coverage = {
        "required_bins": list(DIFFICULTY_BINS),
        "present_bins": sorted(difficulty_counts),
        "missing_bins": [name for name in DIFFICULTY_BINS if difficulty_counts.get(name, 0) == 0],
        "status": "PASS" if all(difficulty_counts.get(name, 0) > 0 for name in DIFFICULTY_BINS) else "FAIL",
    }
    smoothness_missing: dict[str, list[str]] = {}
    for difficulty in DIFFICULTY_BINS:
        if difficulty_counts.get(difficulty, 0) == 0:
            smoothness_missing[difficulty] = list(NEGATIVE_BINS)
            continue
        missing = [name for name in NEGATIVE_BINS if negative_by_difficulty[difficulty].get(name, 0) == 0]
        if missing:
            smoothness_missing[difficulty] = missing
    negative_smoothness = {
        "status": "PASS" if not smoothness_missing else "WARN",
        "missing_negative_bins_by_difficulty": smoothness_missing,
        "value_error_summary_all": _describe(all_value_errors),
        "structure_distance_summary_all": _describe(all_structure_distances),
    }
    status = "PASS"
    if not all(bool(value) for key, value in hard_checks.items() if key.endswith("_ok")):
        status = "FAIL"
    elif difficulty_coverage["status"] != "PASS":
        status = "FAIL"
    elif negative_smoothness["status"] != "PASS":
        status = "WARN"

    summary = {
        "status": status,
        "hard_validity": "PASS" if all(bool(value) for key, value in hard_checks.items() if key.endswith("_ok")) else "FAIL",
        "difficulty_coverage": difficulty_coverage,
        "negative_smoothness": negative_smoothness,
        "thresholds": thresholds.__dict__,
        "hard_checks": hard_checks,
        "row_count": len(rows),
        "total_negatives": total_negatives,
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "negative_bin_counts_by_difficulty": {
            key: dict(sorted(counter.items())) for key, counter in sorted(negative_by_difficulty.items())
        },
        "negative_source_counts_by_difficulty": {
            key: dict(sorted(counter.items())) for key, counter in sorted(source_by_difficulty.items())
        },
        "negative_bin_distribution": dict(sorted(Counter({
            key: sum(counter.get(key, 0) for counter in negative_by_difficulty.values()) for key in NEGATIVE_BINS
        }).items())),
        "value_error_summary_by_negative_bin": {key: _describe(values) for key, values in sorted(value_errors.items())},
        "structure_distance_summary_by_negative_bin": {key: _describe(values) for key, values in sorted(structure_distances.items())},
        "suspicious_rows_count": len(suspicious_rows),
        "recommended_action": "accept" if status == "PASS" else ("inspect" if status == "WARN" else "regenerate_or_fix"),
    }
    tables = {
        "row_records": row_records,
        "duplicate_prompt_ids": duplicate_prompt_ids,
        "duplicate_rows": duplicate_rows,
        "incomplete_rows": incomplete_rows,
        "accidental_correct_rows": accidental_correct_rows,
        "suspicious_rows": suspicious_rows[:500],
        "difficulty_distribution": [
            {"difficulty": key, "rows": value, "fraction": _safe_div(value, len(rows))}
            for key, value in sorted(difficulty_counts.items())
        ],
        "negative_type_distribution": [
            {"difficulty": difficulty, "negative_bin": bin_name, "count": negative_by_difficulty[difficulty].get(bin_name, 0)}
            for difficulty in sorted(difficulty_counts)
            for bin_name in NEGATIVE_BINS
        ],
        "source_mix_by_difficulty": [
            {"difficulty": difficulty, "source": source, "count": count}
            for difficulty, counter in sorted(source_by_difficulty.items())
            for source, count in sorted(counter.items())
        ],
        "negative_summary_by_bin": [
            {
                "negative_bin": key,
                **{f"value_error_{k}": v for k, v in _describe(value_errors.get(key, [])).items()},
                **{f"structure_distance_{k}": v for k, v in _describe(structure_distances.get(key, [])).items()},
            }
            for key in NEGATIVE_BINS
        ],
    }
    return summary, tables


def _write_tables(out_dir: Path, summary: Mapping[str, Any], tables: Mapping[str, list[dict[str, Any]]]) -> None:
    table_dir = out_dir / "tables"
    write_csv(table_dir / "difficulty_distribution.csv", tables["difficulty_distribution"], ["difficulty", "rows", "fraction"])
    write_csv(table_dir / "negative_type_distribution.csv", tables["negative_type_distribution"], ["difficulty", "negative_bin", "count"])
    write_csv(table_dir / "source_mix_by_difficulty.csv", tables["source_mix_by_difficulty"], ["difficulty", "source", "count"])
    write_csv(table_dir / "negative_summary_by_bin.csv", tables["negative_summary_by_bin"], list(tables["negative_summary_by_bin"][0].keys()) if tables["negative_summary_by_bin"] else ["negative_bin"])
    write_csv(table_dir / "incomplete_rows.csv", tables["incomplete_rows"], ["row_id", "difficulty", "negative_count", "covered_bins", "missing_bins", "reasons"])
    write_csv(table_dir / "duplicate_prompt_ids.csv", tables["duplicate_prompt_ids"], ["source_prompt_id", "count"])
    write_csv(table_dir / "duplicate_rows.csv", tables["duplicate_rows"], ["row_id", "duplicate_negative_count"])
    write_csv(table_dir / "accidental_correct_rows.csv", tables["accidental_correct_rows"], ["row_id", "expression", "negative_bin"])

    coverage_rows: list[dict[str, Any]] = []
    neg_by_diff = summary["negative_bin_counts_by_difficulty"]
    for difficulty in DIFFICULTY_BINS:
        row: dict[str, Any] = {
            "difficulty": difficulty,
            "rows": summary["difficulty_counts"].get(difficulty, 0),
            "oracle_positive": summary["difficulty_counts"].get(difficulty, 0),
        }
        for bin_name in NEGATIVE_BINS:
            row[bin_name] = neg_by_diff.get(difficulty, {}).get(bin_name, 0)
        coverage_rows.append(row)
    write_csv(
        table_dir / "positive_negative_coverage.csv",
        coverage_rows,
        ["difficulty", "rows", "oracle_positive", *NEGATIVE_BINS],
    )


def _plot_bar(labels: Sequence[str], values: Sequence[float], title: str, ylabel: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(list(labels), list(values))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_stacked_by_difficulty(summary: Mapping[str, Any], key: str, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = summary[key]
    difficulties = [name for name in DIFFICULTY_BINS if name in payload]
    bins = list(NEGATIVE_BINS)
    fig, ax = plt.subplots(figsize=(9, 5))
    bottoms = [0] * len(difficulties)
    for bin_name in bins:
        values = [payload[difficulty].get(bin_name, 0) for difficulty in difficulties]
        ax.bar(difficulties, values, bottom=bottoms, label=bin_name)
        bottoms = [a + b for a, b in zip(bottoms, values)]
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_histogram(rows: Sequence[dict[str, Any]], field: str, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_bin: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        negatives = row.get("negatives") if isinstance(row.get("negatives"), list) else []
        for negative in negatives:
            if isinstance(negative, Mapping) and field in negative:
                by_bin[_negative_bin(negative)].append(float(negative[field]))
    fig, ax = plt.subplots(figsize=(9, 5))
    for bin_name in NEGATIVE_BINS:
        values = by_bin.get(bin_name, [])
        if values:
            ax.hist(values, bins=30, alpha=0.35, label=bin_name)
    ax.set_title(title)
    ax.set_xlabel(field)
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_negatives_per_prompt(rows: Sequence[dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values_by_diff: list[list[int]] = []
    labels: list[str] = []
    for difficulty in DIFFICULTY_BINS:
        values = [len(row.get("negatives", [])) for row in rows if _difficulty(row) == difficulty]
        if values:
            values_by_diff.append(values)
            labels.append(difficulty)
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        ax.boxplot(values_by_diff, tick_labels=labels)
    except TypeError:
        ax.boxplot(values_by_diff, labels=labels)
    ax.set_title("Negatives per prompt by difficulty")
    ax.set_ylabel("negative count")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_quality_matrix(summary: Mapping[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = ["oracle_positive", *NEGATIVE_BINS]
    matrix: list[list[float]] = []
    neg_by_diff = summary["negative_bin_counts_by_difficulty"]
    for difficulty in DIFFICULTY_BINS:
        row = [float(summary["difficulty_counts"].get(difficulty, 0))]
        row.extend(float(neg_by_diff.get(difficulty, {}).get(bin_name, 0)) for bin_name in NEGATIVE_BINS)
        matrix.append(row)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(list(range(len(columns))))
    ax.set_xticklabels(columns, rotation=35, ha="right")
    ax.set_yticks(list(range(len(DIFFICULTY_BINS))))
    ax.set_yticklabels(DIFFICULTY_BINS)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, f"{int(value)}", ha="center", va="center", fontsize=8)
    ax.set_title("Easy / medium / hard quality coverage matrix")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_figures(out_dir: Path, rows: Sequence[dict[str, Any]], summary: Mapping[str, Any]) -> None:
    figure_dir = out_dir / "figures"
    difficulties = list(summary["difficulty_counts"].keys())
    _plot_bar(
        difficulties,
        [summary["difficulty_counts"][key] for key in difficulties],
        "Difficulty distribution",
        "rows",
        figure_dir / "difficulty_distribution.png",
    )
    _plot_stacked_by_difficulty(
        summary,
        "negative_bin_counts_by_difficulty",
        "Negative type counts by difficulty",
        figure_dir / "negative_type_by_difficulty.png",
    )
    _plot_negatives_per_prompt(rows, figure_dir / "negatives_per_prompt_by_difficulty.png")
    _plot_histogram(rows, "value_error", "Value-error distribution by negative bin", figure_dir / "value_error_histogram.png")
    _plot_histogram(rows, "structure_distance", "Structure-distance distribution by negative bin", figure_dir / "structure_distance_histogram.png")
    _plot_quality_matrix(summary, figure_dir / "easy_medium_hard_quality_matrix.png")

    source_payload = summary["negative_source_counts_by_difficulty"]
    if source_payload:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        difficulties2 = [name for name in DIFFICULTY_BINS if name in source_payload]
        sources = sorted({source for counter in source_payload.values() for source in counter})
        fig, ax = plt.subplots(figsize=(9, 5))
        bottoms = [0] * len(difficulties2)
        for source in sources:
            values = [source_payload[difficulty].get(source, 0) for difficulty in difficulties2]
            ax.bar(difficulties2, values, bottom=bottoms, label=source)
            bottoms = [a + b for a, b in zip(bottoms, values)]
        ax.set_title("Negative source mix by difficulty")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        (figure_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_dir / "source_mix_by_difficulty.png", dpi=160)
        plt.close(fig)


def _sample_rows(rows: Sequence[dict[str, Any]], thresholds: AuditThresholds) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(thresholds.random_seed)
    samples: dict[str, list[dict[str, Any]]] = {}
    for difficulty in DIFFICULTY_BINS:
        candidates = [row for row in rows if _difficulty(row) == difficulty]
        rng.shuffle(candidates)
        samples[f"difficulty_{difficulty}"] = candidates[: thresholds.spotcheck_per_difficulty]
    for bin_name in NEGATIVE_BINS:
        candidates = [
            row for row in rows
            if any(isinstance(neg, Mapping) and _negative_bin(neg) == bin_name for neg in row.get("negatives", []))
        ]
        rng.shuffle(candidates)
        samples[f"negative_bin_{bin_name}"] = candidates[: thresholds.spotcheck_per_negative_bin]
    return samples


def _format_spotcheck_row(row: Mapping[str, Any], index: int) -> str:
    lines = [
        f"### {index}. `{row.get('row_id', row.get('source_prompt_id', 'unknown'))}`",
        "",
        f"- difficulty: `{_difficulty(row)}`",
        f"- numbers: `{row.get('numbers')}`",
        f"- target: `{row.get('target')}`",
        f"- oracle_positive: `{row.get('oracle_positive')}`",
        "- negatives:",
    ]
    for negative in list(row.get("negatives", []))[:8]:
        if not isinstance(negative, Mapping):
            continue
        lines.append(
            "  - "
            f"[{_negative_bin(negative)} / {_negative_source(negative)}] "
            f"`{negative.get('expression')}` "
            f"value_error={negative.get('value_error')} "
            f"structure_distance={negative.get('structure_distance')}"
        )
    lines.append("")
    return "\n".join(lines)


def _write_spotcheck(out_dir: Path, rows: Sequence[dict[str, Any]], tables: Mapping[str, list[dict[str, Any]]], thresholds: AuditThresholds) -> None:
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    samples = _sample_rows(rows, thresholds)
    for name, selected in samples.items():
        path = sample_dir / f"spotcheck_{name}.md"
        body = [f"# Spot check: {name}", ""]
        for index, row in enumerate(selected, start=1):
            body.append(_format_spotcheck_row(row, index))
        path.write_text("\n".join(body), encoding="utf-8")
    write_jsonl(sample_dir / "suspicious_rows.jsonl", tables["suspicious_rows"])


def _write_report(out_dir: Path, summary: Mapping[str, Any]) -> None:
    report = [
        "# Countdown E8 Oracle-Offline Bank V2 Audit",
        "",
        f"- status: **{summary['status']}**",
        f"- recommended_action: `{summary['recommended_action']}`",
        f"- rows: `{summary['row_count']}`",
        f"- total_negatives: `{summary['total_negatives']}`",
        f"- hard_validity: `{summary['hard_validity']}`",
        f"- difficulty_coverage: `{summary['difficulty_coverage']['status']}`",
        f"- negative_smoothness: `{summary['negative_smoothness']['status']}`",
        f"- suspicious_rows_count: `{summary['suspicious_rows_count']}`",
        "",
        "## Hard checks",
        "",
        "| check | value |",
        "|---|---:|",
    ]
    for key, value in summary["hard_checks"].items():
        report.append(f"| `{key}` | `{value}` |")
    report.extend([
        "",
        "## Difficulty distribution",
        "",
        "| difficulty | rows |",
        "|---|---:|",
    ])
    for key, value in summary["difficulty_counts"].items():
        report.append(f"| `{key}` | {value} |")
    report.extend([
        "",
        "## Negative coverage by difficulty",
        "",
        "| difficulty | detail | near_value | mid_value | far_value |",
        "|---|---:|---:|---:|---:|",
    ])
    for difficulty in DIFFICULTY_BINS:
        counter = summary["negative_bin_counts_by_difficulty"].get(difficulty, {})
        report.append(
            f"| `{difficulty}` | {counter.get('detail_wrong', 0)} | "
            f"{counter.get('near_value_wrong', 0)} | {counter.get('mid_value_wrong', 0)} | "
            f"{counter.get('far_value_wrong', 0)} |"
        )
    report.extend([
        "",
        "## Output files",
        "",
        "- `summary.json`",
        "- `tables/*.csv`",
        "- `figures/*.png`",
        "- `samples/spotcheck_*.md`",
        "- `samples/suspicious_rows.jsonl`",
        "",
        "## Interpretation rule",
        "",
        "This audit validates the canonical corpus only. Model-specific surprisal scoring, negative calibration, and RL training must be downstream phases on this fixed corpus.",
        "",
    ])
    (out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def _write_zip(out_dir: Path) -> Path:
    zip_path = out_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(out_dir.parent))
    return zip_path


def run_standard_audit(
    corpus_path: Path,
    out_dir: Path,
    *,
    thresholds: AuditThresholds | None = None,
    make_zip: bool = True,
) -> dict[str, Any]:
    thresholds = thresholds or AuditThresholds()
    rows = read_jsonl(corpus_path)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, tables = _collect_metrics(rows, thresholds)
    summary = {**summary, "corpus_path": str(corpus_path), "audit_dir": str(out_dir)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_tables(out_dir, summary, tables)
    _write_figures(out_dir, rows, summary)
    _write_spotcheck(out_dir, rows, tables, thresholds)
    _write_report(out_dir, summary)
    if make_zip:
        summary = {**summary, "audit_zip": str(_write_zip(out_dir))}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def thresholds_from_config(config: Mapping[str, Any] | None, *, expected_rows: int | None = None) -> AuditThresholds:
    if config is None:
        return AuditThresholds(expected_rows=expected_rows if expected_rows is not None else DEFAULT_EXPECTED_ROWS)
    data = config.get("data", {}) if isinstance(config.get("data"), Mapping) else {}
    neg = config.get("negative_generation", {}) if isinstance(config.get("negative_generation"), Mapping) else {}
    audit = config.get("audit", {}) if isinstance(config.get("audit"), Mapping) else {}
    return AuditThresholds(
        expected_rows=expected_rows if expected_rows is not None else int(data.get("corpus_train_rows", data.get("train_rows", DEFAULT_EXPECTED_ROWS))),
        min_negatives_per_prompt=int(neg.get("min_total_negatives_per_prompt", DEFAULT_MIN_NEGATIVES_PER_PROMPT)),
        min_bins_per_prompt=int(neg.get("min_bins_per_prompt", DEFAULT_MIN_BINS_PER_PROMPT)),
        max_duplicate_negative_rate=float(audit.get("max_duplicate_negative_rate", DEFAULT_MAX_DUPLICATE_NEGATIVE_RATE)),
        max_incomplete_row_rate=float(audit.get("max_incomplete_row_rate", DEFAULT_MAX_INCOMPLETE_ROW_RATE)),
        spotcheck_per_difficulty=int(audit.get("spotcheck_per_difficulty", 10)),
        spotcheck_per_negative_bin=int(audit.get("spotcheck_per_negative_bin", 10)),
        random_seed=int(audit.get("spotcheck_seed", 2026070902)),
    )


def _load_yaml_config(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="Path to oracle-offline bank v2 JSONL corpus.")
    parser.add_argument("--out_dir", required=True, help="Directory for REPORT.md, tables, figures, and samples.")
    parser.add_argument("--config", help="Optional oracle-bank-v2 YAML config for thresholds.")
    parser.add_argument("--expected_rows", type=int, help="Override expected row count.")
    parser.add_argument("--no_zip", action="store_true", help="Do not create audit ZIP next to out_dir.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = _load_yaml_config(Path(args.config)) if args.config else None
    thresholds = thresholds_from_config(config, expected_rows=args.expected_rows)
    summary = run_standard_audit(Path(args.corpus), Path(args.out_dir), thresholds=thresholds, make_zip=not args.no_zip)
    print(json.dumps({"status": summary["status"], "audit_dir": str(args.out_dir), "summary_json": str(Path(args.out_dir) / "summary.json"), "audit_zip": summary.get("audit_zip")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
