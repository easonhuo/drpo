#!/usr/bin/env python3
"""Canonical oracle-offline bank v2 for Countdown E8.

This module builds a model-independent Countdown offline corpus whose positives
are anchored by the oracle solution and whose negatives are stratified by
orthogonal error axes.  It deliberately does not sample from, score with, or
load a learner model.  Model-specific surprisal scoring and method-specific
negative calibration are downstream phases that should read this fixed corpus
rather than rebuilding it.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from drpo import countdown_qwen_arena_onefile as arena

EXPERIMENT_ID = "EXT-C-E8-ORACLE-OFFLINE-BANK-V2-0.5B-01"
CORPUS_VERSION = "oracle-offline-bank-v2.0"
NEGATIVE_BINS = ("detail_wrong", "near_value_wrong", "mid_value_wrong", "far_value_wrong")
OPS = ("+", "-", "*", "/")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_fraction_value(expression: str, numbers: Sequence[int], target: int) -> Fraction | None:
    check = arena.verify_expression(expression, numbers, target)
    if not check.get("valid_format") or not check.get("uses_numbers"):
        return None
    try:
        visitor = arena.ExpressionVerifier()
        return visitor.visit(ast.parse(arena.clean_expression(expression), mode="eval"))
    except Exception:
        return None


def _operator_counts(expression: str) -> Counter[str]:
    return Counter(ch for ch in expression if ch in OPS)


def _string_edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def _difficulty_score(row: dict[str, Any]) -> dict[str, Any]:
    oracle = str(row["oracle"])
    depth = arena.expression_tree_depth(oracle)
    ops = _operator_counts(oracle)
    division_count = ops["/"]
    subtraction_count = ops["-"]
    multiplication_count = ops["*"]
    expression_length = len(arena.clean_expression(oracle))
    target = abs(int(row["target"]))
    score = (
        2.0 * depth
        + 1.5 * division_count
        + 0.75 * subtraction_count
        + 0.25 * multiplication_count
        + 0.02 * target
        + 0.01 * expression_length
    )
    return {
        "score": float(score),
        "oracle_tree_depth": int(depth),
        "operator_counts": dict(sorted(ops.items())),
        "target_abs": int(target),
        "expression_length": int(expression_length),
    }


def assign_difficulty_bins(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for row in rows:
        features = _difficulty_score(row)
        scored.append((float(features["score"]), str(row["id"]), features))
    ranked = sorted(scored, key=lambda item: (item[0], item[1]))
    result: dict[str, dict[str, Any]] = {}
    n = len(ranked)
    for index, (_, row_id, features) in enumerate(ranked):
        if n <= 1:
            bin_name = "medium"
        elif index < n / 3:
            bin_name = "easy"
        elif index < 2 * n / 3:
            bin_name = "medium"
        else:
            bin_name = "hard"
        result[row_id] = {**features, "difficulty_bin": bin_name}
    return result


def _mutate_operator_candidates(expression: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for index, char in enumerate(expression):
        if char not in OPS:
            continue
        for replacement in OPS:
            if replacement == char:
                continue
            mutated = expression[:index] + replacement + expression[index + 1:]
            candidates.append((mutated, "synthetic_operator_flip"))
    return candidates


def _random_expression_candidates(
    numbers: Sequence[int], rng: random.Random, max_candidates: int
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for _ in range(max_candidates):
        # arena.random_expression expects a NumPy generator.  Use a deterministic
        # bridge seed per candidate so this module remains independent from global
        # NumPy RNG state inside tests and one-click runs.
        seed = rng.randrange(0, 2**32 - 1)
        np_rng = arena.np.random.default_rng(seed)
        expression, _ = arena.random_expression(np_rng, list(numbers))
        candidates.append((expression, "synthetic_random_tree"))
    return candidates


def _candidate_record(
    expression: str,
    source: str,
    row: dict[str, Any],
    oracle_structure: str,
    oracle_expression: str,
    tokenizer_free: bool = True,
) -> dict[str, Any] | None:
    numbers = [int(x) for x in row["numbers"]]
    target = int(row["target"])
    check = arena.verify_expression(expression, numbers, target)
    if not check.get("valid_format") or not check.get("uses_numbers") or check.get("correct"):
        return None
    value = _safe_fraction_value(expression, numbers, target)
    if value is None:
        return None
    cleaned = arena.clean_expression(expression)
    try:
        structure = arena.expression_structure(cleaned)
        tree_depth = arena.expression_tree_depth(cleaned)
    except Exception:
        return None
    value_error = abs(float(value - Fraction(target, 1)))
    edit_distance = _string_edit_distance(cleaned, arena.clean_expression(oracle_expression))
    same_structure = structure == oracle_structure
    depth_diff = abs(int(tree_depth) - int(arena.expression_tree_depth(oracle_expression)))
    structure_distance = float(edit_distance + 2 * depth_diff + (0 if same_structure else 3))
    return {
        "expression": cleaned,
        "source": source,
        "valid_format": True,
        "uses_numbers": True,
        "correct": False,
        "value": float(value),
        "value_error": float(value_error),
        "structure": structure,
        "same_oracle_structure": same_structure,
        "tree_depth": int(tree_depth),
        "structure_distance": structure_distance,
        "string_edit_distance_to_oracle": int(edit_distance),
        "depth_diff_to_oracle": int(depth_diff),
    }


def _bin_candidate(candidate: dict[str, Any], cfg: dict[str, Any]) -> str:
    if candidate["source"] == "synthetic_operator_flip":
        return "detail_wrong"
    value_error = float(candidate["value_error"])
    if value_error <= float(cfg["near_value_error_max"]):
        return "near_value_wrong"
    if value_error <= float(cfg["mid_value_error_max"]):
        return "mid_value_wrong"
    return "far_value_wrong"


def _select_stratified_negatives(
    candidates: list[dict[str, Any]], cfg: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_bin_target = {str(k): int(v) for k, v in cfg["per_bin_target"].items()}
    by_bin: dict[str, list[dict[str, Any]]] = {name: [] for name in NEGATIVE_BINS}
    seen: set[str] = set()
    for candidate in candidates:
        expression = candidate["expression"]
        if expression in seen:
            continue
        seen.add(expression)
        bin_name = _bin_candidate(candidate, cfg)
        record = {**candidate, "negative_bin": bin_name}
        by_bin.setdefault(bin_name, []).append(record)
    for bin_name, values in by_bin.items():
        # Stable preference: detail/near prefer smaller value and structure errors;
        # far prefers larger value error while still deterministic.
        if bin_name == "far_value_wrong":
            values.sort(key=lambda item: (-float(item["value_error"]), float(item["structure_distance"]), item["expression"]))
        else:
            values.sort(key=lambda item: (float(item["value_error"]), float(item["structure_distance"]), item["expression"]))
    selected: list[dict[str, Any]] = []
    missing: dict[str, int] = {}
    for bin_name in NEGATIVE_BINS:
        target = per_bin_target.get(bin_name, 0)
        chosen = by_bin.get(bin_name, [])[:target]
        selected.extend(chosen)
        if len(chosen) < target:
            missing[bin_name] = target - len(chosen)
    return selected, {
        "available_by_bin": {name: len(by_bin.get(name, [])) for name in NEGATIVE_BINS},
        "selected_by_bin": dict(Counter(item["negative_bin"] for item in selected)),
        "missing_by_bin": missing,
    }


@dataclass(frozen=True)
class BuildOutputs:
    train_corpus: Path
    val_data: Path
    test_data: Path
    split_manifest: Path
    audit_json: Path
    run_manifest: Path


def build_oracle_corpus(config: dict[str, Any], work_dir: Path, *, force: bool = False) -> BuildOutputs:
    work_dir = work_dir.resolve()
    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "oracle_offline_bank_v2_train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"
    split_manifest_path = data_dir / "split_manifest.json"
    audit_path = work_dir / "oracle_bank_v2_audit.json"
    run_manifest_path = work_dir / "RUN_MANIFEST.json"
    if train_path.exists() and audit_path.exists() and run_manifest_path.exists() and not force:
        return BuildOutputs(train_path, val_path, test_path, split_manifest_path, audit_path, run_manifest_path)

    data_cfg = config["data"]
    neg_cfg = config["negative_generation"]
    train_rows, val_rows, test_rows, split_manifest = arena.generate_structural_splits(
        int(data_cfg["train_rows"]),
        int(data_cfg["validation_rows"]),
        int(data_cfg["test_rows"]),
        int(data_cfg["generation_seed"]),
        n_numbers=4,
    )
    requested_train = int(data_cfg.get("corpus_train_rows", data_cfg["train_rows"]))
    train_rows = train_rows[:requested_train]
    difficulty = assign_difficulty_bins(train_rows)
    rng = random.Random(int(neg_cfg["seed"]))
    output_rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    candidate_attempts = int(neg_cfg["random_candidates_per_prompt"])
    min_total_negatives = int(neg_cfg["min_total_negatives_per_prompt"])
    min_bins = int(neg_cfg["min_bins_per_prompt"])
    for index, row in enumerate(train_rows):
        oracle = str(row["oracle"])
        oracle_structure = str(row["oracle_structure"])
        raw_candidates = _mutate_operator_candidates(oracle)
        raw_candidates.extend(_random_expression_candidates(row["numbers"], rng, candidate_attempts))
        evaluated: list[dict[str, Any]] = []
        seen: set[str] = set()
        for expression, source in raw_candidates:
            if expression in seen:
                continue
            seen.add(expression)
            record = _candidate_record(expression, source, row, oracle_structure, oracle)
            if record is not None:
                evaluated.append(record)
        selected, selection_audit = _select_stratified_negatives(evaluated, neg_cfg)
        selected_bins = set(selection_audit["selected_by_bin"])
        if len(selected) < min_total_negatives or len(selected_bins) < min_bins:
            dropped.append({
                "id": row["id"],
                "reason": "insufficient_stratified_negatives",
                "selected": len(selected),
                "selected_bins": sorted(selected_bins),
                **selection_audit,
            })
            if bool(neg_cfg.get("drop_rows_with_incomplete_bins", False)):
                continue
        row_difficulty = difficulty[str(row["id"])]
        positive_record = {
            "expression": oracle,
            "source": "oracle_positive",
            "correct": True,
            "valid_format": True,
            "uses_numbers": True,
            "structure": oracle_structure,
            "tree_depth": int(row_difficulty["oracle_tree_depth"]),
        }
        output_rows.append({
            "experiment_id": EXPERIMENT_ID,
            "corpus_version": CORPUS_VERSION,
            "row_id": f"oracle_bank_v2_{index:07d}",
            "source_prompt_id": row["id"],
            "split": "train",
            "numbers": row["numbers"],
            "target": int(row["target"]),
            "prompt": row["prompt"],
            "oracle_positive": oracle,
            "oracle_structure": oracle_structure,
            "difficulty": row_difficulty,
            "positives": [positive_record],
            "negatives": selected,
            "negative_bin_counts": dict(Counter(item["negative_bin"] for item in selected)),
            "negative_generation_audit": selection_audit,
            "audit_flags": {
                "has_oracle_positive": True,
                "has_detail_wrong": any(item["negative_bin"] == "detail_wrong" for item in selected),
                "has_near_value_wrong": any(item["negative_bin"] == "near_value_wrong" for item in selected),
                "has_mid_value_wrong": any(item["negative_bin"] == "mid_value_wrong" for item in selected),
                "has_far_value_wrong": any(item["negative_bin"] == "far_value_wrong" for item in selected),
                "model_sampled_negatives_used": False,
                "model_scoring_used": False,
            },
        })
    if not output_rows:
        raise RuntimeError("Oracle corpus builder produced no rows")
    write_jsonl(train_path, output_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(test_path, test_rows)
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8")
    audit = audit_oracle_corpus(output_rows, dropped=dropped)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    run_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "corpus_version": CORPUS_VERSION,
        "status": "built",
        "config": config,
        "outputs": {
            "train_corpus": str(train_path),
            "val_data": str(val_path),
            "test_data": str(test_path),
            "split_manifest": str(split_manifest_path),
            "audit_json": str(audit_path),
            "train_corpus_sha256": sha256_file(train_path),
            "audit_sha256": sha256_file(audit_path),
        },
        "model_dependency": {
            "corpus_generation": "none",
            "model_specific_scoring": "downstream_optional_phase",
            "negative_calibration": "downstream_method_specific_phase",
        },
    }
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return BuildOutputs(train_path, val_path, test_path, split_manifest_path, audit_path, run_manifest_path)


def audit_oracle_corpus(rows: Sequence[dict[str, Any]], *, dropped: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    difficulty_counts: Counter[str] = Counter()
    neg_by_difficulty: dict[str, Counter[str]] = defaultdict(Counter)
    missing_flags: Counter[str] = Counter()
    value_errors: dict[str, list[float]] = defaultdict(list)
    structure_distances: dict[str, list[float]] = defaultdict(list)
    total_negatives = 0
    for row in rows:
        difficulty = row["difficulty"]["difficulty_bin"]
        difficulty_counts[difficulty] += 1
        flags = row["audit_flags"]
        for flag, value in flags.items():
            if flag.startswith("has_") and not value:
                missing_flags[f"{difficulty}:{flag}"] += 1
        for negative in row["negatives"]:
            bin_name = negative["negative_bin"]
            neg_by_difficulty[difficulty][bin_name] += 1
            value_errors[bin_name].append(float(negative["value_error"]))
            structure_distances[bin_name].append(float(negative["structure_distance"]))
            total_negatives += 1

    def describe(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "min": None, "median": None, "max": None, "mean": None}
        ordered = sorted(values)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
        return {
            "count": len(values),
            "min": float(ordered[0]),
            "median": float(median),
            "max": float(ordered[-1]),
            "mean": float(sum(values) / len(values)),
        }

    return {
        "experiment_id": EXPERIMENT_ID,
        "corpus_version": CORPUS_VERSION,
        "rows": len(rows),
        "total_negatives": total_negatives,
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "negative_bin_counts_by_difficulty": {
            key: dict(sorted(counter.items())) for key, counter in sorted(neg_by_difficulty.items())
        },
        "missing_required_flags_by_difficulty": dict(sorted(missing_flags.items())),
        "dropped_rows": len(dropped or []),
        "dropped_examples": list(dropped or [])[:20],
        "value_error_summary_by_negative_bin": {
            key: describe(values) for key, values in sorted(value_errors.items())
        },
        "structure_distance_summary_by_negative_bin": {
            key: describe(values) for key, values in sorted(structure_distances.items())
        },
        "model_dependency": {
            "corpus_generation_uses_model_sampling": False,
            "corpus_generation_uses_model_scoring": False,
            "positives_are_oracle_anchored": True,
            "distance_quality_axes_decoupled": True,
        },
    }


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def cmd_build(args: argparse.Namespace) -> None:
    outputs = build_oracle_corpus(load_config(Path(args.config)), Path(args.work_dir), force=args.force)
    print(json.dumps({
        "status": "PASS",
        "train_corpus": str(outputs.train_corpus),
        "audit_json": str(outputs.audit_json),
        "run_manifest": str(outputs.run_manifest),
    }, indent=2, sort_keys=True))


def cmd_audit(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.corpus))
    audit = audit_oracle_corpus(rows)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build canonical oracle-offline corpus v2")
    build.add_argument("--config", required=True)
    build.add_argument("--work_dir", required=True)
    build.add_argument("--force", action="store_true")
    build.set_defaults(func=cmd_build)
    audit = sub.add_parser("audit", help="Audit an existing oracle corpus v2 JSONL")
    audit.add_argument("--corpus", required=True)
    audit.add_argument("--output_json")
    audit.set_defaults(func=cmd_audit)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
