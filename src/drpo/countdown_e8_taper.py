#!/usr/bin/env python3
"""Countdown E8 continuous-surprisal taper pilot.

This runner implements the registered ``EXT-C-E8-TAPER-0.5B-01`` protocol:

* one frozen Qwen2.5-0.5B reference adapter;
* one common sample-level replay pool with variable negatives per prompt;
* prompt-balanced paired sampling shared by every method;
* an independent calibration split and seed 9134;
* initialization-matched raw negative-gradient L2 for Global and all taper methods;
* fixed 1200-update training for three paired seeds;
* best and terminal task evaluation, current-surprisal quantile diagnostics, and
  separate task/support/numerical terminal audit sections.

Countdown remains an external-validity method pilot. It does not replace C-U1 or
D-U1 controlled mechanism identification, and a fixed horizon is not called
convergence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from threading import Thread
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR

try:  # direct script execution and package import are both supported
    from drpo import countdown_qwen_arena_onefile as arena
except ImportError:  # pragma: no cover - direct execution from src/drpo
    import countdown_qwen_arena_onefile as arena  # type: ignore


EXPERIMENT_ID = "EXT-C-E8-TAPER-0.5B-01"
VERSION = "1.1.0-normalized-distance-deterministic-audit"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "countdown_e8_taper_0p5b.yaml"
METHODS = (
    "positive_only",
    "uncontrolled_negative",
    "global_matched",
    "reciprocal_linear",
    "exponential",
    "squared_distance_exponential",
)
TAPER_METHODS = (
    "reciprocal_linear",
    "exponential",
    "squared_distance_exponential",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise RuntimeError(f"Adapter directory does not exist: {root}")
    members = sorted(root.rglob("*"))
    symlinks = [path for path in members if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"Adapter directory contains a symlink: {symlinks[0]}")
    files = [path for path in members if path.is_file()]
    if not files:
        raise RuntimeError(f"Adapter directory is empty: {root}")
    return {str(path.relative_to(root)): _sha256_file(path) for path in files}


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def source_provenance() -> dict[str, Any]:
    """Record both the E8-TAPER runner and its shared Countdown dependency."""
    runner = Path(__file__).resolve()
    shared = Path(arena.__file__).resolve()
    inherited = arena.source_provenance()
    return {
        "runner_source_file": str(runner),
        "runner_source_sha256": _sha256_file(runner),
        "shared_arena_source_file": str(shared),
        "shared_arena_source_sha256": _sha256_file(shared),
        "git_commit": inherited.get("git_commit"),
        "git_branch": inherited.get("git_branch"),
        "git_dirty": inherited.get("git_dirty"),
        "repository_root": inherited.get("repository_root"),
    }


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Countdown E8 taper config must be a YAML mapping")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Config experiment_id does not match the registered experiment")
    methods = tuple(value.get("methods") or [])
    if methods != METHODS:
        raise ValueError(f"Frozen method order mismatch: {methods!r}")
    seeds = tuple(value["confirmation"]["paired_training_seeds"])
    if seeds != (9234, 10234, 11234):
        raise ValueError(f"Frozen confirmation seed mismatch: {seeds!r}")
    if int(value["calibration"]["development_seed"]) != 9134:
        raise ValueError("Frozen calibration seed must be 9134")
    if int(value["training"]["optimizer_updates"]) != 1200:
        raise ValueError("Frozen update budget must be 1200")
    if not bool(value["training"].get("fixed_update_budget")):
        raise ValueError("E8 TAPER requires a fixed matched update budget")
    calibration = value["calibration"]
    if calibration.get("surprisal_scale_rule") != (
        "calibration_upper_half_median_minus_lower_half_median"
    ):
        raise ValueError("Frozen surprisal scale rule mismatch")
    if float(calibration.get("minimum_surprisal_scale", 0.0)) <= 0:
        raise ValueError("minimum_surprisal_scale must be positive")
    if calibration.get("shared_negative_scale") != (
        "positive_aggregate_gradient_l2_over_"
        "uncontrolled_negative_aggregate_gradient_l2"
    ):
        raise ValueError("Frozen shared negative scale definition mismatch")
    return value


def taper_weight(method: str, distance: torch.Tensor, coefficient: float) -> torch.Tensor:
    """Detached continuous taper weight on the true distance coordinate ``d``.

    The normalized excess surprisal is ``S`` and the distance coordinate is
    ``d = sqrt(S)``.  Therefore ``squared_distance_exponential`` is exactly
    ``exp(-lambda * S)`` rather than a fourth-order distance taper.
    """
    if method == "positive_only":
        return torch.zeros_like(distance)
    if method == "uncontrolled_negative":
        return torch.ones_like(distance)
    if method == "global_matched":
        return torch.full_like(distance, float(coefficient))
    if method == "reciprocal_linear":
        return 1.0 / (1.0 + float(coefficient) * distance)
    if method == "exponential":
        return torch.exp(-float(coefficient) * distance)
    if method == "squared_distance_exponential":
        return torch.exp(-float(coefficient) * distance.square())
    raise ValueError(f"Unknown method: {method}")


def normalized_remoteness(
    seq_lp: torch.Tensor, tau: float, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached normalized excess surprisal ``S`` and ``d=sqrt(S)``."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Surprisal scale must be finite and positive")
    excess = torch.relu(-seq_lp.detach() - float(tau))
    normalized_excess = excess / float(scale)
    distance = torch.sqrt(normalized_excess)
    return normalized_excess, distance


def learner_distance(seq_lp: torch.Tensor, tau: float, scale: float) -> torch.Tensor:
    """Compatibility helper returning the corrected distance coordinate only."""
    return normalized_remoteness(seq_lp, tau, scale)[1]


def calibration_surprisal_scale(
    surprisals: Sequence[float], *, minimum: float
) -> tuple[float, dict[str, float]]:
    """Freeze a robust rarity spread on the independent calibration split.

    The E6 correction used a rare-minus-common median scale.  Countdown has no
    permanent near/far labels, so the calibration samples are sorted once at
    the frozen reference initialization and divided into lower/upper rarity
    halves.  Their median gap is the scale.  This changes no training labels and
    fails closed when the calibration split does not contain a usable spread.
    """
    values = np.asarray([float(value) for value in surprisals], dtype=float)
    if values.size < 4 or not np.isfinite(values).all():
        raise RuntimeError("Calibration requires at least four finite surprisals")
    values.sort()
    midpoint = values.size // 2
    common_median = float(np.median(values[:midpoint]))
    rare_median = float(np.median(values[midpoint:]))
    scale = rare_median - common_median
    if not math.isfinite(scale) or scale < float(minimum):
        raise RuntimeError(
            "Calibration surprisal spread is too small for normalized distance: "
            f"scale={scale}, minimum={minimum}"
        )
    return scale, {
        "common_half_median_surprisal": common_median,
        "rare_half_median_surprisal": rare_median,
        "scale": scale,
    }


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            total += parameter.grad.detach().double().cpu().square().sum()
    return float(total.sqrt())


def _gradient_vectors(
    loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter], *, retain_graph: bool = False
) -> list[torch.Tensor | None]:
    return list(
        torch.autograd.grad(
            loss,
            parameters,
            allow_unused=True,
            retain_graph=retain_graph,
        )
    )


def gradient_norm_dot_cosine(
    left: Sequence[torch.Tensor | None], right: Sequence[torch.Tensor | None]
) -> tuple[float, float, float, float]:
    left_sq = torch.zeros((), dtype=torch.float64)
    right_sq = torch.zeros((), dtype=torch.float64)
    dot = torch.zeros((), dtype=torch.float64)
    for lhs, rhs in zip(left, right):
        lhs_cpu = lhs.detach().double().cpu() if lhs is not None else None
        rhs_cpu = rhs.detach().double().cpu() if rhs is not None else None
        if lhs_cpu is not None:
            left_sq += lhs_cpu.square().sum()
        if rhs_cpu is not None:
            right_sq += rhs_cpu.square().sum()
        if lhs_cpu is not None and rhs_cpu is not None:
            dot += (lhs_cpu * rhs_cpu).sum()
    left_norm = float(left_sq.sqrt())
    right_norm = float(right_sq.sqrt())
    cosine = float(dot / (left_sq.sqrt() * right_sq.sqrt()).clamp_min(1e-30))
    return left_norm, right_norm, float(dot), cosine


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return arena.read_jsonl(path)


def _structure(row: Mapping[str, Any]) -> str:
    return str(row.get("oracle_structure") or arena.expression_structure(str(row["oracle"])))


def balanced_disjoint_prompt_selection(
    rows: Sequence[dict[str, Any]],
    train_count: int,
    calibration_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select disjoint pattern-balanced train and calibration prompt sets."""
    if train_count <= 0 or calibration_count <= 0:
        raise ValueError("Both train and calibration prompt counts must be positive")
    if train_count + calibration_count > len(rows):
        raise ValueError("Requested more prompt rows than available")
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_structure(row)].append(dict(row))
    for group in grouped.values():
        rng.shuffle(group)
    patterns = sorted(grouped)

    def take(count: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        while len(selected) < count:
            progressed = False
            for pattern in patterns:
                group = grouped[pattern]
                if group:
                    # Pop exactly the selected row. This avoids the old cursor-based
                    # deletion bug when a selection stopped part-way through a cycle.
                    selected.append(group.pop())
                    progressed = True
                    if len(selected) == count:
                        break
            if not progressed:
                raise RuntimeError("Pattern-balanced selection exhausted the source rows")
        return selected

    train = take(train_count)
    calibration = take(calibration_count)
    train_ids = {str(row["id"]) for row in train}
    calibration_ids = {str(row["id"]) for row in calibration}
    if train_ids & calibration_ids:
        raise AssertionError("Calibration split overlaps replay-training prompts")
    return train, calibration


def balanced_prompt_take(
    rows: Sequence[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    """Take an exact pattern-balanced subset from already eligible rows."""
    if count <= 0 or count > len(rows):
        raise ValueError(f"Cannot take {count} rows from {len(rows)} eligible rows")
    # Reuse the disjoint selector with a one-row audit reserve, then return the
    # requested side. For the full-size case no reserve exists, so use a local
    # balanced queue directly.
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_structure(row)].append(dict(row))
    for group in grouped.values():
        rng.shuffle(group)
    patterns = sorted(grouped)
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for pattern in patterns:
            if grouped[pattern]:
                selected.append(grouped[pattern].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise RuntimeError("Pattern-balanced eligible-row selection exhausted")
    return selected


def _eligible_wrong_candidate(
    text: str,
    row: Mapping[str, Any],
    seen: set[str],
    allowed_structures: set[str] | None,
) -> dict[str, Any] | None:
    check = arena.verify_expression(text, row["numbers"], row["target"])
    expression = str(check["expression"])
    if not expression or expression in seen:
        return None
    seen.add(expression)
    if not (check["valid_format"] and check["uses_numbers"]):
        return None
    if check["correct"]:
        return None
    try:
        structure = arena.expression_structure(expression)
    except Exception:
        return None
    if allowed_structures is not None and structure not in allowed_structures:
        return None
    return {"expression": expression, "structure": structure}


def replay_pool_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = []
    for row in rows:
        canonical.append(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "oracle": row["oracle"],
                "negatives": [
                    {
                        "expression": item["expression"],
                        "structure": item["structure"],
                        "reference_surprisal": float(item["reference_surprisal"]),
                    }
                    for item in row["negatives"]
                ],
            }
        )
    return _stable_hash(canonical)


def _build_replay_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    *,
    allowed_structures: set[str] | None,
    rollouts: int,
    rounds: int,
    batch_size: int,
    score_batch_size: int,
    max_new_tokens: int,
    max_length: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    arena.seed_all(seed)
    states = [
        {"row": dict(row), "seen": set(), "negatives": []}
        for row in rows
    ]
    diagnostics: Counter[str] = Counter()
    for round_index in range(rounds):
        arena.seed_all(seed + round_index * 100003)
        for start in range(0, len(states), batch_size):
            chunk = states[start : start + batch_size]
            groups = arena.generate_outputs(
                model,
                tokenizer,
                [item["row"]["prompt"] for item in chunk],
                max_new_tokens,
                True,
                temperature,
                top_p,
                rollouts,
            )
            pending: list[tuple[int, dict[str, Any]]] = []
            pairs: list[tuple[str, str]] = []
            for local_index, (state, texts) in enumerate(zip(chunk, groups)):
                diagnostics["generated_candidates"] += len(texts)
                for text in texts:
                    item = _eligible_wrong_candidate(
                        text,
                        state["row"],
                        state["seen"],
                        allowed_structures,
                    )
                    if item is None:
                        diagnostics["rejected_or_duplicate"] += 1
                        continue
                    pending.append((local_index, item))
                    pairs.append((state["row"]["prompt"], item["expression"]))
            if pairs:
                scores = arena.score_completions_batch(
                    model,
                    tokenizer,
                    pairs,
                    max_length,
                    score_batch_size,
                )
                for (local_index, item), score in zip(pending, scores):
                    item["reference_surprisal"] = float(score)
                    chunk[local_index]["negatives"].append(item)
                    diagnostics["retained_candidates"] += 1
        print(
            json.dumps(
                {
                    "stage": "common_replay",
                    "round": round_index + 1,
                    "rounds": rounds,
                    "prompts_with_negative": sum(bool(item["negatives"]) for item in states),
                    "retained_candidates": diagnostics["retained_candidates"],
                }
            ),
            flush=True,
        )
    output: list[dict[str, Any]] = []
    for state in states:
        if not state["negatives"]:
            diagnostics["ineligible_prompts_without_negative"] += 1
            continue
        negatives = sorted(
            state["negatives"],
            key=lambda item: (item["reference_surprisal"], item["expression"]),
        )
        output.append({**state["row"], "negatives": negatives})
    diagnostics["eligible_prompts"] = len(output)
    return output, dict(diagnostics)


def cmd_build_replay(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    arena.seed_all(args.seed)
    tokenizer = arena.load_tokenizer(args.model_path)
    model = arena.load_model(
        args.model_path,
        args.reference_adapter,
        trainable_adapter=False,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        gradient_checkpointing=False,
    )
    model.eval()
    source_rows = _read_jsonl(args.input_data)
    replay_cfg = config["replay"]
    train_target = int(replay_cfg["train_prompt_rows"])
    calibration_target = int(replay_cfg["calibration_prompt_rows"])
    train_candidate_count = int(
        replay_cfg.get("train_candidate_prompt_rows", train_target)
    )
    calibration_candidate_count = int(
        replay_cfg.get("calibration_candidate_prompt_rows", calibration_target)
    )
    if train_candidate_count < train_target or calibration_candidate_count < calibration_target:
        raise ValueError("Replay candidate pools cannot be smaller than frozen target pools")
    train_selected, calibration_selected = balanced_disjoint_prompt_selection(
        source_rows,
        train_candidate_count,
        calibration_candidate_count,
        args.seed,
    )
    allowed_structures = (
        {_structure(row) for row in source_rows}
        if replay_cfg.get("enforce_training_structure_support", True)
        else None
    )
    common = dict(
        allowed_structures=allowed_structures,
        rollouts=int(replay_cfg["rollouts_per_prompt_per_round"]),
        rounds=int(replay_cfg["resample_rounds"]),
        batch_size=int(replay_cfg["generation_batch_size"]),
        score_batch_size=int(replay_cfg["score_batch_size"]),
        max_new_tokens=int(config["model"]["max_new_tokens"]),
        max_length=int(config["model"]["max_length"]),
        temperature=float(replay_cfg["temperature"]),
        top_p=float(replay_cfg["top_p"]),
    )
    train_rows, train_diag = _build_replay_rows(
        model, tokenizer, train_selected, seed=args.seed, **common
    )
    calibration_rows, calibration_diag = _build_replay_rows(
        model, tokenizer, calibration_selected, seed=args.seed + 1, **common
    )
    expected_train = train_target
    expected_calibration = calibration_target
    if len(train_rows) < expected_train or len(calibration_rows) < expected_calibration:
        raise RuntimeError(
            "Common replay reserve was exhausted because too many prompts lacked a legal "
            "generated negative and synthetic fallback is forbidden. "
            f"eligible train={len(train_rows)}/{expected_train} from {train_candidate_count}, "
            f"calibration={len(calibration_rows)}/{expected_calibration} from "
            f"{calibration_candidate_count}."
        )
    train_eligible_count = len(train_rows)
    calibration_eligible_count = len(calibration_rows)
    train_rows = balanced_prompt_take(train_rows, expected_train, args.seed + 200003)
    calibration_rows = balanced_prompt_take(
        calibration_rows, expected_calibration, args.seed + 300007
    )
    _atomic_jsonl(Path(args.train_output), train_rows)
    _atomic_jsonl(Path(args.calibration_output), calibration_rows)
    combined_hash = _stable_hash(
        {
            "train": replay_pool_hash(train_rows),
            "calibration": replay_pool_hash(calibration_rows),
        }
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "source_provenance": source_provenance(),
        "model_path": str(Path(args.model_path).resolve()),
        "reference_adapter": str(Path(args.reference_adapter).resolve()),
        "reference_adapter_hashes": _hash_tree(Path(args.reference_adapter)),
        "source_data": str(Path(args.input_data).resolve()),
        "source_data_sha256": _sha256_file(Path(args.input_data)),
        "train_output": str(Path(args.train_output).resolve()),
        "calibration_output": str(Path(args.calibration_output).resolve()),
        "train_prompt_rows": len(train_rows),
        "calibration_prompt_rows": len(calibration_rows),
        "train_candidate_prompt_rows": train_candidate_count,
        "calibration_candidate_prompt_rows": calibration_candidate_count,
        "train_eligible_prompt_rows_before_exact_selection": train_eligible_count,
        "calibration_eligible_prompt_rows_before_exact_selection": calibration_eligible_count,
        "train_negative_candidates": sum(len(row["negatives"]) for row in train_rows),
        "calibration_negative_candidates": sum(
            len(row["negatives"]) for row in calibration_rows
        ),
        "fixed_negative_count_per_prompt": False,
        "synthetic_negative_fallback": False,
        "prompt_balanced_sampling": True,
        "per_prompt_candidate_counts": {
            "train_min": min(len(row["negatives"]) for row in train_rows),
            "train_max": max(len(row["negatives"]) for row in train_rows),
            "train_mean": mean(len(row["negatives"]) for row in train_rows),
            "calibration_min": min(len(row["negatives"]) for row in calibration_rows),
            "calibration_max": max(len(row["negatives"]) for row in calibration_rows),
            "calibration_mean": mean(len(row["negatives"]) for row in calibration_rows),
        },
        "train_diagnostics": train_diag,
        "calibration_diagnostics": calibration_diag,
        "train_replay_hash": replay_pool_hash(train_rows),
        "calibration_replay_hash": replay_pool_hash(calibration_rows),
        "replay_pool_hash": combined_hash,
        "collector": {
            "rollouts_per_prompt_per_round": common["rollouts"],
            "resample_rounds": common["rounds"],
            "temperature": common["temperature"],
            "top_p": common["top_p"],
            "enforce_training_structure_support": bool(allowed_structures is not None),
        },
    }
    _atomic_json(Path(args.manifest_output), manifest)
    print(json.dumps(manifest, indent=2))


def make_prompt_balanced_sampler_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    total_samples: int,
) -> list[dict[str, int]]:
    """Uniform prompt cycles plus within-prompt negative sampling.

    The exact plan is shared by all methods for one paired seed. Candidate-rich
    prompts therefore receive no extra probability mass.
    """
    if not rows:
        raise ValueError("Sampler plan requires a non-empty replay pool")
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    rng = random.Random(seed)
    order: list[int] = []
    while len(order) < total_samples:
        cycle = list(range(len(rows)))
        rng.shuffle(cycle)
        order.extend(cycle)
    order = order[:total_samples]
    plan = []
    for row_index in order:
        candidates = rows[row_index].get("negatives") or []
        if not candidates:
            raise ValueError("Every replay row must have at least one negative")
        plan.append(
            {
                "prompt_index": int(row_index),
                "negative_index": int(rng.randrange(len(candidates))),
            }
        )
    return plan


def cmd_make_sampler(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    rows = _read_jsonl(args.replay_data)
    train_cfg = config["training"]
    total_samples = (
        int(train_cfg["optimizer_updates"])
        * int(train_cfg["gradient_accumulation_microbatches"])
        * int(train_cfg["micro_batch"])
    )
    plan = make_prompt_balanced_sampler_plan(
        rows, seed=args.seed, total_samples=total_samples
    )
    _atomic_jsonl(Path(args.output), plan)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "seed": args.seed,
        "replay_data": str(Path(args.replay_data).resolve()),
        "replay_sha256": _sha256_file(Path(args.replay_data)),
        "total_samples": len(plan),
        "prompt_count": len(rows),
        "minimum_prompt_count": min(Counter(item["prompt_index"] for item in plan).values()),
        "maximum_prompt_count": max(Counter(item["prompt_index"] for item in plan).values()),
        "plan_sha256": _sha256_file(Path(args.output)),
    }
    _atomic_json(Path(str(args.output) + ".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))


def _encoded_pair(
    tokenizer: Any, row: Mapping[str, Any], negative_index: int, max_length: int
) -> tuple[arena.EncodedExample, arena.EncodedExample]:
    negatives = row.get("negatives") or []
    if not (0 <= negative_index < len(negatives)):
        raise IndexError("Sampler plan negative_index is outside the replay row")
    return (
        arena.encode_prompt_completion(tokenizer, row["prompt"], row["oracle"], max_length),
        arena.encode_prompt_completion(
            tokenizer,
            row["prompt"],
            negatives[negative_index]["expression"],
            max_length,
        ),
    )


def _collate_pairs(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    plan_items: Sequence[Mapping[str, int]],
    max_length: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    positive: list[arena.EncodedExample] = []
    negative: list[arena.EncodedExample] = []
    for item in plan_items:
        pos, neg = _encoded_pair(
            tokenizer,
            rows[int(item["prompt_index"])],
            int(item["negative_index"]),
            max_length,
        )
        positive.append(pos)
        negative.append(neg)
    return (
        arena.pad_encoded(positive, tokenizer.pad_token_id),
        arena.pad_encoded(negative, tokenizer.pad_token_id),
    )


def _deterministic_current_weights(
    model: Any,
    negative_batch: Mapping[str, torch.Tensor],
    *,
    method: str,
    coefficient: float,
    tau: float,
    surprisal_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute learner-relative remoteness with dropout disabled and detach it.

    Training uses a second forward pass in train mode for the gradient-bearing
    negative log-probability.  This keeps the method a pure detached reweighting
    rule rather than a stochastic dropout-conditioned regularizer.
    """
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            stats = arena.completion_stats(model, dict(negative_batch))
            normalized_excess, distance = normalized_remoteness(
                stats["seq_lp"], tau, surprisal_scale
            )
            weights = taper_weight(method, distance, coefficient).detach()
    finally:
        model.train(was_training)
    return normalized_excess.detach(), distance.detach(), weights


def _calibration_sequence_surprisals(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    plan: Sequence[Mapping[str, int]],
    *,
    max_length: int,
    batch_size: int,
) -> list[float]:
    """Measure deterministic reference surprisal on the frozen calibration plan."""
    was_training = bool(model.training)
    model.eval()
    device = next(model.parameters()).device
    values: list[float] = []
    try:
        with torch.no_grad():
            for start in range(0, len(plan), batch_size):
                items = plan[start : start + batch_size]
                _, neg_batch = _collate_pairs(tokenizer, rows, items, max_length)
                stats = arena.completion_stats(
                    model, arena.move_to_device(neg_batch, device)
                )
                values.extend(float(value) for value in (-stats["seq_lp"]).cpu())
    finally:
        model.train(was_training)
    return values


def _objective_gradient_norm(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    plan: Sequence[Mapping[str, int]],
    trainable: Sequence[torch.nn.Parameter],
    *,
    kind: str,
    method: str,
    coefficient: float,
    tau: float,
    surprisal_scale: float,
    max_length: int,
    batch_size: int,
) -> float:
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    total_examples = len(plan)
    for start in range(0, total_examples, batch_size):
        items = plan[start : start + batch_size]
        pos_batch, neg_batch = _collate_pairs(tokenizer, rows, items, max_length)
        if kind == "positive":
            stats = arena.completion_stats(model, arena.move_to_device(pos_batch, device))
            loss = -stats["seq_lp"].mean()
        elif kind == "negative":
            stats = arena.completion_stats(model, arena.move_to_device(neg_batch, device))
            _, distance = normalized_remoteness(
                stats["seq_lp"], tau, surprisal_scale
            )
            weights = taper_weight(method, distance, coefficient)
            loss = (weights * stats["seq_lp"]).mean()
        else:
            raise ValueError(f"Unknown calibration objective kind: {kind}")
        (loss * (len(items) / total_examples)).backward()
    value = _gradient_norm(trainable)
    model.zero_grad(set_to_none=True)
    return value


def calibrate_monotone_coefficient(
    norm_fn: Any,
    target: float,
    *,
    maximum: float,
    steps: int,
    tolerance: float,
) -> tuple[float, float, float]:
    """Match a gradient-norm target without assuming global monotonicity.

    Individual taper weights decrease with the coefficient, but the norm of
    their vector sum can have small non-monotone regions because sample
    gradients cancel.  A deterministic geometric bracket scan therefore finds
    every observed target crossing before bisection refines the best interval.
    """
    if not math.isfinite(target) or target <= 0:
        raise ValueError("Calibration target must be finite and positive")
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("Calibration maximum must be finite and positive")

    grid = [0.0]
    value = min(1e-4, maximum)
    while value < maximum:
        grid.append(value)
        value *= 2.0
    if grid[-1] != maximum:
        grid.append(float(maximum))
    observations = [(coefficient, float(norm_fn(coefficient))) for coefficient in grid]
    if any(not math.isfinite(norm) or norm < 0 for _, norm in observations):
        raise RuntimeError("Calibration norm function returned a non-finite value")
    if observations[0][1] < target:
        raise RuntimeError("Taper norm at lambda=0 is already below the target")

    candidates = list(observations)
    brackets: list[tuple[float, float, float, float]] = []
    for (left, left_norm), (right, right_norm) in zip(observations, observations[1:]):
        left_delta = left_norm - target
        right_delta = right_norm - target
        if left_delta == 0:
            brackets.append((left, left, left_norm, left_norm))
        elif left_delta * right_delta <= 0:
            brackets.append((left, right, left_norm, right_norm))
    if not brackets:
        closest = min(candidates, key=lambda item: abs(math.log(max(item[1], 1e-30) / target)))
        relative_error = abs(closest[1] - target) / target
        if relative_error <= tolerance:
            return float(closest[0]), float(closest[1]), float(relative_error)
        raise RuntimeError(
            "Could not bracket taper target within the frozen coefficient range; "
            f"closest_norm={closest[1]}, target={target}, maximum={maximum}"
        )

    for left, right, left_norm, right_norm in brackets:
        if left == right:
            continue
        left_delta = left_norm - target
        for _ in range(steps):
            middle = 0.5 * (left + right)
            middle_norm = float(norm_fn(middle))
            if not math.isfinite(middle_norm) or middle_norm < 0:
                raise RuntimeError("Calibration norm function returned a non-finite value")
            candidates.append((middle, middle_norm))
            middle_delta = middle_norm - target
            if left_delta * middle_delta <= 0:
                right, right_norm = middle, middle_norm
            else:
                left, left_norm, left_delta = middle, middle_norm, middle_delta

    coefficient, matched = min(
        candidates, key=lambda item: abs(math.log(max(item[1], 1e-30) / target))
    )
    relative_error = abs(matched - target) / target
    if relative_error > tolerance:
        raise RuntimeError(
            f"Calibration relative error {relative_error:.6f} exceeds tolerance {tolerance:.6f}"
        )
    return float(coefficient), float(matched), float(relative_error)


def cmd_calibrate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    calibration_cfg = config["calibration"]
    arena.seed_all(int(calibration_cfg["development_seed"]))
    rows = _read_jsonl(args.calibration_replay)
    expected = int(calibration_cfg["batches"])
    if len(rows) != expected:
        raise RuntimeError(f"Calibration replay must contain exactly {expected} prompts")
    plan = make_prompt_balanced_sampler_plan(
        rows,
        seed=int(calibration_cfg["development_seed"]),
        total_samples=expected,
    )
    tokenizer = arena.load_tokenizer(args.model_path)
    model = arena.load_model(
        args.model_path,
        args.reference_adapter,
        trainable_adapter=True,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        gradient_checkpointing=False,
    )
    # Calibration is gradient-bearing but deterministic: keep dropout disabled.
    model.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Calibration model has no trainable parameters")
    calibration_surprisals = _calibration_sequence_surprisals(
        model,
        tokenizer,
        rows,
        plan,
        max_length=int(config["model"]["max_length"]),
        batch_size=1,
    )
    surprisal_scale, scale_diagnostics = calibration_surprisal_scale(
        calibration_surprisals,
        minimum=float(calibration_cfg["minimum_surprisal_scale"]),
    )
    common = dict(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        plan=plan,
        trainable=trainable,
        tau=float(calibration_cfg["surprisal_threshold_tau"]),
        surprisal_scale=surprisal_scale,
        max_length=int(config["model"]["max_length"]),
        batch_size=1,
    )
    positive_norm = _objective_gradient_norm(
        kind="positive", method="positive_only", coefficient=0.0, **common
    )
    uncontrolled_norm = _objective_gradient_norm(
        kind="negative", method="uncontrolled_negative", coefficient=1.0, **common
    )
    reference_lambda = float(calibration_cfg["inherited_reference_lambda"])
    target_unscaled = _objective_gradient_norm(
        kind="negative", method="exponential", coefficient=reference_lambda, **common
    )
    if any(
        not math.isfinite(value) or value <= 0
        for value in (positive_norm, uncontrolled_norm, target_unscaled)
    ):
        raise RuntimeError("Calibration norms must all be finite and positive")
    negative_scale = positive_norm / uncontrolled_norm
    target_effective = negative_scale * target_unscaled
    coefficients: dict[str, float] = {
        "positive_only": 0.0,
        "uncontrolled_negative": 1.0,
        "global_matched": target_unscaled / uncontrolled_norm,
        "exponential": reference_lambda,
    }
    matched_norms: dict[str, float] = {
        "positive_only": 0.0,
        "uncontrolled_negative": uncontrolled_norm,
        "global_matched": coefficients["global_matched"] * uncontrolled_norm,
        "exponential": target_unscaled,
    }
    errors: dict[str, float] = {
        "global_matched": abs(matched_norms["global_matched"] - target_unscaled)
        / target_unscaled,
        "exponential": 0.0,
    }
    for method in ("reciprocal_linear", "squared_distance_exponential"):
        coefficient, matched, relative_error = calibrate_monotone_coefficient(
            lambda value, method=method: _objective_gradient_norm(
                kind="negative", method=method, coefficient=value, **common
            ),
            target_unscaled,
            maximum=float(calibration_cfg["maximum_lambda"]),
            steps=int(calibration_cfg["bisection_steps"]),
            tolerance=float(calibration_cfg["relative_l2_tolerance"]),
        )
        coefficients[method] = coefficient
        matched_norms[method] = matched
        errors[method] = relative_error
    payload = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "development_seed": int(calibration_cfg["development_seed"]),
        "calibration_replay": str(Path(args.calibration_replay).resolve()),
        "calibration_replay_sha256": _sha256_file(Path(args.calibration_replay)),
        "reference_adapter": str(Path(args.reference_adapter).resolve()),
        "reference_adapter_hashes": _hash_tree(Path(args.reference_adapter)),
        "config": str(Path(args.config).resolve()),
        "config_sha256": _sha256_file(Path(args.config)),
        "surprisal_threshold_tau": float(calibration_cfg["surprisal_threshold_tau"]),
        "surprisal_scale": surprisal_scale,
        "surprisal_scale_rule": calibration_cfg["surprisal_scale_rule"],
        "surprisal_scale_diagnostics": scale_diagnostics,
        "distance_definition": "d=sqrt(max(0,(sequence_surprisal-tau)/scale))",
        "positive_gradient_l2": positive_norm,
        "uncontrolled_negative_gradient_l2": uncontrolled_norm,
        "shared_negative_scale": negative_scale,
        "target_unscaled_negative_gradient_l2": target_unscaled,
        "target_effective_negative_gradient_l2": target_effective,
        "target_definition": "corrected_exponential_linear_distance_lambda_0.7_at_common_initialization",
        "shared_negative_scale_definition": (
            "positive_aggregate_gradient_l2_over_"
            "uncontrolled_negative_aggregate_gradient_l2"
        ),
        "method_coefficients": coefficients,
        "matched_unscaled_negative_gradient_l2": matched_norms,
        "relative_matching_error": errors,
        "confirmation_or_test_metrics_used": False,
        "frozen_before_method_training": True,
        "sampler_plan": plan,
        "source_provenance": source_provenance(),
    }
    _atomic_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2))


def _cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> LambdaLR:
    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, scale)


def _csv_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(arena.csv_safe_row(dict(row)) for row in rows)


def _diagnostic_subset(
    rows: Sequence[dict[str, Any]], plan: Sequence[Mapping[str, int]], count: int
) -> list[dict[str, int]]:
    selected: list[dict[str, int]] = []
    used_prompts: set[int] = set()
    for item in plan:
        prompt_index = int(item["prompt_index"])
        if prompt_index in used_prompts:
            continue
        used_prompts.add(prompt_index)
        selected.append(
            {
                "prompt_index": prompt_index,
                "negative_index": int(item["negative_index"]),
            }
        )
        if len(selected) >= count:
            break
    if not selected:
        raise RuntimeError("Diagnostic subset is empty")
    return selected


def _teacher_forced_summary(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    plan: Sequence[Mapping[str, int]],
    max_length: int,
) -> dict[str, float]:
    device = next(model.parameters()).device
    pos_batch, neg_batch = _collate_pairs(tokenizer, rows, plan, max_length)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            positive = arena.completion_stats(
                model, arena.move_to_device(pos_batch, device)
            )
            negative = arena.completion_stats(
                model, arena.move_to_device(neg_batch, device)
            )
    finally:
        model.train(was_training)
    return {
        "correct_completion_surprisal": float((-positive["seq_lp"]).mean()),
        "correct_completion_entropy": float(positive["entropy"].mean()),
        "negative_completion_surprisal": float((-negative["seq_lp"]).mean()),
        "negative_completion_entropy": float(negative["entropy"].mean()),
    }


def surprisal_bin_diagnostics(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    plan: Sequence[Mapping[str, int]],
    trainable: Sequence[torch.nn.Parameter],
    *,
    method: str,
    coefficient: float,
    negative_scale: float,
    tau: float,
    surprisal_scale: float,
    max_length: int,
    quantile_bins: int,
    checkpoint_kind: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Full-parameter current-surprisal quantile audit on fixed prompt samples."""
    model.eval()
    device = next(model.parameters()).device
    pos_batch, neg_batch = _collate_pairs(tokenizer, rows, plan, max_length)
    pos_batch = arena.move_to_device(pos_batch, device)
    neg_batch = arena.move_to_device(neg_batch, device)
    with torch.no_grad():
        negative = arena.completion_stats(model, neg_batch)
        surprisals = (-negative["seq_lp"]).detach().float().cpu().numpy()
    if len(surprisals) < quantile_bins:
        raise RuntimeError("Diagnostic sample count must be at least the number of bins")
    order = np.argsort(surprisals, kind="stable")
    split_indices = np.array_split(order, quantile_bins)

    positive = arena.completion_stats(model, pos_batch)
    positive_loss = -positive["seq_lp"].mean()
    positive_grads = _gradient_vectors(positive_loss, trainable)
    positive_norm, _, _, _ = gradient_norm_dot_cosine(positive_grads, positive_grads)

    rows_out: list[dict[str, Any]] = []
    weighted_budgets: list[float] = []
    diagnostic_total = len(surprisals)
    for bin_index, indices_np in enumerate(split_indices):
        indices = torch.tensor(indices_np.tolist(), device=device, dtype=torch.long)
        selected_neg = arena.select_tensor_batch(neg_batch, indices)
        selected_pos = arena.select_tensor_batch(pos_batch, indices)

        branch = arena.completion_stats(model, selected_neg)
        normalized_excess, distance = normalized_remoteness(
            branch["seq_lp"], tau, surprisal_scale
        )
        weights = taper_weight(method, distance, coefficient)
        raw_loss = branch["seq_lp"].mean()
        weighted_loss = negative_scale * (weights * branch["seq_lp"]).mean()
        raw_grads = _gradient_vectors(raw_loss, trainable)
        weighted_grads = _gradient_vectors(weighted_loss, trainable)
        _, raw_norm, _, raw_cosine = gradient_norm_dot_cosine(
            positive_grads, raw_grads
        )
        _, weighted_norm, weighted_dot, weighted_cosine = gradient_norm_dot_cosine(
            positive_grads, weighted_grads
        )

        selected_correct = arena.completion_stats(model, selected_pos)
        correct_surprisal = -selected_correct["seq_lp"].mean()
        correct_grads = _gradient_vectors(correct_surprisal, trainable)
        _, _, collateral_dot, _ = gradient_norm_dot_cosine(correct_grads, weighted_grads)
        # Optimizer step is -grad(weighted negative loss). First-order change in
        # correct surprisal is therefore -<grad S_correct, grad L_negative>.
        collateral_effect = -collateral_dot
        # The diagnostic budget is the sum of per-bin gradient norms weighted
        # by the bin's sample fraction. This is an influence allocation metric,
        # not the norm of the globally aggregated gradient (which can cancel).
        weighted_budget = weighted_norm * (len(indices_np) / diagnostic_total)
        weighted_budgets.append(weighted_budget)
        rows_out.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "seed": seed,
                "method": method,
                "checkpoint_kind": checkpoint_kind,
                "bin_index": bin_index,
                "samples_per_bin": len(indices_np),
                "surprisal_min": float(np.min(surprisals[indices_np])),
                "surprisal_max": float(np.max(surprisals[indices_np])),
                "surprisal_mean": float(np.mean(surprisals[indices_np])),
                "normalized_excess_surprisal_mean": float(
                    normalized_excess.mean().detach()
                ),
                "distance_mean": float(distance.mean().detach()),
                "actual_taper_weight_mean": float(weights.mean().detach()),
                "raw_negative_gradient_norm": raw_norm,
                "weighted_negative_gradient_norm": weighted_norm,
                "weighted_negative_gradient_budget": weighted_budget,
                "positive_gradient_norm": positive_norm,
                "positive_negative_gradient_cosine_raw": raw_cosine,
                "positive_negative_gradient_cosine_weighted": weighted_cosine,
                "positive_negative_gradient_dot_weighted": weighted_dot,
                "correct_completion_collateral_effect_per_unit_optimizer_step": collateral_effect,
                "fraction_of_total_negative_gradient_budget": None,
            }
        )
    denominator = sum(weighted_budgets)
    for row, value in zip(rows_out, weighted_budgets):
        row["fraction_of_total_negative_gradient_budget"] = (
            float(value / denominator) if denominator > 0 else 0.0
        )
    summary = {
        "checkpoint_kind": checkpoint_kind,
        "method": method,
        "seed": seed,
        "positive_gradient_norm": positive_norm,
        "weighted_negative_gradient_budget_sum_across_bins": denominator,
        "budget_definition": "sum_of_bin_sample_fraction_times_bin_gradient_norm",
        "bin_budget_fraction_sums_to": sum(
            float(row["fraction_of_total_negative_gradient_budget"]) for row in rows_out
        ),
        **_teacher_forced_summary(model, tokenizer, rows, plan, max_length),
    }
    model.train()
    return rows_out, summary


def _evaluate_validation(
    model: Any,
    tokenizer: Any,
    val_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    config: Mapping[str, Any],
    eval_seed: int,
) -> dict[str, Any]:
    known = {_structure(row) for row in train_rows}
    train_cfg = config["training"]
    return arena.evaluate_rows(
        model,
        tokenizer,
        val_rows[: int(train_cfg["validation_examples"])],
        int(train_cfg["evaluation_batch_size"]),
        int(config["model"]["max_new_tokens"]),
        int(train_cfg["pass_at_k"]),
        eval_seed,
        known,
    )


def _final_window_slope(rows: Sequence[Mapping[str, Any]], key: str, count: int) -> float | None:
    usable = [row for row in rows if row.get(key) not in (None, "", "None")]
    if len(usable) < 2:
        return None
    tail = usable[-max(2, count) :]
    x = np.array([float(row["step"]) for row in tail], dtype=float)
    y = np.array([float(row[key]) for row in tail], dtype=float)
    if np.allclose(x, x[0]):
        return None
    return float(np.polyfit(x, y, 1)[0])


def _validate_training_inputs(
    *,
    config_path: Path,
    reference_adapter: Path,
    replay_path: Path,
    sampler_plan_path: Path,
    calibration_path: Path,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed on stale or cross-run calibration/sampler artifacts."""
    calibration = json.loads(calibration_path.read_text())
    if calibration.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("Calibration file belongs to another experiment")
    if calibration.get("config_sha256") != _sha256_file(config_path):
        raise RuntimeError("Calibration was generated from a different frozen config")
    if calibration.get("reference_adapter_hashes") != _hash_tree(reference_adapter):
        raise RuntimeError("Reference adapter differs from calibration initialization")

    manifest_path = Path(str(sampler_plan_path) + ".manifest.json")
    if not manifest_path.is_file():
        raise RuntimeError("Sampler plan manifest is missing")
    sampler_manifest = json.loads(manifest_path.read_text())
    if sampler_manifest.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("Sampler plan belongs to another experiment")
    if int(sampler_manifest.get("seed", -1)) != int(seed):
        raise RuntimeError("Sampler plan seed does not match the paired training seed")
    if sampler_manifest.get("replay_sha256") != _sha256_file(replay_path):
        raise RuntimeError("Sampler plan was not generated from the supplied replay")
    if sampler_manifest.get("plan_sha256") != _sha256_file(sampler_plan_path):
        raise RuntimeError("Sampler plan content does not match its manifest")
    return calibration, sampler_manifest


def cmd_train_method(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.method not in METHODS:
        raise ValueError(f"Method is not registered: {args.method}")
    arena.seed_all(args.seed)
    replay_path = Path(args.replay_data).resolve()
    sampler_plan_path = Path(args.sampler_plan).resolve()
    calibration_path = Path(args.calibration_json).resolve()
    config_path = Path(args.config).resolve()
    reference_adapter = Path(args.reference_adapter).resolve()
    replay_rows = _read_jsonl(replay_path)
    plan = _read_jsonl(sampler_plan_path)
    calibration, sampler_manifest = _validate_training_inputs(
        config_path=config_path,
        reference_adapter=reference_adapter,
        replay_path=replay_path,
        sampler_plan_path=sampler_plan_path,
        calibration_path=calibration_path,
        seed=args.seed,
    )
    expected_samples = (
        int(config["training"]["optimizer_updates"])
        * int(config["training"]["gradient_accumulation_microbatches"])
        * int(config["training"]["micro_batch"])
    )
    if len(plan) != expected_samples:
        raise RuntimeError(f"Sampler plan length {len(plan)} != {expected_samples}")
    if int(sampler_manifest.get("total_samples", -1)) != expected_samples:
        raise RuntimeError("Sampler manifest total_samples differs from frozen budget")
    tokenizer = arena.load_tokenizer(args.model_path)
    model = arena.load_model(
        args.model_path,
        args.reference_adapter,
        trainable_adapter=True,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        gradient_checkpointing=True,
    )
    model.train()
    device = next(model.parameters()).device
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Method model has no trainable parameters")
    train_cfg = config["training"]
    total_steps = int(train_cfg["optimizer_updates"])
    grad_accum = int(train_cfg["gradient_accumulation_microbatches"])
    micro_batch = int(train_cfg["micro_batch"])
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = _cosine_warmup_scheduler(
        optimizer,
        max(1, int(total_steps * float(train_cfg["warmup_ratio"]))),
        total_steps,
    )
    negative_scale = float(calibration["shared_negative_scale"])
    coefficient = float(calibration["method_coefficients"][args.method])
    tau = float(calibration["surprisal_threshold_tau"])
    surprisal_scale = float(calibration["surprisal_scale"])
    if not math.isfinite(surprisal_scale) or surprisal_scale <= 0:
        raise RuntimeError("Calibration surprisal_scale must be finite and positive")
    val_rows = _read_jsonl(args.val_data)
    source_train_rows = _read_jsonl(args.structure_reference_data)
    eval_seed = int(config["confirmation"]["paired_evaluation_seed_offset"]) + args.seed
    output = Path(args.output_dir)
    arena.ensure_checkpoint_output_is_local_or_ignored(output)
    output.mkdir(parents=True, exist_ok=True)
    best_dir = output / "best_adapter"
    terminal_dir = output / "terminal_adapter"
    last_finite_dir = output / "last_finite_adapter"
    metrics: list[dict[str, Any]] = []
    training_log: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    numerical_failure: str | None = None
    failure_step: int | None = None
    last_finite_step = 0
    best_step = 0
    best_value = -float("inf")
    selection_delta = float(train_cfg["selection_delta"])

    diagnostic_plan = _diagnostic_subset(
        replay_rows, plan, int(config["diagnostics"]["prompt_rows"])
    )
    initial_eval = _evaluate_validation(
        model, tokenizer, val_rows, source_train_rows, config, eval_seed
    )
    initial_teacher = _teacher_forced_summary(
        model,
        tokenizer,
        replay_rows,
        diagnostic_plan,
        int(config["model"]["max_length"]),
    )
    _, initial_neg_batch = _collate_pairs(
        tokenizer, replay_rows, diagnostic_plan, int(config["model"]["max_length"])
    )
    initial_normalized, initial_distance, initial_weights = (
        _deterministic_current_weights(
            model,
            arena.move_to_device(initial_neg_batch, device),
            method=args.method,
            coefficient=coefficient,
            tau=tau,
            surprisal_scale=surprisal_scale,
        )
    )
    initial_row = {
        "step": 0,
        "method": args.method,
        "seed": args.seed,
        "mean_taper_weight": float(initial_weights.mean()),
        "mean_normalized_excess_surprisal": float(initial_normalized.mean()),
        "mean_distance": float(initial_distance.mean()),
        **initial_teacher,
        **initial_eval,
    }
    metrics.append(initial_row)
    best_value = float(initial_eval[str(train_cfg["selection_metric"])])
    checkpoint_records.append(
        arena.save_local_model_checkpoint(model, tokenizer, best_dir, "best", 0)
    )

    cursor = 0
    completed_steps = 0
    for step in range(1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        aggregate = Counter()
        stop = False
        for _ in range(grad_accum):
            items = plan[cursor : cursor + micro_batch]
            cursor += micro_batch
            pos_batch, neg_batch = _collate_pairs(
                tokenizer,
                replay_rows,
                items,
                int(config["model"]["max_length"]),
            )
            positive = arena.completion_stats(
                model, arena.move_to_device(pos_batch, device)
            )
            positive_lp = positive["seq_lp"].mean()
            if args.method == "positive_only":
                normalized_excess = torch.zeros_like(positive["seq_lp"])
                distance = torch.zeros_like(positive["seq_lp"])
                weights = torch.zeros_like(positive["seq_lp"])
                negative_lp = torch.zeros_like(positive_lp)
                loss = -positive_lp
            else:
                neg_batch = arena.move_to_device(neg_batch, device)
                normalized_excess, distance, weights = _deterministic_current_weights(
                    model,
                    neg_batch,
                    method=args.method,
                    coefficient=coefficient,
                    tau=tau,
                    surprisal_scale=surprisal_scale,
                )
                negative = arena.completion_stats(model, neg_batch)
                negative_lp = (weights * negative["seq_lp"]).mean()
                loss = -positive_lp + negative_scale * negative_lp
            if not bool(torch.isfinite(loss)):
                numerical_failure = f"nonfinite_loss_at_step_{step}"
                failure_step = step
                stop = True
                break
            (loss / grad_accum).backward()
            aggregate["loss"] += float(loss.detach()) / grad_accum
            aggregate["positive_lp"] += float(positive_lp.detach()) / grad_accum
            aggregate["negative_lp"] += float(negative_lp.detach()) / grad_accum
            aggregate["mean_taper_weight"] += float(weights.mean().detach()) / grad_accum
            aggregate["mean_normalized_excess_surprisal"] += (
                float(normalized_excess.mean().detach()) / grad_accum
            )
            aggregate["mean_distance"] += float(distance.mean().detach()) / grad_accum
            aggregate["positive_entropy"] += float(positive["entropy"].mean().detach()) / grad_accum
        if stop:
            break
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(train_cfg["maximum_gradient_norm"])
        )
        aggregate["gradient_norm_before_clip_return"] = float(grad_norm)
        if not bool(torch.isfinite(grad_norm)):
            numerical_failure = f"nonfinite_gradient_at_step_{step}"
            failure_step = step
            break
        if not arena.optimizer_step_with_last_finite_guard(optimizer, trainable):
            numerical_failure = f"nonfinite_parameters_at_step_{step}"
            failure_step = step
            break
        scheduler.step()
        completed_steps = step
        last_finite_step = step
        aggregate["step"] = step
        aggregate["method"] = args.method
        aggregate["seed"] = args.seed
        aggregate["learning_rate"] = scheduler.get_last_lr()[0]
        training_log.append(dict(aggregate))
        if step % int(train_cfg["log_every_updates"]) == 0:
            print(json.dumps(dict(aggregate)), flush=True)
        if step % int(train_cfg["evaluation_every_updates"]) == 0 or step == total_steps:
            evaluation = _evaluate_validation(
                model, tokenizer, val_rows, source_train_rows, config, eval_seed
            )
            teacher = _teacher_forced_summary(
                model,
                tokenizer,
                replay_rows,
                diagnostic_plan,
                int(config["model"]["max_length"]),
            )
            row = {
                "step": step,
                "method": args.method,
                "seed": args.seed,
                "mean_taper_weight": aggregate["mean_taper_weight"],
                "mean_normalized_excess_surprisal": aggregate[
                    "mean_normalized_excess_surprisal"
                ],
                "mean_distance": aggregate["mean_distance"],
                **teacher,
                **evaluation,
            }
            metrics.append(row)
            value = float(evaluation[str(train_cfg["selection_metric"])])
            if value > best_value + selection_delta:
                best_value = value
                best_step = step
                checkpoint_records = [
                    item for item in checkpoint_records if item["kind"] != "best"
                ]
                checkpoint_records.append(
                    arena.save_local_model_checkpoint(
                        model, tokenizer, best_dir, "best", step
                    )
                )
            model.train()

    if numerical_failure:
        checkpoint_records.append(
            arena.save_local_model_checkpoint(
                model,
                tokenizer,
                last_finite_dir,
                "last_finite",
                last_finite_step,
            )
        )
        terminal_step: int | None = None
    else:
        checkpoint_records.append(
            arena.save_local_model_checkpoint(
                model,
                tokenizer,
                terminal_dir,
                "terminal",
                completed_steps,
            )
        )
        terminal_step = completed_steps

    _csv_write(output / "metrics.csv", metrics)
    _csv_write(output / "training_log.csv", training_log)

    # Diagnostics reload immutable checkpoint adapters. Release the training model
    # first so a 0.5B run does not unnecessarily hold two model copies on one GPU.
    del model, optimizer, scheduler, trainable
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    diag_rows: list[dict[str, Any]] = []
    diag_summaries: list[dict[str, Any]] = []
    initial_model = arena.load_model(
        args.model_path,
        args.reference_adapter,
        trainable_adapter=True,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        gradient_checkpointing=False,
    )
    initial_trainable = [p for p in initial_model.parameters() if p.requires_grad]
    rows_part, summary_part = surprisal_bin_diagnostics(
        initial_model,
        tokenizer,
        replay_rows,
        diagnostic_plan,
        initial_trainable,
        method=args.method,
        coefficient=coefficient,
        negative_scale=negative_scale,
        tau=tau,
        surprisal_scale=surprisal_scale,
        max_length=int(config["model"]["max_length"]),
        quantile_bins=int(config["diagnostics"]["quantile_bins"]),
        checkpoint_kind="initial",
        seed=args.seed,
    )
    diag_rows.extend(rows_part)
    diag_summaries.append(summary_part)
    del initial_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for checkpoint_kind, adapter in (
        ("best", best_dir),
        ("terminal" if not numerical_failure else "last_finite", terminal_dir if not numerical_failure else last_finite_dir),
    ):
        diag_model = arena.load_model(
            args.model_path,
            str(adapter),
            trainable_adapter=True,
            load_in_4bit=args.load_in_4bit,
            dtype=args.dtype,
            gradient_checkpointing=False,
        )
        diag_trainable = [p for p in diag_model.parameters() if p.requires_grad]
        rows_part, summary_part = surprisal_bin_diagnostics(
            diag_model,
            tokenizer,
            replay_rows,
            diagnostic_plan,
            diag_trainable,
            method=args.method,
            coefficient=coefficient,
            negative_scale=negative_scale,
            tau=tau,
            surprisal_scale=surprisal_scale,
            max_length=int(config["model"]["max_length"]),
            quantile_bins=int(config["diagnostics"]["quantile_bins"]),
            checkpoint_kind=checkpoint_kind,
            seed=args.seed,
        )
        diag_rows.extend(rows_part)
        diag_summaries.append(summary_part)
        del diag_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _csv_write(output / "surprisal_bin_diagnostics.csv", diag_rows)
    _atomic_json(output / "diagnostic_summary.json", diag_summaries)

    final_window = int(config["diagnostics"]["final_window_evaluations"])
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "method": args.method,
        "seed": args.seed,
        "source_provenance": source_provenance(),
        "reference_adapter": str(Path(args.reference_adapter).resolve()),
        "reference_adapter_hashes": _hash_tree(Path(args.reference_adapter)),
        "replay_data": str(Path(args.replay_data).resolve()),
        "replay_sha256": _sha256_file(Path(args.replay_data)),
        "sampler_plan": str(Path(args.sampler_plan).resolve()),
        "sampler_plan_sha256": _sha256_file(Path(args.sampler_plan)),
        "calibration_json": str(Path(args.calibration_json).resolve()),
        "calibration_sha256": _sha256_file(Path(args.calibration_json)),
        "shared_negative_scale": negative_scale,
        "method_coefficient": coefficient,
        "surprisal_threshold_tau": tau,
        "surprisal_scale": surprisal_scale,
        "distance_definition": calibration["distance_definition"],
        "planned_optimizer_updates": total_steps,
        "completed_optimizer_updates": completed_steps,
        "fixed_update_budget": True,
        "best_step": best_step,
        "best_value": best_value,
        "terminal_step": terminal_step,
        "last_finite_step": last_finite_step,
        "failure_detected_at_step": failure_step,
        "numerical_failure": numerical_failure,
        "stop_reason": numerical_failure or "fixed_update_budget_complete",
        "final_window_slopes": {
            key: _final_window_slope(metrics, key, final_window)
            for key in (
                "greedy_success",
                "pass_at_k",
                "valid_rate",
                "correct_completion_entropy",
                "negative_completion_surprisal",
            )
        },
        "checkpoints": checkpoint_records,
        "checkpoint_policy": "persistent-local best plus terminal-or-last-finite",
        "result_status": "pilot",
    }
    _atomic_json(output / "manifest.json", manifest)
    _atomic_json(
        output / "checkpoint_manifest.json",
        {
            "local_only": True,
            "experiment_id": EXPERIMENT_ID,
            "method": args.method,
            "seed": args.seed,
            "checkpoints": checkpoint_records,
        },
    )
    if numerical_failure:
        # A numerical failure is an experimental outcome, not an orchestration
        # failure. Preserve the last-finite checkpoint and let the other paired
        # runs finish so terminal_audit.json can report it separately.
        print(
            json.dumps(
                {
                    "stage": "train_method",
                    "method": args.method,
                    "seed": args.seed,
                    "status": "numerical_failure_recorded",
                    "failure": numerical_failure,
                    "last_finite_step": last_finite_step,
                }
            ),
            flush=True,
        )
        return
    if completed_steps != total_steps:
        raise RuntimeError("Method did not complete the frozen update budget")


def cmd_evaluate_checkpoint(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    tokenizer = arena.load_tokenizer(args.model_path)
    model = arena.load_model(
        args.model_path,
        args.adapter,
        trainable_adapter=False,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        gradient_checkpointing=False,
    )
    rows = _read_jsonl(args.data)
    train_rows = _read_jsonl(args.structure_reference_data)
    known = {_structure(row) for row in train_rows}
    metrics = arena.evaluate_rows(
        model,
        tokenizer,
        rows,
        int(config["training"]["evaluation_batch_size"]),
        int(config["model"]["max_new_tokens"]),
        int(config["training"]["pass_at_k"]),
        args.seed,
        known,
    )
    _atomic_json(Path(args.output), metrics)
    print(json.dumps(metrics, indent=2))


@dataclass(frozen=True)
class StageTask:
    name: str
    command: list[str]
    log_path: Path
    gpu: str | None = None


def _run_task(task: StageTask, repo: Path) -> None:
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if task.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(task.gpu)
    with task.log_path.open("w") as log:
        log.write("COMMAND: " + " ".join(task.command) + "\n")
        if task.gpu is not None:
            log.write(f"CUDA_VISIBLE_DEVICES={task.gpu}\n")
        log.flush()
        result = subprocess.run(
            task.command,
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode:
        raise RuntimeError(
            f"Stage {task.name} failed with exit {result.returncode}; see {task.log_path}"
        )


def _run_group(tasks: Sequence[StageTask], gpu_ids: Sequence[str], repo: Path) -> None:
    pending: queue.Queue[StageTask] = queue.Queue()
    for task in tasks:
        pending.put(task)
    errors: list[BaseException] = []

    def worker(gpu: str) -> None:
        while not errors:
            try:
                original = pending.get_nowait()
            except queue.Empty:
                return
            task = StageTask(original.name, original.command, original.log_path, gpu)
            try:
                _run_task(task, repo)
            except BaseException as exc:
                errors.append(exc)
            finally:
                pending.task_done()

    threads = [Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


def _model_flags(model_path: Path, dtype: str, load_in_4bit: bool) -> list[str]:
    flags = ["--model_path", str(model_path), "--dtype", dtype]
    if load_in_4bit:
        flags.append("--load_in_4bit")
    return flags


def _reference_adapter_from_sft(output: Path) -> Path:
    candidate = output / "best_adapter"
    if not (candidate / "adapter_config.json").is_file():
        raise RuntimeError(f"SFT best adapter is missing: {candidate}")
    return candidate


def _read_metrics_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _collect_summary(
    root: Path,
    config: Mapping[str, Any],
    seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    method_audits: dict[str, Any] = {}
    valid_structure_floor = float(
        config["diagnostics"]["valid_structure_boundary_rate"]
    )
    for method in METHODS:
        method_audits[method] = {}
        for seed in seeds:
            output = root / "methods" / method / str(seed)
            manifest = _read_metrics_json(output / "manifest.json")
            records = []
            for kind in ("best", "terminal"):
                path = output / f"test_metrics_{kind}.json"
                if not path.exists() and kind == "terminal":
                    path = output / "test_metrics_last_finite.json"
                if not path.exists():
                    continue
                task_metrics = _read_metrics_json(path)
                row = {
                    "method": method,
                    "seed": seed,
                    "checkpoint_kind": "last_finite" if "last_finite" in path.name else kind,
                    "best_step": manifest["best_step"],
                    "terminal_step": manifest.get("terminal_step"),
                    "last_finite_step": manifest.get("last_finite_step"),
                    "planned_optimizer_updates": manifest["planned_optimizer_updates"],
                    "completed_optimizer_updates": manifest["completed_optimizer_updates"],
                    "shared_negative_scale": manifest["shared_negative_scale"],
                    "method_coefficient": manifest["method_coefficient"],
                    "numerical_failure": manifest.get("numerical_failure"),
                    **task_metrics,
                }
                summary.append(row)
                records.append(row)
            best = next((row for row in records if row["checkpoint_kind"] == "best"), None)
            terminal = next(
                (
                    row
                    for row in records
                    if row["checkpoint_kind"] in {"terminal", "last_finite"}
                ),
                None,
            )
            diagnostic = _read_metrics_json(output / "diagnostic_summary.json")
            terminal_diag = next(
                (
                    item
                    for item in diagnostic
                    if item["checkpoint_kind"] in {"terminal", "last_finite"}
                ),
                None,
            )
            method_audits[method][str(seed)] = {
                "task_performance": {
                    "best": best,
                    "terminal": terminal,
                    "best_terminal_greedy_gap": (
                        float(best["greedy_success"] - terminal["greedy_success"])
                        if best and terminal
                        else None
                    ),
                    "best_terminal_pass_at_k_gap": (
                        float(best["pass_at_k"] - terminal["pass_at_k"])
                        if best and terminal
                        else None
                    ),
                },
                "valid_support_entropy": {
                    "best_valid_rate": best.get("valid_rate") if best else None,
                    "terminal_valid_rate": terminal.get("valid_rate") if terminal else None,
                    "terminal_correct_completion_entropy": (
                        terminal_diag.get("correct_completion_entropy")
                        if terminal_diag
                        else None
                    ),
                    "terminal_negative_completion_entropy": (
                        terminal_diag.get("negative_completion_entropy")
                        if terminal_diag
                        else None
                    ),
                    "valid_structure_boundary_rate": valid_structure_floor,
                    "terminal_valid_structure_boundary_event": (
                        bool(float(terminal["valid_rate"]) < valid_structure_floor)
                        if terminal
                        else None
                    ),
                    "entropy_reported_continuously_without_binary_threshold": True,
                },
                "numerical": {
                    "nan_inf_failure": bool(manifest.get("numerical_failure")),
                    "failure": manifest.get("numerical_failure"),
                    "last_finite_step": manifest.get("last_finite_step"),
                },
                "fixed_horizon_terminal": {
                    "classification": "fixed_horizon_not_convergence_claim",
                    "final_window_slopes": manifest.get("final_window_slopes"),
                },
            }
    return summary, method_audits


def _paired_effects(summary: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> dict[str, Any]:
    terminal = {
        (str(row["method"]), int(row["seed"])): row
        for row in summary
        if row["checkpoint_kind"] in {"terminal", "last_finite"}
    }
    output: dict[str, Any] = {}
    for method in METHODS:
        if method == "positive_only":
            continue
        greedy = [
            float(terminal[(method, seed)]["greedy_success"])
            - float(terminal[("positive_only", seed)]["greedy_success"])
            for seed in seeds
        ]
        pass_k = [
            float(terminal[(method, seed)]["pass_at_k"])
            - float(terminal[("positive_only", seed)]["pass_at_k"])
            for seed in seeds
        ]
        valid = [
            float(terminal[(method, seed)]["valid_rate"])
            - float(terminal[("positive_only", seed)]["valid_rate"])
            for seed in seeds
        ]
        output[method] = {
            "terminal_greedy_success_difference_vs_positive_only": greedy,
            "mean_terminal_greedy_success_difference_vs_positive_only": mean(greedy),
            "terminal_pass_at_k_difference_vs_positive_only": pass_k,
            "mean_terminal_pass_at_k_difference_vs_positive_only": mean(pass_k),
            "terminal_valid_rate_difference_vs_positive_only": valid,
            "mean_terminal_valid_rate_difference_vs_positive_only": mean(valid),
            "positive_greedy_seed_count": sum(value > 0 for value in greedy),
        }
    return output


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    script = Path(__file__).resolve()
    repo = Path(_git(script.parent, "rev-parse", "--show-toplevel")).resolve()
    head = _git(repo, "rev-parse", "HEAD")
    root = Path(args.work_dir).resolve()
    if (root / "RUN_COMPLETE.json").exists():
        raise RuntimeError("work_dir already contains a completed E8 TAPER run")
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    data = root / "data"
    replay = root / "replay"
    methods_root = root / "methods"
    for path in (logs, data, replay, methods_root):
        path.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path).resolve()
    if not model_path.is_dir():
        raise RuntimeError(f"Model path does not exist: {model_path}")
    gpu_ids = arena.resolve_gpu_ids(args.gpus)
    dtype = "bf16"
    load_in_4bit = False
    model_flags = _model_flags(model_path, dtype, load_in_4bit)
    source = Path(__file__).resolve()
    arena_script = Path(arena.__file__).resolve()
    config_path = Path(args.config).resolve()

    run_config = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "base_commit": head,
        "model_path": str(model_path),
        "gpus": gpu_ids,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "config": config,
        "test_split_access": "after_all_method_training_only",
        "source_provenance": source_provenance(),
    }
    _atomic_json(root / "run_config.json", run_config)

    train_file = data / "train.jsonl"
    val_file = data / "val.jsonl"
    test_file = data / "test.jsonl"
    split_manifest = data / "split_manifest.json"
    _run_task(
        StageTask(
            "preflight",
            [sys.executable, str(arena_script), "preflight", *model_flags, "--seed", "1234"],
            logs / "01_preflight.log",
            gpu_ids[0],
        ),
        repo,
    )
    _run_task(
        StageTask(
            "generate",
            [
                sys.executable,
                str(arena_script),
                "generate",
                "--train",
                str(config["dataset"]["train_rows"]),
                "--val",
                str(config["dataset"]["validation_rows"]),
                "--test",
                str(config["dataset"]["test_rows"]),
                "--n_numbers",
                str(config["dataset"]["numbers_per_problem"]),
                "--seed",
                str(config["dataset"]["generation_seed"]),
                "--train_out",
                str(train_file),
                "--val_out",
                str(val_file),
                "--test_out",
                str(test_file),
                "--manifest_out",
                str(split_manifest),
            ],
            logs / "02_generate.log",
        ),
        repo,
    )

    base_val_json = root / "base_validation.json"
    _run_task(
        StageTask(
            "base_validation",
            [
                sys.executable,
                str(arena_script),
                "evaluate",
                *model_flags,
                "--data",
                str(val_file),
                "--structure_reference_data",
                str(train_file),
                "--batch_size",
                str(config["training"]["evaluation_batch_size"]),
                "--pass_k",
                str(config["training"]["pass_at_k"]),
                "--seed",
                "5000",
                "--output_json",
                str(base_val_json),
            ],
            logs / "03_base_validation.log",
            gpu_ids[0],
        ),
        repo,
    )
    base_metrics = _read_metrics_json(base_val_json)
    reference_dir = root / "reference_adapter"
    base_pass = (
        float(base_metrics["greedy_success"])
        >= float(config["reference"]["base_greedy_success_gate"])
        and float(base_metrics["valid_rate"])
        >= float(config["reference"]["base_valid_rate_gate"])
    )
    if base_pass:
        _run_task(
            StageTask(
                "init_adapter",
                [
                    sys.executable,
                    str(arena_script),
                    "init_adapter",
                    *model_flags,
                    "--output_dir",
                    str(reference_dir),
                    "--seed",
                    str(config["dataset"]["generation_seed"]),
                ],
                logs / "04_init_adapter.log",
                gpu_ids[0],
            ),
            repo,
        )
        initialization_mode = "base_first_untrained_lora"
    else:
        sft_root = root / "sft_adapter"
        _run_task(
            StageTask(
                "sft_reference",
                [
                    sys.executable,
                    str(arena_script),
                    "sft",
                    *model_flags,
                    "--train_data",
                    str(train_file),
                    "--val_data",
                    str(val_file),
                    "--output_dir",
                    str(sft_root),
                    "--epochs",
                    str(config["reference"]["sft_epochs"]),
                    "--min_epochs",
                    str(config["reference"]["sft_min_epochs"]),
                    "--early_stop_patience",
                    str(config["reference"]["sft_early_stop_patience"]),
                    "--micro_batch",
                    str(config["reference"]["sft_micro_batch"]),
                    "--grad_accum",
                    str(config["reference"]["sft_gradient_accumulation"]),
                    "--lr",
                    str(config["reference"]["sft_learning_rate"]),
                    "--warmup_ratio",
                    str(config["reference"]["sft_warmup_ratio"]),
                    "--max_grad_norm",
                    str(config["reference"]["sft_max_gradient_norm"]),
                    "--eval_examples",
                    str(config["dataset"]["validation_rows"]),
                    "--eval_batch",
                    str(config["training"]["evaluation_batch_size"]),
                    "--pass_k",
                    str(config["training"]["pass_at_k"]),
                    "--eval_seed",
                    "5000",
                    "--seed",
                    str(config["dataset"]["generation_seed"]),
                    "--result_status",
                    "pilot",
                ],
                logs / "04_sft_reference.log",
                gpu_ids[0],
            ),
            repo,
        )
        source_adapter = _reference_adapter_from_sft(sft_root)
        shutil.copytree(source_adapter, reference_dir)
        initialization_mode = "sft_fallback_lora"

    reference_val_json = root / "reference_validation.json"
    _run_task(
        StageTask(
            "reference_validation",
            [
                sys.executable,
                str(arena_script),
                "evaluate",
                *model_flags,
                "--adapter",
                str(reference_dir),
                "--data",
                str(val_file),
                "--structure_reference_data",
                str(train_file),
                "--batch_size",
                str(config["training"]["evaluation_batch_size"]),
                "--pass_k",
                str(config["training"]["pass_at_k"]),
                "--seed",
                "5000",
                "--output_json",
                str(reference_val_json),
            ],
            logs / "05_reference_validation.log",
            gpu_ids[0],
        ),
        repo,
    )
    reference_metrics = _read_metrics_json(reference_val_json)

    def reference_passes(metrics: Mapping[str, Any]) -> bool:
        return bool(
            float(metrics["greedy_success"])
            >= float(config["reference"]["trained_greedy_success_gate"])
            and float(metrics["valid_rate"])
            >= float(config["reference"]["trained_valid_rate_gate"])
        )

    reference_gate = reference_passes(reference_metrics)
    if base_pass and not reference_gate:
        # The lower base-first gate only decides whether an untrained LoRA should
        # be tried. If it misses the final method-pilot gate, run the already
        # registered SFT fallback instead of failing an otherwise recoverable run.
        sft_root = root / "sft_adapter_after_base_gate"
        _run_task(
            StageTask(
                "sft_reference_after_base_gate",
                [
                    sys.executable,
                    str(arena_script),
                    "sft",
                    *model_flags,
                    "--train_data",
                    str(train_file),
                    "--val_data",
                    str(val_file),
                    "--output_dir",
                    str(sft_root),
                    "--epochs",
                    str(config["reference"]["sft_epochs"]),
                    "--min_epochs",
                    str(config["reference"]["sft_min_epochs"]),
                    "--early_stop_patience",
                    str(config["reference"]["sft_early_stop_patience"]),
                    "--micro_batch",
                    str(config["reference"]["sft_micro_batch"]),
                    "--grad_accum",
                    str(config["reference"]["sft_gradient_accumulation"]),
                    "--lr",
                    str(config["reference"]["sft_learning_rate"]),
                    "--warmup_ratio",
                    str(config["reference"]["sft_warmup_ratio"]),
                    "--max_grad_norm",
                    str(config["reference"]["sft_max_gradient_norm"]),
                    "--eval_examples",
                    str(config["dataset"]["validation_rows"]),
                    "--eval_batch",
                    str(config["training"]["evaluation_batch_size"]),
                    "--pass_k",
                    str(config["training"]["pass_at_k"]),
                    "--eval_seed",
                    "5000",
                    "--seed",
                    str(config["dataset"]["generation_seed"]),
                    "--result_status",
                    "pilot",
                ],
                logs / "05b_sft_reference_after_base_gate.log",
                gpu_ids[0],
            ),
            repo,
        )
        if reference_dir.exists():
            shutil.rmtree(reference_dir)
        shutil.copytree(_reference_adapter_from_sft(sft_root), reference_dir)
        initialization_mode = "base_attempt_then_sft_fallback_lora"
        _run_task(
            StageTask(
                "reference_validation_after_sft_fallback",
                [
                    sys.executable,
                    str(arena_script),
                    "evaluate",
                    *model_flags,
                    "--adapter",
                    str(reference_dir),
                    "--data",
                    str(val_file),
                    "--structure_reference_data",
                    str(train_file),
                    "--batch_size",
                    str(config["training"]["evaluation_batch_size"]),
                    "--pass_k",
                    str(config["training"]["pass_at_k"]),
                    "--seed",
                    "5000",
                    "--output_json",
                    str(reference_val_json),
                ],
                logs / "05c_reference_validation_after_sft_fallback.log",
                gpu_ids[0],
            ),
            repo,
        )
        reference_metrics = _read_metrics_json(reference_val_json)
        reference_gate = reference_passes(reference_metrics)

    _atomic_json(
        root / "reference_gate.json",
        {
            "initialization_mode": initialization_mode,
            "base_validation": base_metrics,
            "reference_validation": reference_metrics,
            "gate_passed": reference_gate,
        },
    )
    if not reference_gate:
        raise RuntimeError(
            "Reference policy failed the registered 15% greedy / 95% valid gate; "
            "method training is blocked."
        )

    replay_train = replay / "train_replay.jsonl"
    replay_calibration = replay / "calibration_replay.jsonl"
    replay_manifest = root / "replay_pool_manifest.json"
    _run_task(
        StageTask(
            "common_replay",
            [
                sys.executable,
                str(source),
                "build_replay",
                *model_flags,
                "--config",
                str(config_path),
                "--reference_adapter",
                str(reference_dir),
                "--input_data",
                str(train_file),
                "--train_output",
                str(replay_train),
                "--calibration_output",
                str(replay_calibration),
                "--manifest_output",
                str(replay_manifest),
                "--seed",
                str(config["dataset"]["generation_seed"] + 100),
            ],
            logs / "06_common_replay.log",
            gpu_ids[0],
        ),
        repo,
    )

    calibration_json = root / "taper_calibration.json"
    _run_task(
        StageTask(
            "calibration",
            [
                sys.executable,
                str(source),
                "calibrate",
                *model_flags,
                "--config",
                str(config_path),
                "--reference_adapter",
                str(reference_dir),
                "--calibration_replay",
                str(replay_calibration),
                "--output",
                str(calibration_json),
            ],
            logs / "07_calibration.log",
            gpu_ids[0],
        ),
        repo,
    )

    seeds = [int(seed) for seed in config["confirmation"]["paired_training_seeds"]]
    sampler_paths: dict[int, Path] = {}
    for seed in seeds:
        sampler = replay / f"sampler_seed_{seed}.jsonl"
        cmd_make_sampler(
            argparse.Namespace(
                config=str(config_path),
                replay_data=str(replay_train),
                output=str(sampler),
                seed=seed,
            )
        )
        sampler_paths[seed] = sampler

    train_tasks: list[StageTask] = []
    for method in METHODS:
        for seed in seeds:
            output = methods_root / method / str(seed)
            train_tasks.append(
                StageTask(
                    f"train_{method}_{seed}",
                    [
                        sys.executable,
                        str(source),
                        "train_method",
                        *model_flags,
                        "--config",
                        str(config_path),
                        "--reference_adapter",
                        str(reference_dir),
                        "--replay_data",
                        str(replay_train),
                        "--sampler_plan",
                        str(sampler_paths[seed]),
                        "--calibration_json",
                        str(calibration_json),
                        "--val_data",
                        str(val_file),
                        "--structure_reference_data",
                        str(train_file),
                        "--method",
                        method,
                        "--seed",
                        str(seed),
                        "--output_dir",
                        str(output),
                    ],
                    logs / f"08_train_{method}_{seed}.log",
                )
            )
    _run_group(train_tasks, gpu_ids, repo)

    # The test split is first opened here, after every method and seed finished.
    evaluation_tasks: list[StageTask] = []
    for method in METHODS:
        for seed in seeds:
            output = methods_root / method / str(seed)
            manifest = _read_metrics_json(output / "manifest.json")
            checkpoints = [("best", output / "best_adapter")]
            if manifest.get("numerical_failure"):
                checkpoints.append(("last_finite", output / "last_finite_adapter"))
            else:
                checkpoints.append(("terminal", output / "terminal_adapter"))
            for kind, adapter in checkpoints:
                result = output / f"test_metrics_{kind}.json"
                evaluation_tasks.append(
                    StageTask(
                        f"test_{method}_{seed}_{kind}",
                        [
                            sys.executable,
                            str(source),
                            "evaluate_checkpoint",
                            *model_flags,
                            "--config",
                            str(config_path),
                            "--adapter",
                            str(adapter),
                            "--data",
                            str(test_file),
                            "--structure_reference_data",
                            str(train_file),
                            "--seed",
                            str(
                                int(config["confirmation"]["paired_evaluation_seed_offset"])
                                + seed
                            ),
                            "--output",
                            str(result),
                        ],
                        logs / f"09_test_{method}_{seed}_{kind}.log",
                    )
                )
    _run_group(evaluation_tasks, gpu_ids, repo)

    summary, method_audits = _collect_summary(root, config, seeds)
    _csv_write(root / "arena_summary.csv", summary)
    all_diag_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in seeds:
            path = methods_root / method / str(seed) / "surprisal_bin_diagnostics.csv"
            with path.open(newline="") as handle:
                all_diag_rows.extend(dict(row) for row in csv.DictReader(handle))
    _csv_write(root / "surprisal_bin_diagnostics.csv", all_diag_rows)
    paired = _paired_effects(summary, seeds)
    manifests = [
        _read_metrics_json(methods_root / method / str(seed) / "manifest.json")
        for method in METHODS
        for seed in seeds
    ]
    exact_budget = all(
        int(item["completed_optimizer_updates"])
        == int(item["planned_optimizer_updates"])
        == int(config["training"]["optimizer_updates"])
        for item in manifests
    )
    terminal_audit = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "base_commit": head,
        "reference_gate_passed": reference_gate,
        "common_replay_hash": _read_metrics_json(replay_manifest)["replay_pool_hash"],
        "calibration": _read_metrics_json(calibration_json),
        "fixed_optimizer_update_budget": {
            "expected_per_run": int(config["training"]["optimizer_updates"]),
            "run_count": len(manifests),
            "exactly_matched": exact_budget,
        },
        "task_performance": {
            "summary_rows": summary,
            "paired_terminal_effects_vs_positive_only": paired,
        },
        "valid_support_entropy": {
            "per_method_seed": method_audits,
            "valid_structure_boundary_rate": float(
                config["diagnostics"]["valid_structure_boundary_rate"]
            ),
            "valid_structure_boundary_events": sum(
                bool(
                    method_audits[method][str(seed)]["valid_support_entropy"][
                        "terminal_valid_structure_boundary_event"
                    ]
                )
                for method in METHODS
                for seed in seeds
            ),
            "threshold_source": "inherited_reference_valid_rate_gate",
            "raw_valid_rate_and_teacher_forced_entropy_reported": True,
            "entropy_has_no_separate_binary_threshold": True,
        },
        "numerical": {
            "nan_inf_failures": [
                {
                    "method": item["method"],
                    "seed": item["seed"],
                    "failure": item["numerical_failure"],
                }
                for item in manifests
                if item.get("numerical_failure")
            ],
            "reported_separately_from_task_and_support": True,
        },
        "terminal_state": {
            "fixed_horizon_is_not_convergence": True,
            "best_and_terminal_both_reported": True,
            "final_window_slopes_recorded_per_run": True,
            "classification": "pilot_fixed_horizon_terminal_audit",
        },
        "test_split_accessed_only_after_all_training": True,
        "interpretation_limits": [
            "external_method_pilot_not_controlled_causal_identification",
            "no_universal_taper_winner_claim",
            "no_0p5b_to_3b_or_7b_automatic_generalization",
        ],
    }
    _atomic_json(root / "terminal_audit.json", terminal_audit)
    scientific_manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "base_commit": head,
        "result_status": "pilot",
        "model_path": str(model_path),
        "reference_adapter_hashes": _hash_tree(reference_dir),
        "config_sha256": _sha256_file(config_path),
        "data_sha256": {
            "train": _sha256_file(train_file),
            "validation": _sha256_file(val_file),
            "test": _sha256_file(test_file),
            "train_replay": _sha256_file(replay_train),
            "calibration_replay": _sha256_file(replay_calibration),
        },
        "replay_pool_hash": _read_metrics_json(replay_manifest)["replay_pool_hash"],
        "calibration_sha256": _sha256_file(calibration_json),
        "method_seed_runs": len(manifests),
        "test_access_order": "after_all_training",
        "terminal_audit": str(root / "terminal_audit.json"),
    }
    _atomic_json(root / "scientific_run_manifest.json", scientific_manifest)
    complete = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "base_commit": head,
        "status": "terminal_audited",
        "result_status": "pilot",
        "initialization_mode": initialization_mode,
        "reference_validation": reference_metrics,
        "methods": list(METHODS),
        "paired_training_seeds": seeds,
        "fixed_update_budget_complete": exact_budget,
        "test_used_only_after_all_training_finished": True,
        "summary": summary,
        "paired_effects": paired,
        "terminal_audit_present": True,
        "formal_or_universal_ranking_claim": False,
    }
    _atomic_json(root / "RUN_COMPLETE.json", complete)
    _atomic_json(
        root / "pipeline_status.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "status": "terminal_audited",
            "completed_unix": time.time(),
        },
    )
    print(json.dumps(complete, indent=2))
    return 0


def cmd_selftest(_: argparse.Namespace) -> None:
    distance = torch.tensor([0.0, 1.0, 3.0])
    assert torch.allclose(taper_weight("reciprocal_linear", distance, 2.0), torch.tensor([1.0, 1 / 3, 1 / 7]))
    assert torch.allclose(taper_weight("exponential", distance, 1.0), torch.exp(-distance))
    assert torch.allclose(taper_weight("squared_distance_exponential", distance, 1.0), torch.exp(-distance.square()))
    rows = [
        {"id": f"p{i}", "prompt": str(i), "oracle": "1+1", "oracle_structure": "s", "negatives": [{"expression": "1-1"}, {"expression": "1*1"}]}
        for i in range(7)
    ]
    plan_a = make_prompt_balanced_sampler_plan(rows, seed=9234, total_samples=29)
    plan_b = make_prompt_balanced_sampler_plan(rows, seed=9234, total_samples=29)
    assert plan_a == plan_b
    counts = Counter(item["prompt_index"] for item in plan_a)
    assert max(counts.values()) - min(counts.values()) <= 1
    coefficient, matched, error = calibrate_monotone_coefficient(
        lambda value: 10.0 / (1.0 + value),
        2.0,
        maximum=16.0,
        steps=24,
        tolerance=1e-5,
    )
    assert abs(coefficient - 4.0) < 1e-4
    assert abs(matched - 2.0) < 1e-4
    assert error < 1e-5
    print(json.dumps({"selftest": "passed", "experiment_id": EXPERIMENT_ID}))


def _common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="bf16")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    item = sub.add_parser("selftest")
    item.set_defaults(func=cmd_selftest)

    item = sub.add_parser("build_replay")
    _common_model_args(item)
    item.add_argument("--config", default=str(DEFAULT_CONFIG))
    item.add_argument("--reference_adapter", required=True)
    item.add_argument("--input_data", required=True)
    item.add_argument("--train_output", required=True)
    item.add_argument("--calibration_output", required=True)
    item.add_argument("--manifest_output", required=True)
    item.add_argument("--seed", type=int, required=True)
    item.set_defaults(func=cmd_build_replay)

    item = sub.add_parser("make_sampler")
    item.add_argument("--config", default=str(DEFAULT_CONFIG))
    item.add_argument("--replay_data", required=True)
    item.add_argument("--output", required=True)
    item.add_argument("--seed", type=int, required=True)
    item.set_defaults(func=cmd_make_sampler)

    item = sub.add_parser("calibrate")
    _common_model_args(item)
    item.add_argument("--config", default=str(DEFAULT_CONFIG))
    item.add_argument("--reference_adapter", required=True)
    item.add_argument("--calibration_replay", required=True)
    item.add_argument("--output", required=True)
    item.set_defaults(func=cmd_calibrate)

    item = sub.add_parser("train_method")
    _common_model_args(item)
    item.add_argument("--config", default=str(DEFAULT_CONFIG))
    item.add_argument("--reference_adapter", required=True)
    item.add_argument("--replay_data", required=True)
    item.add_argument("--sampler_plan", required=True)
    item.add_argument("--calibration_json", required=True)
    item.add_argument("--val_data", required=True)
    item.add_argument("--structure_reference_data", required=True)
    item.add_argument("--method", choices=METHODS, required=True)
    item.add_argument("--seed", type=int, required=True)
    item.add_argument("--output_dir", required=True)
    item.set_defaults(func=cmd_train_method)

    item = sub.add_parser("evaluate_checkpoint")
    _common_model_args(item)
    item.add_argument("--config", default=str(DEFAULT_CONFIG))
    item.add_argument("--adapter", required=True)
    item.add_argument("--data", required=True)
    item.add_argument("--structure_reference_data", required=True)
    item.add_argument("--seed", type=int, required=True)
    item.add_argument("--output", required=True)
    item.set_defaults(func=cmd_evaluate_checkpoint)

    item = sub.add_parser("run")
    item.add_argument("--model_path", required=True)
    item.add_argument("--work_dir", required=True)
    item.add_argument("--gpus", default="auto")
    item.add_argument("--config", default=str(DEFAULT_CONFIG))
    item.set_defaults(func=cmd_run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = args.func(args)
    except BaseException as exc:
        work_dir = getattr(args, "work_dir", None)
        if work_dir:
            root = Path(work_dir).resolve()
            root.mkdir(parents=True, exist_ok=True)
            failure = {
                "experiment_id": EXPERIMENT_ID,
                "version": VERSION,
                "status": "failed",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "timestamp_unix": time.time(),
            }
            _atomic_json(root / "RUN_FAILED.json", failure)
            _atomic_json(root / "pipeline_status.json", failure)
        raise
    if isinstance(result, int):
        raise SystemExit(result)


if __name__ == "__main__":
    main()
