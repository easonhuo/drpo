"""Prepared-row normalization and deterministic split helpers for E8 multitask runs.

This module is a physical extraction from ``e8_multitask_exp_tuning``. It owns
input/data transformations only; scientific method kernels, optimizer behavior,
and scheduling remain outside this module.
"""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from drpo import e8_experiment_config as experiment_config
from drpo.e8_multitask_p0 import (
    atomic_json,
    atomic_jsonl,
    bank_path,
    model_identity,
    read_jsonl,
    sha256_file,
    stable_config_hash,
    with_smoke_overrides,
)
from drpo.e8_multitask_tasks import TaskInstance, build_adapters, stable_hash

P0_EXPERIMENT_ID = experiment_config.P0_EXPERIMENT_ID
experiment_id = experiment_config.experiment_id


def _is_coldstart(config: Mapping[str, Any]) -> bool:
    return (
        experiment_config.sweep_profile(config)
        == experiment_config.SWEEP_PROFILE_COLDSTART
    )


def _ordered_by_prompt_hash(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    seed: int,
    role: str,
) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: stable_hash(
            {
                "task": task,
                "prompt_id": str(row["prompt_id"]),
                "seed": seed,
                "role": role,
            }
        ),
    )


def _normalize_p0_row(row: Mapping[str, Any]) -> dict[str, Any]:
    negatives = [dict(item) for item in row["negatives"]]
    normalized = dict(row)
    normalized["prompt_id"] = str(row["prompt_id"])
    normalized["oracle_completion"] = str(row["oracle_completion"])
    normalized["negatives"] = [
        {
            **item,
            "negative_id": str(item["negative_id"]),
            "completion": str(item["completion"]),
        }
        for item in negatives
    ]
    return normalized


def _normalize_countdown_train_row(row: Mapping[str, Any]) -> dict[str, Any]:
    negatives = []
    source_negatives = row["negative_bank"] if "negative_bank" in row else row["negatives"]
    for index, item_value in enumerate(source_negatives):
        item = dict(item_value)
        negatives.append(
            {
                **item,
                "negative_id": f"{row['row_id']}_neg_{index:03d}",
                "completion": str(item["expression"]),
                "format_valid": bool(item.get("valid_format", True)),
                "binary_correct": bool(item.get("correct", False)),
                "error_class": str(item.get("negative_bin", item.get("source", "wrong_answer"))),
            }
        )
    return {
        "schema_version": 1,
        "task": "countdown",
        "prompt_id": str(row["row_id"]),
        "source_prompt_id": str(row.get("source_prompt_id", row["row_id"])),
        "prompt": str(row["prompt"]),
        "oracle_completion": str(row["oracle_positive"]),
        "metadata": {
            "numbers": [int(value) for value in row["numbers"]],
            "target": int(row["target"]),
        },
        "negatives": negatives,
        "source_schema": "countdown_oracle_offline_bank_v2",
    }


def _normalize_countdown_validation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    prompt_id = str(row.get("id", row.get("row_id", row.get("source_prompt_id", ""))))
    if not prompt_id:
        raise RuntimeError("Countdown validation row has no stable ID")
    oracle = row.get("oracle", row.get("oracle_positive"))
    if oracle is None:
        raise RuntimeError(f"Countdown validation row {prompt_id} has no oracle")
    return {
        "schema_version": 1,
        "task": "countdown",
        "prompt_id": prompt_id,
        "prompt": str(row["prompt"]),
        "oracle_completion": str(oracle),
        "metadata": {
            "numbers": [int(value) for value in row["numbers"]],
            "target": int(row["target"]),
        },
        "source_schema": "countdown_structural_validation",
    }


def _audit_training_rows(task: str, rows: Sequence[Mapping[str, Any]], expected: int) -> None:
    if len(rows) != expected:
        raise RuntimeError(f"{task} expected {expected} training rows, found {len(rows)}")
    prompt_ids = [str(row["prompt_id"]) for row in rows]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise RuntimeError(f"{task} has duplicate prompt IDs")
    for row in rows:
        negatives = list(row.get("negatives", ()))
        if len(negatives) != 16:
            raise RuntimeError(
                f"{task}/{row['prompt_id']} must have exactly 16 negatives, found {len(negatives)}"
            )
        completions = [str(item["completion"]) for item in negatives]
        if task != "countdown" and len(set(completions)) != 16:
            raise RuntimeError(f"{task}/{row['prompt_id']} has duplicate negative completions")
        if any(bool(item.get("binary_correct", item.get("correct", False))) for item in negatives):
            raise RuntimeError(f"{task}/{row['prompt_id']} contains a verifier-correct negative")


def _audit_partition_prompt_ids(
    task: str,
    partitions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    seen: dict[str, str] = {}
    for partition, rows in partitions.items():
        prompt_ids = [str(row["prompt_id"]) for row in rows]
        if len(set(prompt_ids)) != len(prompt_ids):
            raise RuntimeError(f"{task} has duplicate prompt IDs within {partition}")
        for prompt_id in prompt_ids:
            previous = seen.get(prompt_id)
            if previous is not None:
                raise RuntimeError(
                    f"{task} prompt ID {prompt_id} overlaps {previous} and {partition}"
                )
            seen[prompt_id] = partition


def split_p0_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    config: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    split = config["split"]
    required = (
        int(split["p0_train_rows"]) + int(split["p0_validation_rows"]) + int(split["p0_test_rows"])
    )
    if len(rows) != required:
        raise RuntimeError(f"{task} P0 bank must contain exactly {required} rows")
    ordered = _ordered_by_prompt_hash(
        [_normalize_p0_row(row) for row in rows],
        task=task,
        seed=int(split["hash_seed"]),
        role="p0_tuning_split",
    )
    train_end = int(split["p0_train_rows"])
    validation_end = train_end + int(split["p0_validation_rows"])
    partitions = {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }
    _audit_training_rows(task, partitions["train"], int(split["p0_train_rows"]))
    _audit_partition_prompt_ids(task, partitions)
    return partitions


def split_countdown_rows(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    split = config["split"]
    normalized_train = [_normalize_countdown_train_row(row) for row in train_rows]
    normalized_validation = [_normalize_countdown_validation_row(row) for row in validation_rows]
    if _is_coldstart(config):
        if not bool(split.get("countdown_subsampling_forbidden", False)):
            raise RuntimeError("Paper Countdown forbids wrapper-level train subsampling")
        train = normalized_train
        validation = normalized_validation
    else:
        train = _ordered_by_prompt_hash(
            normalized_train,
            task="countdown",
            seed=int(split["hash_seed"]),
            role="countdown_train_select",
        )[: int(split["countdown_train_rows"])]
        validation = _ordered_by_prompt_hash(
            normalized_validation,
            task="countdown",
            seed=int(split["hash_seed"]),
            role="countdown_validation_select",
        )[: int(split["countdown_validation_rows"])]
    _audit_training_rows("countdown", train, int(split["countdown_train_rows"]))
    if len(validation) != int(split["countdown_validation_rows"]):
        raise RuntimeError("Countdown validation file does not contain the exact frozen rows")
    partitions = {"train": train, "validation": validation}
    _audit_partition_prompt_ids("countdown", partitions)
    return partitions


def _canonical_train_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Translate task schema while preserving the paper all-unique-negative loss."""

    task = str(row["task"])
    prompt_id = str(row["prompt_id"])
    negatives = list(row["negatives"])
    if len(negatives) != 16:
        raise RuntimeError(f"{task}/{prompt_id} canonical conversion requires 16 negatives")
    bank = [
        {
            **dict(item),
            "expression": str(item["completion"]),
        }
        for item in negatives
    ]
    oracle = str(row["oracle_completion"])
    if any(item["expression"] == oracle for item in bank):
        raise RuntimeError(f"{task}/{prompt_id} negative completion matches the positive")
    return {
        **dict(row),
        "id": prompt_id,
        "oracle": oracle,
        "positive": oracle,
        "negative_bank": bank,
        "negative_bank_size": 16,
        "pair_matched": True,
        # The old core uses this only for balanced diagnostics.  Task correctness
        # is supplied by the environment verifier, not Countdown expression parsing.
        "oracle_structure": f"{task}:task_verifier",
        "canonical_training_core": "countdown_e8_alpha1_c_scan.ContinuousUniqueBankDataset",
        "canonical_negative_consumer": "all_unique_negatives_per_prompt",
    }


def _canonical_validation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    task = str(row["task"])
    prompt_id = str(row["prompt_id"])
    oracle = str(row["oracle_completion"])
    return {
        **dict(row),
        "id": prompt_id,
        "oracle": oracle,
        "oracle_structure": f"{task}:task_verifier",
    }


@dataclass(frozen=True)
class TaskInputs:
    task: str
    bank: Path
    reference_adapter: Path | None
    sources_root: Path
    p0_config: Path
    countdown_validation: Path | None = None


def _evenly_spaced_rank_indices(candidate_count: int, selected_count: int = 16) -> tuple[int, ...]:
    if selected_count < 2:
        raise ValueError("Reference-remoteness selection requires at least two selected ranks")
    if candidate_count < selected_count:
        raise ValueError(
            f"Reference-remoteness selection requires >= {selected_count} candidates; "
            f"found {candidate_count}"
        )
    indices = tuple(
        (index * (candidate_count - 1)) // (selected_count - 1) for index in range(selected_count)
    )
    if len(set(indices)) != selected_count or indices[0] != 0 or indices[-1] != candidate_count - 1:
        raise AssertionError("Even rank selection must be unique and include both extremes")
    return indices


def _coverage_first_reference_rank_indices(
    scored: Sequence[Mapping[str, Any]],
    source_negatives: Sequence[Mapping[str, Any]],
    selected_count: int = 16,
) -> tuple[int, ...]:
    if len(scored) < selected_count:
        raise RuntimeError(f"Coverage-first selection needs >= {selected_count} candidates")
    buckets: dict[str, list[int]] = {}
    for rank, item in enumerate(scored):
        buckets.setdefault(str(item["error_class"]), []).append(rank)
    class_order = [str(item["error_class"]) for item in source_negatives]
    queues = {}
    for name in sorted(set(class_order)):
        quota, ranks = class_order.count(name), buckets[name]
        if quota == 1:
            original = next(item for item in source_negatives if str(item["error_class"]) == name)
            canonical = str(original.get("canonical_completion", original["completion"]))
            local = tuple(
                index
                for index, rank in enumerate(ranks)
                if str(scored[rank]["canonical_completion"]) == canonical
            )
            if len(local) != 1:
                raise RuntimeError(f"Source P0 singleton not uniquely reconstructed for {name}")
        else:
            local = _evenly_spaced_rank_indices(len(ranks), quota)
        queues[name] = iter([ranks[index] for index in local])
    selected = tuple(next(queues[name]) for name in class_order)
    if len(set(selected)) != selected_count:
        raise RuntimeError("Coverage-first selector produced duplicate negatives")
    return selected


def _reference_surprisal_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray([float(value) for value in values], dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise RuntimeError("Reference-surprisal audit requires finite non-empty values")
    q25, median, q75 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "min": float(array.min()),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "max": float(array.max()),
        "range": float(array.max() - array.min()),
        "iqr": float(q75 - q25),
    }


def _reference_error_class_audit(
    scored: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    source_negatives: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_classes = [str(item["error_class"]) for item in source_negatives]
    selected_classes = [str(item["error_class"]) for item in selected]
    if selected_classes != source_classes:
        raise RuntimeError("Coverage-first selector changed the July-29 P0 error-class sequence")
    candidate_buckets: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    selected_buckets: dict[str, list[Mapping[str, Any]]] = {}
    for rank, item in enumerate(scored):
        candidate_buckets.setdefault(str(item["error_class"]), []).append((rank, item))
    for item in selected:
        selected_buckets.setdefault(str(item["error_class"]), []).append(item)
    class_audit: dict[str, Any] = {}
    endpoint_total = 0
    endpoint_covered = 0
    for error_class, bucket in sorted(candidate_buckets.items()):
        chosen = selected_buckets.get(error_class, [])
        candidate_ranks = [rank for rank, _ in bucket]
        selected_ranks = [int(item["reference_rank"]) for item in chosen]
        endpoint_ok: bool | None = None
        if len(chosen) >= 2:
            endpoint_total += 1
            endpoint_ok = (
                candidate_ranks[0] in selected_ranks and candidate_ranks[-1] in selected_ranks
            )
            if not endpoint_ok:
                raise RuntimeError(
                    f"Coverage-first selector missed a class-local endpoint: {error_class}"
                )
            endpoint_covered += 1
        class_audit[error_class] = {
            "candidate_count": len(bucket),
            "source_p0_count": source_classes.count(error_class),
            "selected_count": len(chosen),
            "candidate_global_rank_min": candidate_ranks[0],
            "candidate_global_rank_max": candidate_ranks[-1],
            "selected_global_ranks": selected_ranks,
            "candidate_reference_surprisal": _reference_surprisal_summary(
                [float(item["reference_surprisal"]) for _, item in bucket]
            ),
            "selected_reference_surprisal": (
                _reference_surprisal_summary(
                    [float(item["reference_surprisal"]) for item in chosen]
                )
                if chosen
                else None
            ),
            "near_far_endpoint_coverage": endpoint_ok,
        }
    selected_class_count = len(selected_buckets)
    candidate_class_count = len(candidate_buckets)
    selected_ranks = [int(item["reference_rank"]) for item in selected]
    return {
        "source_p0_error_class_sequence": source_classes,
        "selected_error_class_sequence": selected_classes,
        "coverage_sequence_matches_source_p0": True,
        "candidate_error_class_counts": {
            name: len(bucket) for name, bucket in sorted(candidate_buckets.items())
        },
        "source_p0_error_class_counts": {
            name: source_classes.count(name) for name in sorted(set(source_classes))
        },
        "selected_error_class_counts": {
            name: len(bucket) for name, bucket in sorted(selected_buckets.items())
        },
        "candidate_distinct_error_class_count": candidate_class_count,
        "selected_distinct_error_class_count": selected_class_count,
        "error_class_coverage_fraction": selected_class_count / candidate_class_count,
        "singleton_selected_error_class_count": sum(
            len(bucket) == 1 for bucket in selected_buckets.values()
        ),
        "multi_slot_selected_error_class_count": endpoint_total,
        "multi_slot_endpoint_coverage_count": endpoint_covered,
        "global_reference_rank_span_fraction": (
            (max(selected_ranks) - min(selected_ranks)) / (len(scored) - 1)
            if len(scored) > 1
            else 0.0
        ),
        "error_class_reference_surprisal": class_audit,
    }


def _verified_wrong_candidates(
    adapter: Any,
    instance: TaskInstance,
    source_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    generation_seed = int(source_row["generation_seed"])
    rng = random.Random(
        int(
            stable_hash(
                {
                    "task": str(source_row["task"]),
                    "prompt_id": str(source_row["prompt_id"]),
                    "seed": generation_seed,
                }
            )[:16],
            16,
        )
    )
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for mutation in adapter.mutation_candidates(instance, rng):
        result = adapter.verify(
            instance,
            mutation.completion,
            mutation_class=mutation.mutation_class,
        )
        canonical = str(result.canonical_completion)
        if canonical in seen or not adapter.accept_negative(result):
            continue
        seen.add(canonical)
        candidates.append(
            {
                "completion": str(mutation.completion),
                "canonical_completion": canonical,
                "verifier_score": float(result.score),
                "binary_correct": bool(result.correct),
                "format_valid": bool(result.format_valid),
                "error_class": str(result.error_class),
                "verification_details": dict(result.details),
            }
        )
    original = {
        str(item.get("canonical_completion", item["completion"]))
        for item in source_row["negatives"]
    }
    missing = sorted(original - seen)
    if missing:
        raise RuntimeError(
            f"{source_row['task']}/{source_row['prompt_id']} cannot reconstruct the original "
            f"P0 negative universe: {missing[:3]}"
        )
    if len(candidates) < 16:
        raise RuntimeError(
            f"{source_row['task']}/{source_row['prompt_id']} has only {len(candidates)} "
            "deterministic verified wrong candidates"
        )
    return candidates


def resolve_task_inputs(
    config: Mapping[str, Any],
    *,
    p0_work_dir: Path,
    p0_config: Path,
    countdown_bank: Path,
    countdown_validation: Path,
    countdown_adapter: Path | None,
) -> dict[str, TaskInputs]:
    p0_config_value = yaml.safe_load(p0_config.read_text(encoding="utf-8"))
    if (
        not isinstance(p0_config_value, dict)
        or p0_config_value.get("experiment_id") != P0_EXPERIMENT_ID
    ):
        raise RuntimeError("P0 config identity mismatch")
    qualification_path = p0_work_dir / "qualification_audit.json"
    if not qualification_path.is_file():
        raise FileNotFoundError(f"Missing P0 qualification audit: {qualification_path}")
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    if (
        qualification.get("experiment_id") != P0_EXPERIMENT_ID
        or qualification.get("config_hash")
        != stable_config_hash(
            with_smoke_overrides(
                p0_config_value,
                rows=None,
                negatives=None,
            )
        )
        or not qualification.get("passed")
    ):
        raise RuntimeError("P0 bank qualification identity or pass status mismatch")

    result: dict[str, TaskInputs] = {}
    for task_value in config["suite"]["p0_tasks"]:
        task = str(task_value)
        if not qualification.get("tasks", {}).get(task, {}).get("passed", False):
            raise RuntimeError(f"P0 bank is not qualified for {task}")
        result[task] = TaskInputs(
            task=task,
            bank=bank_path(p0_work_dir, task).resolve(),
            reference_adapter=None,
            sources_root=(p0_work_dir / "sources").resolve(),
            p0_config=p0_config.resolve(),
        )
    result["countdown"] = TaskInputs(
        task="countdown",
        bank=countdown_bank.resolve(),
        reference_adapter=(countdown_adapter.resolve() if countdown_adapter is not None else None),
        sources_root=(p0_work_dir / "sources").resolve(),
        p0_config=p0_config.resolve(),
        countdown_validation=countdown_validation.resolve(),
    )
    for task, inputs in result.items():
        if not inputs.bank.is_file():
            raise FileNotFoundError(f"Missing bank for {task}: {inputs.bank}")
        if task == "countdown":
            if inputs.countdown_validation is None or not inputs.countdown_validation.is_file():
                raise FileNotFoundError("Missing Countdown validation file")
            if not _is_coldstart(config) and (
                inputs.reference_adapter is None
                or not (inputs.reference_adapter / "adapter_config.json").is_file()
            ):
                raise FileNotFoundError("Missing supplied Countdown reference adapter")
        if not inputs.sources_root.is_dir():
            raise FileNotFoundError(f"Missing sources root for {task}: {inputs.sources_root}")
    if set(result) != set(config["suite"]["tasks"]):
        raise AssertionError("Resolved inputs do not match the configured suite")
    return result


def write_split_manifest(
    task_inputs: Mapping[str, TaskInputs],
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    task_records: dict[str, Any] = {}
    for task_value in config["suite"]["tasks"]:
        task = str(task_value)
        inputs = task_inputs[task]
        if task == "countdown":
            if inputs.countdown_validation is None:
                raise AssertionError("Countdown validation path missing")
            partitions = split_countdown_rows(
                read_jsonl(inputs.bank),
                read_jsonl(inputs.countdown_validation),
                config=config,
            )
        else:
            partitions = split_p0_rows(read_jsonl(inputs.bank), task=task, config=config)
        task_root = output_root / "splits" / task
        task_root.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        prompt_hashes: dict[str, str] = {}
        counts: dict[str, int] = {}
        for name, values in partitions.items():
            path = task_root / f"{name}.jsonl"
            atomic_jsonl(path, values)
            paths[name] = str(path.resolve())
            counts[name] = len(values)
            prompt_hashes[name] = stable_hash(sorted(str(row["prompt_id"]) for row in values))
        task_records[task] = {
            "bank": str(inputs.bank),
            "bank_sha256": sha256_file(inputs.bank),
            "reference_adapter": (
                str(inputs.reference_adapter) if inputs.reference_adapter is not None else None
            ),
            "reference_adapter_identity": (
                model_identity("unresolved_backbone", str(inputs.reference_adapter))["adapter"]
                if inputs.reference_adapter is not None
                else None
            ),
            "sources_root": str(inputs.sources_root),
            "p0_config": str(inputs.p0_config),
            "p0_config_sha256": sha256_file(inputs.p0_config),
            "countdown_validation_source": (
                str(inputs.countdown_validation) if inputs.countdown_validation else None
            ),
            "countdown_validation_sha256": (
                sha256_file(inputs.countdown_validation) if inputs.countdown_validation else None
            ),
            "counts": counts,
            "prompt_id_hashes": prompt_hashes,
            "paths": paths,
        }
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": stable_config_hash(config),
        "test_access_allowed": False,
        "tasks": task_records,
        "complete": len(task_records) == len(config["suite"]["tasks"]),
        "scientific_status": "not_run",
    }
    atomic_json(output_root / "split_manifest.json", manifest)
    return manifest


def _leaf_values(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, item in value.items():
        child = f"{prefix}.{key}" if prefix else str(key)
        result.update(_leaf_values(item, child))
    return result


def _changed_leaf_paths(original: Mapping[str, Any], derived: Mapping[str, Any]) -> list[str]:
    left = _leaf_values(original)
    right = _leaf_values(derived)
    return sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))


def _atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(dict(value), sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _task_base_config(
    config: Mapping[str, Any],
    *,
    task: str,
    canonical_paths: Mapping[str, Path],
    task_root: Path,
) -> tuple[Path, list[str]]:
    """Materialize effective base runtime without editing the canonical source."""

    base_path = canonical_paths["base_config"]
    original = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(original, dict):
        raise TypeError("Paper base config root must be a mapping")
    historical = experiment_config.is_historical_coldstart_config(config)
    if historical and task == "countdown":
        return base_path, []

    derived = copy.deepcopy(original)
    effective = experiment_config.effective_coldstart_runtime(config, task)
    runtime = config["task_runtime"][task]
    if historical:
        # Preserve the exact wrapper behavior of the three closed historical IDs.
        derived["model"]["max_length"] = int(runtime["max_length"])
        derived["model"]["max_new_tokens"] = int(runtime["max_new_tokens"])
        derived["evaluation"]["batch_size"] = int(runtime["evaluation_batch_size"])
        derived["evaluation"]["pass_ks"] = [8] + [
            int(value) for value in runtime["auxiliary_pass_ks"]
        ]
    else:
        model = effective["model"]
        training = effective["training"]
        evaluation = effective["evaluation"]
        derived["model"].update(
            {
                "max_length": int(model["max_length"]),
                "max_new_tokens": int(model["max_new_tokens"]),
                "dtype": str(model["dtype"]),
                "lora_rank": int(model["lora_rank"]),
                "lora_alpha": int(model["lora_alpha"]),
                "lora_dropout": float(model["lora_dropout"]),
                "gradient_checkpointing": bool(model["gradient_checkpointing"]),
            }
        )
        derived["offline_training"].update(
            {
                "seed": int(effective["initialization_seed"]),
                "steps": int(training["optimizer_updates"]),
                "micro_batch": int(training["micro_batch"]),
                "gradient_accumulation": int(training["gradient_accumulation"]),
                "learning_rate": float(training["learning_rate"]),
                "weight_decay": float(training["weight_decay"]),
                "warmup_ratio": float(training["warmup_ratio"]),
                "maximum_gradient_norm": float(training["max_grad_norm"]),
                "eval_every": int(training["evaluation_every_updates"]),
            }
        )
        derived["evaluation"].update(
            {
                "examples": int(evaluation["examples"]),
                "batch_size": int(evaluation["batch_size"]),
                "pass_ks": [int(value) for value in evaluation["pass_ks"]],
                "seed": int(evaluation["generation_seed"]),
                "sampling_temperature": float(evaluation["sampling_temperature"]),
                "top_p": float(evaluation["top_p"]),
                "greedy_prompt_rows": int(evaluation["greedy_prompt_rows"]),
                "passk_prompt_rows": int(evaluation["passk_prompt_rows"]),
            }
        )
    path = task_root / "paper_base_task_interface.yaml"
    _atomic_yaml(path, derived)
    return path, _changed_leaf_paths(original, derived)


def _task_grid_configs(
    config: Mapping[str, Any],
    *,
    canonical_paths: Mapping[str, Path],
    task_root: Path,
) -> dict[str, dict[str, Any]]:
    """Return historical grids unchanged or generic derived runtime-grid copies."""

    result: dict[str, dict[str, Any]] = {}
    historical = experiment_config.is_historical_coldstart_config(config)
    training = config["training"]
    for name in ("round1_grid", "extension_grid"):
        source = canonical_paths[name]
        original = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(original, dict):
            raise TypeError(f"Paper grid root must be a mapping: {source}")
        if historical:
            runtime_path = source
            changed: list[str] = []
        else:
            derived = copy.deepcopy(original)
            derived["training"]["steps"] = int(training["optimizer_updates"])
            derived["training"]["eval_every"] = int(training["evaluation_every_updates"])
            runtime_path = task_root / f"paper_{name}_runtime.yaml"
            _atomic_yaml(runtime_path, derived)
            changed = _changed_leaf_paths(original, derived)
        result[name] = {
            "path": runtime_path,
            "source": source,
            "changed_fields": changed,
        }
    return result


def _load_task_adapter_and_instances(
    task: str,
    *,
    inputs: TaskInputs,
    validation_rows: Sequence[Mapping[str, Any]],
) -> tuple[Any, dict[str, TaskInstance]]:
    p0_config = yaml.safe_load(inputs.p0_config.read_text(encoding="utf-8"))
    if not isinstance(p0_config, dict):
        raise TypeError("P0 config root is not a mapping")
    adapter_config = copy.deepcopy(p0_config)
    adapter_config["tasks"]["names"] = [task]
    adapter = build_adapters(adapter_config, inputs.sources_root)[task]
    if task == "countdown":
        instances = {
            str(row["prompt_id"]): TaskInstance(
                task="countdown",
                prompt_id=str(row["prompt_id"]),
                prompt=str(row["prompt"]),
                oracle_completion=str(row["oracle_completion"]),
                metadata=dict(row["metadata"]),
                source_entry={},
            )
            for row in validation_rows
        }
        return adapter, instances

    seeds = {int(row["generation_seed"]) for row in validation_rows}
    if len(seeds) != 1:
        raise RuntimeError(f"{task} validation rows do not share one generation seed")
    required_ids = {str(row["prompt_id"]) for row in validation_rows}
    candidate_count = int(p0_config["bank"]["candidate_rows_per_task"])
    instances: dict[str, TaskInstance] = {}
    for instance in adapter.generate_instances(candidate_count, seeds.pop()):
        if instance.prompt_id in required_ids:
            instances[instance.prompt_id] = instance
            if len(instances) == len(required_ids):
                break
    missing = sorted(required_ids - set(instances))
    if missing:
        raise RuntimeError(f"Could not reconstruct {task} validation instances: {missing[:5]}")
    return adapter, instances

def materialize_reference_remoteness_task(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    task: str,
    record: Mapping[str, Any],
    inputs: TaskInputs,
    identity_hash: str,
    score_candidates: Callable[
        [str, Sequence[Mapping[str, Any]], int, int],
        Sequence[float],
    ],
) -> dict[str, Any]:
    """Materialize one fixed reference-remoteness training bank.

    Model/runtime ownership stays with the caller. This helper owns only the
    deterministic candidate selection, audit, and input-artifact materialization.
    """

    train_rows = read_jsonl(Path(record["paths"]["train"]))
    adapter, instances = _load_task_adapter_and_instances(
        task,
        inputs=inputs,
        validation_rows=train_rows,
    )
    derived_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    runtime = config["task_runtime"][task]
    for source_row in train_rows:
        prompt_id = str(source_row["prompt_id"])
        candidates = _verified_wrong_candidates(
            adapter,
            instances[prompt_id],
            source_row,
        )
        scores = list(
            score_candidates(
                str(source_row["prompt"]),
                candidates,
                int(runtime["max_length"]),
                int(runtime["evaluation_batch_size"]),
            )
        )
        scored = [
            {**candidate, "reference_surprisal": float(score)}
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        scored.sort(
            key=lambda item: (
                float(item["reference_surprisal"]),
                stable_hash(
                    {
                        "task": task,
                        "prompt_id": prompt_id,
                        "canonical_completion": item["canonical_completion"],
                    }
                ),
            )
        )
        selected_indices = _coverage_first_reference_rank_indices(
            scored, source_row["negatives"], 16
        )
        selected: list[dict[str, Any]] = []
        for slot, rank in enumerate(selected_indices):
            item = dict(scored[rank])
            item.update(
                {
                    "negative_id": f"{prompt_id}_refrem_{slot:03d}",
                    "reference_rank": int(rank),
                    "reference_candidate_count": len(scored),
                    "reference_rank_role": "provenance_and_diagnostic_only",
                }
            )
            selected.append(item)
        coverage_audit = _reference_error_class_audit(
            scored, selected, list(source_row["negatives"])
        )
        derived = dict(source_row)
        derived["negatives"] = selected
        derived["reference_remoteness_selection"] = {
            "identity_hash": identity_hash,
            "reference_policy": "zero_update_base_plus_fresh_lora",
            "coordinate": "mean_completion_token_surprisal",
            "candidate_count": len(scored),
            "selected_ranks": list(selected_indices),
            "training_weight_uses_reference_rank": False,
            "current_policy_surprisal_recomputed_each_update": True,
        }
        derived_rows.append(derived)
        audit_rows.append(
            {
                "task": task,
                "prompt_id": prompt_id,
                "candidate_count": len(scored),
                "selected_ranks": list(selected_indices),
                "candidate_reference_surprisal": _reference_surprisal_summary(
                    [float(item["reference_surprisal"]) for item in scored]
                ),
                "selected_reference_surprisal": _reference_surprisal_summary(
                    [float(item["reference_surprisal"]) for item in selected]
                ),
                **coverage_audit,
                "coverage_threshold": None,
                "coverage_gate": False,
            }
        )

    root = output_root / "reference_remoteness" / task
    bank_path_value = root / "train.jsonl"
    audit_path = root / "prompt_audit.jsonl"
    atomic_jsonl(bank_path_value, derived_rows)
    atomic_jsonl(audit_path, audit_rows)
    ranges = [
        float(row["selected_reference_surprisal"]["range"])
        for row in audit_rows
    ]
    class_rows = [
        value
        for row in audit_rows
        for value in row["error_class_reference_surprisal"].values()
    ]
    selected_class_rows = [
        value for value in class_rows if int(value["selected_count"]) > 0
    ]
    endpoint_total = sum(
        int(row["multi_slot_selected_error_class_count"]) for row in audit_rows
    )
    endpoint_covered = sum(
        int(row["multi_slot_endpoint_coverage_count"]) for row in audit_rows
    )
    summary = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": stable_config_hash(config),
        "task": task,
        "identity_hash": identity_hash,
        "source_train": record["paths"]["train"],
        "source_train_sha256": sha256_file(Path(record["paths"]["train"])),
        "source_p0_bank_preserved": True,
        "path": str(bank_path_value.resolve()),
        "sha256": sha256_file(bank_path_value),
        "rows": len(derived_rows),
        "selected_negatives_per_prompt": 16,
        "selection": "source_p0_error_class_sequence_then_within_class_reference_rank_spread",
        "candidate_pool": "all_deterministic_verified_wrong_mutations",
        "reference_rank_enters_training_weight": False,
        "current_policy_surprisal_recomputed_each_update": True,
        "coverage_threshold": None,
        "coverage_sequence_matches_all_prompts": all(
            bool(row["coverage_sequence_matches_source_p0"]) for row in audit_rows
        ),
        "error_class_coverage_fraction": _reference_surprisal_summary(
            [float(row["error_class_coverage_fraction"]) for row in audit_rows]
        ),
        "singleton_selected_error_class_instances": sum(
            int(row["singleton_selected_error_class_count"]) for row in audit_rows
        ),
        "multi_slot_selected_error_class_instances": endpoint_total,
        "multi_slot_endpoint_coverage_count": endpoint_covered,
        "multi_slot_endpoint_coverage_fraction": (
            endpoint_covered / endpoint_total if endpoint_total else None
        ),
        "global_reference_rank_span_fraction": _reference_surprisal_summary(
            [float(row["global_reference_rank_span_fraction"]) for row in audit_rows]
        ),
        "candidate_within_class_range": _reference_surprisal_summary(
            [float(row["candidate_reference_surprisal"]["range"]) for row in class_rows]
        ),
        "candidate_within_class_iqr": _reference_surprisal_summary(
            [float(row["candidate_reference_surprisal"]["iqr"]) for row in class_rows]
        ),
        "selected_within_class_range": _reference_surprisal_summary(
            [
                float(row["selected_reference_surprisal"]["range"])
                for row in selected_class_rows
            ]
        ),
        "selected_within_class_iqr": _reference_surprisal_summary(
            [
                float(row["selected_reference_surprisal"]["iqr"])
                for row in selected_class_rows
            ]
        ),
        "selected_range_median": float(np.median(np.asarray(ranges, dtype=float))),
        "prompt_audit": str(audit_path.resolve()),
        "prompt_audit_sha256": sha256_file(audit_path),
        "complete": len(derived_rows) == int(config["split"]["p0_train_rows"]),
        "scientific_status": "not_run",
    }
    if not summary["complete"]:
        raise RuntimeError(f"Reference-remoteness bank is incomplete for {task}")
    atomic_json(root / "summary.json", summary)
    return summary


def write_canonical_cold_inputs(
    config: Mapping[str, Any],
    output_root: Path,
    split_manifest: dict[str, Any],
    *,
    audit_canonical_sources: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    canonical_paths_for_config: Callable[[Mapping[str, Any]], Mapping[str, Path]],
) -> dict[str, Any]:
    """Write canonical cold-start input adapters without owning trainer logic."""

    if not _is_coldstart(config):
        raise RuntimeError("Canonical input conversion is cold-profile only")
    source_audit = dict(audit_canonical_sources(config))
    canonical_paths = canonical_paths_for_config(config)
    records: dict[str, Any] = {}
    for task_value in config["suite"]["tasks"]:
        task = str(task_value)
        record = split_manifest["tasks"][task]
        source_paths = record["paths"]
        task_root = output_root / "canonical_inputs" / task
        task_root.mkdir(parents=True, exist_ok=True)
        if task == "countdown":
            train_path = Path(str(record["bank"])).resolve()
            validation_path = Path(str(record["countdown_validation_source"])).resolve()
            train_rows = read_jsonl(train_path)
            validation_rows = read_jsonl(validation_path)
            exact_countdown_sources = True
            reference_selection_applied = False
            reference_selection_identity = None
        else:
            reference_record = record.get("reference_remoteness_bank")
            if isinstance(reference_record, Mapping):
                reference_path = Path(str(reference_record.get("path", "")))
                if (
                    not reference_path.is_file()
                    or sha256_file(reference_path) != reference_record.get("sha256")
                    or not reference_record.get("complete")
                ):
                    raise RuntimeError(
                        f"Derived reference-remoteness bank identity failed for {task}"
                    )
                train_source = reference_path
                reference_selection_applied = True
                reference_selection_identity = str(reference_record["identity_hash"])
            else:
                train_source = Path(source_paths["train"])
                reference_selection_applied = False
                reference_selection_identity = None
            train_rows = [
                _canonical_train_row(row) for row in read_jsonl(train_source)
            ]
            validation_rows = [
                _canonical_validation_row(row)
                for row in read_jsonl(Path(source_paths["validation"]))
            ]
            train_path = task_root / "train.jsonl"
            validation_path = task_root / "validation.jsonl"
            atomic_jsonl(train_path, train_rows)
            atomic_jsonl(validation_path, validation_rows)
            exact_countdown_sources = False
        sealed_test_path = task_root / "SEALED_TEST_NOT_ACCESSED.jsonl"
        sealed_test_path.parent.mkdir(parents=True, exist_ok=True)
        sealed_test_path.write_text("", encoding="utf-8")
        task_base_config, changed_fields = _task_base_config(
            config,
            task=task,
            canonical_paths=canonical_paths,
            task_root=task_root,
        )
        runtime_grids = _task_grid_configs(
            config,
            canonical_paths=canonical_paths,
            task_root=task_root,
        )
        canonical_record = {
            "train": str(train_path.resolve()),
            "validation": str(validation_path.resolve()),
            "sealed_test": str(sealed_test_path.resolve()),
            "base_config": str(task_base_config.resolve()),
            "base_config_sha256": sha256_file(task_base_config),
            "round1_grid": str(runtime_grids["round1_grid"]["path"].resolve()),
            "round1_grid_sha256": sha256_file(runtime_grids["round1_grid"]["path"]),
            "extension_grid": str(runtime_grids["extension_grid"]["path"].resolve()),
            "extension_grid_sha256": sha256_file(runtime_grids["extension_grid"]["path"]),
            "task_interface_changed_fields": changed_fields,
            "countdown_exact_source_files": exact_countdown_sources,
            "reference_remoteness_bank_applied": reference_selection_applied,
            "reference_remoteness_bank_identity_hash": reference_selection_identity,
            "negative_consumer": "all_unique_negatives_per_prompt",
            "calibration": "forbidden",
            "train_sha256": sha256_file(train_path),
            "validation_sha256": sha256_file(validation_path),
            "sealed_test_sha256": sha256_file(sealed_test_path),
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "test_rows": 0,
        }
        if not experiment_config.is_historical_coldstart_config(config):
            canonical_record["effective_runtime"] = (
                experiment_config.effective_coldstart_runtime(config, task)
            )
            canonical_record["runtime_grid_sources"] = {
                name: {
                    "source": str(runtime_grids[name]["source"].resolve()),
                    "source_sha256": sha256_file(runtime_grids[name]["source"]),
                    "changed_fields": list(runtime_grids[name]["changed_fields"]),
                }
                for name in ("round1_grid", "extension_grid")
            }
        record["canonical_coldstart"] = canonical_record
        records[task] = canonical_record

    split_manifest["canonical_source_audit"] = source_audit
    split_manifest["canonical_coldstart_complete"] = True
    atomic_json(output_root / "split_manifest.json", split_manifest)
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id(config),
        "config_hash": stable_config_hash(config),
        "source_audit": source_audit,
        "tasks": records,
        "test_partition_accessed": False,
        "complete": set(records) == set(config["suite"]["tasks"]),
    }
    atomic_json(output_root / "canonical_inputs" / "manifest.json", manifest)
    return manifest

