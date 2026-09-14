"""Prepared-row normalization and deterministic split helpers for E8 multitask runs.

This module is a physical extraction from ``e8_multitask_exp_tuning``. It owns
input/data transformations only; scientific method kernels, optimizer behavior,
and scheduling remain outside this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from drpo import e8_experiment_config as experiment_config
from drpo.e8_multitask_tasks import stable_hash


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
