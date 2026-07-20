"""Reviewer-facing end-to-end runtime for the Countdown external-validity task.

The mathematical objective, verifier, remoteness coordinate, taper weights, and
model-backed gradient calibration live in :mod:`drpo_reference.categorical.countdown`.
This module adds the selected-model runtime around that stable core: explicit JSON
configuration, lazy Transformers/PEFT loading, replay/calibration validation,
paired prompt-balanced training, finite-state guards, best/terminal checkpoints,
generation evaluation, and simple seed aggregation.

No scientific coordinate is silently selected here. Model identity, LoRA settings,
methods, coefficients inherited by calibration, seeds, update budget, checkpoint
cadence, and evaluation policy must all be supplied in the JSON configuration.
Every output remains reviewer-run evidence with ``formal_result_claim=False``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import traceback
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drpo_reference.categorical.countdown import (
    COUNTDOWN_ACTIVE_TAIL_METHODS,
    COUNTDOWN_ACTIVE_TAIL_TAU_RULE,
    CountdownTrainingItem,
    active_tail_objective_from_model,
    calibrate_active_tail_model,
    calibration_surprisal_scale,
    chat_prompt,
    collate_countdown_training_items,
    clean_expression,
    completion_stats,
    encode_countdown_training_row,
    evaluate_response_batches,
    make_prompt_balanced_sampler_plan,
    move_tensor_batch_to_device,
    parameter_update_norm,
    resolve_active_tail_tau,
    unique_negative_expressions,
    verify_expression,
)
from drpo_reference.common.io import atomic_json, write_csv

COUNTDOWN_REVIEWER_EXPERIMENT_ID = "EXT-C-E8-TAPER-0.5B-01"
COUNTDOWN_CANONICAL_PROTOCOL_ID = "EXT-C-E8-TAPER-0.5B-01-v79"
COUNTDOWN_REVIEWER_RUNNER_VERSION = "0.2.0-canonical-result-coordinate"
_CONFIG_SCHEMA_VERSIONS = (1, 2)
_SELECTION_METRICS = ("greedy_success", "pass_at_k", "valid_rate")
_CANONICAL_METHODS = (
    "positive_only",
    "uncontrolled_negative",
    "global_matched",
    "reciprocal_linear",
    "exponential",
    "squared_distance_exponential",
)
_CANONICAL_SEEDS = (9234, 10234, 11234)
_CANONICAL_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_CANONICAL_PROTOCOL_CONTRACT = {
    "dataset_generation": {
        "seed": 1234,
        "train_rows": 6000,
        "validation_rows": 500,
        "test_rows": 1000,
        "numbers_per_problem": 4,
        "split_protocol": (
            "park_inspired_pattern_first_family_holdout_capacity_audited"
        ),
    },
    "reference": {
        "base_greedy_success_gate": 0.15,
        "base_valid_rate_gate": 0.80,
        "pilot_greedy_success_gate": 0.08,
        "pilot_valid_rate_gate": 0.95,
        "formal_greedy_success_gate": 0.15,
        "formal_valid_rate_gate": 0.95,
        "sft_epochs": 6,
        "sft_min_epochs": 3,
        "sft_early_stop_patience": 2,
        "sft_micro_batch": 1,
        "sft_gradient_accumulation": 32,
        "sft_learning_rate": 0.0002,
        "sft_warmup_ratio": 0.03,
        "sft_max_gradient_norm": 1.0,
        "validation_seed": 5000,
    },
    "replay_collection": {
        "train_prompt_rows": 900,
        "calibration_prompt_rows": 16,
        "train_candidate_prompt_rows": 1800,
        "calibration_candidate_prompt_rows": 32,
        "rollouts_per_prompt_per_round": 12,
        "resample_rounds": 3,
        "generation_batch_size": 4,
        "score_batch_size": 16,
        "temperature": 0.8,
        "top_p": 0.95,
        "synthetic_negative_fallback": False,
        "fixed_negative_count_per_prompt": False,
        "enforce_training_structure_support": True,
        "prompt_balanced_sampling": True,
    },
    "test_access": "after_all_method_training_only",
    "checkpoint_policy": "best_plus_terminal_or_last_finite",
}


@dataclass(frozen=True)
class HFStack:
    AutoModelForCausalLM: Any
    AutoTokenizer: Any
    BitsAndBytesConfig: Any
    get_cosine_schedule_with_warmup: Any
    LoraConfig: Any
    PeftModel: Any
    get_peft_model: Any
    prepare_model_for_kbit_training: Any


@dataclass(frozen=True)
class CountdownReviewerConfig:
    schema_version: int
    protocol_id: str
    protocol_contract: dict[str, Any]
    model_path: str
    model_identity: str | None
    initial_adapter: str | None
    device: str
    dtype: str
    load_in_4bit: bool
    gradient_checkpointing: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    replay_path: Path
    calibration_path: Path
    validation_path: Path
    test_path: Path | None
    structure_reference_path: Path | None
    expected_replay_rows: int | None
    expected_calibration_rows: int | None
    expected_validation_rows: int | None
    expected_test_rows: int | None
    expected_structure_reference_rows: int | None
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    max_length: int
    steps: int
    micro_batch: int
    grad_accum: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    eval_every: int
    checkpoint_every: int
    calibration_prompts: int
    calibration_seed: int
    minimum_surprisal_scale: float
    inherited_exponential_coefficient: float
    maximum_coefficient: float
    bisection_steps: int
    relative_l2_tolerance: float
    minimum_active_distance_fraction: float
    nondegenerate_target_max_ratio: float
    minimum_taper_lambda: float
    eval_batch: int
    eval_examples: int
    max_new_tokens: int
    pass_k: int
    evaluation_seed: int | None
    evaluation_seed_offset: int | None
    selection_metric: str
    selection_delta: float
    sample_temperature: float
    sample_top_p: float
    require_structure_metrics: bool

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        base_dir: Path,
    ) -> "CountdownReviewerConfig":
        schema_version = _int_value(value.get("schema_version"), "schema_version")
        if schema_version not in _CONFIG_SCHEMA_VERSIONS:
            raise ValueError("Countdown config schema_version must be 1 or 2")
        protocol_id = str(value.get("protocol_id") or "custom-reviewer-coordinate")
        protocol_contract_value = value.get("protocol_contract") or {}
        if not isinstance(protocol_contract_value, Mapping):
            raise ValueError("protocol_contract must be a mapping")
        protocol_contract = json.loads(json.dumps(protocol_contract_value))
        model = _mapping(value, "model")
        data = _mapping(value, "data")
        training = _mapping(value, "training")
        calibration = _mapping(value, "calibration")
        evaluation = _mapping(value, "evaluation")
        lora = _mapping(model, "lora")
        expected_rows_value = data.get("expected_rows") or {}
        if not isinstance(expected_rows_value, Mapping):
            raise ValueError("data.expected_rows must be a mapping")
        methods = _unique_strings(value.get("methods"), "methods")
        unknown = sorted(set(methods) - set(COUNTDOWN_ACTIVE_TAIL_METHODS))
        if unknown:
            raise ValueError("unsupported Countdown methods: " + ", ".join(unknown))
        seeds = _unique_ints(value.get("seeds"), "seeds")
        target_modules = _unique_strings(
            lora.get("target_modules"), "lora.target_modules"
        )
        test_value = data.get("test")
        structure_reference_value = data.get("structure_reference")
        initial_adapter = model.get("initial_adapter")
        seed_value = evaluation.get("seed")
        seed_offset_value = evaluation.get("seed_offset")
        config = cls(
            schema_version=schema_version,
            protocol_id=_nonempty_string(protocol_id, "protocol_id"),
            protocol_contract=protocol_contract,
            model_path=_expand_required_string(model.get("path"), "model.path"),
            model_identity=(
                _nonempty_string(model.get("identity"), "model.identity")
                if model.get("identity") is not None
                else None
            ),
            initial_adapter=(
                _expand_required_string(initial_adapter, "model.initial_adapter")
                if initial_adapter is not None
                else None
            ),
            device=_nonempty_string(model.get("device"), "model.device"),
            dtype=_nonempty_string(model.get("dtype"), "model.dtype"),
            load_in_4bit=_strict_bool(model.get("load_in_4bit"), "model.load_in_4bit"),
            gradient_checkpointing=_strict_bool(
                model.get("gradient_checkpointing"), "model.gradient_checkpointing"
            ),
            lora_r=_positive_int(lora.get("r"), "model.lora.r"),
            lora_alpha=_positive_int(lora.get("alpha"), "model.lora.alpha"),
            lora_dropout=_probability(
                lora.get("dropout"), "model.lora.dropout", upper_open=True
            ),
            lora_target_modules=target_modules,
            replay_path=_config_path(data.get("replay"), base_dir, "data.replay"),
            calibration_path=_config_path(
                data.get("calibration"), base_dir, "data.calibration"
            ),
            validation_path=_config_path(
                data.get("validation"), base_dir, "data.validation"
            ),
            test_path=(
                _config_path(test_value, base_dir, "data.test")
                if test_value is not None
                else None
            ),
            structure_reference_path=(
                _config_path(
                    structure_reference_value,
                    base_dir,
                    "data.structure_reference",
                )
                if structure_reference_value is not None
                else None
            ),
            expected_replay_rows=_optional_positive_int(
                expected_rows_value.get("replay"), "data.expected_rows.replay"
            ),
            expected_calibration_rows=_optional_positive_int(
                expected_rows_value.get("calibration"),
                "data.expected_rows.calibration",
            ),
            expected_validation_rows=_optional_positive_int(
                expected_rows_value.get("validation"),
                "data.expected_rows.validation",
            ),
            expected_test_rows=_optional_positive_int(
                expected_rows_value.get("test"), "data.expected_rows.test"
            ),
            expected_structure_reference_rows=_optional_positive_int(
                expected_rows_value.get("structure_reference"),
                "data.expected_rows.structure_reference",
            ),
            methods=methods,
            seeds=seeds,
            max_length=_positive_int(training.get("max_length"), "training.max_length"),
            steps=_positive_int(training.get("steps"), "training.steps"),
            micro_batch=_positive_int(
                training.get("micro_batch"), "training.micro_batch"
            ),
            grad_accum=_positive_int(training.get("grad_accum"), "training.grad_accum"),
            learning_rate=_positive_float(
                training.get("learning_rate"), "training.learning_rate"
            ),
            weight_decay=_nonnegative_float(
                training.get("weight_decay"), "training.weight_decay"
            ),
            warmup_ratio=_probability(
                training.get("warmup_ratio"),
                "training.warmup_ratio",
                upper_open=False,
            ),
            max_grad_norm=_positive_float(
                training.get("max_grad_norm"), "training.max_grad_norm"
            ),
            eval_every=_positive_int(training.get("eval_every"), "training.eval_every"),
            checkpoint_every=_positive_int(
                training.get("checkpoint_every"), "training.checkpoint_every"
            ),
            calibration_prompts=_positive_int(
                calibration.get("prompts"), "calibration.prompts"
            ),
            calibration_seed=_int_value(calibration.get("seed"), "calibration.seed"),
            minimum_surprisal_scale=_positive_float(
                calibration.get("minimum_surprisal_scale"),
                "calibration.minimum_surprisal_scale",
            ),
            inherited_exponential_coefficient=_positive_float(
                calibration.get("inherited_exponential_coefficient"),
                "calibration.inherited_exponential_coefficient",
            ),
            maximum_coefficient=_positive_float(
                calibration.get("maximum_coefficient"),
                "calibration.maximum_coefficient",
            ),
            bisection_steps=_positive_int(
                calibration.get("bisection_steps"), "calibration.bisection_steps"
            ),
            relative_l2_tolerance=_probability(
                calibration.get("relative_l2_tolerance"),
                "calibration.relative_l2_tolerance",
                upper_open=False,
            ),
            minimum_active_distance_fraction=_probability(
                calibration.get("minimum_active_distance_fraction"),
                "calibration.minimum_active_distance_fraction",
                upper_open=False,
                lower_open=True,
            ),
            nondegenerate_target_max_ratio=_probability(
                calibration.get("nondegenerate_target_max_ratio"),
                "calibration.nondegenerate_target_max_ratio",
                upper_open=True,
                lower_open=True,
            ),
            minimum_taper_lambda=_positive_float(
                calibration.get("minimum_taper_lambda"),
                "calibration.minimum_taper_lambda",
            ),
            eval_batch=_positive_int(
                evaluation.get("batch_size"), "evaluation.batch_size"
            ),
            eval_examples=_positive_int(
                evaluation.get("examples"), "evaluation.examples"
            ),
            max_new_tokens=_positive_int(
                evaluation.get("max_new_tokens"), "evaluation.max_new_tokens"
            ),
            pass_k=_positive_int(evaluation.get("pass_k"), "evaluation.pass_k"),
            evaluation_seed=(
                _int_value(seed_value, "evaluation.seed")
                if seed_value is not None
                else None
            ),
            evaluation_seed_offset=(
                _int_value(seed_offset_value, "evaluation.seed_offset")
                if seed_offset_value is not None
                else None
            ),
            selection_metric=_nonempty_string(
                evaluation.get("selection_metric"), "evaluation.selection_metric"
            ),
            selection_delta=_nonnegative_float(
                evaluation.get("selection_delta", 0.0),
                "evaluation.selection_delta",
            ),
            sample_temperature=_positive_float(
                evaluation.get("sample_temperature"),
                "evaluation.sample_temperature",
            ),
            sample_top_p=_probability(
                evaluation.get("sample_top_p"),
                "evaluation.sample_top_p",
                upper_open=False,
                lower_open=True,
            ),
            require_structure_metrics=_strict_bool(
                evaluation.get("require_structure_metrics", False),
                "evaluation.require_structure_metrics",
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("model.dtype must be bf16, fp16, or fp32")
        if self.selection_metric not in _SELECTION_METRICS:
            raise ValueError(
                "evaluation.selection_metric must be one of "
                + ", ".join(_SELECTION_METRICS)
            )
        if self.eval_every > self.steps or self.checkpoint_every > self.steps:
            raise ValueError(
                "evaluation/checkpoint cadence cannot exceed training steps"
            )
        if self.calibration_prompts < 2:
            raise ValueError("calibration.prompts must be at least two")
        if self.test_path is not None and self.test_path == self.validation_path:
            raise ValueError("validation and test paths must be distinct")
        if (self.evaluation_seed is None) == (self.evaluation_seed_offset is None):
            raise ValueError(
                "evaluation must define exactly one of seed or seed_offset"
            )
        if self.require_structure_metrics and self.structure_reference_path is None:
            raise ValueError(
                "structure_reference is required when structure metrics are enabled"
            )
        if self.protocol_id == COUNTDOWN_CANONICAL_PROTOCOL_ID:
            self._validate_canonical_coordinate()

    def _validate_canonical_coordinate(self) -> None:
        failures: list[str] = []

        def require(condition: bool, message: str) -> None:
            if not condition:
                failures.append(message)

        require(self.schema_version == 2, "canonical schema_version must be 2")
        require(
            self.protocol_contract == _CANONICAL_PROTOCOL_CONTRACT,
            "canonical protocol_contract differs from the registered v79 freeze",
        )
        require(
            self.model_identity == "Qwen2.5-0.5B-Instruct",
            "canonical model identity must be Qwen2.5-0.5B-Instruct",
        )
        require(
            self.initial_adapter is not None, "canonical reference adapter is required"
        )
        require(self.device.startswith("cuda"), "canonical runtime requires CUDA")
        require(self.dtype == "bf16", "canonical dtype must be bf16")
        require(not self.load_in_4bit, "canonical v79 uses BF16, not 4-bit loading")
        require(
            self.gradient_checkpointing, "canonical gradient checkpointing is required"
        )
        require(self.lora_r == 32, "canonical LoRA rank must be 32")
        require(self.lora_alpha == 64, "canonical LoRA alpha must be 64")
        require(self.lora_dropout == 0.05, "canonical LoRA dropout must be 0.05")
        require(
            self.lora_target_modules == _CANONICAL_LORA_TARGETS,
            "canonical LoRA target modules differ",
        )
        require(self.expected_replay_rows == 900, "canonical replay rows must be 900")
        require(
            self.expected_calibration_rows == 16,
            "canonical calibration rows must be 16",
        )
        require(
            self.expected_validation_rows == 500,
            "canonical validation rows must be 500",
        )
        require(self.expected_test_rows == 1000, "canonical test rows must be 1000")
        require(
            self.expected_structure_reference_rows == 6000,
            "canonical structure-reference rows must be 6000",
        )
        require(self.methods == _CANONICAL_METHODS, "canonical method order differs")
        require(self.seeds == _CANONICAL_SEEDS, "canonical paired seeds differ")
        require(self.max_length == 256, "canonical max_length must be 256")
        require(self.steps == 1200, "canonical update budget must be 1200")
        require(self.micro_batch == 1, "canonical micro_batch must be 1")
        require(self.grad_accum == 8, "canonical grad_accum must be 8")
        require(self.learning_rate == 0.00005, "canonical learning rate differs")
        require(self.weight_decay == 0.01, "canonical weight decay differs")
        require(self.warmup_ratio == 0.03, "canonical warmup ratio differs")
        require(self.max_grad_norm == 1.0, "canonical gradient norm differs")
        require(self.eval_every == 100, "canonical evaluation cadence differs")
        require(self.calibration_prompts == 16, "canonical calibration prompts differ")
        require(self.calibration_seed == 9134, "canonical calibration seed differs")
        require(
            self.minimum_surprisal_scale == 1.0e-6, "canonical minimum scale differs"
        )
        require(
            self.inherited_exponential_coefficient == 0.7,
            "canonical inherited exponential coefficient differs",
        )
        require(self.maximum_coefficient == 64.0, "canonical maximum lambda differs")
        require(self.bisection_steps == 24, "canonical bisection steps differ")
        require(self.relative_l2_tolerance == 0.01, "canonical L2 tolerance differs")
        require(
            self.minimum_active_distance_fraction == 0.25,
            "canonical minimum active fraction differs",
        )
        require(
            self.nondegenerate_target_max_ratio == 0.995,
            "canonical nondegenerate target ratio differs",
        )
        require(self.minimum_taper_lambda == 1.0e-6, "canonical minimum lambda differs")
        require(self.eval_batch == 8, "canonical evaluation batch differs")
        require(self.eval_examples == 500, "canonical validation example count differs")
        require(self.max_new_tokens == 80, "canonical max_new_tokens differs")
        require(self.pass_k == 8, "canonical pass@k must be 8")
        require(
            self.evaluation_seed_offset == 700000,
            "canonical paired evaluation seed offset differs",
        )
        require(
            self.selection_metric == "greedy_success",
            "canonical selection metric differs",
        )
        require(self.selection_delta == 0.002, "canonical selection delta differs")
        require(
            self.sample_temperature == 0.8, "canonical sampling temperature differs"
        )
        require(self.sample_top_p == 0.95, "canonical top_p differs")
        require(
            self.require_structure_metrics, "canonical structure metrics are required"
        )
        if failures:
            raise ValueError(
                "canonical Countdown coordinate mismatch: " + "; ".join(failures)
            )

    def evaluation_seed_for(self, training_seed: int) -> int:
        if self.evaluation_seed_offset is not None:
            return int(training_seed) + int(self.evaluation_seed_offset)
        if self.evaluation_seed is None:
            raise AssertionError("validated config has no evaluation seed")
        return int(self.evaluation_seed)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "protocol_contract": self.protocol_contract,
            "model": {
                "path": self.model_path,
                "identity": self.model_identity,
                "initial_adapter": self.initial_adapter,
                "device": self.device,
                "dtype": self.dtype,
                "load_in_4bit": self.load_in_4bit,
                "gradient_checkpointing": self.gradient_checkpointing,
                "lora": {
                    "r": self.lora_r,
                    "alpha": self.lora_alpha,
                    "dropout": self.lora_dropout,
                    "target_modules": list(self.lora_target_modules),
                },
            },
            "data": {
                "replay": str(self.replay_path),
                "calibration": str(self.calibration_path),
                "validation": str(self.validation_path),
                "test": str(self.test_path) if self.test_path is not None else None,
                "structure_reference": (
                    str(self.structure_reference_path)
                    if self.structure_reference_path is not None
                    else None
                ),
                "expected_rows": {
                    "replay": self.expected_replay_rows,
                    "calibration": self.expected_calibration_rows,
                    "validation": self.expected_validation_rows,
                    "test": self.expected_test_rows,
                    "structure_reference": self.expected_structure_reference_rows,
                },
            },
            "methods": list(self.methods),
            "seeds": list(self.seeds),
            "training": {
                "max_length": self.max_length,
                "steps": self.steps,
                "micro_batch": self.micro_batch,
                "grad_accum": self.grad_accum,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_ratio": self.warmup_ratio,
                "max_grad_norm": self.max_grad_norm,
                "eval_every": self.eval_every,
                "checkpoint_every": self.checkpoint_every,
            },
            "calibration": {
                "prompts": self.calibration_prompts,
                "seed": self.calibration_seed,
                "tau_rule": COUNTDOWN_ACTIVE_TAIL_TAU_RULE,
                "minimum_surprisal_scale": self.minimum_surprisal_scale,
                "inherited_exponential_coefficient": (
                    self.inherited_exponential_coefficient
                ),
                "maximum_coefficient": self.maximum_coefficient,
                "bisection_steps": self.bisection_steps,
                "relative_l2_tolerance": self.relative_l2_tolerance,
                "minimum_active_distance_fraction": (
                    self.minimum_active_distance_fraction
                ),
                "nondegenerate_target_max_ratio": self.nondegenerate_target_max_ratio,
                "minimum_taper_lambda": self.minimum_taper_lambda,
            },
            "evaluation": {
                "batch_size": self.eval_batch,
                "examples": self.eval_examples,
                "max_new_tokens": self.max_new_tokens,
                "pass_k": self.pass_k,
                "seed": self.evaluation_seed,
                "seed_offset": self.evaluation_seed_offset,
                "selection_metric": self.selection_metric,
                "selection_delta": self.selection_delta,
                "sample_temperature": self.sample_temperature,
                "sample_top_p": self.sample_top_p,
                "require_structure_metrics": self.require_structure_metrics,
            },
        }


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return result


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _int_value(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _positive_int(value: Any, name: str) -> int:
    result = _int_value(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _probability(
    value: Any,
    name: str,
    *,
    upper_open: bool,
    lower_open: bool = False,
) -> float:
    result = _finite_float(value, name)
    lower_ok = result > 0.0 if lower_open else result >= 0.0
    upper_ok = result < 1.0 if upper_open else result <= 1.0
    if not lower_ok or not upper_ok:
        left = "(" if lower_open else "["
        right = ")" if upper_open else "]"
        raise ValueError(f"{name} must lie in {left}0, 1{right}")
    return result


def _unique_strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    items = tuple(_nonempty_string(item, name) for item in value)
    if not items or len(set(items)) != len(items):
        raise ValueError(f"{name} must be non-empty and duplicate-free")
    return items


def _unique_ints(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    items = tuple(_int_value(item, name) for item in value)
    if not items or len(set(items)) != len(items):
        raise ValueError(f"{name} must be non-empty and duplicate-free")
    return items


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _expand_required_string(value: Any, name: str) -> str:
    raw = _nonempty_string(value, name)
    expanded = os.path.expanduser(os.path.expandvars(raw))
    if "$" in expanded:
        raise ValueError(f"{name} contains an unresolved environment variable")
    return expanded


def _config_path(value: Any, base_dir: Path, name: str) -> Path:
    expanded = _expand_required_string(value, name)
    path = Path(expanded)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_countdown_config(path: str | Path) -> CountdownReviewerConfig:
    config_path = Path(path).expanduser().resolve()
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Countdown reviewer config must be a JSON object")
    return CountdownReviewerConfig.from_mapping(value, base_dir=config_path.parent)


def _load_hf_stack() -> HFStack:
    try:
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
    except ImportError as exc:
        raise RuntimeError(
            "Countdown runtime dependencies are missing. Install the optional "
            "paper_code[countdown] dependencies."
        ) from exc
    return HFStack(
        AutoModelForCausalLM=transformers.AutoModelForCausalLM,
        AutoTokenizer=transformers.AutoTokenizer,
        BitsAndBytesConfig=transformers.BitsAndBytesConfig,
        get_cosine_schedule_with_warmup=(transformers.get_cosine_schedule_with_warmup),
        LoraConfig=peft.LoraConfig,
        PeftModel=peft.PeftModel,
        get_peft_model=peft.get_peft_model,
        prepare_model_for_kbit_training=peft.prepare_model_for_kbit_training,
    )


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {value}")
    return device


def _dtype(value: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[value]


def _normalized_identity_text(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _validate_model_identity(config: CountdownReviewerConfig) -> dict[str, Any]:
    expected = config.model_identity
    if expected is None:
        return {"expected_identity": None, "identity_verified": False}
    model_root = Path(config.model_path).expanduser()
    hints = [config.model_path]
    model_type: str | None = None
    has_chat_template: bool | None = None
    if model_root.is_dir():
        model_config_path = model_root / "config.json"
        tokenizer_config_path = model_root / "tokenizer_config.json"
        if not model_config_path.is_file():
            raise RuntimeError("canonical model directory has no config.json")
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        tokenizer_config = (
            json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
            if tokenizer_config_path.is_file()
            else {}
        )
        model_type = str(model_config.get("model_type") or "")
        has_chat_template = bool(tokenizer_config.get("chat_template"))
        hints.extend(
            [
                str(model_config.get("_name_or_path") or ""),
                str(model_config.get("architectures") or ""),
                str(tokenizer_config.get("name_or_path") or ""),
                model_root.name,
            ]
        )
        if config.protocol_id == COUNTDOWN_CANONICAL_PROTOCOL_ID:
            if model_type != "qwen2":
                raise RuntimeError(
                    f"canonical model_type must be qwen2, observed {model_type!r}"
                )
            if not has_chat_template:
                raise RuntimeError("canonical tokenizer must define a chat template")
    identity_text = _normalized_identity_text(" ".join(hints))
    expected_text = _normalized_identity_text(expected)
    required_fragments = ("qwen25", "05b", "instruct")
    verified = expected_text in identity_text or all(
        fragment in identity_text for fragment in required_fragments
    )
    if not verified:
        raise RuntimeError(
            f"model identity does not match Qwen2.5-0.5B-Instruct; hints={hints!r}"
        )
    return {
        "expected_identity": expected,
        "identity_verified": True,
        "model_type": model_type,
        "has_chat_template": has_chat_template,
        "identity_hints": hints,
    }


def _hash_tree(root: Path) -> dict[str, str]:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {resolved}")
    members = sorted(resolved.rglob("*"))
    symlinks = [item for item in members if item.is_symlink()]
    if symlinks:
        raise RuntimeError(f"adapter directory contains a symlink: {symlinks[0]}")
    files = [item for item in members if item.is_file()]
    if not files:
        raise RuntimeError(f"adapter directory is empty: {resolved}")
    return {str(item.relative_to(resolved)): _sha256_file(item) for item in files}


def _adapter_identity(config: CountdownReviewerConfig) -> dict[str, Any]:
    if config.initial_adapter is None:
        return {"path": None, "hashes": None, "prepared_reference_required": False}
    root = Path(config.initial_adapter).expanduser().resolve()
    hashes = _hash_tree(root)
    if config.protocol_id == COUNTDOWN_CANONICAL_PROTOCOL_ID:
        if "adapter_config.json" not in hashes:
            raise RuntimeError("canonical reference adapter has no adapter_config.json")
        if not ({"adapter_model.safetensors", "adapter_model.bin"} & set(hashes)):
            raise RuntimeError("canonical reference adapter has no adapter weights")
    return {
        "path": str(root),
        "hashes": hashes,
        "prepared_reference_required": (
            config.protocol_id == COUNTDOWN_CANONICAL_PROTOCOL_ID
        ),
    }


def _load_tokenizer(stack: HFStack, model_path: str) -> Any:
    tokenizer = stack.AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
        raise RuntimeError("Countdown tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_trainable_model(
    stack: HFStack,
    config: CountdownReviewerConfig,
    *,
    adapter_path: str | None = None,
) -> Any:
    device = _resolve_device(config.device)
    if config.load_in_4bit and device.type != "cuda":
        raise RuntimeError("4-bit Countdown loading requires CUDA")
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": _dtype(config.dtype),
    }
    if config.load_in_4bit:
        kwargs["device_map"] = {"": device.index or 0}
        kwargs["quantization_config"] = stack.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_dtype(config.dtype),
            bnb_4bit_use_double_quant=True,
        )
    model = stack.AutoModelForCausalLM.from_pretrained(config.model_path, **kwargs)
    if not config.load_in_4bit:
        model = model.to(device)
    else:
        model = stack.prepare_model_for_kbit_training(model)
    selected_adapter = adapter_path or config.initial_adapter
    if selected_adapter is not None:
        model = stack.PeftModel.from_pretrained(
            model,
            selected_adapter,
            is_trainable=True,
        )
    else:
        model = stack.get_peft_model(
            model,
            stack.LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(config.lora_target_modules),
            ),
        )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False
    return model


def _trainable_parameters(model: Any) -> list[torch.nn.Parameter]:
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("Countdown model exposes no trainable parameters")
    return parameters


def _trainable_state_digest(model: Any) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count += 1
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        tensor = parameter.detach().float().cpu().contiguous().numpy()
        digest.update(tensor.tobytes())
    if count == 0:
        raise RuntimeError("cannot hash an empty trainable state")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{resolved}:{line_number} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL file is empty: {resolved}")
    return rows


def _normalize_training_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized = dict(row)
        if not isinstance(normalized.get("positive"), str):
            oracle = normalized.get("oracle")
            if isinstance(oracle, str):
                normalized["positive"] = oracle
        if "negative_bank" not in normalized:
            negatives = normalized.get("negatives")
            if isinstance(negatives, Sequence) and not isinstance(
                negatives, (str, bytes)
            ):
                normalized["negative_bank"] = list(negatives)
        normalized_rows.append(normalized)
    return normalized_rows


def _require_row_count(
    rows: Sequence[Mapping[str, Any]],
    expected: int | None,
    name: str,
) -> None:
    if expected is not None and len(rows) != expected:
        raise ValueError(f"{name} row count mismatch: {len(rows)} != {expected}")


def _validate_structure_reference_rows(
    rows: Sequence[Mapping[str, Any]],
) -> set[str]:
    structures: set[str] = set()
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        oracle = row.get("oracle") or row.get("positive")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"structure_reference[{index}] has no prompt")
        if not isinstance(oracle, str) or not oracle:
            raise ValueError(f"structure_reference[{index}] has no oracle")
        structures.add(str(row.get("oracle_structure") or expression_structure(oracle)))
    if not structures:
        raise ValueError("structure_reference has no canonical structures")
    return structures


def _validate_training_rows(rows: Sequence[Mapping[str, Any]], name: str) -> None:
    prompts: set[str] = set()
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        positive = row.get("positive")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{name}[{index}] has no prompt")
        if prompt in prompts:
            raise ValueError(f"{name} contains duplicate prompt: {prompt}")
        prompts.add(prompt)
        if not isinstance(positive, str) or not positive:
            raise ValueError(f"{name}[{index}] has no positive completion")
        unique_negative_expressions(row)


def _validate_evaluation_rows(rows: Sequence[Mapping[str, Any]], name: str) -> None:
    prompts: set[str] = set()
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        numbers = row.get("numbers")
        target = row.get("target")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{name}[{index}] has no prompt")
        if prompt in prompts:
            raise ValueError(f"{name} contains duplicate prompt: {prompt}")
        prompts.add(prompt)
        if not isinstance(numbers, Sequence) or isinstance(numbers, (str, bytes)):
            raise ValueError(f"{name}[{index}] has no number sequence")
        if not numbers or any(
            isinstance(item, bool) or not isinstance(item, int) for item in numbers
        ):
            raise ValueError(f"{name}[{index}] numbers must be non-empty integers")
        if isinstance(target, bool) or not isinstance(target, int):
            raise ValueError(f"{name}[{index}] target must be an integer")


def _validate_structure_evaluation_rows(
    rows: Sequence[Mapping[str, Any]],
    name: str,
) -> None:
    for index, row in enumerate(rows):
        oracle = row.get("oracle")
        oracle_structure = row.get("oracle_structure")
        if not isinstance(oracle, str) and not isinstance(oracle_structure, str):
            raise ValueError(
                f"{name}[{index}] requires oracle or oracle_structure for "
                "canonical structure metrics"
            )


def _assert_prompt_disjoint(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    left_name: str,
    right_name: str,
) -> None:
    overlap = {str(row["prompt"]) for row in left} & {
        str(row["prompt"]) for row in right
    }
    if overlap:
        example = sorted(overlap)[0]
        raise ValueError(f"{left_name} and {right_name} prompt sets overlap: {example}")


def _prepare_training_items(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> tuple[list[CountdownTrainingItem], list[dict[str, Any]]]:
    items: list[CountdownTrainingItem] = []
    sampler_rows: list[dict[str, Any]] = []
    for row in rows:
        unique = unique_negative_expressions(row)
        normalized = dict(row)
        normalized["negative_bank"] = unique
        items.append(encode_countdown_training_row(normalized, tokenizer, max_length))
        sampler_rows.append({"negatives": unique})
    return items, sampler_rows


def _selected_batch(
    encoded: Sequence[CountdownTrainingItem],
    plan: Sequence[Mapping[str, int]],
    start: int,
    batch_size: int,
    pad_id: int,
) -> dict[str, Any]:
    selected: list[CountdownTrainingItem] = []
    for coordinate in plan[start : start + batch_size]:
        prompt_index = int(coordinate["prompt_index"])
        negative_index = int(coordinate["negative_index"])
        item = encoded[prompt_index]
        selected.append(
            CountdownTrainingItem(
                positive=item.positive,
                bank=(item.bank[negative_index],),
                unique_count=1,
                raw_bank_count=1,
            )
        )
    if len(selected) != batch_size:
        raise RuntimeError("prompt-balanced plan ended before the requested batch")
    return collate_countdown_training_items(selected, pad_id)


def _packed_to_device(
    packed: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    return {
        "positive": move_tensor_batch_to_device(packed["positive"], device),
        "bank": move_tensor_batch_to_device(packed["bank"], device),
        "bank_row_index": packed["bank_row_index"].to(device),
        "unique_counts": packed["unique_counts"].to(device),
        "raw_bank_counts": packed["raw_bank_counts"].to(device),
    }


def _calibration_subset(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    if count > len(rows):
        raise ValueError(
            f"calibration.prompts={count} exceeds calibration rows={len(rows)}"
        )
    indices = list(range(len(rows)))
    random.Random(int(seed)).shuffle(indices)
    return [rows[index] for index in indices[:count]]


def _calibrate_for_seed(
    model: Any,
    tokenizer: Any,
    config: CountdownReviewerConfig,
    calibration_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    subset = _calibration_subset(
        calibration_rows,
        config.calibration_prompts,
        config.calibration_seed,
    )
    items = [
        encode_countdown_training_row(row, tokenizer, config.max_length)
        for row in subset
    ]
    packed = collate_countdown_training_items(items, tokenizer.pad_token_id)
    device = next(model.parameters()).device
    packed_device = _packed_to_device(packed, device)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            negative_stats = completion_stats(model, packed_device["bank"])
            surprisals = (-negative_stats["seq_lp"].detach()).float().cpu().tolist()
    finally:
        model.train(was_training)
    surprisal_scale, scale_diagnostics = calibration_surprisal_scale(
        surprisals,
        minimum=config.minimum_surprisal_scale,
    )
    tau, tau_rule = resolve_active_tail_tau(
        COUNTDOWN_ACTIVE_TAIL_TAU_RULE,
        scale_diagnostics,
    )
    calibration = calibrate_active_tail_model(
        model,
        packed_device,
        _trainable_parameters(model),
        tau=tau,
        surprisal_scale=surprisal_scale,
        inherited_exponential_coefficient=(config.inherited_exponential_coefficient),
        maximum_coefficient=config.maximum_coefficient,
        bisection_steps=config.bisection_steps,
        relative_l2_tolerance=config.relative_l2_tolerance,
        minimum_active_distance_fraction=(config.minimum_active_distance_fraction),
        nondegenerate_target_max_ratio=config.nondegenerate_target_max_ratio,
        minimum_taper_lambda=config.minimum_taper_lambda,
    )
    result = dict(calibration)
    result.update(
        {
            "seed": int(seed),
            "calibration_subset_seed": int(config.calibration_seed),
            "calibration_prompt_count": len(subset),
            "surprisal_scale": float(surprisal_scale),
            "surprisal_scale_diagnostics": scale_diagnostics,
            "tau": float(tau),
            "tau_rule": tau_rule,
            "initial_trainable_state_sha256": _trainable_state_digest(model),
            "confirmation_or_test_metrics_used": False,
            "formal_result_claim": False,
        }
    )
    coefficients = result.get("method_coefficients")
    if not isinstance(coefficients, Mapping):
        raise RuntimeError("active-tail calibration returned no coefficient mapping")
    if not isinstance(result.get("shared_negative_scale"), (int, float)):
        raise RuntimeError("active-tail calibration returned no shared_negative_scale")
    return result


def _gradient_state(parameters: Sequence[torch.nn.Parameter]) -> tuple[float, bool]:
    total = torch.zeros((), dtype=torch.float64)
    finite = True
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all()):
            finite = False
        total += gradient.detach().double().cpu().square().sum()
    return float(torch.sqrt(total).item()), finite


def _parameters_finite(parameters: Sequence[torch.nn.Parameter]) -> bool:
    return all(
        bool(torch.isfinite(parameter.detach()).all()) for parameter in parameters
    )


def _restore_parameters(
    snapshots: Sequence[torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
) -> None:
    if len(snapshots) != len(parameters):
        raise ValueError("parameter snapshots and live parameters must align")
    with torch.no_grad():
        for snapshot, parameter in zip(snapshots, parameters, strict=True):
            parameter.copy_(snapshot.to(device=parameter.device, dtype=parameter.dtype))
    if not _parameters_finite(parameters):
        raise FloatingPointError("failed to restore the last finite parameters")


def _save_adapter_checkpoint(
    model: Any,
    tokenizer: Any,
    destination: Path,
    metadata: Mapping[str, Any],
) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(temporary)
    tokenizer.save_pretrained(temporary)
    atomic_json(temporary / "CHECKPOINT.json", dict(metadata))
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)


_AST_OP = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}


class _PatternNode:
    def __init__(
        self,
        op: str,
        children: list["_PatternNode"] | None = None,
        sign: str = "+",
    ) -> None:
        self.op = op
        self.children = children or []
        self.sign = sign
        self.weight = (
            1 if not self.children else sum(child.weight for child in self.children)
        )


def _flip_sign(sign: str) -> str:
    return "-" if sign == "+" else "+"


def _pattern_tree_from_ast(
    node: ast.AST,
    symbols: Iterable[str] | None = None,
) -> _PatternNode:
    symbol_iter = iter(symbols or (chr(ord("A") + index) for index in range(26)))

    def convert(current: ast.AST) -> _PatternNode:
        if isinstance(current, ast.Expression):
            return convert(current.body)
        if isinstance(current, (ast.Constant, ast.Name)):
            if isinstance(current, ast.Constant) and (
                isinstance(current.value, bool) or not isinstance(current.value, int)
            ):
                raise ValueError("only integer leaves are supported")
            return _PatternNode(next(symbol_iter))
        if isinstance(current, ast.BinOp) and type(current.op) in _AST_OP:
            return _PatternNode(
                _AST_OP[type(current.op)],
                [convert(current.left), convert(current.right)],
            )
        raise ValueError(f"unsupported structure node: {type(current).__name__}")

    return convert(node)


def _raw_pattern_shape(node: _PatternNode) -> str:
    if not node.children:
        return "L"
    return (
        node.op
        + "("
        + ",".join(child.sign + _raw_pattern_shape(child) for child in node.children)
        + ")"
    )


def _generic_pattern_tree(node: _PatternNode) -> _PatternNode:
    if not node.children:
        return node
    node.children = [_generic_pattern_tree(child) for child in node.children]
    if node.op in {"-", "/"}:
        node.children[1].sign = _flip_sign(node.children[1].sign)
    family = {"+", "-"} if node.op in {"+", "-"} else {"*", "/"}
    merged: list[_PatternNode] = []
    for child in node.children:
        if child.op in family:
            for grandchild in child.children:
                if child.sign == "-":
                    grandchild.sign = _flip_sign(grandchild.sign)
                merged.append(grandchild)
        else:
            merged.append(child)
    node.children = merged
    node.op = "+" if node.op in {"+", "-"} else "*"
    node.weight = sum(child.weight for child in node.children)
    node.children.sort(
        key=lambda child: (
            child.sign,
            -child.weight,
            -sum(grandchild.sign == "+" for grandchild in child.children),
            _raw_pattern_shape(child),
        )
    )
    return node


def _canonical_pattern_string(tree: _PatternNode) -> str:
    symbols: dict[str, str] = {}

    def render(node: _PatternNode) -> str:
        if not node.children:
            if node.op not in symbols:
                symbols[node.op] = chr(ord("A") + len(symbols))
            return symbols[node.op]
        parts: list[str] = []
        for index, child in enumerate(node.children):
            child_text = render(child)
            if child.children and node.op == "*" and child.op == "+":
                child_text = f"({child_text})"
            if index > 0:
                if child.sign == "-":
                    parts.append("-" if node.op == "+" else "/")
                else:
                    parts.append(node.op)
            elif child.sign == "-":
                parts.append("-" if node.op == "+" else "1/")
            parts.append(child_text)
        return "".join(parts)

    return render(tree)


def expression_structure(text: str) -> str:
    expression = clean_expression(text)
    if not expression:
        raise ValueError("empty expression")
    tree = _pattern_tree_from_ast(ast.parse(expression, mode="eval"))
    return _canonical_pattern_string(_generic_pattern_tree(tree))


def _pattern_metrics(
    rows: Sequence[Mapping[str, Any]],
    greedy_outputs: Sequence[str],
    sampled_outputs: Sequence[Sequence[str]],
    known_structures: set[str],
) -> dict[str, Any]:
    target_structures = {
        str(row.get("oracle_structure") or expression_structure(str(row["oracle"])))
        for row in rows
    }
    heldout_targets = target_structures - known_structures
    greedy_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"attempts": 0, "correct": 0}
    )
    sampled_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"attempts": 0, "correct": 0}
    )
    observed_correct: set[str] = set()
    greedy_correct_structures: set[str] = set()
    sampled_correct_structures: set[str] = set()
    greedy_unseen_presence: list[float] = []
    greedy_unseen_success: list[float] = []
    sampled_unseen_presence: list[float] = []
    sampled_unseen_success: list[float] = []

    def record(
        counts: dict[str, dict[str, int]],
        pattern: str,
        correct: bool,
    ) -> None:
        counts[pattern]["attempts"] += 1
        counts[pattern]["correct"] += int(correct)

    for row, greedy_text, samples in zip(
        rows,
        greedy_outputs,
        sampled_outputs,
        strict=True,
    ):
        greedy_check = verify_expression(
            greedy_text,
            row["numbers"],
            int(row["target"]),
        )
        greedy_presence = False
        greedy_success = False
        if greedy_check["valid_format"] and greedy_check["uses_numbers"]:
            try:
                pattern = expression_structure(str(greedy_check["expression"]))
                if pattern in heldout_targets:
                    record(greedy_counts, pattern, bool(greedy_check["correct"]))
                greedy_presence = pattern not in known_structures
                if greedy_check["correct"]:
                    observed_correct.add(pattern)
                    greedy_correct_structures.add(pattern)
                    greedy_success = greedy_presence
            except Exception:
                pass
        greedy_unseen_presence.append(float(greedy_presence))
        greedy_unseen_success.append(float(greedy_success))

        any_presence = False
        any_success = False
        for sample in samples:
            check = verify_expression(sample, row["numbers"], int(row["target"]))
            if not (check["valid_format"] and check["uses_numbers"]):
                continue
            try:
                pattern = expression_structure(str(check["expression"]))
            except Exception:
                continue
            if pattern in heldout_targets:
                record(sampled_counts, pattern, bool(check["correct"]))
            if pattern not in known_structures:
                any_presence = True
            if check["correct"]:
                observed_correct.add(pattern)
                sampled_correct_structures.add(pattern)
                if pattern not in known_structures:
                    any_success = True
        sampled_unseen_presence.append(float(any_presence))
        sampled_unseen_success.append(float(any_success))

    def precision(
        counts: Mapping[str, Mapping[str, int]],
    ) -> tuple[float, float, int, int]:
        attempts = sum(int(item["attempts"]) for item in counts.values())
        correct = sum(int(item["correct"]) for item in counts.values())
        per_pattern = [
            float(item["correct"] / item["attempts"])
            for item in counts.values()
            if item["attempts"] > 0
        ]
        return (
            float(correct / attempts) if attempts else 0.0,
            float(np.mean(per_pattern)) if per_pattern else 0.0,
            attempts,
            correct,
        )

    def per_pattern(
        counts: Mapping[str, Mapping[str, int]],
    ) -> dict[str, dict[str, float | int | None]]:
        result: dict[str, dict[str, float | int | None]] = {}
        for pattern in sorted(heldout_targets):
            attempts = int(counts.get(pattern, {}).get("attempts", 0))
            correct = int(counts.get(pattern, {}).get("correct", 0))
            result[pattern] = {
                "attempts": attempts,
                "correct": correct,
                "precision": float(correct / attempts) if attempts else None,
            }
        return result

    greedy_micro, greedy_macro, greedy_attempts, greedy_correct = precision(
        greedy_counts
    )
    sampled_micro, sampled_macro, sampled_attempts, sampled_correct = precision(
        sampled_counts
    )
    heldout_correct = observed_correct & heldout_targets
    greedy_heldout_correct = greedy_correct_structures & heldout_targets
    sampled_heldout_correct = sampled_correct_structures & heldout_targets
    denominator = len(heldout_targets)
    return {
        "greedy_unseen_structure_presence": float(np.mean(greedy_unseen_presence)),
        "greedy_unseen_structure_success": float(np.mean(greedy_unseen_success)),
        "pass_at_k_unseen_structure_presence": float(np.mean(sampled_unseen_presence)),
        "pass_at_k_unseen_structure": float(np.mean(sampled_unseen_success)),
        "pass_at_k_unseen_structure_success": float(np.mean(sampled_unseen_success)),
        "unique_correct_structures": float(len(observed_correct)),
        "heldout_pattern_coverage": (
            float(len(heldout_correct) / denominator) if denominator else 0.0
        ),
        "greedy_heldout_pattern_coverage": (
            float(len(greedy_heldout_correct) / denominator) if denominator else 0.0
        ),
        "sampled_heldout_pattern_coverage": (
            float(len(sampled_heldout_correct) / denominator) if denominator else 0.0
        ),
        "greedy_heldout_pattern_precision_micro": greedy_micro,
        "greedy_heldout_pattern_precision_macro": greedy_macro,
        "sampled_heldout_pattern_precision_micro": sampled_micro,
        "sampled_heldout_pattern_precision_macro": sampled_macro,
        "heldout_pattern_precision": sampled_micro,
        "heldout_pattern_family_coverage": (
            float(len(heldout_correct) / denominator) if denominator else 0.0
        ),
        "heldout_pattern_family_precision_micro": sampled_micro,
        "heldout_pattern_family_precision_macro": sampled_macro,
        "heldout_pattern_attempts": float(sampled_attempts),
        "greedy_heldout_pattern_attempts": float(greedy_attempts),
        "greedy_heldout_pattern_correct": float(greedy_correct),
        "sampled_heldout_pattern_attempts": float(sampled_attempts),
        "sampled_heldout_pattern_correct": float(sampled_correct),
        "heldout_patterns_observed_correct": float(len(heldout_correct)),
        "correct_heldout_patterns": float(len(heldout_correct)),
        "heldout_patterns_total": float(denominator),
        "per_pattern_precision": {
            "greedy": per_pattern(greedy_counts),
            "sampled": per_pattern(sampled_counts),
        },
    }


def _generation_batches(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_return_sequences: int,
) -> list[list[str]]:
    outputs: list[list[str]] = []
    device = next(model.parameters()).device
    for start in range(0, len(prompts), batch_size):
        chunk = list(prompts[start : start + batch_size])
        rendered = [chat_prompt(tokenizer, prompt) for prompt in chunk]
        previous_padding = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            batch = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
        finally:
            tokenizer.padding_side = previous_padding
        tensors = {name: tensor.to(device) for name, tensor in batch.items()}
        kwargs: dict[str, Any] = {
            **tensors,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "num_return_sequences": int(num_return_sequences),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if do_sample:
            kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})
        generated = model.generate(**kwargs)
        prompt_length = tensors["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
        )
        for index in range(len(chunk)):
            first = index * num_return_sequences
            outputs.append(decoded[first : first + num_return_sequences])
    return outputs


def evaluate_countdown_model(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    config: CountdownReviewerConfig,
    *,
    seed: int,
    known_structures: set[str] | None = None,
) -> dict[str, Any]:
    selected = list(rows[: min(len(rows), config.eval_examples)])
    prompts = [str(row["prompt"]) for row in selected]
    was_training = bool(model.training)
    cache_value = getattr(model.config, "use_cache", False)
    checkpointing_was_enabled = bool(getattr(model, "is_gradient_checkpointing", False))
    model.eval()
    if checkpointing_was_enabled and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    _seed_all(seed)
    try:
        with torch.no_grad():
            greedy = _generation_batches(
                model,
                tokenizer,
                prompts,
                batch_size=config.eval_batch,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                num_return_sequences=1,
            )
            sampled = _generation_batches(
                model,
                tokenizer,
                prompts,
                batch_size=config.eval_batch,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.pass_k > 1,
                temperature=config.sample_temperature,
                top_p=config.sample_top_p,
                num_return_sequences=config.pass_k,
            )
    finally:
        model.config.use_cache = cache_value
        if checkpointing_was_enabled and hasattr(
            model, "gradient_checkpointing_enable"
        ):
            model.gradient_checkpointing_enable()
        model.train(was_training)
    greedy_text = [group[0] for group in greedy]
    metrics = evaluate_response_batches(selected, greedy_text, sampled)
    if known_structures is not None:
        metrics.update(
            _pattern_metrics(selected, greedy_text, sampled, known_structures)
        )
    elif config.require_structure_metrics:
        raise RuntimeError("canonical evaluation has no structure-reference set")
    return {
        **metrics,
        "evaluation_seed": int(seed),
        "selection_metric": config.selection_metric,
        "selection_delta": config.selection_delta,
        "checkpoint_is_terminal_audit": False,
    }


def _train_one_method(
    model: Any,
    tokenizer: Any,
    config: CountdownReviewerConfig,
    encoded_replay: Sequence[CountdownTrainingItem],
    sampler_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    known_structures: set[str] | None,
    *,
    method: str,
    seed: int,
    output: Path,
    stack: HFStack,
) -> dict[str, Any]:
    coefficients = calibration["method_coefficients"]
    coefficient = float(coefficients[method])
    shared_negative_scale = float(calibration["shared_negative_scale"])
    tau = float(calibration["tau"])
    surprisal_scale = float(calibration["surprisal_scale"])
    initial_digest = _trainable_state_digest(model)
    if initial_digest != calibration["initial_trainable_state_sha256"]:
        raise RuntimeError("method model does not match the calibrated initialization")
    total_samples = config.steps * config.grad_accum * config.micro_batch
    plan = make_prompt_balanced_sampler_plan(
        sampler_rows,
        seed=seed,
        total_samples=total_samples,
    )
    parameters = _trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    warmup_steps = int(config.steps * config.warmup_ratio)
    scheduler = stack.get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps,
        config.steps,
    )
    output.mkdir(parents=True, exist_ok=False)
    metrics_rows: list[dict[str, Any]] = []
    best_value = -float("inf")
    best_step = 0
    best_checkpoint = output / "best_adapter"
    terminal_checkpoint = output / "terminal_adapter"
    last_finite_checkpoint = output / "last_finite_adapter"
    initial_metrics = evaluate_countdown_model(
        model,
        tokenizer,
        validation_rows,
        config,
        seed=config.evaluation_seed_for(seed),
        known_structures=known_structures,
    )
    metrics_rows.append({"step": 0, "method": method, **initial_metrics})
    best_value = float(initial_metrics[config.selection_metric])
    _save_adapter_checkpoint(
        model,
        tokenizer,
        best_checkpoint,
        {
            "step": 0,
            "method": method,
            "kind": "best",
            "selection_metric": config.selection_metric,
            "selection_value": best_value,
            "formal_result_claim": False,
        },
    )
    _save_adapter_checkpoint(
        model,
        tokenizer,
        last_finite_checkpoint,
        {
            "step": 0,
            "method": method,
            "kind": "last_finite",
            "formal_result_claim": False,
        },
    )
    device = next(model.parameters()).device
    plan_offset = 0
    numerical_failure: dict[str, Any] | None = None
    last_finite_step = 0
    model.train()
    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        before_update: list[torch.Tensor] | None = None
        accumulated = {
            "loss": 0.0,
            "positive_lp": 0.0,
            "weighted_negative_lp": 0.0,
            "weight_mean": 0.0,
            "distance_mean": 0.0,
        }
        try:
            for _ in range(config.grad_accum):
                packed = _selected_batch(
                    encoded_replay,
                    plan,
                    plan_offset,
                    config.micro_batch,
                    tokenizer.pad_token_id,
                )
                plan_offset += config.micro_batch
                packed_device = _packed_to_device(packed, device)
                terms = active_tail_objective_from_model(
                    model,
                    packed_device,
                    method=method,
                    coefficient=coefficient,
                    shared_negative_scale=shared_negative_scale,
                    tau=tau,
                    surprisal_scale=surprisal_scale,
                )
                loss = terms["loss"]
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite Countdown loss")
                (loss / config.grad_accum).backward()
                accumulated["loss"] += float(loss.detach()) / config.grad_accum
                accumulated["positive_lp"] += (
                    float(terms["positive_lp"].detach()) / config.grad_accum
                )
                accumulated["weighted_negative_lp"] += (
                    float(terms["effective_weighted_negative_lp"].detach())
                    / config.grad_accum
                )
                if terms["weights"].numel():
                    accumulated["weight_mean"] += (
                        float(terms["weights"].mean()) / config.grad_accum
                    )
                    accumulated["distance_mean"] += (
                        float(terms["distance"].mean()) / config.grad_accum
                    )
            raw_gradient_l2, gradients_finite = _gradient_state(parameters)
            if not gradients_finite:
                raise FloatingPointError("non-finite Countdown gradient")
            torch.nn.utils.clip_grad_norm_(
                parameters,
                config.max_grad_norm,
                error_if_nonfinite=True,
            )
            post_clip_gradient_l2, post_clip_finite = _gradient_state(parameters)
            if not post_clip_finite:
                raise FloatingPointError("non-finite post-clip Countdown gradient")
            before_update = [
                parameter.detach().float().cpu().clone() for parameter in parameters
            ]
            optimizer.step()
            optimizer_update_l2 = parameter_update_norm(before_update, parameters)
            scheduler.step()
            if not _parameters_finite(parameters):
                raise FloatingPointError("non-finite Countdown parameter")
            last_finite_step = step
        except (FloatingPointError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and "non-finite" not in str(exc).lower():
                raise
            if before_update is not None:
                _restore_parameters(before_update, parameters)
            numerical_failure = {
                "event": "nan_inf_numerical_failure",
                "step": int(step),
                "last_finite_step": int(last_finite_step),
                "error": f"{type(exc).__name__}: {exc}",
            }
            break
        row = {
            "step": step,
            "method": method,
            **accumulated,
            "raw_gradient_l2": raw_gradient_l2,
            "post_clip_gradient_l2": post_clip_gradient_l2,
            "optimizer_update_l2": optimizer_update_l2,
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        if step % config.eval_every == 0 or step == config.steps:
            evaluation = evaluate_countdown_model(
                model,
                tokenizer,
                validation_rows,
                config,
                seed=config.evaluation_seed_for(seed),
                known_structures=known_structures,
            )
            row.update(evaluation)
            value = float(evaluation[config.selection_metric])
            if value > best_value + config.selection_delta:
                best_value = value
                best_step = step
                _save_adapter_checkpoint(
                    model,
                    tokenizer,
                    best_checkpoint,
                    {
                        "step": step,
                        "method": method,
                        "kind": "best",
                        "selection_metric": config.selection_metric,
                        "selection_value": best_value,
                        "formal_result_claim": False,
                    },
                )
            model.train()
        metrics_rows.append(row)
        if step % config.checkpoint_every == 0 or step == config.steps:
            _save_adapter_checkpoint(
                model,
                tokenizer,
                last_finite_checkpoint,
                {
                    "step": step,
                    "method": method,
                    "kind": "last_finite",
                    "formal_result_claim": False,
                },
            )
    _save_adapter_checkpoint(
        model,
        tokenizer,
        terminal_checkpoint,
        {
            "step": last_finite_step,
            "method": method,
            "kind": "terminal",
            "fixed_horizon_is_convergence": False,
            "formal_result_claim": False,
        },
    )
    write_csv(output / "training_metrics.csv", metrics_rows)
    summary = {
        "experiment_id": COUNTDOWN_REVIEWER_EXPERIMENT_ID,
        "runner_version": COUNTDOWN_REVIEWER_RUNNER_VERSION,
        "method": method,
        "seed": int(seed),
        "status": "failed" if numerical_failure else "completed",
        "completed_steps": int(last_finite_step),
        "requested_steps": int(config.steps),
        "best_step": int(best_step),
        "best_validation_value": float(best_value),
        "selection_metric": config.selection_metric,
        "selection_delta": config.selection_delta,
        "evaluation_seed": config.evaluation_seed_for(seed),
        "coefficient": coefficient,
        "shared_negative_scale": shared_negative_scale,
        "tau": tau,
        "surprisal_scale": surprisal_scale,
        "initial_trainable_state_sha256": initial_digest,
        "numerical_failure": numerical_failure,
        "nan_inf_numerical_failure": numerical_failure is not None,
        "task_performance_collapse": None,
        "support_or_probability_boundary": None,
        "formal_result_claim": False,
        "method_ranking_claim_allowed": False,
        "fixed_horizon_is_convergence": False,
    }
    atomic_json(output / "TRAINING_SUMMARY.json", summary)
    if numerical_failure is not None:
        atomic_json(output / "RUN_FAILED.json", summary)
    return summary


def _release_model(model: Any) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_checkpoint_for_evaluation(
    stack: HFStack,
    config: CountdownReviewerConfig,
    checkpoint: Path,
) -> tuple[Any, Any]:
    tokenizer = _load_tokenizer(stack, config.model_path)
    model = _load_trainable_model(stack, config, adapter_path=str(checkpoint))
    return model, tokenizer


def _evaluate_saved_checkpoint(
    stack: HFStack,
    config: CountdownReviewerConfig,
    checkpoint: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    kind: str,
    known_structures: set[str] | None = None,
) -> dict[str, Any]:
    model, tokenizer = _load_checkpoint_for_evaluation(stack, config, checkpoint)
    try:
        metrics = evaluate_countdown_model(
            model,
            tokenizer,
            rows,
            config,
            seed=seed,
            known_structures=known_structures,
        )
    finally:
        _release_model(model)
    return {"checkpoint_kind": kind, **metrics}


def _aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["method"]), []).append(run)
    methods: dict[str, Any] = {}
    for method, entries in sorted(grouped.items()):
        completed = [entry for entry in entries if entry.get("status") == "completed"]
        payload: dict[str, Any] = {
            "requested_seed_count": len(entries),
            "completed_seed_count": len(completed),
            "failed_seed_count": len(entries) - len(completed),
        }
        for checkpoint_kind in ("best", "terminal"):
            metrics = [entry.get(f"{checkpoint_kind}_test") for entry in completed]
            usable = [metric for metric in metrics if isinstance(metric, Mapping)]
            if not usable:
                continue
            payload[checkpoint_kind] = {}
            for metric_name in _SELECTION_METRICS:
                values = np.asarray(
                    [float(metric[metric_name]) for metric in usable],
                    dtype=np.float64,
                )
                payload[checkpoint_kind][metric_name] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "values": values.tolist(),
                }
        methods[method] = payload
    return {
        "experiment_id": COUNTDOWN_REVIEWER_EXPERIMENT_ID,
        "runner_version": COUNTDOWN_REVIEWER_RUNNER_VERSION,
        "methods": methods,
        "formal_result_claim": False,
        "method_ranking_claim_allowed": False,
    }


def run_countdown(
    *,
    config_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    config_file = Path(config_path).expanduser().resolve()
    config = load_countdown_config(config_file)
    output = Path(output_root).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"Countdown output is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Countdown output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    stack = _load_hf_stack()
    model_identity = _validate_model_identity(config)
    adapter_identity = _adapter_identity(config)

    replay_rows = _normalize_training_rows(_read_jsonl(config.replay_path))
    calibration_rows = _normalize_training_rows(_read_jsonl(config.calibration_path))
    validation_rows = _read_jsonl(config.validation_path)
    _require_row_count(replay_rows, config.expected_replay_rows, "replay")
    _require_row_count(
        calibration_rows,
        config.expected_calibration_rows,
        "calibration",
    )
    _require_row_count(
        validation_rows,
        config.expected_validation_rows,
        "validation",
    )
    _validate_training_rows(replay_rows, "replay")
    _validate_training_rows(calibration_rows, "calibration")
    _validate_evaluation_rows(validation_rows, "validation")
    if config.require_structure_metrics:
        _validate_structure_evaluation_rows(validation_rows, "validation")
    _assert_prompt_disjoint(replay_rows, calibration_rows, "replay", "calibration")
    _assert_prompt_disjoint(replay_rows, validation_rows, "replay", "validation")
    _assert_prompt_disjoint(
        calibration_rows,
        validation_rows,
        "calibration",
        "validation",
    )

    known_structures: set[str] | None = None
    structure_reference_rows: list[dict[str, Any]] | None = None
    if config.structure_reference_path is not None:
        structure_reference_rows = _read_jsonl(config.structure_reference_path)
        _require_row_count(
            structure_reference_rows,
            config.expected_structure_reference_rows,
            "structure_reference",
        )
        known_structures = _validate_structure_reference_rows(structure_reference_rows)

    input_identity = {
        "config_sha256": _sha256_file(config_file),
        "replay_sha256": _sha256_file(config.replay_path),
        "calibration_sha256": _sha256_file(config.calibration_path),
        "validation_sha256": _sha256_file(config.validation_path),
        "structure_reference_sha256": (
            _sha256_file(config.structure_reference_path)
            if config.structure_reference_path is not None
            else None
        ),
        "test_sha256": None,
        "test_accessed_before_training": False,
    }
    manifest = {
        "experiment_id": COUNTDOWN_REVIEWER_EXPERIMENT_ID,
        "runner_version": COUNTDOWN_REVIEWER_RUNNER_VERSION,
        "scope": "reviewer_facing_countdown_training_and_evaluation",
        "config": config.as_manifest(),
        "model_identity": model_identity,
        "reference_adapter_identity": adapter_identity,
        "input_identity": input_identity,
        "output_root": str(output),
        "formal_result_claim": False,
        "method_ranking_claim_allowed": False,
        "registered_result_affecting_coordinate_bound": (
            config.protocol_id == COUNTDOWN_CANONICAL_PROTOCOL_ID
        ),
        "final_manuscript_coordinate_frozen": False,
        "reviewer_code_migration_closed": (
            config.protocol_id == COUNTDOWN_CANONICAL_PROTOCOL_ID
        ),
        "scientific_experiment_completed": False,
        "scientific_status": "pilot_not_run",
        "countdown_replaces_du1_controlled_identification": False,
    }
    atomic_json(output / "RUN_MANIFEST.json", manifest)
    all_runs: list[dict[str, Any]] = []
    tokenizer = _load_tokenizer(stack, config.model_path)
    encoded_replay, sampler_rows = _prepare_training_items(
        replay_rows, tokenizer, config.max_length
    )
    for seed in config.seeds:
        seed_root = output / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=False)
        _seed_all(seed)
        calibration_model = _load_trainable_model(stack, config)
        try:
            calibration = _calibrate_for_seed(
                calibration_model,
                tokenizer,
                config,
                calibration_rows,
                seed=seed,
            )
        except Exception as exc:
            calibration_failure = {
                "experiment_id": COUNTDOWN_REVIEWER_EXPERIMENT_ID,
                "runner_version": COUNTDOWN_REVIEWER_RUNNER_VERSION,
                "seed": int(seed),
                "status": "failed",
                "event": "calibration_failure",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "formal_result_claim": False,
            }
            atomic_json(seed_root / "CALIBRATION_FAILED.json", calibration_failure)
            for method in config.methods:
                method_failure = {**calibration_failure, "method": method}
                method_root = seed_root / method
                method_root.mkdir(parents=True, exist_ok=False)
                atomic_json(method_root / "RUN_FAILED.json", method_failure)
                all_runs.append(method_failure)
            continue
        finally:
            _release_model(calibration_model)
        atomic_json(seed_root / "CALIBRATION.json", calibration)
        for method in config.methods:
            method_root = seed_root / method
            _seed_all(seed)
            model = _load_trainable_model(stack, config)
            try:
                summary = _train_one_method(
                    model,
                    tokenizer,
                    config,
                    encoded_replay,
                    sampler_rows,
                    validation_rows,
                    calibration,
                    known_structures,
                    method=method,
                    seed=seed,
                    output=method_root,
                    stack=stack,
                )
            except Exception as exc:
                failure = {
                    "experiment_id": COUNTDOWN_REVIEWER_EXPERIMENT_ID,
                    "runner_version": COUNTDOWN_REVIEWER_RUNNER_VERSION,
                    "method": method,
                    "seed": int(seed),
                    "status": "failed",
                    "event": "runtime_failure",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "task_performance_collapse": None,
                    "support_or_probability_boundary": None,
                    "nan_inf_numerical_failure": isinstance(exc, FloatingPointError),
                    "formal_result_claim": False,
                }
                method_root.mkdir(parents=True, exist_ok=True)
                atomic_json(method_root / "RUN_FAILED.json", failure)
                summary = failure
            finally:
                _release_model(model)
            all_runs.append(dict(summary))

    test_rows: list[dict[str, Any]] | None = None
    test_input_failure: dict[str, Any] | None = None
    if config.test_path is not None:
        try:
            test_rows = _read_jsonl(config.test_path)
            _require_row_count(test_rows, config.expected_test_rows, "test")
            _validate_evaluation_rows(test_rows, "test")
            if config.require_structure_metrics:
                _validate_structure_evaluation_rows(test_rows, "test")
            _assert_prompt_disjoint(replay_rows, test_rows, "replay", "test")
            _assert_prompt_disjoint(
                calibration_rows,
                test_rows,
                "calibration",
                "test",
            )
            _assert_prompt_disjoint(
                validation_rows,
                test_rows,
                "validation",
                "test",
            )
            input_identity["test_sha256"] = _sha256_file(config.test_path)
        except Exception as exc:
            test_rows = None
            test_input_failure = {
                "event": "environment_invalid_or_evaluation_input_unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "training_outputs_preserved": True,
                "formal_result_claim": False,
            }
            atomic_json(output / "TEST_INPUT_FAILED.json", test_input_failure)
    if test_rows is not None:
        for run in all_runs:
            if run.get("status") != "completed":
                continue
            seed = int(run["seed"])
            method = str(run["method"])
            method_root = output / f"seed_{seed}" / method
            evaluation_seed = config.evaluation_seed_for(seed)
            try:
                best = _evaluate_saved_checkpoint(
                    stack,
                    config,
                    method_root / "best_adapter",
                    test_rows,
                    seed=evaluation_seed,
                    kind="best",
                    known_structures=known_structures,
                )
                terminal = _evaluate_saved_checkpoint(
                    stack,
                    config,
                    method_root / "terminal_adapter",
                    test_rows,
                    seed=evaluation_seed,
                    kind="terminal",
                    known_structures=known_structures,
                )
            except Exception as exc:
                run["training_status"] = "completed"
                run["status"] = "failed"
                run["event"] = "test_evaluation_failure"
                run["evaluation_error"] = f"{type(exc).__name__}: {exc}"
                run["evaluation_traceback"] = traceback.format_exc()
                atomic_json(method_root / "RUN_FAILED.json", run)
                continue
            run["best_test"] = best
            run["terminal_test"] = terminal
            atomic_json(method_root / "BEST_TEST.json", best)
            atomic_json(method_root / "TERMINAL_TEST.json", terminal)
            atomic_json(method_root / "RUN_COMPLETE.json", run)
    elif test_input_failure is not None:
        for run in all_runs:
            if run.get("status") != "completed":
                continue
            seed = int(run["seed"])
            method = str(run["method"])
            method_root = output / f"seed_{seed}" / method
            run["training_status"] = "completed"
            run["status"] = "failed"
            run["event"] = test_input_failure["event"]
            run["evaluation_error"] = test_input_failure["error"]
            atomic_json(method_root / "RUN_FAILED.json", run)
    else:
        for run in all_runs:
            if run.get("status") != "completed":
                continue
            seed = int(run["seed"])
            method = str(run["method"])
            method_root = output / f"seed_{seed}" / method
            atomic_json(method_root / "RUN_COMPLETE.json", run)
    manifest["input_identity"] = input_identity
    manifest["test_accessed_after_training"] = bool(config.test_path is not None)
    manifest["test_input_failure"] = test_input_failure
    atomic_json(output / "RUN_MANIFEST.json", manifest)
    aggregate = _aggregate_runs(all_runs)
    aggregate["protocol_id"] = config.protocol_id
    aggregate["reviewer_code_migration_closed"] = (
        config.protocol_id == COUNTDOWN_CANONICAL_PROTOCOL_ID
    )
    atomic_json(output / "SUMMARY.json", aggregate)
    suite_complete = all(run.get("status") == "completed" for run in all_runs)
    completion = {
        "experiment_id": COUNTDOWN_REVIEWER_EXPERIMENT_ID,
        "protocol_id": config.protocol_id,
        "runner_version": COUNTDOWN_REVIEWER_RUNNER_VERSION,
        "status": "completed" if suite_complete else "partial_failure",
        "requested_runs": len(config.methods) * len(config.seeds),
        "completed_runs": sum(run.get("status") == "completed" for run in all_runs),
        "failed_runs": sum(run.get("status") != "completed" for run in all_runs),
        "test_configured": config.test_path is not None,
        "test_input_failure": test_input_failure,
        "reviewer_code_migration_closed": (
            config.protocol_id == COUNTDOWN_CANONICAL_PROTOCOL_ID
        ),
        "scientific_experiment_completed": False,
        "scientific_status": "pilot_not_run",
        "formal_result_claim": False,
        "method_ranking_claim_allowed": False,
        "terminal_scientific_audit_included": False,
    }
    atomic_json(output / "RUN_COMPLETE.json", completion)
    return {"completion": completion, "summary": aggregate, "runs": all_runs}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="drpo-reference countdown",
        description="Run the reviewer-facing Countdown training lifecycle.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_countdown(config_path=args.config, output_root=args.output)
    return 0


__all__ = [
    "COUNTDOWN_CANONICAL_PROTOCOL_ID",
    "COUNTDOWN_REVIEWER_EXPERIMENT_ID",
    "COUNTDOWN_REVIEWER_RUNNER_VERSION",
    "CountdownReviewerConfig",
    "evaluate_countdown_model",
    "load_countdown_config",
    "main",
    "run_countdown",
]


if __name__ == "__main__":
    raise SystemExit(main())
