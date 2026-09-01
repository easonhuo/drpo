"""Config interpretation and fail-closed validation for E8 sweep orchestration.

This module deliberately does not implement a loss, trainer, model forward pass, or
scientific update.  Reviewed YAML defines a new cold-start-family experiment;
validation here is limited to schema/type/range checks, implementation capability,
self-consistency, historical-ID immutability, and repository-path safety.
"""

from __future__ import annotations

import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from drpo.e8_multitask_tasks import TASK_NAMES, stable_hash

RHO_EXPERIMENT_ID = "EXT-C-E8-MULTITASK-EXP-TUNING-01"
DENSE_EXPERIMENT_ID = "EXT-C-E8-MULTITASK-EXP-LAMBDA-DENSE-01"
COLDSTART_EXPERIMENT_ID = "EXT-C-E8-MULTITASK-EXP-COLDSTART-01"
LAMBDA_COMPLETION_EXPERIMENT_ID = "EXT-C-E8-MULTITASK-EXP-LAMBDA-COMPLETION-01"
LAMBDA_CURVE_COMPLETION_EXPERIMENT_ID = "EXT-C-E8-MULTITASK-EXP-LAMBDA-CURVE-COMPLETION-02"
P0_EXPERIMENT_ID = "EXT-C-E8-MULTITASK-P0-01"

SWEEP_PROFILE_RHO = "nine_task_rho_v1"
SWEEP_PROFILE_DENSE = "task_lambda_dense_v1"
SWEEP_PROFILE_COLDSTART = "eight_task_coldstart_lambda_v1"

HISTORICAL_CONFIG_IDENTITIES = {
    COLDSTART_EXPERIMENT_ID: (
        "configs/e8_multitask_exp_coldstart.yaml",
        "b02d2f64e19d8a169e7b5c8e3116826608d6502f",
    ),
    LAMBDA_COMPLETION_EXPERIMENT_ID: (
        "configs/e8_multitask_exp_lambda_completion.yaml",
        "4ee383012fd38a3e4a5fee002030f2fe9a86adac",
    ),
    LAMBDA_CURVE_COMPLETION_EXPERIMENT_ID: (
        "configs/e8_multitask_exp_lambda_curve_completion.yaml",
        "118759c719df94de0583ecf4f831f8d63629925d",
    ),
}

CANONICAL_COLDSTART_BLOB_SHAS = {
    "arena": "d8a04f3ae3edd08042aa1004b4cbf927fc5cea72",
    "scan_common": "572f6ad98bf063c88e52a4594fde892842c4fe15",
    "scan_runtime": "b4ad8581f0afd6e4d24069524f909eaa1b0c9563",
    "scan_trainer": "e026afbefc09205bb1632b5dd1bd6db33b5df358",
    "paper_common": "720415583e9e372fafa5aa3520e07de04e6494d8",
    "paper_runtime": "a57cd88287daf95864f7e30caf658473a32d3602",
    "base_config": "10f27f32719298376bdc7be7e01023626c6ad3f8",
    "round1_grid": "e6d70895ad9e4caceb029425fbed523b8530c2d3",
    "extension_grid": "e7657ef7c8fbb1ae81e0a7b9dd6f4b9cea32262d",
    "bank_generator": "545119fdf8e560b5e81a862e1e48134dd52ac869",
    "bank_config": "d1873efae15c778d2472a927206d8620aa43be71",
    "bank_converter": "a935e2d721b06437568556040475736bbf45ceee",
    "p0_bank_pipeline": "3482967fe656156500f4598f16f5e7031e198d48",
    "p0_task_adapters": "454f3076171ee25636109c33f5a177ee2201b5f8",
    "p0_config": "14605ae3a79f18e435feafd3927bc21485edbbc9",
    "p0_launcher": "ffcad2a64cb2f42906cae67dabdcc98c3eb46ff0",
    "result_reference": "972a67867aafb5ddea6e1625bacd337b6939f097",
}

CANONICAL_COLDSTART_PATHS = {
    "arena": "src/drpo/countdown_qwen_arena_onefile.py",
    "scan_common": "src/drpo/countdown_e8_alpha1_c_scan_common.py",
    "scan_runtime": "src/drpo/countdown_e8_alpha1_c_scan_runtime.py",
    "scan_trainer": "src/drpo/countdown_e8_alpha1_c_scan_trainer.py",
    "paper_common": "src/drpo/countdown_e8_alpha1_highc_scan_common.py",
    "paper_runtime": "src/drpo/countdown_e8_alpha1_highc_scan_runtime.py",
    "base_config": "configs/countdown_e8_base_rl_replay_0p5b.yaml",
    "round1_grid": "configs/countdown_e8_oracle_offline_v2_alpha1_highc_scan_0p5b.yaml",
    "extension_grid": "configs/countdown_e8_oracle_offline_v2_linear_c_extension_0p5b.yaml",
    "bank_generator": "src/drpo/countdown_e8_oracle_bank_v2.py",
    "bank_config": "configs/countdown_e8_oracle_offline_bank_v2_0p5b.yaml",
    "bank_converter": "scripts/v2_bank_convert.py",
    "p0_bank_pipeline": "src/drpo/e8_multitask_p0.py",
    "p0_task_adapters": "src/drpo/e8_multitask_tasks.py",
    "p0_config": "configs/e8_multitask_p0.yaml",
    "p0_launcher": "scripts/run_e8_multitask_p0.sh",
    "result_reference": (
        "experiments/results/e8_paper_aligned_linear_scan_round1_pilot/RESULT_SUMMARY.json"
    ),
}

COUNTDOWN_PAPER_COEFFICIENTS = frozenset(
    (
        0.051293294,
        0.105360516,
        0.162518929,
        0.223143551,
        0.287682072,
        0.430782916,
        0.693147181,
        0.916290732,
        1.203972804,
        1.386294361,
        1.609437912,
        1.897119985,
        2.302585093,
        2.995732274,
        0.01,
        0.025,
        0.04,
        3.506557897,
        4.605170186,
        5.298317367,
        6.907755279,
        9.210340372,
    )
)
COUNTDOWN_PAPER_SEEDS = frozenset((4000, 5000))
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def experiment_id(config: Mapping[str, Any]) -> str:
    value = config.get("experiment_id")
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError("experiment_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    return value


def sweep_profile(config: Mapping[str, Any]) -> str:
    return str(config.get("sweep", {}).get("profile", SWEEP_PROFILE_RHO))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")  # noqa: TRY004
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")  # noqa: TRY004
    return value


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")  # noqa: TRY004
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite numeric scalar")  # noqa: TRY004
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite numeric scalar")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")  # noqa: TRY004
    return value


def task_lambdas(config: Mapping[str, Any], task: str) -> tuple[float, ...]:
    raw = _sequence(config["sweep"]["task_lambda"][task], f"{task} lambda grid")
    values = tuple(_number(value, f"{task} lambda value") for value in raw)
    if any(value <= 0.0 for value in values):
        raise ValueError(f"{task} lambda values must be strictly positive")
    return values


def _validate_scalar_types(config: Mapping[str, Any]) -> None:
    integer_fields = {
        "reference": ("optimizer_updates", "validation_rows_seen", "test_rows_seen"),
        "initialization": ("optimizer_updates", "seed"),
        "model": ("lora_rank", "lora_alpha", "max_length", "max_new_tokens"),
        "split": (
            "p0_train_rows",
            "p0_validation_rows",
            "p0_test_rows",
            "countdown_train_rows",
            "countdown_validation_rows",
            "hash_seed",
        ),
        "training": (
            "optimizer_updates",
            "micro_batch",
            "gradient_accumulation",
            "evaluation_every_updates",
        ),
        "evaluation": (
            "greedy_prompt_rows",
            "passk_prompt_rows",
            "pass_k",
            "batch_size",
            "max_new_tokens",
            "generation_seed",
        ),
        "negative_sampling": ("negatives_per_prompt",),
    }
    for section, fields in integer_fields.items():
        values = _mapping(config.get(section), section)
        for field in fields:
            _integer(values.get(field), f"{section}.{field}")

    for section, fields in {
        "model": ("lora_dropout",),
        "training": ("learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm"),
        "evaluation": ("sampling_temperature", "top_p"),
    }.items():
        values = _mapping(config.get(section), section)
        for field in fields:
            _number(values.get(field), f"{section}.{field}")

    for section, fields in {
        "parent": ("qualified_banks_required",),
        "initialization": ("external_adapter_allowed", "deterministic_fresh_lora"),
        "model": ("gradient_checkpointing",),
        "split": ("test_access_allowed", "countdown_subsampling_forbidden"),
        "training": ("early_stopping", "terminal_adapter_required"),
        "negative_sampling": (
            "near_far_selection",
            "selection_stop_gradient",
            "weight_sum_normalization",
            "gradient_budget_matching",
        ),
        "remoteness_calibration": (
            "enabled",
            "detached",
            "extra_square",
            "gradient_rms_matching",
        ),
        "selection": ("finite_required", "report_grid_edge"),
    }.items():
        values = _mapping(config.get(section), section)
        for field in fields:
            _boolean(values.get(field), f"{section}.{field}")

    training = config["training"]
    late = tuple(
        _integer(value, "training.late_window_updates entry", positive=True)
        for value in _sequence(training.get("late_window_updates"), "training.late_window_updates")
    )
    if not late or tuple(sorted(set(late))) != late:
        raise ValueError("training.late_window_updates must be sorted and unique")
    if late[-1] > training["optimizer_updates"]:
        raise ValueError("training.late_window_updates may not exceed optimizer_updates")
    optimizer_updates = _integer(
        training["optimizer_updates"], "training.optimizer_updates", positive=True
    )
    evaluation_every = _integer(
        training["evaluation_every_updates"],
        "training.evaluation_every_updates",
        positive=True,
    )
    if any(update != optimizer_updates and update % evaluation_every != 0 for update in late):
        raise ValueError(
            "training.late_window_updates must fall on configured evaluation updates "
            "or the terminal optimizer update"
        )

    model = config["model"]
    for field in ("lora_rank", "lora_alpha", "max_length", "max_new_tokens"):
        _integer(model[field], f"model.{field}", positive=True)
    dropout = float(model["lora_dropout"])
    if not 0.0 <= dropout < 1.0:
        raise ValueError("model.lora_dropout must be in [0, 1)")

    evaluation = config["evaluation"]
    tuple(
        _integer(value, "evaluation.auxiliary_pass_ks entry", positive=True)
        for value in _sequence(evaluation.get("auxiliary_pass_ks"), "evaluation.auxiliary_pass_ks")
    )
    if not 0.0 < float(evaluation["top_p"]) <= 1.0:
        raise ValueError("evaluation.top_p must be in (0, 1]")
    if float(evaluation["sampling_temperature"]) <= 0.0:
        raise ValueError("evaluation.sampling_temperature must be positive")

    if int(config["training"]["optimizer_updates"]) <= 0:
        raise ValueError("training.optimizer_updates must be positive")
    for field in ("micro_batch", "gradient_accumulation", "evaluation_every_updates"):
        if int(config["training"][field]) <= 0:
            raise ValueError(f"training.{field} must be positive")
    for field in (
        "greedy_prompt_rows",
        "passk_prompt_rows",
        "pass_k",
        "batch_size",
        "max_new_tokens",
    ):
        if int(config["evaluation"][field]) <= 0:
            raise ValueError(f"evaluation.{field} must be positive")
    if float(config["training"]["learning_rate"]) <= 0.0:
        raise ValueError("training.learning_rate must be positive")
    if not 0.0 <= float(config["training"]["warmup_ratio"]) < 1.0:
        raise ValueError("training.warmup_ratio must be in [0, 1)")
    if float(config["training"]["weight_decay"]) < 0.0:
        raise ValueError("training.weight_decay must be non-negative")
    if float(config["training"]["max_grad_norm"]) <= 0.0:
        raise ValueError("training.max_grad_norm must be positive")


def _validate_implementation_contract(config: Mapping[str, Any]) -> None:
    if config.get("parent", {}).get("experiment_id") != P0_EXPERIMENT_ID:
        raise ValueError("Cold-start parent must remain the P0 experiment")
    if config["parent"].get("qualified_banks_required") is not True:
        raise ValueError("Cold-start requires qualified P0 banks")

    reference = config["reference"]
    if (
        reference.get("checkpoint_kind") != "fresh_lora_from_base_model"
        or reference.get("optimizer_updates") != 0
        or reference.get("validation_rows_seen") != 0
        or reference.get("test_rows_seen") != 0
    ):
        raise ValueError("Cold-start reference must remain a zero-update fresh LoRA")

    initialization = config["initialization"]
    if (
        initialization.get("source") != "base_model"
        or initialization.get("optimizer_updates") != 0
        or initialization.get("external_adapter_allowed") is not False
        or initialization.get("deterministic_fresh_lora") is not True
    ):
        raise ValueError(
            "Cold-start initialization must remain zero-update deterministic fresh LoRA"
        )

    model = config["model"]
    if (
        model.get("base_model") != "Qwen/Qwen2.5-0.5B-Instruct"
        or model.get("revision") != "7ae557604adf67be50417f59c2c2f167def9a775"
        or model.get("parameterization") != "lora"
        or model.get("dtype") != "auto"
    ):
        raise ValueError("Cold-start model/parameterization is not implemented by this runner")

    if config["split"].get("test_access_allowed") is not False:
        raise ValueError("Tuning must forbid test access")
    if config["split"].get("countdown_subsampling_forbidden") is not True:
        raise ValueError("Countdown subsampling is not implemented by this cold-start family")
    if config["training"].get("early_stopping") is not False:
        raise ValueError("Early stopping is not implemented by this cold-start runner")
    if config["training"].get("terminal_adapter_required") is not True:
        raise ValueError("Cold-start requires a terminal adapter")

    evaluation = config["evaluation"]
    if (
        evaluation.get("primary_checkpoint_policy") != "late_window_and_terminal"
        or evaluation.get("best_checkpoint_role") != "supplementary_only"
    ):
        raise ValueError("Cold-start checkpoint-selection mode is not implemented")

    negative = config["negative_sampling"]
    if (
        negative.get("negatives_per_prompt") != 16
        or negative.get("consumer") != "all_unique_negatives_per_prompt"
        or negative.get("deduplicate_rule") != "first_canonical_completion_occurrence"
        or negative.get("denominator") != "unique_negative_count_per_prompt"
        or negative.get("near_far_selection") is not False
        or negative.get("selection_stop_gradient") is not True
        or negative.get("weight_sum_normalization") is not False
        or negative.get("gradient_budget_matching") is not False
    ):
        raise ValueError("Cold-start negative-consumer mode is not implemented")

    reference_bank = _mapping(
        negative.get("reference_remoteness_bank"),
        "negative_sampling.reference_remoteness_bank",
    )
    expected_reference_bank = {
        "enabled": True,
        "scope": "non_countdown_training_only",
        "source_candidates": "all_deterministic_verified_wrong_mutations",
        "reference_policy": "zero_update_base_plus_fresh_lora",
        "coordinate": "mean_completion_token_surprisal",
        "selected_negatives_per_prompt": 16,
        "selection": "source_p0_error_class_sequence_then_within_class_reference_rank_spread",
        "coverage_threshold": None,
        "reference_rank_role": "provenance_and_diagnostic_only",
        "static_reference_rank_enters_training_weight": False,
        "current_policy_surprisal_recomputed_each_update": True,
        "original_p0_bank_preserved": True,
        "audit_quantiles": ["min", "q25", "median", "q75", "max"],
    }
    if dict(reference_bank) != expected_reference_bank:
        raise ValueError("Reference-remoteness bank mode is not implemented")

    calibration = config["remoteness_calibration"]
    if (
        calibration.get("enabled") is not False
        or calibration.get("mode") != "paper_linear_surprisal_no_calibration"
        or calibration.get("coordinate") != "current_sequence_surprisal_div_2"
        or calibration.get("detached") is not True
        or calibration.get("extra_square") is not False
        or calibration.get("gradient_rms_matching") is not False
    ):
        raise ValueError("Cold-start remoteness-calibration mode is not implemented")


def _validate_task_runtime(config: Mapping[str, Any], tasks: tuple[str, ...]) -> None:
    runtime = _mapping(config.get("task_runtime"), "task_runtime")
    whitelist = (
        "model.max_length",
        "model.max_new_tokens",
        "evaluation.batch_size",
        "evaluation.greedy_prompt_rows",
        "evaluation.passk_prompt_rows",
        "evaluation.pass_ks",
    )
    if tuple(runtime.get("override_whitelist", ())) != whitelist or set(runtime) != {
        "override_whitelist",
        *tasks,
    }:
        raise ValueError("task_runtime must use only the implemented override fields")
    expected_fields = {
        "max_length",
        "max_new_tokens",
        "evaluation_batch_size",
        "greedy_prompt_rows",
        "passk_prompt_rows",
        "auxiliary_pass_ks",
    }
    for task in tasks:
        values = _mapping(runtime[task], f"task_runtime.{task}")
        if set(values) != expected_fields:
            raise ValueError(f"task_runtime.{task} contains an unsupported field")
        for field in expected_fields - {"auxiliary_pass_ks"}:
            _integer(values[field], f"task_runtime.{task}.{field}", positive=True)
        tuple(
            _integer(value, f"task_runtime.{task}.auxiliary_pass_ks entry", positive=True)
            for value in _sequence(
                values["auxiliary_pass_ks"], f"task_runtime.{task}.auxiliary_pass_ks"
            )
        )


def is_historical_coldstart_config(config: Mapping[str, Any]) -> bool:
    return experiment_id(config) in HISTORICAL_CONFIG_IDENTITIES


def effective_coldstart_runtime(config: Mapping[str, Any], task: str) -> dict[str, Any]:
    """Resolve the values that the canonical cold-start runtime must actually consume."""

    tasks = tuple(str(value) for value in config.get("suite", {}).get("tasks", ()))
    if task not in tasks:
        raise ValueError(f"Unknown cold-start task: {task}")
    runtime = _mapping(config["task_runtime"][task], f"task_runtime.{task}")
    model = _mapping(config["model"], "model")
    training = _mapping(config["training"], "training")
    evaluation = _mapping(config["evaluation"], "evaluation")
    auxiliary = [int(value) for value in runtime["auxiliary_pass_ks"]]
    pass_k = int(evaluation["pass_k"])
    return {
        "initialization_seed": int(config["initialization"]["seed"]),
        "model": {
            "parameterization": str(model["parameterization"]),
            "dtype": str(model["dtype"]),
            "lora_rank": int(model["lora_rank"]),
            "lora_alpha": int(model["lora_alpha"]),
            "lora_dropout": float(model["lora_dropout"]),
            "gradient_checkpointing": bool(model["gradient_checkpointing"]),
            "max_length": int(runtime["max_length"]),
            "max_new_tokens": int(runtime["max_new_tokens"]),
        },
        "training": {
            "optimizer_updates": int(training["optimizer_updates"]),
            "micro_batch": int(training["micro_batch"]),
            "gradient_accumulation": int(training["gradient_accumulation"]),
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "warmup_ratio": float(training["warmup_ratio"]),
            "max_grad_norm": float(training["max_grad_norm"]),
            "evaluation_every_updates": int(training["evaluation_every_updates"]),
            "late_window_updates": [int(value) for value in training["late_window_updates"]],
        },
        "evaluation": {
            "examples": max(int(runtime["greedy_prompt_rows"]), int(runtime["passk_prompt_rows"])),
            "batch_size": int(runtime["evaluation_batch_size"]),
            "greedy_prompt_rows": int(runtime["greedy_prompt_rows"]),
            "passk_prompt_rows": int(runtime["passk_prompt_rows"]),
            "pass_k": pass_k,
            "auxiliary_pass_ks": auxiliary,
            "pass_ks": [pass_k, *auxiliary],
            "sampling_temperature": float(evaluation["sampling_temperature"]),
            "top_p": float(evaluation["top_p"]),
            "generation_seed": int(evaluation["generation_seed"]),
            "max_new_tokens": int(runtime["max_new_tokens"]),
        },
    }


def _validate_runtime_authority_consistency(
    config: Mapping[str, Any], tasks: tuple[str, ...]
) -> None:
    runtime = config["task_runtime"]
    countdown = runtime["countdown"]
    model = config["model"]
    evaluation = config["evaluation"]
    expected_pairs = (
        (model["max_length"], countdown["max_length"], "model.max_length"),
        (model["max_new_tokens"], countdown["max_new_tokens"], "model.max_new_tokens"),
        (
            evaluation["max_new_tokens"],
            countdown["max_new_tokens"],
            "evaluation.max_new_tokens",
        ),
        (
            evaluation["batch_size"],
            countdown["evaluation_batch_size"],
            "evaluation.batch_size",
        ),
        (
            evaluation["greedy_prompt_rows"],
            countdown["greedy_prompt_rows"],
            "evaluation.greedy_prompt_rows",
        ),
        (
            evaluation["passk_prompt_rows"],
            countdown["passk_prompt_rows"],
            "evaluation.passk_prompt_rows",
        ),
    )
    for configured, task_value, label in expected_pairs:
        if int(configured) != int(task_value):
            raise ValueError(f"{label} must match task_runtime.countdown")
    if tuple(int(value) for value in evaluation["auxiliary_pass_ks"]) != tuple(
        int(value) for value in countdown["auxiliary_pass_ks"]
    ):
        raise ValueError("evaluation.auxiliary_pass_ks must match task_runtime.countdown")
    if int(countdown["greedy_prompt_rows"]) != int(countdown["passk_prompt_rows"]):
        raise ValueError(
            "Countdown canonical evaluator currently requires equal greedy/pass-k prompt budgets"
        )
    if int(evaluation["pass_k"]) != 8:
        raise ValueError("Cold-start canonical reporting currently implements pass_k=8")
    for task in tasks:
        auxiliary = tuple(int(value) for value in runtime[task]["auxiliary_pass_ks"])
        if len(set(auxiliary)) != len(auxiliary) or any(value != 64 for value in auxiliary):
            raise ValueError(
                f"task_runtime.{task}.auxiliary_pass_ks currently supports only optional pass@64"
            )


def _validate_sweep(config: Mapping[str, Any], tasks: tuple[str, ...]) -> None:
    sweep = _mapping(config.get("sweep"), "sweep")
    if sweep.get("method") != "exponential":
        raise ValueError("Cold-start sweep.method must be exponential")
    if sweep.get("parameterization") not in ("paper_coefficient_c", "paper_lambda_c1"):
        raise ValueError("Unsupported cold-start parameterization")
    task_lambda = _mapping(sweep.get("task_lambda"), "sweep.task_lambda")
    if set(task_lambda) != set(tasks):
        raise ValueError("Cold-start task_lambda must contain the exact nine tasks")

    countdown_values = task_lambdas(config, "countdown")
    if len(set(countdown_values)) != len(countdown_values):
        raise ValueError("Countdown coefficient grid contains duplicates")
    if any(value not in COUNTDOWN_PAPER_COEFFICIENTS for value in countdown_values):
        raise ValueError("Countdown coefficients exceed the implemented paper-worker domain")

    countdown_seeds = tuple(
        _integer(value, "Countdown seed offset")
        for value in _sequence(sweep.get("countdown_seed_offsets", ()), "countdown_seed_offsets")
    )
    if len(set(countdown_seeds)) != len(countdown_seeds):
        raise ValueError("Countdown seed offsets must be unique")
    if any(seed not in COUNTDOWN_PAPER_SEEDS for seed in countdown_seeds):
        raise ValueError("Countdown seed exceeds the implemented paper-worker domain")
    if countdown_seeds and not countdown_values:
        raise ValueError("Scheduled Countdown seeds require a non-empty coefficient grid")

    countdown_positive = _boolean(
        sweep.get("countdown_include_positive_only", True),
        "countdown_include_positive_only",
    )
    include_global = _boolean(
        sweep.get("include_global_endpoint", False),
        "include_global_endpoint",
    )
    positive_seeds = tuple(
        _integer(value, "Transfer Positive-only seed offset")
        for value in _sequence(
            sweep.get("transfer_positive_only_seed_offsets", ()),
            "transfer_positive_only_seed_offsets",
        )
    )
    if len(set(positive_seeds)) != len(positive_seeds):
        raise ValueError("Transfer Positive-only seed offsets must be unique")
    _integer(sweep.get("task_transfer_seed_offset"), "task_transfer_seed_offset")
    _integer(sweep.get("tuning_seed"), "tuning_seed")

    transfer_tasks = set(tasks) - {"countdown"}
    hashes = _mapping(sweep.get("task_grid_hashes"), "task_grid_hashes")
    provenance = _mapping(sweep.get("task_grid_provenance"), "task_grid_provenance")
    if set(hashes) != transfer_tasks or set(provenance) != transfer_tasks:
        raise ValueError("Task-grid hash/provenance maps must cover the exact eight transfer tasks")

    active_transfer: list[str] = []
    for task in transfer_tasks:
        values = task_lambdas(config, task)
        if len(set(values)) != len(values):
            raise ValueError(f"{task} lambda grid contains duplicates")
        if stable_hash(list(values)) != str(hashes[task]):
            raise ValueError(f"{task} lambda grid does not match task_grid_hashes")
        if not str(provenance[task]).strip():
            raise ValueError(f"{task} task-grid provenance must be non-empty")
        if values:
            active_transfer.append(task)

    expanded = len(countdown_seeds) * (1 + int(countdown_positive) + len(countdown_values)) + sum(
        len(positive_seeds) + int(include_global) + len(task_lambdas(config, task))
        for task in active_transfer
    )
    expected = _integer(sweep.get("expected_cells"), "sweep.expected_cells")
    if expanded <= 0:
        raise ValueError("Cold-start sweep must schedule at least one scientific cell")
    if expected != expanded:
        raise ValueError("sweep.expected_cells must match the expanded scientific matrix")

    reporting = _mapping(config.get("reporting"), "reporting")
    if tuple(reporting.get("separate_events", ())) != (
        "task_performance",
        "valid_or_structure_diagnostic",
        "nan_inf_numerical_failure",
    ):
        raise ValueError("reporting must preserve the three terminal event classes")
    if reporting.get("other_tasks_task_interface_adaptation_only") is not True:
        raise ValueError("reporting must preserve task-interface-only adaptation")
    if reporting.get("transfer_exp_scope") != "single_seed_response_shape_localization":
        raise ValueError("Unsupported transfer reporting scope")
    if reporting.get("positive_only_seed_count_per_transfer_task") != len(positive_seeds):
        raise ValueError("reporting Positive-only seed count must match configured seeds")
    for field in (
        "convergence_claim_allowed",
        "significance_claim_allowed",
        "method_ranking_allowed",
    ):
        if reporting.get(field) is not False:
            raise ValueError(f"reporting.{field} must remain false")
    expected_role = (
        "diagnostic_regression_sentinel_not_result_gate"
        if countdown_seeds
        else "predecessor_only_no_new_scientific_cells"
    )
    if reporting.get("countdown_role") != expected_role:
        raise ValueError("reporting Countdown role does not match scheduled cells")
    role = reporting.get("scientific_role")
    if role is not None and (not isinstance(role, str) or not role.startswith("external_validity")):
        raise ValueError("reporting.scientific_role must remain external validity")
    causal = reporting.get("causal_identification_environment")
    if causal is not None and causal != "D-U1":
        raise ValueError("reporting causal-identification pointer may only be D-U1")


def _validate_canonical_and_execution(config: Mapping[str, Any]) -> None:
    canonical = _mapping(config.get("canonical_coldstart"), "canonical_coldstart")
    if canonical.get("paths") != CANONICAL_COLDSTART_PATHS:
        raise ValueError("Cold-start canonical source paths are not implemented")
    if canonical.get("expected_git_blob_shas") != CANONICAL_COLDSTART_BLOB_SHAS:
        raise ValueError("Cold-start canonical source identities drifted")
    if (
        canonical.get("scientific_kernel") != "import_only_no_loss_reimplementation"
        or canonical.get("initialization") != "qwen_pretrained_base_plus_fresh_lora"
        or canonical.get("formula")
        != "alpha_times_exp_minus_c_times_current_sequence_surprisal_div_2"
        or canonical.get("countdown_entry") != "countdown_e8_alpha1_highc_scan_runtime.worker"
        or canonical.get("transfer_entry") != "countdown_e8_alpha1_c_scan_trainer.train_cell"
    ):
        raise ValueError("Cold-start canonical trainer/dispatch contract drifted")

    selection = config["selection"]
    if (
        selection.get("primary_metric") != "validation_late_window_pass8_mean"
        or selection.get("finite_required") is not True
        or selection.get("report_grid_edge") is not True
        or selection.get("terminal_valid_rate_role") != "diagnostic_only_not_selection_eligibility"
        or tuple(selection.get("tie_breakers", ()))
        != (
            "validation_terminal_pass8",
            "validation_late_window_greedy_mean",
            "smaller_lambda",
        )
    ):
        raise ValueError("Cold-start selection mode is not implemented")

    execution = _mapping(config.get("execution"), "execution")
    if _integer(execution.get("max_concurrent_cells"), "execution.max_concurrent_cells") != 16:
        raise ValueError("Cold-start scheduler currently implements exactly 16 slots")
    gpu_ids = tuple(
        _integer(value, "execution.gpu_ids entry")
        for value in _sequence(execution.get("gpu_ids"), "execution.gpu_ids")
    )
    if (
        gpu_ids != tuple(range(8))
        or _integer(execution.get("slots_per_gpu"), "execution.slots_per_gpu") != 2
    ):
        raise ValueError("Cold-start runner currently implements 8 GPUs x 2 slots")
    if execution.get("scheduler") != "dynamic_slot_queue":
        raise ValueError("Cold-start runner requires dynamic_slot_queue")
    safety = {
        "wave_barriers": False,
        "identity_checked_resume": True,
        "retry_incomplete_requires_explicit_flag": True,
        "fail_closed": True,
        "test_partition_forbidden": True,
    }
    for field, expected in safety.items():
        if _boolean(execution.get(field), f"execution.{field}") is not expected:
            raise ValueError(f"execution.{field} violates the recovery safety contract")
    if execution.get("oom_policy") != "fail_cell_no_automatic_scientific_parameter_mutation":
        raise ValueError("Cold-start OOM policy may not mutate scientific parameters")


def validate_coldstart_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version: 1")
    current_id = experiment_id(config)
    if sweep_profile(config) != SWEEP_PROFILE_COLDSTART:
        raise ValueError("validate_coldstart_config requires the cold-start profile")
    if current_id in (RHO_EXPERIMENT_ID, DENSE_EXPERIMENT_ID):
        raise ValueError("RHO/DENSE historical IDs may not be reused for cold-start")

    tasks = tuple(str(task) for task in config.get("suite", {}).get("tasks", ()))
    expected = set(TASK_NAMES)
    if len(tasks) != 9 or len(set(tasks)) != 9 or set(tasks) != expected:
        raise ValueError("Cold-start suite must be Countdown plus the exact eight P0 tasks")
    if set(config["suite"].get("p0_tasks", ())) != expected - {"countdown"}:
        raise ValueError("Cold-start suite.p0_tasks must be the exact eight P0 tasks")
    if tuple(config["suite"].get("external_tasks", ())) != ("countdown",):
        raise ValueError("Countdown must be the only cold-start external task")

    _validate_scalar_types(config)
    _validate_implementation_contract(config)
    _validate_task_runtime(config, tasks)
    _validate_runtime_authority_consistency(config, tasks)
    _validate_sweep(config, tasks)
    _validate_canonical_and_execution(config)


def validate_profile_experiment_id(config: Mapping[str, Any]) -> None:
    current_id = experiment_id(config)
    profile = sweep_profile(config)
    if profile == SWEEP_PROFILE_RHO and current_id != RHO_EXPERIMENT_ID:
        raise ValueError(f"{SWEEP_PROFILE_RHO} requires experiment_id={RHO_EXPERIMENT_ID}")
    if profile == SWEEP_PROFILE_DENSE and current_id != DENSE_EXPERIMENT_ID:
        raise ValueError(f"{SWEEP_PROFILE_DENSE} requires experiment_id={DENSE_EXPERIMENT_ID}")
    if profile == SWEEP_PROFILE_COLDSTART:
        validate_coldstart_config(config)
    elif profile not in (SWEEP_PROFILE_RHO, SWEEP_PROFILE_DENSE):
        raise ValueError(f"Unsupported sweep profile: {profile}")


def repo_relative_config(path: str | Path, repo_root: str | Path) -> tuple[Path, str]:
    root = Path(repo_root).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"config path escapes repository: {resolved}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"config file is missing: {resolved}")
    return resolved, relative


def require_tracked_config(path: str | Path, repo_root: str | Path) -> tuple[Path, str, str]:
    resolved, relative = repo_relative_config(path, repo_root)
    root = Path(repo_root).resolve()
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise ValueError(f"config is not Git-tracked: {relative}")
    blob = subprocess.check_output(
        ["git", "hash-object", "--", relative], cwd=root, text=True
    ).strip()
    return resolved, relative, blob


def validate_historical_config_identity(
    path: str | Path,
    config: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> None:
    expected = HISTORICAL_CONFIG_IDENTITIES.get(experiment_id(config))
    if expected is None:
        return
    resolved, relative = repo_relative_config(path, repo_root)
    expected_path, expected_blob = expected
    if relative != expected_path:
        raise ValueError(f"Historical experiment_id requires canonical config path {expected_path}")
    observed = subprocess.check_output(
        ["git", "hash-object", str(resolved)],
        cwd=Path(repo_root).resolve(),
        text=True,
    ).strip()
    if observed != expected_blob:
        raise ValueError(
            f"Historical experiment_id config identity drifted: expected {expected_blob}, found {observed}"
        )
