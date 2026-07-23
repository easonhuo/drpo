#!/usr/bin/env python3
"""Autotuned launcher adapter for reviewed fixed-profile E8 scans."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping

from drpo import countdown_e8_alpha1_highc_scan_common as highc

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_LAUNCHER = (
    _REPO_ROOT
    / "scripts"
    / "run_countdown_e8_oracle_offline_v2_alpha1_c_scan_auto.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_e8_alpha1_c_scan_auto_base", _BASE_LAUNCHER
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load base launcher: {_BASE_LAUNCHER}")
_base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_base)

CANONICAL_DPO_EXPERIMENT_ID = (
    "EXT-C-E8-ORACLE-OFFLINE-V2-CANONICAL-DPO-BETA-SCAN-0.5B-01"
)
CANONICAL_DPO_POINTS = tuple(
    ("canonical_dpo", 1.0, beta) for beta in (0.03, 0.1, 0.3, 1.0)
)
CANONICAL_DPO_SEED_OFFSETS = (4000, 5000)


def _validate_canonical_dpo_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != CANONICAL_DPO_EXPERIMENT_ID:
        raise ValueError("Canonical DPO experiment_id mismatch")
    if config.get("result_status") != "pilot":
        raise ValueError("Canonical DPO beta scan must remain a pilot")
    if config.get("registration_state") != "dev_code_first_unregistered":
        raise ValueError("Canonical DPO beta scan must remain code-first unregistered")
    if config.get("method_identity") != "canonical_sigmoid_dpo_frozen_initial_reference":
        raise ValueError("Canonical DPO method identity changed")

    model = config.get("model", {})
    expected_model = {
        "shared_frozen_backbone": True,
        "policy_adapter": "default",
        "reference_adapter": "reference",
        "copy_policy_adapter_to_reference_at_initialization": True,
        "reference_trainable": False,
        "disable_dropout": True,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(f"Canonical DPO model field changed: {key}")

    bank = config.get("bank", {})
    if bank.get("use_all_unique_negatives") is not True:
        raise ValueError("Canonical DPO must use every unique negative")
    if bank.get("behavior_policy_logged") is not False:
        raise ValueError("E8 V2 bank must not be relabeled as logged behavior data")
    if bank.get("explicit_near_far_training_classes") is not False:
        raise ValueError("Canonical DPO must not introduce near/far classes")
    if bank.get("extreme_selection_forbidden") is not True:
        raise ValueError("Canonical DPO hard/extreme negative selection is forbidden")

    preference = config.get("preference_data", {})
    if preference.get("chosen") != "oracle_completion":
        raise ValueError("Canonical DPO chosen completion must remain the oracle")
    if preference.get("rejected") != "all_unique_verifier_wrong_completions":
        raise ValueError("Canonical DPO rejected set changed")
    if preference.get("pair_aggregation") != "mean_within_prompt_then_mean_prompts":
        raise ValueError("Canonical DPO prompt-balanced aggregation changed")
    if preference.get("hard_negative_mining") is not False:
        raise ValueError("Canonical DPO hard-negative mining is forbidden")
    if float(preference.get("label_smoothing", float("nan"))) != 0.0:
        raise ValueError("Canonical DPO label smoothing must remain zero")

    reference = config.get("reference_policy", {})
    expected_reference = {
        "role": "exact_frozen_initial_policy",
        "adapter": "reference",
        "copied_from_policy_before_update_1": True,
        "update_frequency_per_policy_step": 0,
        "receives_gradient": False,
        "precomputed_log_probabilities": False,
    }
    for key, expected in expected_reference.items():
        if reference.get(key) != expected:
            raise ValueError(f"Canonical DPO reference field changed: {key}")

    objective = config.get("objective", {})
    if objective.get("loss_type") != "sigmoid_dpo":
        raise ValueError("Canonical DPO loss must remain sigmoid DPO")
    if objective.get("beta_parameter") != "dpo_beta":
        raise ValueError("Canonical DPO beta coordinate changed")
    if objective.get("sequence_log_probability") != "full_completion_summed_log_probability":
        raise ValueError("Canonical DPO must use summed completion log probability")
    if objective.get("reference_detached") is not True:
        raise ValueError("Canonical DPO reference must remain detached")
    if objective.get("distance_control") is not False:
        raise ValueError("Canonical DPO must not add remoteness control")
    if objective.get("value_network") is not False:
        raise ValueError("Canonical DPO must not add a value network")

    sweep = config.get("sweep", {})
    configured = tuple(
        (str(item["family"]), float(item["alpha"]), float(item["coefficient"]))
        for item in sweep.get("parameter_points", ())
    )
    if configured != CANONICAL_DPO_POINTS:
        raise ValueError("Canonical DPO beta points changed")
    if tuple(int(value) for value in sweep.get("seed_offsets", ())) != CANONICAL_DPO_SEED_OFFSETS:
        raise ValueError("Canonical DPO development seed offsets changed")
    if int(sweep.get("unique_parameter_points", -1)) != 4:
        raise ValueError("Canonical DPO scan requires four beta points")
    if int(sweep.get("cells", -1)) != 8:
        raise ValueError("Canonical DPO scan requires eight paired cells")
    if sweep.get("cartesian_product") is not False:
        raise ValueError("Canonical DPO scan must remain an explicit point list")

    training = config.get("training", {})
    if int(training.get("steps", -1)) != 1200 or training.get("early_stop") is not False:
        raise ValueError("Canonical DPO requires a fixed 1200-step horizon")
    if int(training.get("eval_every", -1)) != 100:
        raise ValueError("Canonical DPO Greedy/Pass@8 cadence must remain 100")
    if int(training.get("pass64_every", -1)) != 200:
        raise ValueError("Canonical DPO Pass@64 cadence must remain 200")
    if training.get("denominator") != "unique_negative_count_per_prompt":
        raise ValueError("Canonical DPO denominator must remain prompt balanced")
    if training.get("normalize_by_weight_sum") is not False:
        raise ValueError("Canonical DPO weight-sum normalization is forbidden")
    if training.get("gradient_budget_matching") is not False:
        raise ValueError("Canonical DPO gradient-budget matching is forbidden")
    if float(training.get("initial_pair_margin_max_abs_tolerance", -1.0)) != 1.0e-5:
        raise ValueError("Canonical DPO initial pair-margin tolerance must remain 1e-5")

    execution = config.get("execution", {})
    if execution.get("default_gpus") != [0, 1]:
        raise ValueError("Canonical DPO profile requires GPU 0-1")
    if int(execution.get("parallel_cells_per_gpu", -1)) != 1:
        raise ValueError("Canonical DPO requires one cell per GPU")
    if int(execution.get("expected_full_waves", -1)) != 4:
        raise ValueError("Canonical DPO scan requires four full waves")
    if execution.get("identity_checked_resume") is not True:
        raise ValueError("Canonical DPO requires identity-checked resume")
    liveness = execution.get("liveness", {})
    if liveness.get("representative_family") != "canonical_dpo":
        raise ValueError("Canonical DPO liveness family changed")
    if float(liveness.get("representative_c", float("nan"))) != 0.1:
        raise ValueError("Canonical DPO liveness must use beta=0.1")
    if liveness.get("checkpoint_reload_required") is not True:
        raise ValueError("Canonical DPO liveness must require checkpoint reload")

    evaluation = config.get("evaluation", {})
    if evaluation.get("split_role") != "structurally_disjoint_held_out_evaluation":
        raise ValueError("Canonical DPO split role changed")
    if evaluation.get("enters_training_loss") is not False:
        raise ValueError("Canonical DPO held-out evaluation must not enter training loss")
    if evaluation.get("separate_test_split_access") is not False:
        raise ValueError("Canonical DPO test access must remain disabled")
    if evaluation.get("evaluated_adapter") != "default":
        raise ValueError("Canonical DPO must evaluate the policy adapter")
    if evaluation.get("primary_reporting_metric") != "late_window_pass_at_8":
        raise ValueError("Canonical DPO primary reporting metric changed")
    if evaluation.get("secondary_reporting_metric") != "terminal_pass_at_8":
        raise ValueError("Canonical DPO secondary reporting metric changed")
    if evaluation.get("paper_facing_checkpoint_policy") != "late_window_and_terminal":
        raise ValueError("Canonical DPO checkpoint policy changed")
    if evaluation.get("best_checkpoint_metric_is_supplementary_only") is not True:
        raise ValueError("Canonical DPO best checkpoint must remain supplementary")


def _install_canonical_dpo_profile(grid_config: str | Path | None) -> None:
    if grid_config is None:
        return
    config = highc.load_yaml(grid_config)
    if config.get("experiment_id") != CANONICAL_DPO_EXPERIMENT_ID:
        return
    if CANONICAL_DPO_EXPERIMENT_ID in highc._PROFILES:
        return

    highc.CANONICAL_DPO_BETA_SCAN_EXPERIMENT_ID = CANONICAL_DPO_EXPERIMENT_ID
    highc.CANONICAL_DPO_POINTS = CANONICAL_DPO_POINTS
    highc._PROFILES[CANONICAL_DPO_EXPERIMENT_ID] = {
        "experiment_id": CANONICAL_DPO_EXPERIMENT_ID,
        "version": "0.1.0-dev-code-first-canonical-dpo",
        "default_grid_config": (
            "configs/countdown_e8_oracle_offline_v2_canonical_dpo_beta_scan_0p5b.yaml"
        ),
        "parameter_points": CANONICAL_DPO_POINTS,
        "seed_offsets": CANONICAL_DPO_SEED_OFFSETS,
        "expected_points": 4,
        "expected_cells": 8,
        "requires_positive_only": False,
        "kind": "canonical_dpo",
    }

    original_validate = highc.validate_grid_config
    original_parameter_points = highc.parameter_points
    original_build_cells = highc.build_cells

    def validate_grid_config(value: Mapping[str, Any]) -> None:
        if value.get("experiment_id") == CANONICAL_DPO_EXPERIMENT_ID:
            _validate_canonical_dpo_config(value)
            return
        original_validate(value)

    def parameter_points(value: Mapping[str, Any]) -> tuple[Any, ...]:
        if value.get("experiment_id") == CANONICAL_DPO_EXPERIMENT_ID:
            _validate_canonical_dpo_config(value)
            return CANONICAL_DPO_POINTS
        return original_parameter_points(value)

    def build_cells(value: Mapping[str, Any]) -> tuple[Any, ...]:
        if value.get("experiment_id") != CANONICAL_DPO_EXPERIMENT_ID:
            return original_build_cells(value)
        points = parameter_points(value)
        cells = tuple(
            highc.Cell(
                alpha=alpha,
                coefficient=beta,
                seed_offset=seed_offset,
                family=family,
            )
            for family, alpha, beta in points
            for seed_offset in CANONICAL_DPO_SEED_OFFSETS
        )
        if len(cells) != 8 or len({cell.name for cell in cells}) != 8:
            raise AssertionError("Canonical DPO scan must produce eight unique cells")
        return cells

    highc.validate_grid_config = validate_grid_config
    highc.parameter_points = parameter_points
    highc.build_cells = build_cells


def _grid_config_from_argv(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token == "--grid_config" and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _required_device_count() -> int:
    if highc.EXPERIMENT_ID in {
        highc.JOINT_FITTED_REFERENCE_TOPR_DENSE_EXPERIMENT_ID,
        CANONICAL_DPO_EXPERIMENT_ID,
    }:
        return 2
    return 8


def _core_command(args, command: str, *, selected_ids: list[str]) -> list[str]:
    required_devices = _required_device_count()
    if len(selected_ids) != required_devices:
        if required_devices == 8:
            raise RuntimeError("paper-aligned scan requires all eight configured GPUs")
        raise RuntimeError("this fixed dual-adapter profile requires exactly two configured GPUs")
    result = [
        sys.executable,
        str(
            _REPO_ROOT
            / "src"
            / "drpo"
            / "countdown_e8_alpha1_highc_scan_runtime.py"
        ),
        command,
        "--model_path",
        str(Path(args.model_path).resolve()),
        "--work_dir",
        str(Path(args.work_dir).resolve()),
        "--bank",
        str(Path(args.bank).resolve()),
        "--val",
        str(Path(args.val).resolve()),
        "--base_config",
        str(Path(args.base_config).resolve()),
        "--grid_config",
        str(Path(args.grid_config).resolve()),
    ]
    if command == "run":
        result.extend(
            [
                "--gpus",
                ",".join(selected_ids),
                "--runtime-slots-per-gpu",
                str(
                    highc.load_yaml(args.grid_config)["execution"][
                        "parallel_cells_per_gpu"
                    ]
                ),
            ]
        )
    return result


_base._core_command = _core_command
build_parser = _base.build_parser


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    grid_config = _grid_config_from_argv(tokens)
    _install_canonical_dpo_profile(grid_config)
    if grid_config is None:
        highc.activate()
    else:
        highc.activate_for_grid_config(grid_config)
    if highc.EXPERIMENT_ID == highc.ASYMRE_DELTAV_EXPERIMENT_ID:
        _base.ADAPTER_ID = "e8_asymre_deltav_scan_cuda_dev_v1"
    elif highc.EXPERIMENT_ID == highc.C_EXTENSION_EXPERIMENT_ID:
        _base.ADAPTER_ID = "e8_linear_c_extension_cuda_dev_v1"
    elif highc.EXPERIMENT_ID == highc.JOINT_FITTED_REFERENCE_TOPR_DENSE_EXPERIMENT_ID:
        _base.ADAPTER_ID = "e8_joint_fitted_reference_beta_topr_dense_cuda_dev_v1"
    elif highc.EXPERIMENT_ID == CANONICAL_DPO_EXPERIMENT_ID:
        _base.ADAPTER_ID = "e8_canonical_dpo_beta_scan_cuda_dev_v1"
    else:
        _base.ADAPTER_ID = "e8_alpha1_highc_scan_cuda_dev_v1"
    return _base.main(tokens)


if __name__ == "__main__":
    raise SystemExit(main())
