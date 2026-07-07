from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "drpo" / "countdown_e8_capacity_diag.py"
SPEC = importlib.util.spec_from_file_location("countdown_e8_capacity_diag", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

LAUNCHER_PATH = ROOT / "scripts" / "run_countdown_e8_capacity_diag.py"
LAUNCHER_SPEC = importlib.util.spec_from_file_location("run_countdown_e8_capacity_diag", LAUNCHER_PATH)
assert LAUNCHER_SPEC and LAUNCHER_SPEC.loader
launcher = importlib.util.module_from_spec(LAUNCHER_SPEC)
sys.modules[LAUNCHER_SPEC.name] = launcher
LAUNCHER_SPEC.loader.exec_module(launcher)


def test_capacity_config_freezes_scope_and_methods() -> None:
    config = module.load_config(ROOT / "configs" / "countdown_e8_onpolicy_capacity_diag_0p5b.yaml")
    assert config["experiment_id"] == module.EXPERIMENT_ID
    assert tuple(config["methods"]) == module.METHODS
    assert config["methods"] == [
        "lora_sft_only",
        "same_lora_rft",
        "fresh_lora_rft",
        "full_param_rft",
        "full_param_sft_only",
    ]
    assert config["data"]["split_protocol"] == "structural_family_holdout"
    assert (config["data"]["train_rows"], config["data"]["validation_rows"], config["data"]["test_rows"]) == (6000, 500, 1000)
    assert config["scope_guards"]["uses_negative_updates"] is False
    assert config["scope_guards"]["uses_taper_methods"] is False
    assert config["scope_guards"]["uses_frozen_offpolicy_replay"] is False
    assert config["scope_guards"]["full_param_branches_are_capacity_diagnostics_only"] is True
    assert config["parallel"]["enabled"] is True


def test_branch_jobs_parallelize_independent_branches_not_attempts() -> None:
    config = module.load_config(ROOT / "configs" / "countdown_e8_onpolicy_capacity_diag_0p5b.yaml")
    jobs = module.branch_jobs(config, ["0", "1", "2", "3"])
    assert [job["branch"] for job in jobs] == [
        "same_lora_rft",
        "fresh_lora_rft",
        "full_param_rft",
        "full_param_sft_only",
    ]
    assert [job["gpu"] for job in jobs] == ["0", "1", "2", "3"]
    assert {job["seed"] for job in jobs if job["branch"] != "full_param_sft_only"} == {2026070701}


def test_exploration_summary_reports_diversity_and_correct_rate() -> None:
    rows = [
        {"prompt_id": "a", "completion": "1 + 2 + 3 + 4"},
        {"prompt_id": "a", "completion": "1+2+3+4"},
        {"prompt_id": "b", "completion": "(1+4)*(3-1)"},
    ]
    summary = module.exploration_summary(rows, sampled_completions=12)
    assert summary["selected_correct_total"] == 3
    assert summary["unique_correct_expressions"] == 2
    assert summary["prompts_with_selected_correct"] == 2
    assert summary["selected_correct_rate"] == pytest.approx(0.25)
    assert summary["mean_unique_correct_per_prompt"] == pytest.approx(1.0)


def test_sft_helper_config_is_lora_only_for_shared_reference() -> None:
    config = module.load_config(ROOT / "configs" / "countdown_e8_onpolicy_capacity_diag_0p5b.yaml")
    shaped = module._as_unpolished_config(config)
    assert shaped["experiment_id"] == module.onp.EXPERIMENT_ID
    assert shaped["model"]["parameterization"] == "lora"
    assert shaped["policy_update"]["uses_negative_updates"] is False
    assert shaped["reference"]["sft_seed"] == config["reference"]["lora_sft_seed"]


def test_launcher_binds_capacity_experiment_and_parallel_args() -> None:
    assert launcher.EXPERIMENT_ID == module.EXPERIMENT_ID
    parser = launcher.build_parser()
    args = parser.parse_args([
        "--model_path", "/tmp/model",
        "--work_dir", "/tmp/run",
        "--gpu_ids", "0,1,2,3",
        "--sft_adapter_path", "/tmp/sft/best_adapter",
    ])
    assert args.gpu_ids == "0,1,2,3"
    assert args.sft_adapter_path == "/tmp/sft/best_adapter"
