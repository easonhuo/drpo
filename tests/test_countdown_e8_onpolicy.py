from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "drpo" / "countdown_e8_onpolicy.py"
SPEC = importlib.util.spec_from_file_location("countdown_e8_onpolicy", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

LAUNCHER_PATH = ROOT / "scripts" / "run_countdown_e8_onpolicy.py"
LAUNCHER_SPEC = importlib.util.spec_from_file_location(
    "run_countdown_e8_onpolicy", LAUNCHER_PATH
)
assert LAUNCHER_SPEC and LAUNCHER_SPEC.loader
launcher = importlib.util.module_from_spec(LAUNCHER_SPEC)
sys.modules[LAUNCHER_SPEC.name] = launcher
LAUNCHER_SPEC.loader.exec_module(launcher)


def _row(index: int = 0) -> dict:
    return {
        "id": f"p{index}",
        "prompt": "Numbers: 1, 2, 3, 4\nTarget: 10",
        "numbers": [1, 2, 3, 4],
        "target": 10,
        "oracle": "1+2+3+4",
    }


def test_unpolished_config_is_positive_only_lora_and_not_taper() -> None:
    config = module.load_config(ROOT / "configs" / "countdown_e8_onpolicy_0p5b_unpolished.yaml")
    assert config["experiment_id"] == module.EXPERIMENT_ID
    assert tuple(config["methods"]) == module.METHODS
    assert config["methods"] == ["sft_only", "onpolicy_rft_positive_only"]
    assert config["model"]["parameterization"] == "lora"
    assert config["policy_update"]["continuation_parameterization"] == "same_lora_adapter"
    assert config["policy_update"]["uses_negative_updates"] is False
    assert config["policy_update"]["uses_taper_methods"] is False
    assert config["policy_update"]["uses_frozen_offpolicy_replay"] is False
    assert config["data"]["split_protocol"] == "structural_family_holdout"
    assert config["data"]["train_rows"] == 6000
    assert config["data"]["validation_rows"] == 500
    assert config["data"]["test_rows"] == 1000
    assert config["reference"]["sft_min_epochs"] == 3
    assert config["reference"]["sft_early_stop_patience"] == 2
    assert config["reference"]["external_adapter_reuse_allowed"] is True
    assert config["confirmation"]["paired_training_seeds"] == [
        2026070701,
        2026070702,
        2026070703,
    ]


def test_generate_or_load_data_uses_structural_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = module.load_config(ROOT / "configs" / "countdown_e8_onpolicy_0p5b_unpolished.yaml")
    calls: list[tuple[int, int, int, int, int]] = []

    def fake_generate(train: int, val: int, test: int, seed: int, n_numbers: int):
        calls.append((train, val, test, seed, n_numbers))
        train_rows = [_row(0)]
        val_rows = [_row(1)]
        test_rows = [_row(2)]
        return train_rows, val_rows, test_rows, {"protocol": "fake_structural"}

    monkeypatch.setattr(module.arena, "generate_structural_splits", fake_generate)
    paths = module.generate_or_load_data(tmp_path, config)
    assert calls == [(6000, 500, 1000, 1234, 4)]
    assert paths["split_manifest"].is_file()
    assert "fake_structural" in paths["split_manifest"].read_text()


def test_generate_or_load_data_rejects_partial_existing_split(tmp_path: Path) -> None:
    config = module.load_config(ROOT / "configs" / "countdown_e8_onpolicy_0p5b_unpolished.yaml")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n")
    with pytest.raises(RuntimeError, match="partially populated"):
        module.generate_or_load_data(tmp_path, config)


def test_resolve_adapter_path_accepts_adapter_or_best_parent(tmp_path: Path) -> None:
    direct = tmp_path / "adapter"
    direct.mkdir()
    (direct / "adapter_config.json").write_text("{}")
    assert module.resolve_adapter_path(direct) == direct.resolve()
    parent = tmp_path / "parent"
    best = parent / "best_adapter"
    best.mkdir(parents=True)
    (best / "adapter_config.json").write_text("{}")
    assert module.resolve_adapter_path(parent) == best.resolve()
    with pytest.raises(FileNotFoundError):
        module.resolve_adapter_path(tmp_path / "missing")


def test_prompt_attempt_plan_is_deterministic_and_covers_fixed_shape() -> None:
    first = module.prompt_attempt_plan(5, seed=17, attempts=7, prompts_per_attempt=3)
    second = module.prompt_attempt_plan(5, seed=17, attempts=7, prompts_per_attempt=3)
    assert first == second
    assert len(first) == 7
    assert all(len(chunk) == 3 for chunk in first)
    assert all(0 <= index < 5 for chunk in first for index in chunk)
    assert len({index for chunk in first for index in chunk}) == 5


def test_prompt_attempt_plan_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        module.prompt_attempt_plan(0, seed=1, attempts=1, prompts_per_attempt=1)
    with pytest.raises(ValueError):
        module.prompt_attempt_plan(5, seed=1, attempts=0, prompts_per_attempt=1)
    with pytest.raises(ValueError):
        module.prompt_attempt_plan(5, seed=1, attempts=1, prompts_per_attempt=0)


def test_select_correct_completions_uses_verifier_and_deduplicates() -> None:
    selected = module.select_correct_completions(
        _row(),
        ["1 + 2 + 3 + 4", "1+2+3+4", "1+2+3-4", "not an expression"],
        max_per_prompt=2,
    )
    assert len(selected) == 1
    assert selected[0]["completion"].replace(" ", "") == "1+2+3+4"
    assert selected[0]["prompt_id"] == "p0"


def test_terminal_audit_separates_signal_sparsity_from_numerical_failure() -> None:
    audit = module.terminal_audit(
        [
            {"method": "sft_only"},
            {
                "method": "onpolicy_rft_positive_only",
                "seed": 1,
                "sampling_attempts": 10,
                "skipped_attempts": 4,
                "numerical_failure": None,
            },
            {
                "method": "onpolicy_rft_positive_only",
                "seed": 2,
                "sampling_attempts": 10,
                "skipped_attempts": 0,
                "numerical_failure": "nonfinite_loss_at_attempt_3",
            },
        ]
    )
    assert audit["status"] == "pilot_incomplete_numerical_failure"
    assert audit["onpolicy_seed_count"] == 2
    assert audit["max_skip_fraction"] == pytest.approx(0.4)
    assert audit["numerical_failure_count"] == 1
    assert audit["task_performance_collapse_checked_separately"] is True
    assert audit["support_or_structure_boundary_checked_separately"] is True
    assert audit["nan_inf_numerical_failure_checked_separately"] is True


def test_launcher_binds_registered_experiment_id_and_reuse_arg() -> None:
    assert launcher.EXPERIMENT_ID == module.EXPERIMENT_ID
    parser = launcher.build_parser()
    args = parser.parse_args(
        [
            "--model_path",
            "/tmp/model",
            "--work_dir",
            "/tmp/run",
            "--sft_adapter_path",
            "/tmp/sft/best_adapter",
        ]
    )
    assert args.sft_adapter_path == "/tmp/sft/best_adapter"
