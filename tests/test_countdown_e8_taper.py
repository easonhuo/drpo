from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "drpo" / "countdown_e8_taper.py"
SPEC = importlib.util.spec_from_file_location("countdown_e8_taper", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

LAUNCHER_PATH = ROOT / "scripts" / "run_countdown_e8_taper.py"
LAUNCHER_SPEC = importlib.util.spec_from_file_location(
    "run_countdown_e8_taper", LAUNCHER_PATH
)
assert LAUNCHER_SPEC and LAUNCHER_SPEC.loader
launcher = importlib.util.module_from_spec(LAUNCHER_SPEC)
sys.modules[LAUNCHER_SPEC.name] = launcher
LAUNCHER_SPEC.loader.exec_module(launcher)


def _row(index: int, structure: str = "s", candidates: int = 2) -> dict:
    return {
        "id": f"p{index}",
        "prompt": f"prompt {index}",
        "numbers": [1, 2, 3, 4],
        "target": 10,
        "oracle": "1+2+3+4",
        "oracle_structure": structure,
        "negatives": [
            {"expression": f"1+2+3-{4 + offset}"}
            for offset in range(candidates)
        ],
    }


def test_frozen_config_matches_registered_protocol() -> None:
    config = module.load_config(ROOT / "configs" / "countdown_e8_taper_0p5b.yaml")
    assert config["experiment_id"] == module.EXPERIMENT_ID
    assert tuple(config["methods"]) == module.METHODS
    assert config["confirmation"]["paired_training_seeds"] == [9234, 10234, 11234]
    assert config["calibration"]["development_seed"] == 9134
    assert config["calibration"]["surprisal_threshold_tau"] == 2.0
    assert config["calibration"]["surprisal_scale_rule"] == (
        "calibration_upper_half_median_minus_lower_half_median"
    )
    assert config["calibration"]["shared_negative_scale"] == (
        "positive_aggregate_gradient_l2_over_"
        "uncontrolled_negative_aggregate_gradient_l2"
    )
    assert config["training"]["optimizer_updates"] == 1200
    assert config["training"]["early_stopping_changes_training_horizon"] is False
    assert config["replay"]["train_prompt_rows"] == 1500
    assert config["replay"]["calibration_prompt_rows"] == 16
    assert config["replay"]["train_candidate_prompt_rows"] >= 1500
    assert config["replay"]["calibration_candidate_prompt_rows"] >= 16
    assert config["replay"]["synthetic_negative_fallback"] is False


def test_continuous_tapers_have_registered_shapes() -> None:
    distance = torch.tensor([0.0, 1.0, 2.0, 4.0])
    reciprocal = module.taper_weight("reciprocal_linear", distance, 2.0)
    exponential = module.taper_weight("exponential", distance, 0.7)
    squared = module.taper_weight("squared_distance_exponential", distance, 0.7)
    assert torch.allclose(reciprocal, torch.tensor([1.0, 1 / 3, 1 / 5, 1 / 9]))
    assert torch.allclose(exponential, torch.exp(-0.7 * distance))
    assert torch.allclose(squared, torch.exp(-0.7 * distance.square()))
    assert torch.all(reciprocal[:-1] >= reciprocal[1:])
    assert torch.all(exponential[:-1] >= exponential[1:])
    assert torch.all(squared[:-1] >= squared[1:])
    assert squared[-1] < exponential[-1] < reciprocal[-1]
    seq_lp = torch.tensor([-1.0, -2.0, -3.5, -8.0], requires_grad=True)
    normalized, distance_out = module.normalized_remoteness(seq_lp, 2.0, 1.5)
    assert torch.allclose(normalized, torch.tensor([0.0, 0.0, 1.0, 4.0]))
    assert torch.allclose(distance_out, torch.tensor([0.0, 0.0, 1.0, 2.0]))
    assert distance_out.requires_grad is False
    assert torch.allclose(
        module.taper_weight("squared_distance_exponential", distance_out, 0.7),
        torch.exp(-0.7 * normalized),
    )


def test_calibration_surprisal_scale_uses_frozen_half_median_gap() -> None:
    scale, diagnostics = module.calibration_surprisal_scale(
        [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0], minimum=1e-6
    )
    assert scale == pytest.approx(9.0)
    assert diagnostics["common_half_median_surprisal"] == pytest.approx(2.5)
    assert diagnostics["rare_half_median_surprisal"] == pytest.approx(11.5)
    with pytest.raises(RuntimeError):
        module.calibration_surprisal_scale([1.0, 1.0, 1.0, 1.0], minimum=1e-6)


def test_deterministic_weight_pass_disables_dropout_and_restores_mode(monkeypatch) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode: bool = True):
            self.training = mode
            return self

    model = FakeModel()

    def fake_stats(current_model, _batch):
        value = -99.0 if current_model.training else -5.0
        return {"seq_lp": torch.tensor([value])}

    monkeypatch.setattr(module.arena, "completion_stats", fake_stats)
    normalized, distance, weights = module._deterministic_current_weights(
        model,
        {"input_ids": torch.tensor([[1]])},
        method="exponential",
        coefficient=1.0,
        tau=2.0,
        surprisal_scale=3.0,
    )
    assert model.training is True
    assert normalized.item() == pytest.approx(1.0)
    assert distance.item() == pytest.approx(1.0)
    assert weights.item() == pytest.approx(torch.exp(torch.tensor(-1.0)).item())


def test_pattern_balanced_selection_is_disjoint_without_cursor_loss() -> None:
    rows = [_row(i, "a" if i % 2 == 0 else "b") for i in range(20)]
    train, calibration = module.balanced_disjoint_prompt_selection(rows, 13, 5, 17)
    assert len(train) == 13
    assert len(calibration) == 5
    train_ids = {row["id"] for row in train}
    calibration_ids = {row["id"] for row in calibration}
    assert train_ids.isdisjoint(calibration_ids)
    assert len(train_ids | calibration_ids) == 18
    train_counts = Counter(row["oracle_structure"] for row in train)
    assert abs(train_counts["a"] - train_counts["b"]) <= 1


def test_exact_eligible_take_is_deterministic_and_balanced() -> None:
    rows = [_row(i, "a" if i % 3 else "b") for i in range(30)]
    first = module.balanced_prompt_take(rows, 17, 99)
    second = module.balanced_prompt_take(rows, 17, 99)
    assert first == second
    assert len({row["id"] for row in first}) == 17
    counts = Counter(row["oracle_structure"] for row in first)
    assert abs(counts["a"] - counts["b"]) <= 1


def test_wrong_candidate_filter_rejects_correct_invalid_and_duplicates() -> None:
    row = _row(0)
    seen: set[str] = set()
    allowed = {module.arena.expression_structure("1+2+3-4")}
    accepted = module._eligible_wrong_candidate("1+2+3-4", row, seen, allowed)
    assert accepted is not None
    assert module._eligible_wrong_candidate("1+2+3-4", row, seen, allowed) is None
    assert module._eligible_wrong_candidate("1+2+3+4", row, seen, allowed) is None
    assert module._eligible_wrong_candidate("1+2", row, seen, allowed) is None


def test_sampler_is_paired_prompt_uniform_and_not_candidate_weighted() -> None:
    rows = [_row(0, candidates=1), _row(1, candidates=25), _row(2, candidates=3)]
    first = module.make_prompt_balanced_sampler_plan(rows, seed=9234, total_samples=101)
    second = module.make_prompt_balanced_sampler_plan(rows, seed=9234, total_samples=101)
    assert first == second
    prompt_counts = Counter(item["prompt_index"] for item in first)
    assert max(prompt_counts.values()) - min(prompt_counts.values()) <= 1
    assert all(0 <= item["negative_index"] < len(rows[item["prompt_index"]]["negatives"]) for item in first)


def test_calibration_bisection_matches_target_and_rejects_unbracketed() -> None:
    coefficient, matched, error = module.calibrate_monotone_coefficient(
        lambda value: 10.0 / (1.0 + value),
        2.0,
        maximum=16.0,
        steps=24,
        tolerance=1e-5,
    )
    assert coefficient == pytest.approx(4.0, abs=1e-4)
    assert matched == pytest.approx(2.0, abs=1e-4)
    assert error < 1e-5
    with pytest.raises(RuntimeError):
        module.calibrate_monotone_coefficient(
            lambda value: 10.0,
            2.0,
            maximum=4.0,
            steps=4,
            tolerance=0.01,
        )


def test_training_artifact_identity_checks_fail_closed(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text((ROOT / "configs" / "countdown_e8_taper_0p5b.yaml").read_text())
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter.bin").write_bytes(b"adapter")
    replay = tmp_path / "replay.jsonl"
    replay.write_text('{"id":"p0"}\n')
    plan = tmp_path / "plan.jsonl"
    plan.write_text('{"prompt_index":0,"negative_index":0}\n')
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        __import__("json").dumps(
            {
                "experiment_id": module.EXPERIMENT_ID,
                "config_sha256": module._sha256_file(config),
                "reference_adapter_hashes": module._hash_tree(adapter),
            }
        )
    )
    manifest = {
        "experiment_id": module.EXPERIMENT_ID,
        "seed": 9234,
        "replay_sha256": module._sha256_file(replay),
        "plan_sha256": module._sha256_file(plan),
        "total_samples": 1,
    }
    Path(str(plan) + ".manifest.json").write_text(__import__("json").dumps(manifest))
    loaded_calibration, loaded_manifest = module._validate_training_inputs(
        config_path=config,
        reference_adapter=adapter,
        replay_path=replay,
        sampler_plan_path=plan,
        calibration_path=calibration,
        seed=9234,
    )
    assert loaded_calibration["experiment_id"] == module.EXPERIMENT_ID
    assert loaded_manifest["seed"] == 9234
    with pytest.raises(RuntimeError, match="seed"):
        module._validate_training_inputs(
            config_path=config, reference_adapter=adapter, replay_path=replay,
            sampler_plan_path=plan, calibration_path=calibration, seed=10234
        )
    plan.write_text('{"prompt_index":1,"negative_index":0}\n')
    with pytest.raises(RuntimeError, match="content"):
        module._validate_training_inputs(
            config_path=config, reference_adapter=adapter, replay_path=replay,
            sampler_plan_path=plan, calibration_path=calibration, seed=9234
        )


def test_one_click_launcher_and_registry_are_ready_but_not_run() -> None:
    assert launcher.EXPERIMENT_ID == module.EXPERIMENT_ID
    parser = launcher.build_parser()
    args = parser.parse_args(["--model_path", "/tmp/model", "--work_dir", "/tmp/run"])
    assert args.gpus == "auto"
    source = LAUNCHER_PATH.read_text()
    for required in (
        "RUN_COMPLETE.json",
        "terminal_audit.json",
        "scientific_run_manifest.json",
        "arena_summary.csv",
        "taper_calibration.json",
        "replay_pool_manifest.json",
        "surprisal_bin_diagnostics.csv",
    ):
        assert required in source
    registry = yaml.safe_load((ROOT / "experiments" / "registry.yaml").read_text())
    entry = {row["id"]: row for row in registry["experiments"]}[module.EXPERIMENT_ID]
    assert entry["status"] == "not_run"
    assert entry["implementation_state"] == "implemented"
    assert entry["execution_gate"]["state"] == "ready"
    assert entry["code_entrypoint"] == "src/drpo/countdown_e8_taper.py"
    assert entry["operator_entrypoint"] == "scripts/run_countdown_e8_taper.py"
