from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_countdown_v46_online_replay.py"
SPEC = importlib.util.spec_from_file_location("countdown_v46_online_replay", MODULE_PATH)
assert SPEC and SPEC.loader
v46 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v46
SPEC.loader.exec_module(v46)


def _row(round_index: int, index: int) -> dict:
    return {
        "id": f"r{round_index}_{index}",
        "collector_round": round_index,
        "prompt": "p",
    }


def test_protocol_constants_freeze_true_online_offpolicy_design() -> None:
    assert v46.EXPERIMENT_ID == "EXT-C-E8-V4.6-ONLINE-OFFPOLICY-REPLAY"
    assert v46.PREDECESSOR_ID == "EXT-C-E8-V4.5-OFFLINE-BANK-TUNING"
    assert v46.FROZEN_SOURCE_ID == "EXT-C-E8-V4.4-OFFLINE-BANK"
    assert v46.METHODS == (
        "frozen_positive_only",
        "frozen_dynamic",
        "online_positive_only",
        "online_dynamic",
    )
    assert v46.CONFIRM_SEEDS == (6234, 7234, 8234)
    assert v46.COLLECTION_PHASES == 4
    assert v46.REFRESH_ROWS == 1000
    assert v46.ONLINE_BANK_SIZE == 16
    assert v46.REPLAY_WINDOW == 3
    assert v46.FRESH_MICROBATCHES == v46.STALE_MICROBATCHES == 4


def test_update_budget_is_exact_and_nearly_balanced() -> None:
    plan = v46.split_update_budget(1131, 4)
    assert plan == (283, 283, 283, 282)
    assert sum(plan) == 1131
    assert max(plan) - min(plan) == 1
    with pytest.raises(ValueError):
        v46.split_update_budget(3, 4)


def test_replay_age_plan_is_warmup_then_exact_half_stale() -> None:
    warmup = v46.replay_age_plan(0)
    assert warmup["off_policy"] is False
    assert warmup["fresh_microbatches"] == 8
    assert warmup["stale_microbatches"] == 0

    phase_three = v46.replay_age_plan(3)
    assert phase_three["off_policy"] is True
    assert phase_three["fresh_microbatches"] == 4
    assert phase_three["stale_microbatches"] == 4
    assert phase_three["eligible_stale_rounds"] == [1, 2]


def test_stale_replay_sampling_is_deterministic_and_excludes_current_round() -> None:
    rounds = [
        [_row(0, index) for index in range(3)],
        [_row(1, index) for index in range(3)],
        [_row(2, index) for index in range(3)],
    ]
    first = v46.deterministic_stale_rows(rounds, phase=2, target=8, seed=7)
    second = v46.deterministic_stale_rows(rounds, phase=2, target=8, seed=7)
    assert first == second
    assert len(first) == 8
    assert {row["collector_round"] for row in first} <= {0, 1}
    assert all(row["collector_round"] < 2 for row in first)


def test_parser_and_registry_match_registered_pilot() -> None:
    parser = v46.build_parser()
    args = parser.parse_args(
        [
            "--model_path",
            "/tmp/model",
            "--predecessor_work_dir",
            "/tmp/v45",
            "--work_dir",
            "/tmp/v46",
        ]
    )
    assert args.gpus == "auto"
    assert args.online_worker is False

    registry = yaml.safe_load((ROOT / "experiments" / "registry.yaml").read_text())
    experiments = {row["id"]: row for row in registry["experiments"]}
    entry = experiments[v46.EXPERIMENT_ID]
    assert entry["status"] == "not_run"
    assert entry["execution_class"] == "pilot"
    assert entry["predecessor"] == v46.PREDECESSOR_ID
    assert entry["one_click_entrypoint"] == "scripts/run_countdown_v46_online_replay.py"
    assert entry["design"]["cells"] == list(v46.METHODS)
    assert entry["online_replay_protocol"]["collection_phases"] == 4
    assert entry["online_replay_protocol"]["post_warmup_stale_fraction"] == 0.5
    assert entry["confirmation_protocol"]["training_seeds"] == [6234, 7234, 8234]


def test_guard_declares_required_outputs_and_sources() -> None:
    source = MODULE_PATH.read_text()
    for required in ("RUN_COMPLETE.json", "terminal_audit.json", "arena_summary.csv"):
        assert f'"--required-output",\n        "{required}"' in source
    for required_source in (
        "scripts/run_countdown_v46_online_replay.py",
        "src/drpo/countdown_qwen_arena_onefile.py",
        "docs/handoff.md",
        "experiments/registry.yaml",
    ):
        assert f'"--source-file",\n        "{required_source}"' in source


def test_actual_selected_bank_gradient_diagnostics_are_implemented() -> None:
    core = (ROOT / "src" / "drpo" / "countdown_qwen_arena_onefile.py").read_text()
    assert "bank_selected_near_negative_gradient_norm_raw" in core
    assert "bank_selected_far_negative_gradient_norm_raw" in core
    assert "bank_selected_far_over_near_gradient_norm_ratio" in core
    assert 'branch_metrics(selected_near_batch, "bank_selected_near")' in core
    assert 'branch_metrics(selected_far_batch, "bank_selected_far")' in core


def test_frozen_cells_cannot_early_stop_before_matched_budget() -> None:
    command = v46._frozen_train_command(
        python="python3",
        runner=Path("runner.py"),
        model=Path("model"),
        reference=Path("reference"),
        offline=Path("offline.jsonl"),
        val=Path("val.jsonl"),
        train=Path("train.jsonl"),
        output=Path("output"),
        plan={"micro_batch": 2, "eval_batch": 4},
        calibration=Path("calibration.json"),
        seed=6234,
        selected_alpha=1.0,
        selected_lambda=0.7,
        positive_only=False,
        total_steps=120,
        eval_every=20,
    )

    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert value("--steps") == "120"
    assert value("--min_steps") == "120"
    assert value("--eval_every") == "20"


def test_online_worker_records_replay_age_and_matches_eval_schedule() -> None:
    source = MODULE_PATH.read_text()
    assert '"replay_age": phase - int(row["collector_round"])' in source
    assert '"replay_age": 0' in source
    assert "global_step % args.eval_every == 0" in source
    assert '"optimizer_update_budget_exactly_matched"' in source
