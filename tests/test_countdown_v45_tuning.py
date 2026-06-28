from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_countdown_v45_tuning.py"
SPEC = importlib.util.spec_from_file_location("countdown_v45_tuning", MODULE_PATH)
assert SPEC and SPEC.loader
v45 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v45
SPEC.loader.exec_module(v45)


def _record(
    seed: int,
    *,
    best_greedy: float,
    best_pass: float,
    best_valid: float = 0.97,
    terminal_greedy: float | None = None,
    terminal_pass: float | None = None,
    terminal_valid: float = 0.97,
    numerical_failure: bool = False,
) -> dict:
    return {
        "seed": seed,
        "numerical_failure": numerical_failure,
        "best_metrics": {
            "greedy_success": best_greedy,
            "pass_at_k": best_pass,
            "valid_rate": best_valid,
        },
        "terminal_metrics": {
            "greedy_success": best_greedy if terminal_greedy is None else terminal_greedy,
            "pass_at_k": best_pass if terminal_pass is None else terminal_pass,
            "valid_rate": terminal_valid,
        },
    }


def test_registered_grid_and_seed_roles_are_frozen() -> None:
    assert v45.EXPERIMENT_ID == "EXT-C-E8-V4.5-OFFLINE-BANK-TUNING"
    assert v45.PREDECESSOR_ID == "EXT-C-E8-V4.4-OFFLINE-BANK"
    assert v45.ALPHA_CANDIDATES == (0.5, 1.0, 1.5, 2.0)
    assert v45.LAMBDA_CANDIDATES == (0.3, 0.7, 1.2)
    assert v45.ALPHA_STAGE_LAMBDA == 0.7
    assert v45.TUNING_SEEDS == (1234, 2234)
    assert v45.CONFIRM_SEEDS == (3234, 4234, 5234)
    assert v45.VALID_RATE_FLOOR == 0.95


def test_candidate_summary_and_validation_gate() -> None:
    summary = v45.summarize_candidate([
        _record(1, best_greedy=0.10, best_pass=0.20, terminal_greedy=0.09),
        _record(2, best_greedy=0.14, best_pass=0.22, terminal_greedy=0.11),
    ])
    assert summary["eligible"] is True
    assert summary["mean_best_greedy_success"] == pytest.approx(0.12)
    assert summary["mean_best_pass_at_k"] == pytest.approx(0.21)
    assert summary["mean_terminal_greedy_success"] == pytest.approx(0.10)

    low_valid = v45.summarize_candidate([
        _record(1, best_greedy=0.20, best_pass=0.30, best_valid=0.94),
    ])
    failed = v45.summarize_candidate([
        _record(1, best_greedy=0.20, best_pass=0.30, numerical_failure=True),
    ])
    assert low_valid["eligible"] is False
    assert failed["eligible"] is False


def test_candidate_selection_is_lexicographic_and_conservative() -> None:
    summaries = {
        0.5: v45.summarize_candidate([_record(1, best_greedy=0.10, best_pass=0.30)]),
        1.0: v45.summarize_candidate([_record(1, best_greedy=0.11, best_pass=0.20)]),
        2.0: v45.summarize_candidate([_record(1, best_greedy=0.50, best_pass=0.50, best_valid=0.90)]),
    }
    assert v45.choose_candidate(summaries, conservative_tie="smaller") == 1.0

    tied = {
        0.5: v45.summarize_candidate([_record(1, best_greedy=0.11, best_pass=0.20)]),
        1.0: v45.summarize_candidate([_record(1, best_greedy=0.11, best_pass=0.20)]),
    }
    assert v45.choose_candidate(tied, conservative_tie="smaller") == 0.5
    assert v45.choose_candidate(tied, conservative_tie="larger") == 1.0


def test_train_command_applies_alpha_only_to_dynamic_method(tmp_path: Path) -> None:
    common = dict(
        python=sys.executable,
        runner=ROOT / "src" / "drpo" / "countdown_qwen_arena_onefile.py",
        model=tmp_path / "model",
        reference=tmp_path / "reference",
        offline=tmp_path / "offline.jsonl",
        val=tmp_path / "val.jsonl",
        train=tmp_path / "train.jsonl",
        output=tmp_path / "output",
        plan={"micro_batch": 1, "eval_batch": 2},
        calibration=tmp_path / "calibration.json",
        seed=1234,
        taper_lambda=0.7,
        steps=10,
        min_steps=4,
        eval_every=2,
    )
    dynamic = v45._train_command(alpha=1.5, positive_only=False, **common)
    assert dynamic[dynamic.index("--method") + 1] == "bank_dynamic_controlled_negative"
    assert dynamic[dynamic.index("--negative_scale_multiplier") + 1] == "1.5"
    assert dynamic[dynamic.index("--exp_lambda") + 1] == "0.7"

    positive = v45._train_command(alpha=None, positive_only=True, **common)
    assert positive[positive.index("--method") + 1] == "positive_only"
    assert "--negative_scale_multiplier" not in positive
    assert "--negative_calibration_json" not in positive


def test_cli_and_registry_match_the_registered_validation_only_pilot() -> None:
    parser = v45.build_parser()
    args = parser.parse_args([
        "--model_path", "/tmp/model",
        "--predecessor_work_dir", "/tmp/v44",
        "--work_dir", "/tmp/v45",
    ])
    assert args.gpus == "auto"
    assert args.inside_guard is False

    registry = yaml.safe_load((ROOT / "experiments" / "registry.yaml").read_text())
    experiments = {row["id"]: row for row in registry["experiments"]}
    entry = experiments[v45.EXPERIMENT_ID]
    assert entry["status"] == "not_run"
    assert entry["execution_class"] == "pilot"
    assert entry["predecessor"] == v45.PREDECESSOR_ID
    assert entry["one_click_entrypoint"] == "scripts/run_countdown_v45_tuning.py"
    assert entry["parameterization"]["online_rollout_during_training"] is False
    assert entry["tuning_protocol"]["test_split_access"] == (
        "only_after_all_validation_selection_is_frozen"
    )
    assert entry["tuning_protocol"]["stage_a_global_negative_strength"]["values"] == [
        0.5, 1.0, 1.5, 2.0
    ]
    assert entry["tuning_protocol"]["stage_b_exponential_taper"]["values"] == [
        0.3, 0.7, 1.2
    ]
    assert entry["confirmation_protocol"]["untouched_training_seeds"] == [
        3234, 4234, 5234
    ]


def test_guard_declares_terminal_outputs_and_v45_sources() -> None:
    text = MODULE_PATH.read_text()
    for required in ("RUN_COMPLETE.json", "terminal_audit.json", "arena_summary.csv"):
        assert f'"--required-output", "{required}"' in text
    for source in (
        "scripts/run_countdown_v45_tuning.py",
        "src/drpo/countdown_qwen_arena_onefile.py",
        "docs/handoff.md",
        "experiments/registry.yaml",
    ):
        assert f'"--source-file", "{source}"' in text


def test_predecessor_paths_match_v44_outputs_and_adapter_hash_is_stable(tmp_path: Path) -> None:
    source = MODULE_PATH.read_text()
    assert 'val = data_dir / "val.jsonl"' in source
    assert 'data_dir / "validation.jsonl"' not in source
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 32}')
    nested = adapter / "nested"
    nested.mkdir()
    (nested / "weights.safetensors").write_bytes(b"weights")
    first = v45._hash_tree(adapter)
    second = v45._hash_tree(adapter)
    assert first == second
    assert sorted(first) == ["adapter_config.json", "nested/weights.safetensors"]


def test_frozen_adapter_hash_rejects_symlinks(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    target = tmp_path / "target.bin"
    target.write_bytes(b"x")
    (adapter / "linked.bin").symlink_to(target)
    with pytest.raises(RuntimeError, match="symlinks"):
        v45._hash_tree(adapter)
