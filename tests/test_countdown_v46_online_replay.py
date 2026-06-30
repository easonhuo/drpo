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


class _FakeModel:
    def __init__(self) -> None:
        from types import SimpleNamespace

        self.training = True
        self.is_gradient_checkpointing = True
        self.config = SimpleNamespace(use_cache=False)
        self.disable_calls = 0
        self.enable_calls = 0

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def gradient_checkpointing_disable(self) -> None:
        self.disable_calls += 1
        self.is_gradient_checkpointing = False

    def gradient_checkpointing_enable(self) -> None:
        self.enable_calls += 1
        self.is_gradient_checkpointing = True

    def enable_input_require_grads(self) -> None:
        pass


class _FakeArena:
    def __init__(self, fail_first_generate: bool = False) -> None:
        self.fail_first_generate = fail_first_generate
        self.generate_calls: list[list[str]] = []
        self.per_prompt_count: dict[str, int] = {}

    def generate_outputs(
        self,
        model,
        tokenizer,
        prompts,
        max_new_tokens,
        do_sample,
        temperature,
        top_p,
        num_return_sequences,
    ):
        self.generate_calls.append(list(prompts))
        if self.fail_first_generate:
            self.fail_first_generate = False
            # Advance all RNG streams to ensure the durable pre-call state, rather
            # than the interrupted in-memory state, is the recovery authority.
            import random

            import numpy as np
            import torch

            random.random()
            np.random.random()
            torch.rand(1)
            raise RuntimeError("synthetic collector interruption")
        groups = []
        for prompt in prompts:
            start = self.per_prompt_count.get(prompt, 0)
            groups.append(
                [f"{prompt}_wrong_{start + offset}" for offset in range(num_return_sequences)]
            )
            self.per_prompt_count[prompt] = start + num_return_sequences
        return groups

    @staticmethod
    def verify_expression(text, numbers, target):
        return {
            "expression": text,
            "valid_format": True,
            "uses_numbers": True,
            "correct": text.endswith("_correct"),
            "value": None,
        }

    @staticmethod
    def expression_structure(expression):
        return "shape"

    @staticmethod
    def score_completions_batch(model, tokenizer, pairs, max_length, batch_size):
        values = []
        for _, expression in pairs:
            tail = expression.rsplit("_", 1)[-1]
            values.append(float(tail) if tail.isdigit() else 0.25)
        return values

    @staticmethod
    def candidate_metadata(item, tokenizer):
        return {
            "text": item["expression"],
            "surprisal": float(item["surprisal"]),
            "structure": item["structure"],
        }

    @staticmethod
    def select_fixed_negative_bank(candidates, near_item, far_item, bank_size):
        return list(candidates[:bank_size])

    @staticmethod
    def serialize_negative_bank(bank):
        return [dict(item) for item in bank]

    @staticmethod
    def read_jsonl(path):
        import json

        return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _collector_rows(count: int) -> list[dict]:
    return [
        {
            "id": f"row_{index}",
            "prompt": f"prompt_{index}",
            "numbers": [1, 2, 3, 4],
            "target": 10,
            "oracle": f"oracle_{index}",
            "oracle_structure": "shape",
        }
        for index in range(count)
    ]


def _run_fake_collector(arena, model, rows, **kwargs):
    return v46.collect_online_replay_rows(
        arena,
        model,
        object(),
        rows,
        {"shape"},
        target_rows=len(rows),
        collector_round=0,
        collector_step=0,
        collector_seed=123,
        collector_method="online_positive_only",
        collector_policy_digest="digest",
        bank_size=2,
        rollouts=1,
        resample_rounds=1,
        batch_size=len(rows),
        score_batch_size=8,
        heartbeat_every=1,
        **kwargs,
    )


def test_collector_resamples_all_unresolved_prompts_as_one_batch() -> None:
    arena = _FakeArena()
    model = _FakeModel()
    rows, manifest = _run_fake_collector(arena, model, _collector_rows(2))

    assert len(rows) == 2
    assert arena.generate_calls == [
        ["prompt_1", "prompt_0"],
        ["prompt_1", "prompt_0"],
    ]
    assert manifest["initial_generation_calls"] == 1
    assert manifest["resample_generation_calls"] == 1
    assert manifest["collector_implementation"].startswith("batched_resample")
    assert model.disable_calls == model.enable_calls == 1
    assert model.training is True
    assert model.config.use_cache is False


def test_collector_preserves_durable_state_and_resumes_after_interruption(tmp_path) -> None:
    partial = tmp_path / "round_0.partial.jsonl"
    progress = tmp_path / "round_0.progress.json"
    state = tmp_path / "round_0.collector_state.pt"
    rows = _collector_rows(1)
    interrupted_arena = _FakeArena(fail_first_generate=True)

    with pytest.raises(RuntimeError, match="synthetic collector interruption"):
        _run_fake_collector(
            interrupted_arena,
            _FakeModel(),
            rows,
            partial_path=partial,
            progress_path=progress,
            state_path=state,
        )

    assert partial.is_file() and state.is_file() and progress.is_file()
    failure = __import__("json").loads(progress.read_text())
    assert failure["event"] == "collector_interrupted"
    assert failure["durable_source_cursor"] == 0
    assert failure["recovery_state_preserved"] is True

    resumed_arena = _FakeArena()
    recovered, manifest = _run_fake_collector(
        resumed_arena,
        _FakeModel(),
        rows,
        partial_path=partial,
        progress_path=progress,
        state_path=state,
    )
    assert len(recovered) == 1
    assert manifest["resumed_from_partial"] is True
    assert resumed_arena.generate_calls == [["prompt_0"], ["prompt_0"]]


def test_sequence_surprisal_only_matches_existing_definition() -> None:
    import torch
    from types import SimpleNamespace

    sys.path.insert(0, str(ROOT / "src"))
    from drpo import countdown_qwen_arena_onefile as arena

    class Model:
        def __call__(self, input_ids, attention_mask, use_cache=False):
            batch, length = input_ids.shape
            logits = torch.arange(
                batch * length * 7, dtype=torch.float32
            ).reshape(batch, length, 7) / 11.0
            return SimpleNamespace(logits=logits)

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]),
        "attention_mask": torch.ones((2, 4), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 3, 4], [-100, 3, 4, 5]]),
    }
    expected = -arena.completion_stats(Model(), batch)["seq_lp"]
    actual = arena.sequence_surprisal_only(Model(), batch)
    torch.testing.assert_close(actual, expected)


def test_online_worker_only_accepts_narrow_phase_zero_recovery_shape(tmp_path) -> None:
    output = tmp_path / "worker"
    replay = output / "replay"
    replay.mkdir(parents=True)
    assert v46._is_phase_zero_collector_recovery(output) is True

    (output / "best_adapter").mkdir()
    (replay / "round_0.partial.jsonl").write_text("")
    v46._atomic_torch_save(replay / "round_0.collector_state.pt", {"x": 1})
    assert v46._is_phase_zero_collector_recovery(output) is True

    (replay / "round_0.jsonl").write_text("")
    assert v46._is_phase_zero_collector_recovery(output) is False
