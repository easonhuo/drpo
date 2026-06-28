from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "drpo" / "countdown_qwen_arena_onefile.py"
SPEC = importlib.util.spec_from_file_location("countdown_arena", MODULE_PATH)
assert SPEC and SPEC.loader
arena = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arena
SPEC.loader.exec_module(arena)


def test_verifier_and_park_style_canonicalization() -> None:
    expression = "(1 + 2) * (3 + 4)"
    result = arena.verify_expression(expression, [1, 2, 3, 4], 21)
    assert result["correct"]
    assert arena.expression_structure(expression) == arena.expression_structure("(2 + 1) * (4 + 3)")
    assert arena.expression_structure("(1 + 2) + (3 + 4)") == arena.expression_structure("1 + (2 + (3 + 4))")
    assert arena.expression_structure("(1 * 2) * (3 * 4)") == arena.expression_structure("1 * (2 * (3 * 4))")
    assert arena.expression_structure("1 - (2 - 3)") != arena.expression_structure("(1 - 2) - 3")
    assert arena.expression_tree_depth(expression) == 2


def test_canonical_catalog_and_family_holdout_are_well_formed() -> None:
    catalog = arena.canonical_pattern_catalog(4)
    assert len(catalog) == 96
    val_seed, test_seed, val_patterns, test_patterns = arena.choose_disjoint_holdout_families(
        seed=9, val_count=15, test_count=15
    )
    assert val_seed != test_seed
    assert val_patterns
    assert test_patterns
    assert val_patterns.isdisjoint(test_patterns)
    assert val_patterns <= set(catalog)
    assert test_patterns <= set(catalog)


def test_pattern_first_split_is_disjoint_balanced_and_leakage_aware() -> None:
    train, val, test, manifest = arena.generate_structural_splits(80, 18, 18, seed=9)
    train_patterns = {row["oracle_structure"] for row in train}
    val_patterns = {row["oracle_structure"] for row in val}
    test_patterns = {row["oracle_structure"] for row in test}
    assert not train_patterns.intersection(val_patterns)
    assert not train_patterns.intersection(test_patterns)
    assert not val_patterns.intersection(test_patterns)
    train_keys = {(tuple(sorted(row["numbers"])), row["target"]) for row in train}
    val_keys = {(tuple(sorted(row["numbers"])), row["target"]) for row in val}
    test_keys = {(tuple(sorted(row["numbers"])), row["target"]) for row in test}
    assert not train_keys.intersection(val_keys)
    assert not train_keys.intersection(test_keys)
    assert not val_keys.intersection(test_keys)
    assert manifest["protocol"] == "park_inspired_pattern_first_family_holdout_capacity_audited"
    assert manifest["terminology"] == "held-out canonical pattern-family generalization"
    assert set(manifest["negative_training_allowed_patterns"]) == set(manifest["train_patterns"])
    assert manifest["structure_sets_disjoint"] is True
    assert manifest["cross_split_problem_keys_disjoint"] is True
    for split in ("train", "val", "test"):
        counts = list(manifest["per_pattern_counts"][split].values())
        assert max(counts) - min(counts) <= 1
        assert manifest["balance_summary"][split]["max_minus_min"] <= 1


def test_valid_wrong_fallback_respects_allowed_patterns() -> None:
    row = {
        "id": "x",
        "numbers": [1, 2, 3, 4],
        "target": 21,
        "oracle": "(1 + 2) * (3 + 4)",
    }
    allowed = set(arena.canonical_pattern_catalog(4))
    expression = arena.make_valid_wrong_expression(
        row, random.Random(3), allowed_patterns=allowed
    )
    result = arena.verify_expression(expression, row["numbers"], row["target"])
    assert result["valid_format"]
    assert result["uses_numbers"]
    assert not result["correct"]
    assert arena.expression_structure(expression) in allowed


def test_matched_pair_selection_never_falls_back_to_unmatched_extrema() -> None:
    candidates = [
        {"surprisal": 1.0, "token_length": 7, "tree_depth": 2, "value_error": 4.0},
        {"surprisal": 1.2, "token_length": 20, "tree_depth": 4, "value_error": 50.0},
        {"surprisal": 4.0, "token_length": 8, "tree_depth": 3, "value_error": 6.0},
    ]
    near, far, matched = arena.select_matched_negative_pair(
        candidates,
        min_surprisal_gap=0.5,
        max_token_length_diff=2,
        max_tree_depth_diff=1,
        max_value_error_ratio=4.0,
    )
    assert matched
    assert near is not None and far is not None
    assert near["surprisal"] == 1.0
    assert far["surprisal"] == 4.0

    near, far, matched = arena.select_matched_negative_pair(
        candidates[:2],
        min_surprisal_gap=0.5,
        max_token_length_diff=2,
        max_tree_depth_diff=1,
        max_value_error_ratio=4.0,
    )
    assert not matched
    assert near is None and far is None


def test_fixed_negative_bank_is_unique_spans_surprisal_and_keeps_pair() -> None:
    candidates = [
        {
            "text": f"expr_{index:02d}",
            "structure": f"pattern_{index:02d}",
            "surprisal": float(index),
            "token_length": 5 + index % 3,
            "tree_depth": 2 + index % 2,
            "value_error": 1.0 + index,
        }
        for index in range(24)
    ]
    bank = arena.select_fixed_negative_bank(
        candidates, candidates[3], candidates[20], bank_size=16
    )
    assert len(bank) == 16
    assert len({item["text"] for item in bank}) == 16
    assert candidates[3]["text"] in {item["text"] for item in bank}
    assert candidates[20]["text"] in {item["text"] for item in bank}
    surprises = [item["surprisal"] for item in bank]
    assert surprises == sorted(surprises)
    assert min(surprises) == 0.0
    assert max(surprises) == 23.0


def test_current_bank_extremes_ignore_construction_time_identity() -> None:
    bank_stats = {
        "seq_lp": torch.tensor([[-1.0], [-4.0], [-2.0], [-8.0], [-3.0], [-6.0]]),
        "token_lp": torch.arange(6, dtype=torch.float32).reshape(6, 1),
        "token_mask": torch.ones((6, 1), dtype=torch.bool),
        "lengths": torch.ones(6, dtype=torch.long),
        "entropy": torch.arange(6, dtype=torch.float32),
        "score": torch.arange(6, dtype=torch.float32),
        "token_score": torch.arange(6, dtype=torch.float32).reshape(6, 1),
    }
    near, far, near_slot, far_slot = arena.select_current_bank_extremes(
        bank_stats, batch_size=2, bank_size=3
    )
    assert near_slot.tolist() == [0, 1]
    assert far_slot.tolist() == [1, 0]
    assert near["seq_lp"].squeeze(-1).tolist() == [-1.0, -3.0]
    assert far["seq_lp"].squeeze(-1).tolist() == [-4.0, -8.0]

    moved = dict(bank_stats)
    moved["seq_lp"] = torch.tensor([[-7.0], [-4.0], [-1.0], [-2.0], [-9.0], [-6.0]])
    _, _, moved_near_slot, moved_far_slot = arena.select_current_bank_extremes(
        moved, batch_size=2, bank_size=3
    )
    assert moved_near_slot.tolist() == [2, 0]
    assert moved_far_slot.tolist() == [0, 1]


def test_offline_collator_flattens_a_uniform_negative_bank() -> None:
    def encoded(value: int) -> arena.EncodedExample:
        return arena.EncodedExample([value], [value])

    rows = []
    for offset in (0, 10):
        rows.append({
            "positive": encoded(offset + 1),
            "near": encoded(offset + 2),
            "far": encoded(offset + 3),
            "bank": [encoded(offset + 4), encoded(offset + 5), encoded(offset + 6)],
        })
    packed = arena.make_offline_collator(0)(rows)
    assert packed["bank_size"] == 3
    assert packed["bank"]["input_ids"].squeeze(-1).tolist() == [4, 5, 6, 14, 15, 16]


def test_weighted_sequence_logprob_does_not_renormalize_removed_weight() -> None:
    stats = {
        "token_lp": torch.tensor([[-1.0, -3.0]]),
        "token_mask": torch.tensor([[True, True]]),
        "lengths": torch.tensor([2]),
    }
    value = arena.weighted_sequence_logprob(stats, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(value, torch.tensor([-0.5]))


def test_dynamic_control_tapers_current_surprisal_for_both_branches() -> None:
    near = {"token_lp": torch.tensor([[-1.0, -5.0]])}
    far = {"token_lp": torch.tensor([[-1.0, -5.0]])}
    near_weights, far_weights = arena.controlled_negative_token_weights(
        "dynamic_controlled_negative", near, far, exp_lambda=0.7, surprisal_threshold=2.0
    )
    assert torch.allclose(near_weights, far_weights)
    assert near_weights[0, 0] == 1.0
    assert 0.0 < near_weights[0, 1] < 1.0


def test_static_control_keeps_initial_near_branch_untapered() -> None:
    near = {"token_lp": torch.tensor([[-10.0]])}
    far = {"token_lp": torch.tensor([[-10.0]])}
    near_weights, far_weights = arena.controlled_negative_token_weights(
        "controlled_negative", near, far, exp_lambda=0.7, surprisal_threshold=2.0
    )
    assert near_weights.item() == 1.0
    assert 0.0 < far_weights.item() < 1.0


def test_dynamic_taper_weight_decreases_when_same_negative_moves_farther() -> None:
    near_now = {"token_lp": torch.tensor([[-2.5]])}
    near_later = {"token_lp": torch.tensor([[-8.0]])}
    weight_now = arena.detached_token_surprisal_taper(near_now, 0.7, 2.0)
    weight_later = arena.detached_token_surprisal_taper(near_later, 0.7, 2.0)
    assert weight_later.item() < weight_now.item()


def test_checkpoint_inventory_records_hash_and_local_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "terminal_adapter"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    inventory = arena.checkpoint_inventory(checkpoint, "terminal", 12)
    assert inventory["local_only"] is True
    assert inventory["step"] == 12
    assert inventory["total_size_bytes"] == 7
    assert len(inventory["files"][0]["sha256"]) == 64


def test_run_defaults_are_base_first_bf16_lora_with_offline_bank() -> None:
    parser = arena.build_parser()
    args = parser.parse_args([
        "run",
        "--model_path", "/tmp/model",
        "--work_dir", "/tmp/run",
    ])
    assert args.methods == (
        "positive_only,dynamic_controlled_negative,bank_dynamic_controlled_negative,"
        "bank_global_matched,bank_uncontrolled_negative"
    )
    assert args.memory_mode == "bf16"
    assert args.gpus == "auto"
    assert args.gpu is None
    assert args.pair_resample_rounds == 8
    assert args.min_base_success == 0.15
    assert args.min_base_valid == 0.80

    build = parser.parse_args([
        "build_offline",
        "--model_path", "/tmp/model",
        "--input_data", "/tmp/train.jsonl",
        "--split_manifest", "/tmp/split.json",
        "--output_data", "/tmp/offline.jsonl",
    ])
    assert build.negative_bank_size == 16
    assert build.min_negative_candidates == 16



def test_argparse_namespace_manifest_filter_removes_func_and_paths(tmp_path: Path) -> None:
    parser = arena.build_parser()
    args = parser.parse_args([
        "sft",
        "--model_path", "/tmp/model",
        "--train_data", "/tmp/train.jsonl",
        "--val_data", "/tmp/val.jsonl",
        "--output_dir", str(tmp_path / "sft"),
    ])
    args.extra_path = tmp_path / "artifact"
    payload = arena.serializable_namespace(args)
    assert "func" not in payload
    assert payload["extra_path"] == str(tmp_path / "artifact")
    json.dumps(payload)


def test_balanced_6000_offline_quotas_and_nested_subsets() -> None:
    patterns = [f"pattern_{index:02d}" for index in range(48)]
    quotas = arena.balanced_pattern_quotas(patterns, 6000)
    assert set(quotas.values()) == {125}
    rows = [
        {
            "id": f"{pattern}_{index:03d}",
            "oracle_structure": pattern,
            "oracle": "1 + 2 + 3 + 4",
        }
        for pattern in patterns
        for index in range(125)
    ]
    subsets = arena.build_nested_balanced_subsets(rows, [1500, 3000, 6000])
    assert [len(subsets[size]) for size in (1500, 3000, 6000)] == [1500, 3000, 6000]
    assert {row["id"] for row in subsets[1500]} <= {row["id"] for row in subsets[3000]}
    assert {row["id"] for row in subsets[3000]} <= {row["id"] for row in subsets[6000]}
    for size, subset in subsets.items():
        counts: dict[str, int] = {}
        for row in subset:
            counts[row["oracle_structure"]] = counts.get(row["oracle_structure"], 0) + 1
        assert max(counts.values()) - min(counts.values()) <= 1


def test_v4_4_sft_bank_and_method_diagnostic_defaults_are_frozen() -> None:
    parser = arena.build_parser()
    sft = parser.parse_args([
        "sft",
        "--model_path", "/tmp/model",
        "--train_data", "/tmp/train.jsonl",
        "--val_data", "/tmp/val.jsonl",
        "--output_dir", "/tmp/sft",
    ])
    assert sft.epochs == 6
    assert sft.min_epochs == 3
    assert sft.early_stop_patience == 2
    assert sft.parameterization == "lora"

    method = parser.parse_args([
        "train_method",
        "--model_path", "/tmp/model",
        "--offline_data", "/tmp/offline.jsonl",
        "--val_data", "/tmp/val.jsonl",
        "--output_dir", "/tmp/method",
        "--method", "controlled_negative",
    ])
    assert method.diagnostic_examples == 32
    assert method.diagnostic_gradient_examples == 8
    assert method.diagnostic_batch == 8

    run = parser.parse_args([
        "run",
        "--model_path", "/tmp/model",
        "--work_dir", "/tmp/run",
    ])
    assert arena.EXPERIMENT_ID == "EXT-C-E8-V4.4-OFFLINE-BANK"
    assert run.min_sft_success == 0.15
    assert run.min_sft_valid == 0.95
    assert run.min_mechanism_success == 0.08
    assert run.min_mechanism_valid == 0.95

def test_negative_budget_calibration_is_non_outcome_based_and_shared() -> None:
    negative_scale, gamma = arena.calibration_scales_from_rms(
        positive_rms=2.0, controlled_rms=1.5, uncontrolled_rms=4.0
    )
    assert negative_scale == 0.5
    assert gamma == 0.375


def test_global_matched_parser_requires_explicit_calibration_at_training_time() -> None:
    parser = arena.build_parser()
    args = parser.parse_args([
        "train_method",
        "--model_path", "/tmp/model",
        "--offline_data", "/tmp/offline.jsonl",
        "--val_data", "/tmp/val.jsonl",
        "--output_dir", "/tmp/out",
        "--method", "global_matched",
    ])
    assert args.method == "global_matched"
    assert args.negative_calibration_json is None
    assert args.negative_scale is None


def test_legacy_alpha_cli_is_rejected_instead_of_silently_reused() -> None:
    parser = arena.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "train_method",
            "--model_path", "/tmp/model",
            "--offline_data", "/tmp/offline.jsonl",
            "--val_data", "/tmp/val.jsonl",
            "--output_dir", "/tmp/out",
            "--method", "uncontrolled_negative",
            "--alpha", "0.7",
        ])


def test_unseen_structure_presence_is_not_mislabeled_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "id": "heldout-1",
        "numbers": [1, 2, 3, 4],
        "target": 24,
        "prompt": "Numbers: 1, 2, 3, 4\nTarget: 24",
        "oracle": "(1 + 3) * (2 + 4)",
    }
    known_structures = {arena.expression_structure("1 * 2 * 3 * 4")}
    heldout_pattern = arena.expression_structure(row["oracle"])
    calls = iter([
        [["1 + 2 + 3 + 4"]],
        [["(1 + 3) * (2 + 4)", "(1 + 2) * (3 + 4)"]],
    ])

    def fake_generate_outputs(*args, **kwargs):  # type: ignore[no-untyped-def]
        return next(calls)

    class FakeModel:
        training = True

        def train(self) -> None:
            self.training = True

    monkeypatch.setattr(arena, "generate_outputs", fake_generate_outputs)
    metrics = arena.evaluate_rows(
        FakeModel(),
        tokenizer=None,
        rows=[row],
        batch_size=1,
        max_new_tokens=32,
        pass_k=2,
        seed=7,
        known_structures=known_structures,
    )

    assert metrics["greedy_unseen_structure_presence"] == 1.0
    assert metrics["greedy_unseen_structure_success"] == 0.0
    assert metrics["pass_at_k_unseen_structure_presence"] == 1.0
    assert metrics["pass_at_k_unseen_structure"] == 1.0
    assert metrics["pass_at_k_unseen_structure_success"] == 1.0
    assert metrics["heldout_pattern_coverage"] == 1.0
    assert metrics["heldout_pattern_family_coverage"] == 1.0
    assert metrics["greedy_heldout_pattern_attempts"] == 0.0
    assert metrics["sampled_heldout_pattern_attempts"] == 2.0
    assert metrics["sampled_heldout_pattern_correct"] == 1.0
    assert metrics["sampled_heldout_pattern_precision_micro"] == 0.5
    assert metrics["sampled_heldout_pattern_precision_macro"] == 0.5
    assert metrics["heldout_pattern_precision"] == 0.5
    assert metrics["heldout_pattern_family_precision_micro"] == 0.5
    assert metrics["heldout_pattern_family_precision_macro"] == 0.5
    assert metrics["correct_heldout_patterns"] == 1.0
    assert metrics["per_pattern_precision"]["sampled"][heldout_pattern] == {
        "attempts": 2,
        "correct": 1,
        "precision": 0.5,
    }


def test_zero_attempt_pattern_precision_is_null_not_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "id": "heldout-2",
        "numbers": [1, 2, 3, 4],
        "target": 24,
        "prompt": "Numbers: 1, 2, 3, 4\nTarget: 24",
        "oracle": "(1 + 3) * (2 + 4)",
    }
    known_structures = {arena.expression_structure("1 * 2 * 3 * 4")}
    heldout_pattern = arena.expression_structure(row["oracle"])
    calls = iter([
        [["1 * 2 * 3 * 4"]],
        [["1 * 2 * 3 * 4"]],
    ])

    monkeypatch.setattr(arena, "generate_outputs", lambda *args, **kwargs: next(calls))

    class FakeModel:
        training = False

        def train(self) -> None:
            self.training = True

    metrics = arena.evaluate_rows(
        FakeModel(), None, [row], 1, 32, 1, 7, known_structures
    )
    assert metrics["per_pattern_precision"]["greedy"][heldout_pattern]["attempts"] == 0
    assert metrics["per_pattern_precision"]["greedy"][heldout_pattern]["precision"] is None
    assert metrics["per_pattern_precision"]["sampled"][heldout_pattern]["precision"] is None


def test_trainable_parameter_snapshot_restores_exact_last_finite_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    snapshot = arena.snapshot_trainable_parameters([parameter])
    with torch.no_grad():
        parameter.copy_(torch.tensor([float("nan"), 99.0]))
    assert not arena._trainable_parameters_finite([parameter])
    arena.restore_trainable_parameters([parameter], snapshot)
    assert arena._trainable_parameters_finite([parameter])
    assert torch.equal(parameter.detach(), torch.tensor([1.0, -2.0]))

    optimizer = torch.optim.SGD([parameter], lr=1.0)
    parameter.grad = torch.tensor([float("inf"), 0.0])
    applied = arena.optimizer_step_with_last_finite_guard(optimizer, [parameter])
    assert applied is False
    assert torch.equal(parameter.detach(), torch.tensor([1.0, -2.0]))


def test_child_result_status_defaults_safe_and_accepts_top_level_value() -> None:
    parser = arena.build_parser()
    sft = parser.parse_args([
        "sft",
        "--model_path", "/tmp/model",
        "--train_data", "/tmp/train.jsonl",
        "--val_data", "/tmp/val.jsonl",
        "--output_dir", "/tmp/sft",
    ])
    assert sft.result_status == "standalone_unclassified"

    method = parser.parse_args([
        "train_method",
        "--model_path", "/tmp/model",
        "--offline_data", "/tmp/offline.jsonl",
        "--val_data", "/tmp/val.jsonl",
        "--output_dir", "/tmp/method",
        "--method", "positive_only",
        "--result_status", "engineering_smoke",
    ])
    assert method.result_status == "engineering_smoke"


def test_csv_safe_row_serializes_nested_pattern_diagnostics() -> None:
    converted = arena.csv_safe_row({"scalar": 1.0, "nested": {"b": 2, "a": 1}})
    assert converted["scalar"] == 1.0
    assert converted["nested"] == '{"a": 1, "b": 2}'


def test_gpu_resolution_is_automatic_deterministic_and_validated() -> None:
    assert arena.resolve_gpu_ids(
        "auto", visible_env="2,4,6", device_count=3
    ) == ["2", "4", "6"]
    assert arena.resolve_gpu_ids(
        "4,6", visible_env="2,4,6", device_count=3
    ) == ["4", "6"]
    with pytest.raises(ValueError, match="Duplicate"):
        arena.resolve_gpu_ids("2,2", visible_env="2,4", device_count=2)
    with pytest.raises(ValueError, match="not visible"):
        arena.resolve_gpu_ids("7", visible_env="2,4", device_count=2)


def test_registered_model_identity_uses_metadata_not_only_directory_name(tmp_path: Path) -> None:
    model = tmp_path / "Qwen2.5-0.5B"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "intermediate_size": 4864,
        "vocab_size": 151936,
        "_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
    }))
    (model / "tokenizer_config.json").write_text(json.dumps({
        "name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "chat_template": "{{ messages }}",
    }))
    metadata = arena.read_model_metadata(str(model))
    assert metadata["registered_instruct_identity"] is True
    assert metadata["has_chat_template"] is True

    (model / "tokenizer_config.json").write_text(json.dumps({"name_or_path": "Qwen/Qwen2.5-0.5B"}))
    metadata = arena.read_model_metadata(str(model))
    assert metadata["registered_instruct_identity"] is False


def test_parallel_stage_queue_preserves_fifo_per_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[tuple[str, str]] = []

    def fake_run_stage(
        argv: list[str], log_path: Path, *, gpu_id: str | None = None,
        stage_name: str | None = None, stream_output: bool = True,
    ) -> None:
        observed.append((str(gpu_id), str(stage_name)))

    monkeypatch.setattr(arena, "_run_stage", fake_run_stage)
    tasks = [
        arena.StageTask("gpu0_first", ["selftest"], tmp_path / "a.log", "0"),
        arena.StageTask("gpu1_only", ["selftest"], tmp_path / "b.log", "1"),
        arena.StageTask("gpu0_second", ["selftest"], tmp_path / "c.log", "0"),
    ]
    arena._run_stage_group(tasks)
    gpu0 = [name for gpu, name in observed if gpu == "0"]
    assert gpu0 == ["gpu0_first", "gpu0_second"]
    assert ("1", "gpu1_only") in observed


def test_effective_negative_scale_applies_registered_multiplier() -> None:
    assert arena.effective_negative_scale(0.2, 1.5) == pytest.approx(0.3)
    for base, multiplier in [
        (0.0, 1.0), (-1.0, 1.0), (float("nan"), 1.0),
        (1.0, 0.0), (1.0, -1.0), (1.0, float("inf")),
    ]:
        with pytest.raises(ValueError):
            arena.effective_negative_scale(base, multiplier)


def test_train_method_negative_scale_multiplier_defaults_to_one() -> None:
    parser = arena.build_parser()
    args = parser.parse_args([
        "train_method",
        "--model_path", "/tmp/model",
        "--offline_data", "/tmp/offline.jsonl",
        "--val_data", "/tmp/val.jsonl",
        "--output_dir", "/tmp/method",
        "--method", "bank_dynamic_controlled_negative",
    ])
    assert args.negative_scale_multiplier == 1.0
