from __future__ import annotations

import importlib.util
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


def test_weighted_sequence_logprob_does_not_renormalize_removed_weight() -> None:
    stats = {
        "token_lp": torch.tensor([[-1.0, -3.0]]),
        "token_mask": torch.tensor([[True, True]]),
        "lengths": torch.tensor([2]),
    }
    value = arena.weighted_sequence_logprob(stats, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(value, torch.tensor([-0.5]))


def test_checkpoint_inventory_records_hash_and_local_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "terminal_adapter"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    inventory = arena.checkpoint_inventory(checkpoint, "terminal", 12)
    assert inventory["local_only"] is True
    assert inventory["step"] == 12
    assert inventory["total_size_bytes"] == 7
    assert len(inventory["files"][0]["sha256"]) == 64


def test_run_defaults_are_base_first_bf16_lora_with_global_match() -> None:
    parser = arena.build_parser()
    args = parser.parse_args([
        "run",
        "--model_path", "/tmp/model",
        "--work_dir", "/tmp/run",
    ])
    assert args.methods == (
        "positive_only,controlled_negative,uncontrolled_negative,global_matched"
    )
    assert args.memory_mode == "bf16"
    assert args.pair_resample_rounds == 3
    assert args.min_base_success == 0.15
    assert args.min_base_valid == 0.80


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
