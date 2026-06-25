from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "drpo" / "countdown_qwen_arena_onefile.py"
SPEC = importlib.util.spec_from_file_location("countdown_arena", MODULE_PATH)
assert SPEC and SPEC.loader
arena = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arena
SPEC.loader.exec_module(arena)


def test_verifier_and_structure_signature() -> None:
    expr = "(1 + 2) * (3 + 4)"
    result = arena.verify_expression(expr, [1, 2, 3, 4], 21)
    assert result["correct"]
    assert arena.expression_structure(expr) == arena.expression_structure("(2 + 1) * (4 + 3)")
    assert arena.expression_tree_depth(expr) == 2


def test_structural_split_is_disjoint() -> None:
    train, val, test, manifest = arena.generate_structural_splits(50, 15, 15, seed=9)
    train_s = {row["oracle_structure"] for row in train}
    val_s = {row["oracle_structure"] for row in val}
    test_s = {row["oracle_structure"] for row in test}
    assert not train_s.intersection(val_s)
    assert not train_s.intersection(test_s)
    assert not val_s.intersection(test_s)
    train_k = {(tuple(sorted(row["numbers"])), row["target"]) for row in train}
    val_k = {(tuple(sorted(row["numbers"])), row["target"]) for row in val}
    test_k = {(tuple(sorted(row["numbers"])), row["target"]) for row in test}
    assert not train_k.intersection(val_k)
    assert not train_k.intersection(test_k)
    assert not val_k.intersection(test_k)
    assert manifest["structure_sets_disjoint"] is True


def test_valid_wrong_fallback_preserves_numbers() -> None:
    row = {
        "id": "x",
        "numbers": [1, 2, 3, 4],
        "target": 21,
        "oracle": "(1 + 2) * (3 + 4)",
    }
    expression = arena.make_valid_wrong_expression(row, random.Random(3))
    result = arena.verify_expression(expression, row["numbers"], row["target"])
    assert result["valid_format"]
    assert result["uses_numbers"]
    assert not result["correct"]


def test_matched_pair_selection_uses_surprisal_gap_and_controls() -> None:
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
    assert near["surprisal"] == 1.0
    assert far["surprisal"] == 4.0


def test_weighted_sequence_logprob_does_not_renormalize_removed_weight() -> None:
    stats = {
        "token_lp": torch.tensor([[-1.0, -3.0]]),
        "token_mask": torch.tensor([[True, True]]),
        "lengths": torch.tensor([2]),
    }
    value = arena.weighted_sequence_logprob(stats, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(value, torch.tensor([-0.5]))


def test_run_defaults_are_base_first_minimal_methods() -> None:
    parser = arena.build_parser()
    args = parser.parse_args([
        "run",
        "--model_path", "/tmp/model",
        "--work_dir", "/tmp/run",
    ])
    assert args.methods == "positive_only,controlled_negative,uncontrolled_negative"
    assert args.min_base_success == 0.15
    assert args.min_base_valid == 0.80
