from __future__ import annotations

import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from drpo_reference.categorical.countdown import (
    IGNORE_INDEX,
    clean_expression,
    collate_countdown_training_items,
    completion_statistics_from_logits,
    countdown_objective_from_model,
    countdown_training_objective,
    countdown_weight_diagnostics,
    encode_countdown_training_row,
    encode_prompt_completion,
    evaluate_response_batches,
    normalized_sequence_surprisal,
    pad_encoded,
    paper_aligned_linear_weights,
    parameter_update_norm,
    unique_negative_expressions,
    verifier_category,
    verify_expression,
    weighted_sequence_logprob,
)
from drpo_reference.common import (
    atomic_json,
    cpu_generator,
    read_csv,
    seed_all,
    write_csv,
)


class _CharacterTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = False,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return "|".join(message["content"] for message in messages) + "|ASSISTANT:"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


class _CountingLogitModel(torch.nn.Module):
    def __init__(self, vocabulary_size: int = 128) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.linspace(-0.2, 0.2, vocabulary_size))
        self.calls = 0

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        assert attention_mask.shape == input_ids.shape
        assert use_cache is False
        self.calls += 1
        batch, sequence = input_ids.shape
        logits = self.bias.view(1, 1, -1).expand(batch, sequence, -1)
        return SimpleNamespace(logits=logits)


def _training_rows() -> list[dict[str, Any]]:
    return [
        {
            "prompt": "Numbers: 1, 2, 3, 4\nTarget: 10",
            "positive": "1 + 2 + 3 + 4",
            "negative_bank": [
                {"expression": "Answer: 1 + 2"},
                "1 + 2",
                {"expression": "3 * 4"},
            ],
        },
        {
            "prompt": "Numbers: 2, 2, 3, 5\nTarget: 12",
            "positive": "2 + 2 + 3 + 5",
            "negative_bank": ["2 * 5", {"expression": "Answer: 2 * 5"}],
        },
    ]


def test_seed_all_matches_legacy_seed_order() -> None:
    seed_all(20260624)
    actual = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )
    random.seed(20260624)
    np.random.seed(20260624)
    torch.manual_seed(20260624)
    expected = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )
    assert actual == expected


def test_cpu_generator_is_independent_and_repeatable() -> None:
    first = torch.randint(0, 1000, (12,), generator=cpu_generator(17))
    second = torch.randint(0, 1000, (12,), generator=cpu_generator(17))
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_atomic_json_matches_legacy_text_format(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "payload.json"
    payload = {"ascii": 1, "中文": [True, None]}
    atomic_json(path, payload)
    expected = json.dumps(payload, indent=2, ensure_ascii=False)
    assert path.read_text(encoding="utf-8") == expected
    assert not path.with_suffix(".json.tmp").exists()


def test_csv_round_trip_preserves_first_seen_field_order(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    rows = [
        {"seed": 1, "reward": 0.5},
        {"seed": 2, "status": "stable", "reward": 0.7},
    ]
    write_csv(path, rows)
    assert path.read_text(encoding="utf-8").splitlines()[0] == "seed,reward,status"
    assert read_csv(path) == [
        {"seed": "1", "reward": "0.5", "status": ""},
        {"seed": "2", "reward": "0.7", "status": "stable"},
    ]


def test_countdown_cleaner_verifier_and_categories_match_stable_contract() -> None:
    text = "<think>ignore me</think>\n<answer>(1 + 2) * (3 + 4)</answer>."
    assert clean_expression(text) == "(1 + 2) * (3 + 4)"

    correct = verify_expression(text, [1, 2, 3, 4], 21)
    mismatch = verify_expression("1 + 2 + 3", [1, 2, 3, 4], 6)
    wrong = verify_expression("1 + 2 + 3 + 4", [1, 2, 3, 4], 11)
    invalid = verify_expression("__import__('os')", [1, 2, 3, 4], 10)

    assert correct["correct"] is True
    assert verifier_category(correct) == "correct"
    assert mismatch["valid_format"] is True
    assert mismatch["uses_numbers"] is False
    assert verifier_category(mismatch) == "number_mismatch"
    assert wrong["uses_numbers"] is True
    assert wrong["correct"] is False
    assert verifier_category(wrong) == "arithmetic_wrong"
    assert invalid["valid_format"] is False
    assert verifier_category(invalid) == "invalid_format"


def test_countdown_encoding_masks_prompt_and_requires_completion() -> None:
    tokenizer = _CharacterTokenizer()
    encoded = encode_prompt_completion(
        tokenizer,
        "Numbers: 1, 2, 3, 4\nTarget: 10",
        "Answer: 1 + 2 + 3 + 4.",
        max_length=4096,
    )
    active = [label for label in encoded.labels if label != IGNORE_INDEX]
    expected_completion = tokenizer(
        "1 + 2 + 3 + 4" + tokenizer.eos_token,
        add_special_tokens=False,
    )["input_ids"]
    assert active == expected_completion
    assert encoded.labels[: -len(active)] == [IGNORE_INDEX] * (
        len(encoded.labels) - len(active)
    )

    prefix_only_length = len(encoded.input_ids) - len(active)
    with pytest.raises(ValueError, match="no completion token"):
        encode_prompt_completion(
            tokenizer,
            "Numbers: 1, 2, 3, 4\nTarget: 10",
            "1 + 2 + 3 + 4",
            max_length=prefix_only_length,
        )


def test_countdown_padding_and_completion_statistics_use_completion_only() -> None:
    tokenizer = _CharacterTokenizer()
    first = encode_prompt_completion(tokenizer, "p", "1 + 2", 1024)
    second = encode_prompt_completion(tokenizer, "longer prompt", "3 * 4", 1024)
    padded = pad_encoded((first, second), pad_id=0)
    assert padded["input_ids"].shape == padded["labels"].shape
    assert padded["attention_mask"].shape == padded["labels"].shape
    assert bool((padded["labels"][padded["attention_mask"] == 0] == IGNORE_INDEX).all())

    logits = torch.zeros((1, 4, 3), dtype=torch.float64)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 1, 2]])
    stats = completion_statistics_from_logits(logits, labels)
    expected_log_probability = -math.log(3.0)
    assert stats["token_mask"].tolist() == [[False, True, True]]
    assert stats["lengths"].tolist() == [2]
    assert stats["seq_lp"].item() == pytest.approx(expected_log_probability)
    assert stats["entropy"].item() == pytest.approx(math.log(3.0))
    assert stats["score"].item() == pytest.approx(math.sqrt(2.0 / 3.0))

    weighted = weighted_sequence_logprob(
        stats,
        torch.tensor([[0.0, 1.0, 0.0]]),
    )
    assert weighted.item() == pytest.approx(expected_log_probability / 2.0)


def test_countdown_linear_surprisal_weights_are_detached_and_not_squared() -> None:
    sequence_log_probability = torch.tensor(
        [-2.0, -4.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    coordinate = normalized_sequence_surprisal(sequence_log_probability)
    actual = paper_aligned_linear_weights(
        sequence_log_probability,
        alpha=0.5,
        coefficient=0.7,
    )
    expected = 0.5 * torch.exp(
        -0.7 * torch.tensor([1.0, 2.0], dtype=torch.float64)
    )
    squared_alternative = 0.5 * torch.exp(
        -0.7 * torch.tensor([1.0, 4.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        coordinate,
        torch.tensor([1.0, 2.0], dtype=torch.float64),
    )
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, squared_alternative)
    assert coordinate.requires_grad is False
    assert actual.requires_grad is False


def test_countdown_unique_bank_uses_cleaned_first_occurrence() -> None:
    row: dict[str, Any] = {
        "negative_bank": [
            {"expression": "Answer: 1 + 2"},
            "1 + 2",
            {"expression": "```python\n3 * 4\n```"},
        ]
    }
    assert unique_negative_expressions(row) == ["1 + 2", "3 * 4"]


def test_countdown_training_collator_preserves_unique_negative_denominator() -> None:
    tokenizer = _CharacterTokenizer()
    items = [
        encode_countdown_training_row(row, tokenizer, 4096)
        for row in _training_rows()
    ]
    assert [item.unique_count for item in items] == [2, 1]
    assert [item.raw_bank_count for item in items] == [3, 2]

    packed = collate_countdown_training_items(items, pad_id=0)
    assert packed["bank"]["input_ids"].shape[0] == 3
    assert packed["bank_row_index"].tolist() == [0, 0, 1]
    assert packed["unique_counts"].tolist() == [2, 1]
    assert packed["raw_bank_counts"].tolist() == [3, 2]


def test_countdown_training_objective_matches_legacy_formula() -> None:
    positive = torch.tensor([-0.5, -0.8], dtype=torch.float64)
    negative = torch.tensor([-2.0, -4.0, -3.0], dtype=torch.float64)
    row_index = torch.tensor([0, 0, 1])
    counts = torch.tensor([2, 1])
    alpha = 0.5
    coefficient = 0.7

    actual = countdown_training_objective(
        positive,
        alpha=alpha,
        coefficient=coefficient,
        negative_sequence_log_probability=negative,
        row_index=row_index,
        unique_counts=counts,
    )
    legacy_weights = (
        alpha * torch.exp(-coefficient * ((-negative.detach()).clamp_min(0.0) / 2.0))
    ).detach()
    legacy_negative = torch.stack(
        [
            (legacy_weights[:2] * negative[:2]).sum() / 2.0,
            legacy_weights[2] * negative[2],
        ]
    ).mean()
    legacy_loss = -(positive.mean() - legacy_negative)

    torch.testing.assert_close(actual["weights"], legacy_weights)
    torch.testing.assert_close(actual["weighted_negative_lp"], legacy_negative)
    torch.testing.assert_close(actual["loss"], legacy_loss)
    assert actual["negative_evaluated"] is True

    positive_only = countdown_training_objective(
        positive,
        alpha=0.0,
        coefficient=9.0,
    )
    torch.testing.assert_close(positive_only["loss"], -positive.mean())
    assert positive_only["negative_evaluated"] is False
    assert positive_only["weights"].numel() == 0


def test_countdown_model_objective_skips_bank_forward_for_positive_only() -> None:
    tokenizer = _CharacterTokenizer()
    items = [
        encode_countdown_training_row(row, tokenizer, 4096)
        for row in _training_rows()
    ]
    packed = collate_countdown_training_items(items, pad_id=0)
    model = _CountingLogitModel()

    positive_only = countdown_objective_from_model(
        model,
        packed,
        alpha=0.0,
        coefficient=0.0,
    )
    assert model.calls == 1
    assert positive_only["negative_stats"] is None
    assert positive_only["negative_evaluated"] is False

    model.calls = 0
    controlled = countdown_objective_from_model(
        model,
        packed,
        alpha=1.0,
        coefficient=0.7,
    )
    assert model.calls == 2
    assert controlled["negative_stats"] is not None
    assert controlled["negative_evaluated"] is True


def test_countdown_weight_diagnostics_match_frozen_bank_semantics() -> None:
    sequence_log_probability = torch.tensor([-2.0, -4.0, -6.0])
    weights = paper_aligned_linear_weights(
        sequence_log_probability,
        alpha=1.0,
        coefficient=0.7,
    )
    diagnostics = countdown_weight_diagnostics(
        sequence_log_probability,
        weights,
        unique_counts=torch.tensor([2, 1]),
        raw_bank_counts=torch.tensor([3, 2]),
    )
    assert diagnostics["negative_surprisal_mean"] == pytest.approx(4.0)
    assert diagnostics["u_mean"] == pytest.approx(2.0)
    assert diagnostics["unique_negative_count_mean"] == pytest.approx(1.5)
    assert diagnostics["raw_bank_count_mean"] == pytest.approx(2.5)
    assert diagnostics["duplicates_removed_mean"] == pytest.approx(1.0)
    assert diagnostics["weight_mean"] == pytest.approx(float(weights.mean()))


def test_countdown_first_adamw_update_matches_manual_legacy_objective() -> None:
    public_parameter = torch.nn.Parameter(torch.tensor([0.2, -0.3]))
    legacy_parameter = torch.nn.Parameter(public_parameter.detach().clone())
    public_optimizer = torch.optim.AdamW(
        [public_parameter],
        lr=1.0e-2,
        weight_decay=0.01,
    )
    legacy_optimizer = torch.optim.AdamW(
        [legacy_parameter],
        lr=1.0e-2,
        weight_decay=0.01,
    )
    row_index = torch.tensor([0, 0, 1])
    counts = torch.tensor([2, 1])
    alpha = 0.5
    coefficient = 0.7

    def log_probabilities(
        parameter: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positive = torch.stack(
            [
                -((parameter[0] - 0.5).square() + 0.2),
                -((parameter[1] + 0.1).square() + 0.3),
            ]
        )
        negative = torch.stack(
            [
                -((parameter[0] + 0.4).square() + 0.5),
                -((parameter[1] - 0.6).square() + 1.0),
                -((parameter.sum()).square() + 0.7),
            ]
        )
        return positive, negative

    public_positive, public_negative = log_probabilities(public_parameter)
    public_terms = countdown_training_objective(
        public_positive,
        alpha=alpha,
        coefficient=coefficient,
        negative_sequence_log_probability=public_negative,
        row_index=row_index,
        unique_counts=counts,
    )
    public_optimizer.zero_grad(set_to_none=True)
    public_terms["loss"].backward()
    torch.nn.utils.clip_grad_norm_([public_parameter], 1.0)
    public_before = [public_parameter.detach().cpu().clone()]
    public_optimizer.step()

    legacy_positive, legacy_negative = log_probabilities(legacy_parameter)
    legacy_weights = (
        alpha
        * torch.exp(
            -coefficient * ((-legacy_negative.detach()).clamp_min(0.0) / 2.0)
        )
    ).detach()
    legacy_negative_term = torch.stack(
        [
            (legacy_weights[:2] * legacy_negative[:2]).sum() / 2.0,
            legacy_weights[2] * legacy_negative[2],
        ]
    ).mean()
    legacy_loss = -(legacy_positive.mean() - legacy_negative_term)
    legacy_optimizer.zero_grad(set_to_none=True)
    legacy_loss.backward()
    torch.nn.utils.clip_grad_norm_([legacy_parameter], 1.0)
    legacy_before = [legacy_parameter.detach().cpu().clone()]
    legacy_optimizer.step()

    torch.testing.assert_close(public_parameter, legacy_parameter, rtol=0.0, atol=0.0)
    assert parameter_update_norm(
        public_before,
        [public_parameter],
    ) == pytest.approx(
        parameter_update_norm(legacy_before, [legacy_parameter]),
        abs=0.0,
    )


def test_countdown_response_metrics_are_lightweight_and_nonfinal() -> None:
    rows = [
        {"numbers": [1, 2, 3, 4], "target": 21},
        {"numbers": [1, 2, 3, 4], "target": 10},
    ]
    metrics = evaluate_response_batches(
        rows,
        greedy_outputs=["(1 + 2) * (3 + 4)", "1 + 2 + 3"],
        sampled_outputs=[
            ["(1 + 2) * (3 + 4)", "1 + 2 + 3 + 4"],
            ["1 + 2 + 3", "1 + 2 + 3 + 4"],
        ],
    )
    assert metrics["n_eval"] == 2
    assert metrics["pass_k"] == 2
    assert metrics["greedy_success"] == pytest.approx(0.5)
    assert metrics["pass_at_k"] == pytest.approx(1.0)
    assert metrics["valid_rate"] == pytest.approx(0.5)
    assert metrics["greedy_verifier_categories"] == {
        "correct": 1,
        "number_mismatch": 1,
    }
    assert metrics["formal_result_claim"] is False
    assert metrics["final_countdown_protocol_frozen"] is False
