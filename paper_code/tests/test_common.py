from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from drpo_reference.categorical.countdown import (
    IGNORE_INDEX,
    clean_expression,
    completion_statistics_from_logits,
    encode_prompt_completion,
    pad_encoded,
    prompt_balanced_mean,
    verify_expression,
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


def test_countdown_adapter_verifier_is_task_local_only() -> None:
    text = "<think>ignore me</think>\n<answer>(1 + 2) * (3 + 4)</answer>."
    assert clean_expression(text) == "(1 + 2) * (3 + 4)"

    correct = verify_expression(text, [1, 2, 3, 4], 21)
    mismatch = verify_expression("1 + 2 + 3", [1, 2, 3, 4], 6)
    wrong = verify_expression("1 + 2 + 3 + 4", [1, 2, 3, 4], 11)
    invalid = verify_expression("__import__('os')", [1, 2, 3, 4], 10)

    assert correct["correct"] is True
    assert mismatch["valid_format"] is True
    assert mismatch["uses_numbers"] is False
    assert wrong["uses_numbers"] is True
    assert wrong["correct"] is False
    assert invalid["valid_format"] is False


def test_structured_encoding_masks_prompt_and_padding() -> None:
    tokenizer = _CharacterTokenizer()
    encoded = encode_prompt_completion(
        tokenizer,
        "Numbers: 1, 2, 3, 4\nTarget: 10",
        "1 + 2 + 3 + 4",
        max_length=4096,
    )
    active = [label for label in encoded.labels if label != IGNORE_INDEX]
    expected_completion = tokenizer(
        "1 + 2 + 3 + 4" + tokenizer.eos_token,
        add_special_tokens=False,
    )["input_ids"]
    assert active == expected_completion

    second = encode_prompt_completion(tokenizer, "longer prompt", "3 * 4", 4096)
    padded = pad_encoded((encoded, second), pad_id=0)
    assert padded["input_ids"].shape == padded["labels"].shape
    assert padded["attention_mask"].shape == padded["labels"].shape
    assert bool(
        (
            padded["labels"][padded["attention_mask"] == 0]
            == IGNORE_INDEX
        ).all()
    )


def test_completion_statistics_use_completion_tokens_only() -> None:
    logits = torch.zeros((1, 4, 3), dtype=torch.float64)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 1, 2]])
    stats = completion_statistics_from_logits(logits, labels)
    expected_token_log_probability = -math.log(3.0)

    assert stats["token_mask"].tolist() == [[False, True, True]]
    assert stats["lengths"].tolist() == [2]
    assert stats["mean_logprob"].item() == pytest.approx(
        expected_token_log_probability
    )
    assert stats["sum_logprob"].item() == pytest.approx(
        2.0 * expected_token_log_probability
    )


def test_prompt_balanced_mean_is_not_response_count_weighted() -> None:
    values = torch.tensor([1.0, 3.0, 10.0])
    row_index = torch.tensor([0, 0, 1])
    counts = torch.tensor([2, 1])
    actual = prompt_balanced_mean(values, row_index, counts)
    expected = ((1.0 + 3.0) / 2.0 + 10.0) / 2.0
    assert float(actual) == pytest.approx(expected)
