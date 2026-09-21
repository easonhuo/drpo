from __future__ import annotations

import pytest
import torch

from drpo_reference.categorical.structured_generation import (
    CountdownAdapter,
    build_task_bank,
    dpo_objective,
    split_bank,
)


def test_countdown_uses_common_bank_and_split_path() -> None:
    adapter = CountdownAdapter()
    rows, instances = build_task_bank(
        adapter,
        candidate_rows=10,
        accepted_rows=6,
        negatives_per_prompt=4,
        seed=17,
    )
    assert len(rows) == 6
    assert len(instances) == 6
    assert all(row["task"] == "countdown" for row in rows)
    assert all(len(row["negatives"]) == 4 for row in rows)
    assert all(
        "surprisal" not in negative and "taper_weight" not in negative
        for row in rows
        for negative in row["negatives"]
    )

    split = split_bank(
        rows,
        train_rows=3,
        validation_rows=2,
        test_rows=1,
        seed=23,
    )
    assert {name: len(values) for name, values in split.items()} == {
        "train": 3,
        "validation": 2,
        "test": 1,
    }
    prompt_ids = [row["prompt_id"] for values in split.values() for row in values]
    assert len(set(prompt_ids)) == 6


def test_dpo_is_prompt_balanced_over_unique_rejections() -> None:
    policy_positive = torch.tensor([-1.0, -2.0])
    policy_negative = torch.tensor([-3.0, -5.0, -4.0])
    reference_positive = torch.tensor([-1.5, -2.5])
    reference_negative = torch.tensor([-2.5, -4.5, -3.5])
    row_index = torch.tensor([0, 0, 1])
    counts = torch.tensor([2, 1])
    beta = 0.2

    loss = dpo_objective(
        policy_positive,
        policy_negative,
        reference_positive,
        reference_negative,
        row_index,
        counts,
        beta=beta,
    )
    margin = (
        policy_positive[row_index]
        - policy_negative
        - reference_positive[row_index]
        + reference_negative
    )
    pair_losses = -torch.nn.functional.logsigmoid(beta * margin)
    expected = torch.stack([pair_losses[:2].mean(), pair_losses[2:].mean()]).mean()
    assert float(loss) == pytest.approx(float(expected))
