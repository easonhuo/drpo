from __future__ import annotations

import math

import pytest
import torch

from drpo_reference.categorical.countdown import (
    CountdownAdapter,
    asymre_objective,
    build_task_bank,
    dpo_objective,
    drpo_weights,
    split_bank,
    topr_policy_objective,
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
    prompt_ids = [
        row["prompt_id"]
        for values in split.values()
        for row in values
    ]
    assert len(set(prompt_ids)) == 6


def test_drpo_weight_is_one_near_threshold_and_decays_far() -> None:
    mean_logprob = torch.tensor([-1.0, -2.0, -4.0], requires_grad=True)
    weights = drpo_weights(
        mean_logprob,
        threshold=2.0,
        scale=2.0,
        coefficient=1.5,
    )
    assert weights.requires_grad is False
    assert weights.tolist()[:2] == pytest.approx([1.0, 1.0])
    assert float(weights[2]) == pytest.approx(math.exp(-1.5))


def test_asymre_branch_balanced_signed_reward_objective() -> None:
    positive = torch.tensor([-1.0, -3.0])
    negative = torch.tensor([-2.0, -4.0, -6.0, -8.0])
    row_index = torch.tensor([0, 0, 1, 1])
    counts = torch.tensor([2, 2])
    loss = asymre_objective(
        positive,
        negative,
        row_index,
        counts,
        delta_v=-0.5,
    )
    positive_mean = -2.0
    negative_prompt_mean = ((-2.0 - 4.0) / 2 + (-6.0 - 8.0) / 2) / 2
    expected_objective = 0.5 * (
        1.5 * positive_mean + (-0.5) * negative_prompt_mean
    )
    assert float(loss) == pytest.approx(-expected_objective)


def test_joint_topr_uses_detached_full_sequence_ratio_weight() -> None:
    positive = torch.tensor([-1.0])
    negative_mean = torch.tensor([-2.0, -4.0], requires_grad=True)
    policy_sum = torch.tensor([-4.0, -2.0], requires_grad=True)
    reference_sum = torch.tensor([-2.0, -4.0])
    row_index = torch.tensor([0, 0])
    counts = torch.tensor([2])

    _, weights = topr_policy_objective(
        positive,
        negative_mean,
        policy_sum,
        reference_sum,
        row_index,
        counts,
        beta=0.5,
    )
    assert weights.requires_grad is False
    assert weights.tolist() == pytest.approx([math.exp(-1.0), 1.0])


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
    expected = torch.stack(
        [pair_losses[:2].mean(), pair_losses[2:].mean()]
    ).mean()
    assert float(loss) == pytest.approx(float(expected))
