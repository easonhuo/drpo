from __future__ import annotations

from drpo_reference.categorical.structured_generation import (
    CountdownAdapter,
    build_task_bank,
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
