from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from drpo_reference.common import (
    atomic_json,
    cpu_generator,
    read_csv,
    seed_all,
    write_csv,
)


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
