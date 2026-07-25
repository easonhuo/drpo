from __future__ import annotations

import pytest

from drpo.e7_sqexp_gae_aggregate import EXPECTED_PAIRS, _pair_rows
from drpo.e7_sqexp_gae_protocol import (
    ACTOR_MODES,
    COEFFICIENTS,
    EXPECTED_DATASETS,
    EXPECTED_SEEDS,
)


def _rows():
    rows = []
    for dataset in EXPECTED_DATASETS:
        for seed in EXPECTED_SEEDS:
            for actor in ACTOR_MODES:
                for coefficient in (None, *COEFFICIENTS):
                    for estimator, offset in (("td", 0.0), ("gae", 1.5)):
                        rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "actor_update_mode": actor,
                                "exp_coefficient": coefficient,
                                "advantage_estimator": estimator,
                                "score_at_500k": 10.0 + offset,
                                "late_window_mean_800k_1m": 20.0 + offset,
                                "final_score": 30.0 + offset,
                                "best_score": 40.0 + offset,
                            }
                        )
    return rows


def test_pair_rows_is_exact_and_never_imputes() -> None:
    rows = _rows()
    paired = _pair_rows(rows)
    assert len(paired) == EXPECTED_PAIRS
    assert {row["gae_minus_td_late_mean"] for row in paired} == {1.5}
    rows.pop()
    with pytest.raises(RuntimeError, match="missing paired TD/GAE cell"):
        _pair_rows(rows)
