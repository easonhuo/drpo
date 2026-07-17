from __future__ import annotations

import math

from drpo_reference.common.aggregate import aggregate_rows, audit_run_matrix


def test_aggregate_rows_preserves_event_counts() -> None:
    rows = [
        {
            "seed": 1,
            "method": "a",
            "reward": 1.0,
            "task_event": False,
            "boundary_event": True,
        },
        {
            "seed": 2,
            "method": "a",
            "reward": 3.0,
            "task_event": True,
            "boundary_event": False,
        },
    ]
    aggregate = aggregate_rows(
        rows,
        group_keys=("method",),
        event_fields=("task_event", "boundary_event"),
        bootstrap_samples=200,
    )
    assert len(aggregate) == 1
    row = aggregate[0]
    assert row["n_runs"] == 2
    assert row["n_seeds"] == 2
    assert row["reward_mean"] == 2.0
    assert math.isfinite(row["reward_ci_low"])
    assert row["task_event_count"] == 1
    assert row["task_event_rate"] == 0.5
    assert row["boundary_event_count"] == 1
    assert row["boundary_event_rate"] == 0.5


def test_audit_run_matrix_reports_missing_duplicate_and_incomplete() -> None:
    audit = audit_run_matrix(
        [
            {"seed": 1, "method": "a", "reward": 1.0},
            {"seed": 1, "method": "a", "reward": 1.0},
            {"seed": 2, "method": "unexpected"},
        ],
        identity_fields=("seed", "method"),
        expected_identities=((1, "a"), (2, "a")),
        required_fields=("seed", "method", "reward"),
    )
    assert not audit["passed"]
    assert audit["missing_identities"] == [(2, "a")]
    assert audit["unexpected_identities"] == [(2, "unexpected")]
    assert audit["duplicate_identities"] == [(1, "a")]
    assert audit["incomplete_runs"] == [
        {
            "identity": (2, "unexpected"),
            "missing_fields": ["reward"],
        }
    ]
