from __future__ import annotations

import pytest

from drpo import e7_ppo_w0_runtime_autotune as autotune


def test_candidate_grid_includes_half_fallback_three_quarters_and_ceiling() -> None:
    assert autotune.candidate_workers(186, 60) == [60, 93, 140, 186]
    assert autotune.candidate_workers(96, 60) == [48, 60, 72, 96]


def test_retained_peak_selects_smallest_near_peak_candidate() -> None:
    selected, rule = autotune.select_from_throughput(
        [
            {"concurrency": 48, "aggregate_updates_per_second": 100.0, "valid": True},
            {"concurrency": 60, "aggregate_updates_per_second": 103.0, "valid": True},
            {"concurrency": 72, "aggregate_updates_per_second": 104.0, "valid": True},
            {"concurrency": 96, "aggregate_updates_per_second": 101.0, "valid": True},
        ],
        retention_fraction=0.97,
    )
    assert selected == 60
    assert rule["peak_aggregate_updates_per_second"] == pytest.approx(104.0)


def test_retained_peak_rejects_all_failed_candidates() -> None:
    with pytest.raises(
        autotune.RuntimeResourceError,
        match="no resource-valid concurrency candidate completed",
    ):
        autotune.select_from_throughput(
            [
                {
                    "concurrency": 96,
                    "aggregate_updates_per_second": 0.0,
                    "valid": False,
                }
            ],
            retention_fraction=0.97,
        )


def test_probe_template_changes_only_runtime_terminal_evaluation() -> None:
    run_spec = {
        "trainer_argv_template": [
            "--steps",
            "{steps}",
            "--eval_interval",
            "50000",
            "--eval_episodes",
            "10",
        ]
    }
    result = autotune._probe_trainer_template(run_spec, probe_steps=5000)
    assert result[result.index("--eval_interval") + 1] == "5000"
    assert result[result.index("--eval_episodes") + 1] == "1"
    assert run_spec["trainer_argv_template"][3] == "50000"
    assert run_spec["trainer_argv_template"][5] == "10"
