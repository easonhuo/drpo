from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "drpo"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cu1_taper_near_retention_formal as near  # noqa: E402


def _experiment() -> dict:
    registry = yaml.safe_load((ROOT / "experiments" / "registry.yaml").read_text())
    return next(
        row
        for row in registry["experiments"]
        if row["id"] == near.EXPERIMENT_ID
    )


def test_protocol_freezes_exact_families_retentions_and_seed_firewall() -> None:
    protocol = near.PROTOCOL
    assert protocol.development_seeds == tuple(range(5))
    assert protocol.formal_seeds == tuple(range(90, 110))
    assert protocol.near_region_boundary == 5.0
    assert protocol.reference_distance == 5.0
    assert protocol.retention_levels == (0.75, 0.50, 0.25)
    assert protocol.families == (
        "reciprocal_linear",
        "reciprocal_quadratic",
        "exponential",
        "squared_distance_exponential",
    )
    assert len(near.method_configs(protocol)) == 14
    assert near.method_configs(protocol)[:2] == [
        ("positive_only", None),
        ("unweighted", None),
    ]


def test_all_taper_families_have_unit_origin_and_monotone_decay() -> None:
    distances = torch.tensor([0.0, 1.0, 2.5, 5.0, 10.0], dtype=torch.float64)
    for family in near.PROTOCOL.families:
        weights = near.taper_weight_from_coefficient(
            distances,
            family=family,
            coefficient=1.3,
            reference_distance=near.PROTOCOL.reference_distance,
        )
        assert weights[0].item() == pytest.approx(1.0)
        assert torch.all(weights[:-1] >= weights[1:])
        assert torch.all((weights > 0.0) & (weights <= 1.0))

    far = torch.tensor([1e6], dtype=torch.float64)
    reciprocal_quadratic = near.taper_weight_from_coefficient(
        far, "reciprocal_quadratic", 1.0, near.PROTOCOL.reference_distance
    )
    squared_exp = near.taper_weight_from_coefficient(
        far, "squared_distance_exponential", 1.0, near.PROTOCOL.reference_distance
    )
    assert reciprocal_quadratic.item() > 0.0
    assert squared_exp.item() == 0.0


def test_deterministic_bisection_matches_each_near_retention_target() -> None:
    distances = torch.linspace(0.1, 5.0, 1000, dtype=torch.float64)
    coefficients: dict[str, float] = {}
    for family in near.PROTOCOL.families:
        coefficient, achieved = near.solve_matching_coefficient(
            distances,
            family=family,
            target_retention=0.75,
            protocol=near.PROTOCOL,
        )
        coefficients[family] = coefficient
        assert coefficient > 0.0
        assert achieved == pytest.approx(0.75, abs=near.PROTOCOL.calibration_tolerance)
        repeat, repeat_achieved = near.solve_matching_coefficient(
            distances,
            family=family,
            target_retention=0.75,
            protocol=near.PROTOCOL,
        )
        assert repeat == coefficient
        assert repeat_achieved == achieved
    assert len({round(value, 12) for value in coefficients.values()}) == 4


def test_registry_activates_only_near_retention_without_claiming_results() -> None:
    row = _experiment()
    assert row["execution_gate"]["state"] == "ready"
    assert row["implementation_state"] == "implemented"
    assert row["formal_execution"]["activation_state"] == "active"
    assert row["formal_execution"]["runner_archive_policy"]["mode"] == "forbid"
    assert row["matching_contract"]["total_negative_gradient_budget_matched"] is False
    assert row["candidate_families"] == [
        "reciprocal_linear",
        "reciprocal_quadratic",
        "current_exponential",
        "squared_distance_exponential",
    ]
    assert row["protocol"]["formal_paired_seeds"] == list(range(90, 110))
    assert row["scientific_status"] == "not_run"
    assert row["evidence"]["formal_run_started"] is False
    assert row["evidence"]["raw_complete"] is False
    assert row["next_gate"]["state"] == "blocked_until_near_retention_delivered"
    assert row["no_method_winner_assumed"] is True


def test_runner_text_uses_held_out_context_not_ood_claim() -> None:
    source = (SRC / "cu1_taper_near_retention_formal.py").read_text()
    assert "same-distribution held-out-context generalization" in source
    assert "not an OOD protocol" in source
    assert "universal family winner" in source
    assert "does not match total negative-update budget" in source


def test_engineering_smoke_writes_complete_non_scientific_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    subprocess.run(
        [
            sys.executable,
            str(SRC / "cu1_taper_near_retention_formal.py"),
            "--output-dir",
            str(output),
            "--base-commit",
            "ce5964a0c16b12626ceb81fa9813fff14893c612",
            "--smoke",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        timeout=180,
    )

    complete = json.loads((output / "RUN_COMPLETE.json").read_text())
    audit = json.loads((output / "terminal_audit.json").read_text())
    freeze = json.loads((output / "formal_protocol_freeze.json").read_text())
    calibration = json.loads((output / "calibration.json").read_text())

    assert complete["formal_run_started"] is False
    assert complete["result_status"] == "engineering_smoke"
    assert complete["coverage_checks_passed"] is True
    assert complete["aggregate_rows"] == 6
    assert audit["scientific_status"] == "not run / 尚未运行"
    assert freeze["budget_matching_performed"] is False
    assert set(freeze["candidate_formulas"]) == {
        "reciprocal_linear",
        "reciprocal_quadratic",
        "exponential",
        "squared_distance_exponential",
        "u",
    }
    assert max(row["absolute_error"] for row in calibration["calibrations"]) <= 1e-6
