from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "drpo"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cu1_taper_budget_match_formal as budget  # noqa: E402
import cu1_taper_near_retention_formal as near  # noqa: E402
import drpo_cu1_e1_e4_oneclick as experiment  # noqa: E402


def _entry(experiment_id: str) -> dict:
    registry = yaml.safe_load((ROOT / "experiments" / "registry.yaml").read_text())
    return next(row for row in registry["experiments"] if row["id"] == experiment_id)


def test_budget_protocol_freezes_primary_coordinate_and_seed_firewall() -> None:
    protocol = budget.PROTOCOL
    assert protocol.development_seeds == tuple(range(5))
    assert protocol.formal_seeds == tuple(range(110, 130))
    assert protocol.target_retention == 0.75
    assert protocol.maximum_steps == 8000
    assert protocol.matched_methods == (
        "reciprocal_linear",
        "reciprocal_quadratic",
        "exponential",
        "squared_distance_exponential",
        "global_stepwise_scale",
    )
    assert budget.method_names(protocol) == (
        "positive_only",
        "unweighted_boundary",
        *protocol.matched_methods,
    )


def test_gradient_rescaling_matches_target_without_changing_direction() -> None:
    torch.manual_seed(0)
    actor = experiment.GaussianActor().to(experiment.DEVICE)
    params = list(actor.parameters())
    positive = tuple(torch.randn_like(parameter) for parameter in params)
    negative = tuple(torch.randn_like(parameter) for parameter in params)
    raw_norm = float(experiment.norm_tuple(negative).item())
    target = 0.37 * raw_norm
    scale = target / raw_norm
    budget._set_combined_gradient(actor, positive, negative, scale)
    realized_negative = experiment.scale_tuple(negative, scale)
    assert float(experiment.norm_tuple(realized_negative).item()) == pytest.approx(
        target, rel=1e-6, abs=1e-8
    )
    for original, scaled in zip(negative, realized_negative):
        assert torch.dot(original.reshape(-1), scaled.reshape(-1)).item() > 0


def _flatten_gradients(values: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in values if value is not None])


@pytest.mark.parametrize(
    "family",
    [
        "reciprocal_linear",
        "reciprocal_quadratic",
        "exponential",
        "squared_distance_exponential",
        "unweighted",
    ],
)
def test_log_taper_formulas_match_frozen_weight_formulas(family: str) -> None:
    distance = torch.tensor([0.1, 1.0, 5.0, 20.0], dtype=torch.float64)
    coefficient = 0.47
    expected = near.taper_weight_from_coefficient(
        distance,
        family,
        coefficient,
        budget.PROTOCOL.reference_distance,
    )
    actual = torch.exp(
        budget.log_taper_weight_from_coefficient(
            distance,
            family,
            coefficient,
            budget.PROTOCOL.reference_distance,
        )
    )
    assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_common_log_weight_shift_preserves_matched_gradient_direction() -> None:
    torch.manual_seed(3)
    actor = experiment.GaussianActor().to(experiment.DEVICE)
    environment = experiment.make_environment(110)
    ids = torch.arange(64, device=experiment.DEVICE)
    ordinary = budget.negative_gradient_representation(
        actor,
        environment.train,
        ids,
        "squared_distance_exponential",
        0.46629415374943506,
        budget.PROTOCOL,
    )
    shifted = budget.negative_gradient_representation(
        actor,
        environment.train,
        ids,
        "squared_distance_exponential",
        0.46629415374943506,
        budget.PROTOCOL,
        force_stabilization=True,
    )
    assert ordinary.stabilization_used is False
    assert shifted.stabilization_used is True
    target = 0.37
    ordinary_vector = _flatten_gradients(ordinary.gradients)
    shifted_vector = _flatten_gradients(shifted.gradients)
    ordinary_matched = ordinary_vector * (target / ordinary.matching_norm)
    shifted_matched = shifted_vector * (target / shifted.matching_norm)
    assert torch.allclose(ordinary_matched, shifted_matched, rtol=2e-5, atol=2e-6)
    cosine = torch.nn.functional.cosine_similarity(
        ordinary_vector,
        shifted_vector,
        dim=0,
    )
    assert float(cosine.item()) > 0.999999


def test_vanishing_tail_recovers_nonzero_direction_after_float32_underflow() -> None:
    torch.manual_seed(5)
    actor = experiment.GaussianActor().to(experiment.DEVICE)
    with torch.no_grad():
        actor.log_std_head.weight.zero_()
        actor.log_std_head.bias.fill_(-8.0)
    environment = experiment.make_environment(110)
    ids = torch.arange(32, device=experiment.DEVICE)
    ordinary_loss = budget.weighted_negative_loss(
        actor,
        environment.train,
        ids,
        "squared_distance_exponential",
        0.46629415374943506,
        budget.PROTOCOL,
    )
    ordinary_grad = budget.taper_base.gradient_tuple(
        ordinary_loss,
        actor,
        retain_graph=False,
    )
    assert float(experiment.norm_tuple(ordinary_grad).item()) == 0.0

    recovered = budget.negative_gradient_representation(
        actor,
        environment.train,
        ids,
        "squared_distance_exponential",
        0.46629415374943506,
        budget.PROTOCOL,
    )
    assert recovered.stabilization_used is True
    assert recovered.common_log_weight_shift < -100.0
    assert recovered.matching_norm > 0.0
    assert torch.isfinite(_flatten_gradients(recovered.gradients)).all()
    target = 3.25
    realized = recovered.matching_norm * (target / recovered.matching_norm)
    assert realized == pytest.approx(target, rel=1e-12, abs=1e-12)


def test_terminal_audit_rejects_nan_inf_even_with_complete_coverage() -> None:
    protocol = replace(
        budget.PROTOCOL,
        formal_seeds=(110,),
    )
    rows = []
    for method in budget.method_names(protocol):
        rows.append(
            {
                "seed": 110,
                "method": method,
                "maximum_budget_relative_error": 0.0,
                "direction_stabilization_steps": 0,
                "task_performance_collapse_event": False,
                "support_or_variance_boundary_event": False,
                "nan_inf_numerical_event": method == "exponential",
            }
        )
    audit = budget.build_terminal_audit(rows, protocol, "0" * 40, smoke=False)
    assert audit["coverage_checks_passed"] is False
    assert audit["all_checks_passed"] is False
    assert audit["scientific_status"] == "not run / 尚未运行"
    numerical = next(
        check for check in audit["checks"] if check["name"] == "no_nan_inf_numerical_failure"
    )
    assert numerical["passed"] is False


def test_registry_deposits_budget_result_and_preserves_later_gates() -> None:
    row = _entry(budget.EXPERIMENT_ID)
    assert row["execution_gate"]["state"] == "blocked"
    assert row["implementation_state"] == "implemented"
    assert row["formal_execution"]["activation_state"] == "blocked"
    assert row["scientific_status"] == "finite_step_validated"
    assert row["evidence"]["completed_runs"] == 140
    assert row["evidence"]["terminal_audited"] is True
    assert row["formal_execution"]["entrypoint"] == ("src/drpo/cu1_taper_budget_match_formal.py")
    contract = row["budget_contract"]
    assert contract["primary_mode"] == "stepwise_raw_negative_gradient_l2_before_Adam"
    assert contract["reference_method"] == "reciprocal_linear_at_near_retention_0_75"
    assert contract["Adam_parameter_update_norm_matched"] is False
    assert contract["Adam_parameter_update_norm_logged"] is True
    assert row["protocol"]["formal_paired_seeds"] == list(range(110, 130))

    convergence = _entry("C-U1-E4-TAPER-CONV-01")
    confirmation = _entry("C-U1-E4-TAPER-CONFIRM-01")
    assert convergence["execution_gate"]["state"] == "blocked"
    assert convergence["terminal_contract"]["maximum_total_steps"] == 32000
    assert confirmation["execution_gate"]["state"] == "blocked"
    assert confirmation["confirmation_contract"]["exact_untouched_seeds"] == list(range(130, 150))


def test_budget_engineering_smoke_writes_complete_non_scientific_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "smoke"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(SRC),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    subprocess.run(
        [
            sys.executable,
            str(SRC / "cu1_taper_budget_match_formal.py"),
            "--output-dir",
            str(output),
            "--base-commit",
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "--smoke",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        timeout=180,
    )
    complete = json.loads((output / "RUN_COMPLETE.json").read_text())
    scientific = json.loads((output / "scientific_run_manifest.json").read_text())
    audit = json.loads((output / "terminal_audit.json").read_text())
    budget_audit = json.loads((output / "budget_audit.json").read_text())
    freeze = json.loads((output / "formal_protocol_freeze.json").read_text())
    assert complete["formal_run_started"] is False
    assert complete["result_status"] == "engineering_smoke"
    assert complete["coverage_checks_passed"] is True
    assert scientific["experiment_id"] == budget.EXPERIMENT_ID
    assert scientific["base_commit"] == complete["base_commit"]
    assert scientific["runs_completed"] == 7
    assert scientific["expected_runs"] == 7
    assert scientific["primary_budget_coordinate"] == (
        "stepwise_raw_negative_gradient_l2_before_Adam"
    )
    assert scientific["Adam_parameter_update_norm_matched"] is False
    assert scientific["OOD_claim_allowed"] is False
    assert audit["scientific_status"] == "not run / 尚未运行"
    assert budget_audit["maximum_relative_error"] <= 1e-6
    assert budget_audit["Adam_parameter_update_norm_matched"] is False
    stabilization = budget_audit["numerical_direction_stabilization"]
    assert stabilization["reference_reciprocal_linear_steps"] == 0
    assert stabilization["common_factor_only"] is True
    assert stabilization["taper_formulas_unchanged"] is True
    assert freeze["primary_budget_coordinate"] == ("stepwise_raw_negative_gradient_l2_before_Adam")
    numerical_contract = freeze["numerical_representation_contract"]
    assert numerical_contract["taper_formula_changed"] is False
    assert numerical_contract["gradient_direction_changed"] is False
    assert numerical_contract["reference_schedule_recentered"] is False
    assert freeze["OOD_claim_allowed"] is False
