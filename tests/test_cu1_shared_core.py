from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "drpo"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cu1_core  # noqa: E402
import cu1_distance_taper_formal as taper  # noqa: E402
import cu1_e1_componentwise_rerun as componentwise  # noqa: E402
import drpo_cu1_e1_e4_oneclick as experiment  # noqa: E402


def _manual_base(states: torch.Tensor) -> torch.Tensor:
    first = 0.70 * torch.tanh(
        0.85 * states[:, 0]
        - 0.30 * states[:, 1] * states[:, 2]
        + 0.20 * torch.sin(1.6 * states[:, 3])
    )
    second = 0.65 * torch.tanh(
        -0.50 * states[:, 1]
        + 0.35 * torch.cos(1.1 * states[:, 4])
        + 0.22 * states[:, 0] * states[:, 5]
    )
    return torch.stack([first, second], dim=1)


def _manual_direction(states: torch.Tensor) -> torch.Tensor:
    angle = 1.15 * torch.tanh(
        0.75 * states[:, 0] + 0.50 * states[:, 2] - 0.30 * states[:, 5]
    )
    angle = angle + 0.30 * torch.sin(1.35 * states[:, 1])
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)


def test_shared_environment_matches_frozen_reference_geometry() -> None:
    protocol = replace(
        experiment.Protocol(),
        n_train_states=32,
        n_test_states=24,
        hidden_dim=8,
    )
    environment = cu1_core.make_environment(
        31415,
        protocol,
        torch.device("cpu"),
        torch.float32,
    )
    train = environment.train

    expected_plus = _manual_base(train.s)
    expected_direction = _manual_direction(train.s)
    expected_star = expected_plus + protocol.gap_to_unseen_optimum * expected_direction

    torch.testing.assert_close(train.a_plus, expected_plus)
    torch.testing.assert_close(train.direction, expected_direction)
    torch.testing.assert_close(train.a_star, expected_star)
    torch.testing.assert_close(
        train.positive_actions.mean(dim=1),
        train.a_plus,
        atol=2e-6,
        rtol=0.0,
    )
    assert cu1_core.audit_environment(environment, protocol)["passed"] is True


def test_runners_share_environment_actor_and_output_component_implementation() -> None:
    assert experiment.Split is cu1_core.Split
    assert experiment.Environment is cu1_core.Environment
    assert issubclass(experiment.GaussianActor, cu1_core.GaussianActor)
    assert componentwise.gaussian_output_components is cu1_core.gaussian_output_components
    assert not hasattr(taper, "GaussianActor")
    assert not hasattr(taper, "make_environment")


def test_distance_tapers_share_one_distance_and_reference_attenuation() -> None:
    reference = 5.0
    rho = 0.25
    distances = torch.tensor([0.5 * reference, reference, 2.0 * reference])
    linear = cu1_core.distance_taper_weight(
        distances,
        family="reciprocal_linear",
        rho=rho,
        reference_distance=reference,
    )
    quadratic = cu1_core.distance_taper_weight(
        distances,
        family="reciprocal_quadratic",
        rho=rho,
        reference_distance=reference,
    )
    exponential = cu1_core.distance_taper_weight(
        distances,
        family="exponential",
        rho=rho,
        reference_distance=reference,
    )

    torch.testing.assert_close(linear[1], torch.tensor(rho))
    torch.testing.assert_close(quadratic[1], torch.tensor(rho))
    torch.testing.assert_close(exponential[1], torch.tensor(rho))
    assert quadratic[0] > linear[0]
    assert quadratic[2] < linear[2]


def test_quadratic_reciprocal_is_critical_for_exact_gaussian_output_score() -> None:
    distances = torch.tensor([100.0, 1000.0, 10000.0], dtype=torch.float64)
    dimension = 2
    sigma = 1.0
    exact_joint = torch.sqrt(
        distances.square() / sigma**2
        + (distances.square() - dimension).square()
    )
    reference = 1.0
    rho = 0.5
    linear = exact_joint * cu1_core.distance_taper_weight(
        distances,
        family="reciprocal_linear",
        rho=rho,
        reference_distance=reference,
    )
    quadratic = exact_joint * cu1_core.distance_taper_weight(
        distances,
        family="reciprocal_quadratic",
        rho=rho,
        reference_distance=reference,
    )
    assert linear[-1] / linear[-2] > 9.9
    assert math.isclose(float(quadratic[-1] / quadratic[-2]), 1.0, rel_tol=3e-6)


def test_positive_only_stationarity_uses_absolute_positive_gradient() -> None:
    original_protocol = experiment.P
    try:
        experiment.P = replace(
            original_protocol,
            n_train_states=16,
            n_test_states=16,
            hidden_dim=8,
        )
        environment = experiment.make_environment(7)
        experiment.seed_all(7)
        actor = experiment.GaussianActor()
        diagnostics = taper.full_field_diagnostics(
            actor,
            environment.train,
            "positive_only",
            1.0,
            taper.TaperProtocol(formal_seeds=(7,)),
        )
    finally:
        experiment.P = original_protocol

    assert diagnostics["stationarity_residual_kind"] == "absolute_positive_gradient_norm"
    assert diagnostics["stationarity_residual"] == diagnostics["positive_gradient_norm"]
    assert math.isnan(float(diagnostics["normalized_field_residual"]))



def test_gradient_diagnostic_records_output_and_full_parameter_components() -> None:
    original_protocol = experiment.P
    try:
        experiment.P = replace(
            original_protocol,
            n_train_states=8,
            n_test_states=8,
            hidden_dim=8,
        )
        environment = experiment.make_environment(11)
        experiment.seed_all(11)
        actor = experiment.GaussianActor()
        protocol = taper.TaperProtocol(formal_seeds=(11,), probe_states=2)
        rows, summary = taper.per_sample_gradient_diagnostic(
            11,
            "initial",
            actor,
            environment,
            "reciprocal_quadratic",
            0.25,
            protocol,
        )
    finally:
        experiment.P = original_protocol

    assert rows
    for key in (
        "raw_output_mean_gradient_norm",
        "raw_output_log_scale_gradient_abs",
        "raw_output_joint_gradient_norm",
        "weighted_output_joint_gradient_norm",
        "raw_full_parameter_gradient_norm",
        "weighted_full_parameter_gradient_norm",
    ):
        assert key in rows[0]
    assert "far_near_weighted_output_joint_ratio" in summary
    assert "far_near_weighted_gradient_ratio" in summary

def test_formal_taper_experiment_records_finite_step_result() -> None:
    registry = yaml.safe_load((ROOT / "experiments" / "registry.yaml").read_text())
    entry = next(item for item in registry["experiments"] if item["id"] == taper.EXPERIMENT_ID)

    assert entry["status"] == "finite_step_validated"
    assert entry["evidence"]["scientific_status"] == "finite_step_validated"
    assert entry["evidence"]["actual_runs"] == 220
    assert entry["evidence"]["terminal_audit_all_checks_passed"] is False
    assert entry["protocol"]["formal_held_out_seeds"] == list(range(70, 90))
    assert entry["protocol"]["initialization_source"] == "positive_only_adam_2000_step_checkpoint"
    assert entry["protocol"]["e2_post_2000_terminal_audit_checkpoint_used"] is False
    assert entry["depends_on_delivered_experiments"] == [
        "C-U1-E3-ADAM-RERUN",
        "C-U1-E4-ADAM-RERUN",
    ]
    assert entry["formulas"]["normalized_distance"] == "u=d/d_ref"
    assert entry["formulas"]["common_reference_alignment"] == "w(u=1)=rho"
    assert entry["formulas"]["negative_gradient_budget_matching"] is False


def test_smoke_terminal_audit_cannot_claim_scientific_validation() -> None:
    protocol = taper.TaperProtocol(formal_seeds=(70,), sensitivity_rhos=())
    audit = taper.build_terminal_audit(
        [],
        {},
        protocol,
        base_commit="0" * 40,
        smoke=True,
    )
    assert audit["execution_status"] == "engineering_smoke"
    assert audit["scientific_status"] == "not run / 尚未运行"
    assert audit["all_checks_passed"] is False  # no runs were supplied


def test_two_times_terminal_audit_is_never_truncated() -> None:
    assert taper.two_times_audit_target(4000, 8000) == 8000
    assert taper.two_times_audit_target(4001, 8000) is None
    assert taper.two_times_audit_target(8000, 8000) is None


def test_formal_terminal_audit_rejects_unresolved_fixed_horizon() -> None:
    protocol = taper.TaperProtocol(formal_seeds=(70,), sensitivity_rhos=())
    summaries = []
    for family, rho in taper.method_configs(protocol):
        summaries.append(
            {
                "seed": 70,
                "family": family,
                "rho": rho,
                "stop_reason": "maximum_steps",
                "task_performance_collapse_event": False,
                "support_or_variance_boundary_event": False,
                "nan_inf_numerical_event": False,
            }
        )
    audit = taper.build_terminal_audit(
        summaries,
        {"paired_seeds": 1},
        protocol,
        base_commit="0" * 40,
        smoke=False,
    )
    checks = {row["name"]: row for row in audit["checks"]}
    assert checks["all_runs_terminally_resolved"]["passed"] is False
    assert audit["all_checks_passed"] is False
    assert audit["scientific_status"] == "finite-step validated / 有限训练步数验证"
