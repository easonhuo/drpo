from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import drpo_reference.experiments as public_experiments
from drpo_reference.controls import (
    TaperFamily,
    far_mask,
    gradient_l2_norm,
    near_mask,
    normalized_excess_surprisal,
    point_retention_coefficient,
    scale_to_match_norm,
    surprisal_distance,
    taper_weight,
)
from drpo_reference.experiments.d4rl import (
    D4RL_REVIEWER_METHOD_IDS,
    LEGACY_PILOT_CANONICAL_ALPHA,
    LEGACY_PILOT_EXPONENTIAL_COEFFICIENT,
    LEGACY_PILOT_METHOD_PROFILE,
    LEGACY_PILOT_RECIPROCAL_LINEAR_COEFFICIENT,
    LEGACY_PILOT_RECIPROCAL_QUADRATIC_COEFFICIENT,
    LEGACY_PILOT_REFERENCE_DISTANCE,
    canonical_method_negative_factors,
    canonical_standardized_action_distance,
    resolve_d4rl_reviewer_methods,
)
from drpo_reference.external.d4rl_tasks import resolve_d4rl_task


def test_cu1_point_retention_formulas_match_legacy_definitions() -> None:
    distance = torch.tensor([0.0, 2.5, 5.0, 7.5], dtype=torch.float64)
    rho = 0.25
    reference = 5.0
    normalized = distance / reference
    expected = {
        TaperFamily.RECIPROCAL_LINEAR: 1.0 / (1.0 + (1.0 / rho - 1.0) * normalized),
        TaperFamily.RECIPROCAL_QUADRATIC: 1.0
        / (1.0 + (1.0 / rho - 1.0) * normalized.square()),
        TaperFamily.EXPONENTIAL_LINEAR: torch.exp(-(-math.log(rho)) * normalized),
    }
    for family, legacy in expected.items():
        coefficient = point_retention_coefficient(
            family, retention=rho, reference_distance=reference
        )
        actual = taper_weight(distance, family=family, coefficient=coefficient)
        torch.testing.assert_close(actual, legacy, rtol=0.0, atol=1.0e-12)
        assert actual[2].item() == pytest.approx(rho, abs=1.0e-12)


def test_du1_v4_distance_coordinate_matches_legacy_formulas() -> None:
    normalized_excess = torch.tensor([0.0, 0.25, 1.0, 4.0], dtype=torch.float64)
    distance = torch.sqrt(normalized_excess)
    rho = 0.25
    reciprocal = 1.0 / rho - 1.0
    exponential = -math.log(rho)
    expected = {
        TaperFamily.RECIPROCAL_LINEAR: 1.0 / (1.0 + reciprocal * torch.sqrt(normalized_excess)),
        TaperFamily.RECIPROCAL_QUADRATIC: 1.0 / (1.0 + reciprocal * normalized_excess),
        TaperFamily.EXPONENTIAL_QUADRATIC: torch.exp(-exponential * normalized_excess),
    }
    for family, legacy in expected.items():
        coefficient = point_retention_coefficient(family, retention=rho)
        actual = taper_weight(distance, family=family, coefficient=coefficient)
        torch.testing.assert_close(actual, legacy, rtol=0.0, atol=1.0e-12)
        assert actual[2].item() == pytest.approx(rho, abs=1.0e-12)


def test_countdown_paper_aligned_weight_is_linear_in_normalized_excess() -> None:
    log_probability = torch.tensor([-1.0, -3.0, -5.0], dtype=torch.float64)
    normalized = normalized_excess_surprisal(
        log_probability, threshold=1.0, scale=2.0
    )
    distance = surprisal_distance(log_probability, threshold=1.0, scale=2.0)
    coefficient = 0.7
    actual = taper_weight(
        distance,
        family=TaperFamily.EXPONENTIAL_QUADRATIC,
        coefficient=coefficient,
    )
    torch.testing.assert_close(actual, torch.exp(-coefficient * normalized))


def test_remoteness_weights_are_detached_by_default() -> None:
    log_probability = torch.tensor([-1.0, -3.0], requires_grad=True)
    distance = surprisal_distance(log_probability, threshold=0.5, scale=2.0)
    weight = taper_weight(
        distance,
        family=TaperFamily.EXPONENTIAL_QUADRATIC,
        coefficient=1.5,
    )
    assert not distance.requires_grad
    assert not weight.requires_grad


def test_hard_masks_are_complementary_and_boundary_is_near() -> None:
    distance = torch.tensor([0.0, 4.999, 5.0, 5.001, 10.0])
    near = near_mask(distance, threshold=5.0)
    far = far_mask(distance, threshold=5.0)
    assert near.tolist() == [True, True, True, False, False]
    assert far.tolist() == [False, False, False, True, True]
    assert torch.equal(~near, far)


def test_raw_gradient_norm_and_budget_scale() -> None:
    target = [torch.tensor([3.0, 4.0]), None]
    source = [torch.tensor([1.5, 2.0])]
    assert gradient_l2_norm(target).item() == pytest.approx(5.0)
    assert gradient_l2_norm(source).item() == pytest.approx(2.5)
    scale = scale_to_match_norm(target, source)
    assert scale == pytest.approx(2.0)
    scaled = [source[0] * scale]
    assert gradient_l2_norm(scaled).item() == pytest.approx(5.0)


def test_budget_scale_fails_closed_for_nonzero_target_and_zero_source() -> None:
    with pytest.raises(ZeroDivisionError):
        scale_to_match_norm([torch.tensor([1.0])], [torch.tensor([0.0])])


def test_invalid_coordinates_fail_closed() -> None:
    with pytest.raises(ValueError):
        taper_weight(torch.tensor([-1.0]), family="reciprocal_linear", coefficient=1.0)
    with pytest.raises(ValueError):
        normalized_excess_surprisal(torch.tensor([-1.0]), threshold=0.0, scale=0.0)
    with pytest.raises(ValueError):
        near_mask(torch.tensor([float("nan")]), threshold=1.0)


def test_d4rl_reviewer_method_catalog_is_explicit_and_nonfinal() -> None:
    assert D4RL_REVIEWER_METHOD_IDS == (
        "exprank",
        "positive_only",
        "signed",
        "global",
        "reciprocal_linear",
        "reciprocal_quadratic",
        "exponential",
    )
    default = resolve_d4rl_reviewer_methods(None, method_profile=None)
    assert tuple(method.method_id for method in default) == ("exprank",)
    assert default[0].source_profile == "canonical-exprank"
    assert default[0].profile_is_final is False

    with pytest.raises(ValueError, match="explicit method_profile"):
        resolve_d4rl_reviewer_methods(
            ("exprank", "positive_only"),
            method_profile=None,
        )

    methods = resolve_d4rl_reviewer_methods(
        D4RL_REVIEWER_METHOD_IDS,
        method_profile=LEGACY_PILOT_METHOD_PROFILE,
    )
    assert tuple(method.method_id for method in methods) == D4RL_REVIEWER_METHOD_IDS
    assert all(method.profile_is_final is False for method in methods)
    assert methods[1].effective_alpha == 0.0
    assert methods[2].effective_alpha == pytest.approx(0.11)
    assert methods[3].effective_alpha == pytest.approx(0.011)


def test_d4rl_legacy_control_factors_match_registered_pilot_formulas() -> None:
    negative_advantages = torch.tensor(
        [-4.0, -1.0, -3.0, -2.0],
        dtype=torch.float64,
    )
    distances = torch.tensor([0.0, 2.0, 4.0, 6.0], dtype=torch.float64)
    methods = {
        method.method_id: method
        for method in resolve_d4rl_reviewer_methods(
            D4RL_REVIEWER_METHOD_IDS,
            method_profile=LEGACY_PILOT_METHOD_PROFILE,
        )
    }

    positive = canonical_method_negative_factors(
        negative_advantages,
        distances,
        method=methods["positive_only"],
        exprank_temperature=5.0,
    )
    signed = canonical_method_negative_factors(
        negative_advantages,
        distances,
        method=methods["signed"],
        exprank_temperature=5.0,
    )
    global_factor = canonical_method_negative_factors(
        negative_advantages,
        distances,
        method=methods["global"],
        exprank_temperature=5.0,
    )
    torch.testing.assert_close(positive, torch.zeros_like(positive))
    torch.testing.assert_close(
        signed,
        torch.full_like(signed, LEGACY_PILOT_CANONICAL_ALPHA),
    )
    torch.testing.assert_close(
        global_factor,
        torch.full_like(global_factor, LEGACY_PILOT_CANONICAL_ALPHA * 0.1),
    )

    normalized = distances / LEGACY_PILOT_REFERENCE_DISTANCE
    expected = {
        "reciprocal_linear": (
            LEGACY_PILOT_CANONICAL_ALPHA
            * 0.1
            / (
                1.0
                + LEGACY_PILOT_RECIPROCAL_LINEAR_COEFFICIENT
                * normalized
            )
        ),
        "reciprocal_quadratic": (
            LEGACY_PILOT_CANONICAL_ALPHA
            * 0.1
            / (
                1.0
                + LEGACY_PILOT_RECIPROCAL_QUADRATIC_COEFFICIENT
                * normalized.square()
            )
        ),
        "exponential": (
            LEGACY_PILOT_CANONICAL_ALPHA
            * 0.1
            * torch.exp(
                -LEGACY_PILOT_EXPONENTIAL_COEFFICIENT * normalized
            )
        ),
    }
    for method_id, expected_factor in expected.items():
        actual = canonical_method_negative_factors(
            negative_advantages,
            distances,
            method=methods[method_id],
            exprank_temperature=5.0,
        )
        torch.testing.assert_close(actual, expected_factor)


def test_d4rl_standardized_distance_is_detached() -> None:
    mean = torch.tensor([[0.0, 0.0], [1.0, -1.0]], requires_grad=True)
    log_std = torch.zeros_like(mean, requires_grad=True)
    actions = torch.tensor([[3.0, 4.0], [1.0, 1.0]], requires_grad=True)
    distance = canonical_standardized_action_distance(mean, log_std, actions)
    assert distance.tolist() == pytest.approx([math.sqrt(12.5), math.sqrt(2.0)])
    assert distance.requires_grad is False


def test_d4rl_public_runner_executes_explicit_method_axis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = resolve_d4rl_task("halfcheetah-medium-v2")
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    dataset_path = dataset_root / task.dataset_basename
    dataset_path.write_bytes(b"test-dataset")

    monkeypatch.setattr(
        public_experiments,
        "validate_dataset_path",
        lambda *args, **kwargs: {
            "task_id": task.task_id,
            "path": str(dataset_path),
            "identity_verified": False,
        },
    )
    monkeypatch.setattr(
        public_experiments,
        "load_hopper_hdf5",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        public_experiments,
        "prepare_canonical_locomotion_dataset",
        lambda data: SimpleNamespace(
            size=32,
            observation_dim=3,
            action_dim=2,
        ),
    )
    calls: list[tuple[str, Path]] = []

    def fake_train(**kwargs: object) -> dict[str, object]:
        method = kwargs["method"]
        run_root = Path(kwargs["output_root"])
        calls.append((getattr(method, "method_id"), run_root))
        checkpoint = run_root / "ckpts" / "step_0000003.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        return {
            "loss_records": [{"step": 1, "loss": 1.0}],
            "checkpoints": [str(checkpoint)],
            "agent": object(),
        }

    monkeypatch.setattr(
        public_experiments,
        "train_canonical_method",
        fake_train,
    )
    output = tmp_path / "output"
    summary = public_experiments.run_d4rl(
        dataset_root=dataset_root,
        output_root=output,
        task_ids=(task.task_id,),
        methods=("exprank", "positive_only", "exponential"),
        method_profile=LEGACY_PILOT_METHOD_PROFILE,
        seeds=(7, 8),
        steps=3,
        batch_size=4,
        device="cpu",
        smoke=False,
    )

    assert summary["expected_runs"] == 6
    assert summary["completed_runs"] == 6
    assert set(summary["tasks"][task.task_id]["methods"]) == {
        "exprank",
        "positive_only",
        "exponential",
    }
    assert {method_id for method_id, _path in calls} == {
        "exprank",
        "positive_only",
        "exponential",
    }
    for method_id, run_root in calls:
        assert run_root == (
            output / task.task_id / method_id / run_root.name
        )
    assert summary["manifest"]["final_method_matrix_frozen"] is False
    assert summary["method_ranking_claim_allowed"] is False
