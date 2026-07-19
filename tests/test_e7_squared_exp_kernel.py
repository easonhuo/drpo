from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from drpo import e7_canonical_injection as canonical
from drpo import e7_squared_exp_night as night
from drpo.e7_squared_exp_kernel import (
    THRESHOLDED_FORMULA,
    install_squared_exponential_kernel,
    squared_exponential_factor,
)
from drpo.e7_squared_exp_night_bootstrap import (
    _internal_control,
    _validate_weight_control,
)


TUNING_GRID = Path("configs/e7_bench_joint_gae_tuning_p1_c.json")
HISTORICAL_GRID = Path("configs/e7_squared_exp_night_v1.json")


def test_squared_exponential_factor_matches_registered_formula() -> None:
    distance = torch.tensor([0.0, 1.0, 2.0, 4.0])
    actual = squared_exponential_factor(
        distance,
        coefficient=1.0,
        reference_distance=2.0,
    )
    expected = torch.exp(-torch.tensor([0.0, 0.25, 1.0, 4.0]))
    assert torch.allclose(actual, expected)


def test_thresholded_factor_preserves_near_field_and_zero_tau_is_legacy() -> None:
    distance = torch.tensor([0.0, 1.0, 2.0, 4.0])
    thresholded = squared_exponential_factor(
        distance,
        coefficient=0.5,
        reference_distance=2.0,
        remoteness_threshold=1.0,
    )
    expected = torch.exp(-0.5 * torch.tensor([0.0, 0.0, 0.0, 3.0]))
    assert torch.allclose(thresholded, expected)

    implicit_zero = squared_exponential_factor(
        distance,
        coefficient=0.5,
        reference_distance=2.0,
    )
    explicit_zero = squared_exponential_factor(
        distance,
        coefficient=0.5,
        reference_distance=2.0,
        remoteness_threshold=0.0,
    )
    torch.testing.assert_close(implicit_zero, explicit_zero, rtol=0.0, atol=0.0)


def test_squared_kernel_preserves_near_and_suppresses_far_relative_to_linear() -> None:
    near = 0.5
    far = 2.0
    squared_near = math.exp(-(near**2))
    linear_near = math.exp(-near)
    squared_far = math.exp(-(far**2))
    linear_far = math.exp(-far)
    assert squared_near > linear_near
    assert squared_far < linear_far


def test_install_squared_kernel_changes_only_exponential_and_restores() -> None:
    distance = torch.tensor([1.0, 2.0])
    exponential = canonical.NegativeControl(
        method="exponential",
        negative_scale=1.0,
        canonical_alpha=0.11,
        reference_distance=2.0,
        exponential_coefficient=1.0,
    )
    reciprocal = canonical.NegativeControl(
        method="reciprocal_quadratic",
        negative_scale=1.0,
        canonical_alpha=0.11,
        reference_distance=2.0,
        reciprocal_quadratic_coefficient=1.0,
    )
    original = canonical.taper_factor
    linear_value = original(distance, exponential)
    reciprocal_value = original(distance, reciprocal)
    with install_squared_exponential_kernel(remoteness_threshold=0.25):
        assert canonical.taper_factor is not original
        assert torch.allclose(
            canonical.taper_factor(distance, exponential),
            torch.exp(-torch.tensor([0.0, 0.75])),
        )
        assert torch.allclose(
            canonical.taper_factor(distance, reciprocal),
            reciprocal_value,
        )
    assert canonical.taper_factor is original
    assert torch.allclose(canonical.taper_factor(distance, exponential), linear_value)


def test_squared_kernel_rejects_invalid_inputs() -> None:
    distance = torch.tensor([1.0])
    with pytest.raises(ValueError, match="coefficient"):
        squared_exponential_factor(
            distance,
            coefficient=-1.0,
            reference_distance=2.0,
        )
    with pytest.raises(ValueError, match="remoteness_threshold"):
        squared_exponential_factor(
            distance,
            coefficient=1.0,
            reference_distance=2.0,
            remoteness_threshold=-1.0,
        )
    with pytest.raises(FloatingPointError, match="distance"):
        squared_exponential_factor(
            torch.tensor([float("nan")]),
            coefficient=1.0,
            reference_distance=2.0,
        )


def _run_spec() -> dict[str, object]:
    return {
        "datasets": [
            {
                "id": dataset,
                "path": f"/tmp/{dataset}.hdf5",
                "sha256": "0" * 64,
            }
            for dataset in night.TUNING_DATASETS
        ],
        "seeds": list(night.TUNING_SEEDS),
    }


def test_p1_grid_builds_exact_198_branch_common_c_matrix() -> None:
    try:
        night.configure_execution(TUNING_GRID)
        grid, _ = night.load_grid(TUNING_GRID)
        branches = night.build_branches(
            SimpleNamespace(expected_canonical_alpha=0.11),
            _run_spec(),
            grid,
        )
        assert len(branches) == 198
        assert {branch.dataset.id for branch in branches} == set(
            night.TUNING_DATASETS
        )
        assert {branch.seed for branch in branches} == {200, 201}
        assert not ({branch.seed for branch in branches} & set(night.HELD_OUT_SEEDS))
        for dataset in night.TUNING_DATASETS:
            for seed in night.TUNING_SEEDS:
                group = [
                    branch
                    for branch in branches
                    if branch.dataset.id == dataset and branch.seed == seed
                ]
                assert len(group) == 11
                assert sum(
                    branch.template_values["weight_method"] == "positive_only"
                    for branch in group
                ) == 1
                assert sum(
                    branch.template_values["weight_method"] == "uncontrolled"
                    for branch in group
                ) == 1
    finally:
        night.configure_execution(HISTORICAL_GRID)


def test_p1_public_c_maps_to_existing_exponential_slope(tmp_path: Path) -> None:
    try:
        night.configure_execution(TUNING_GRID)
        grid, _ = night.load_grid(TUNING_GRID)
        contract = SimpleNamespace(
            expected_canonical_alpha=0.11,
            source_root=tmp_path,
        )
        branch = next(
            item
            for item in night.build_branches(contract, _run_spec(), grid)
            if item.template_values["weight_method"] == "thresholded_exponential"
            and float(item.template_values["remoteness_scale"]) == 4.0
        )
        _, config = night.branch_command(
            contract_path=tmp_path / "contract.json",
            contract=contract,
            branch=branch,
            branch_dir=tmp_path / "branch",
            trainer_argv_template=["--steps", "{steps}"],
        )
        public = _validate_weight_control(config["weight_control"])
        internal = _internal_control(public, 0.11)
        assert public["formula"] == THRESHOLDED_FORMULA
        assert public["remoteness_threshold"] == 0.0
        assert public["remoteness_scale"] == 4.0
        assert public["taper_lambda"] == 1.0
        assert public["derived_exp_coefficient"] == 0.25
        assert internal.method == "exponential"
        assert internal.exponential_coefficient == 0.25
        assert math.isclose(internal.effective_alpha, 1.0)
    finally:
        night.configure_execution(HISTORICAL_GRID)


def test_p1_full_run_requires_explicit_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(night.TUNING_FULL_RUN_ENV, raising=False)
    try:
        with pytest.raises(RuntimeError, match=night.TUNING_FULL_RUN_ENV):
            night.main(["run", "--grid", str(TUNING_GRID)])
    finally:
        night.configure_execution(HISTORICAL_GRID)


def test_p1_authorized_run_uses_existing_runner_and_aggregator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setenv(night.TUNING_FULL_RUN_ENV, "1")
    monkeypatch.setattr(night.base, "main", lambda argv: calls.append(("run", argv)) or 0)
    monkeypatch.setattr(
        night, "aggregate_results", lambda work: calls.append(("aggregate", work))
    )
    night.main(
        [
            "run",
            "--grid",
            str(TUNING_GRID),
            "--work-dir",
            str(tmp_path),
        ]
    )
    assert calls == [
        (
            "run",
            [
                "run",
                "--grid",
                str(TUNING_GRID),
                "--work-dir",
                str(tmp_path),
            ],
        ),
        ("aggregate", str(tmp_path)),
    ]
