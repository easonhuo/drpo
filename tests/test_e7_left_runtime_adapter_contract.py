from __future__ import annotations

import math

import pytest

from drpo import e7_ppo_w0_runtime_autotune as legacy
from drpo import e7_squared_exp_night as pilot
from drpo import e7_squared_exp_night_runtime_autotune as runtime


@pytest.mark.parametrize(
    ("profile_id", "expected_predecessor", "expected_v3", "expected_coefficient"),
    (
        (
            pilot.TUNING_PROFILE_ID,
            "e7_joint_gae_thresholded_p2_left_cpu_v2",
            "e7_joint_gae_thresholded_p2_left_cpu_v3",
            10.0,
        ),
        (
            pilot.P3_PROFILE_ID,
            "e7_joint_gae_thresholded_p3_left_saturation_cpu_v2",
            "e7_joint_gae_thresholded_p3_left_saturation_cpu_v3",
            1000.0,
        ),
    ),
)
def test_left_profiles_install_under_v3_runtime_without_changing_workload(
    monkeypatch: pytest.MonkeyPatch,
    profile_id: str,
    expected_predecessor: str,
    expected_v3: str,
    expected_coefficient: float,
) -> None:
    monkeypatch.setattr(pilot, "_ACTIVE_EXPERIMENT_ID", pilot.GAE_EXPERIMENT_ID)
    monkeypatch.setattr(pilot, "_ACTIVE_PROFILE_ID", profile_id)
    monkeypatch.setattr(pilot, "_LIVENESS_STEPS", None)

    profile = pilot.active_runtime_profile()
    assert profile["adapter_id"] == expected_predecessor
    assert runtime._v3_adapter_id(profile) == expected_v3  # noqa: SLF001
    assert profile["dataset"] == pilot.GAE_LIVENESS_DATASET
    assert profile["seed"] == pilot.GAE_LIVENESS_SEED
    assert profile["actor_update_mode"] == "a2c"
    assert profile["advantage_estimator"] == "gae"
    assert math.isclose(float(profile["weight_at_zero"]), 1.0)
    assert math.isclose(float(profile["exp_coefficient"]), expected_coefficient)
    assert math.isclose(float(profile["gae_lambda"]), pilot.GAE_LAMBDA)

    previous_adapter_id = legacy.ADAPTER_ID
    with runtime._installed_adapter():  # noqa: SLF001
        assert legacy.ADAPTER_ID == expected_v3
    assert legacy.ADAPTER_ID == previous_adapter_id
