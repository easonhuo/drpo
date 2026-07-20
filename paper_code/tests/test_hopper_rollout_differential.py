from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from drpo import e7_hopper_q2 as legacy
from drpo_reference.external import hopper_rollout as reference
from drpo_reference.external.hopper_data import Normalizer
from drpo_reference.external.hopper_models import SquashedGaussianPolicy
from drpo_reference.external.hopper_protocol import HopperProtocol


class FakeSpace:
    def __init__(
        self,
        low: tuple[float, ...] = (-1.0, -1.0),
        high: tuple[float, ...] = (1.0, 1.0),
    ) -> None:
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.seeds: list[int] = []

    def seed(self, seed: int) -> None:
        self.seeds.append(int(seed))

    def sample(self) -> np.ndarray:
        return np.asarray([0.25, -0.25], dtype=np.float32)


class FakeSpec:
    max_episode_steps = 3


class FakePreflightEnv:
    action_space = FakeSpace()
    spec = FakeSpec()

    def __init__(self, *, five_tuple: bool = False) -> None:
        self.steps = 0
        self.five_tuple = five_tuple
        self.reset_seeds: list[int | None] = []
        self.actions: list[np.ndarray] = []
        self.closed = False

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, int | None]]:
        self.steps = 0
        self.reset_seeds.append(seed)
        return (
            np.zeros(3, dtype=np.float32),
            {"seed": seed},
        )

    def step(self, action: np.ndarray):
        self.steps += 1
        self.actions.append(np.asarray(action).copy())
        done = self.steps >= 2
        observation = np.full(3, self.steps, dtype=np.float32)
        if self.five_tuple:
            return observation, 1.0, False, done, {"ok": True}
        return observation, 1.0, done, {"ok": True}

    def close(self) -> None:
        self.closed = True


class FakeEvaluationEnv:
    def __init__(self) -> None:
        self.action_space = FakeSpace(
            low=(-0.2, -0.2),
            high=(0.2, 0.2),
        )
        self.spec = SimpleNamespace(max_episode_steps=2)
        self.steps = 0
        self.reset_seeds: list[int | None] = []
        self.actions: list[np.ndarray] = []
        self.closed = False

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, int | None]]:
        self.steps = 0
        self.reset_seeds.append(seed)
        return (
            np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            {"seed": seed},
        )

    def step(self, action: np.ndarray):
        self.steps += 1
        action = np.asarray(action, dtype=np.float32)
        self.actions.append(action.copy())
        observation = np.asarray(
            [1.0 + self.steps, 2.0, 3.0],
            dtype=np.float32,
        )
        reward = 2.0 + float(np.sum(action))
        return observation, reward, self.steps >= 2, {}

    def close(self) -> None:
        self.closed = True


def _open_metadata(env_id: str) -> dict[str, object]:
    return {
        "backend": "gymnasium_mujoco",
        "evaluation_env_id": env_id,
        "legacy_d4rl_fallback": "forbidden",
    }


def _preflight_kwargs(output_dir: Path) -> dict[str, object]:
    return {
        "backend": "gymnasium_mujoco",
        "dataset_id": "hopper-medium-replay-v2",
        "env_id": "Hopper-v4",
        "expected_observation_dim": 3,
        "expected_action_dim": 2,
        "seed": 7,
        "max_steps": 5,
        "normalized_score_percent": True,
        "reference_min_score": 0.0,
        "reference_max_score": 10.0,
        "output_dir": output_dir,
        "required": True,
        "process_isolated": False,
        "timeout_seconds": 30,
    }


def _policy_pair() -> tuple[
    legacy.SquashedGaussianPolicy,
    SquashedGaussianPolicy,
]:
    torch.manual_seed(13)
    old = legacy.SquashedGaussianPolicy(
        3,
        2,
        (4,),
        -5.0,
        2.0,
        1.0e-6,
    )
    torch.manual_seed(13)
    new = SquashedGaussianPolicy(
        3,
        2,
        (4,),
        -5.0,
        2.0,
        1.0e-6,
    )
    for name, expected in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name], expected)
    return old, new


def test_rollout_protocol_fields_are_frozen() -> None:
    protocol = HopperProtocol()
    assert protocol.rollout_backend == "gymnasium_mujoco"
    assert protocol.rollout_dataset_id == "hopper-medium-replay-v2"
    assert protocol.env_id == "Hopper-v4"
    assert protocol.process_isolated_preflight is True
    assert protocol.rollout_preflight_timeout_seconds == 120
    assert protocol.rollout_preflight_max_steps == 2_000
    assert protocol.rollout_required is True


@pytest.mark.parametrize(
    ("raw", "minimum", "maximum", "percent"),
    [
        (-20.272305, -20.272305, 3234.3, True),
        (3234.3, -20.272305, 3234.3, True),
        ((-20.272305 + 3234.3) / 2.0, -20.272305, 3234.3, True),
        (4.0, 0.0, 8.0, False),
    ],
)
def test_reference_normalization_matches_legacy(
    raw: float,
    minimum: float,
    maximum: float,
    percent: bool,
) -> None:
    expected = legacy.normalize_d4rl_reference_score(
        raw,
        minimum,
        maximum,
        percent=percent,
    )
    actual = reference.normalize_d4rl_reference_score(
        raw,
        minimum,
        maximum,
        percent=percent,
    )
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    ("raw", "minimum", "maximum"),
    [
        (float("nan"), 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 2.0, 1.0),
    ],
)
def test_reference_normalization_failures_match(
    raw: float,
    minimum: float,
    maximum: float,
) -> None:
    with pytest.raises(ValueError) as expected:
        legacy.normalize_d4rl_reference_score(
            raw,
            minimum,
            maximum,
            percent=True,
        )
    with pytest.raises(ValueError) as actual:
        reference.normalize_d4rl_reference_score(
            raw,
            minimum,
            maximum,
            percent=True,
        )
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("five_tuple", [False, True])
def test_preflight_matches_authoritative_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    five_tuple: bool,
) -> None:
    old_env = FakePreflightEnv(five_tuple=five_tuple)
    new_env = FakePreflightEnv(five_tuple=five_tuple)
    monkeypatch.setattr(
        legacy,
        "_open_gymnasium_mujoco_env",
        lambda env_id: (old_env, _open_metadata(env_id)),
    )
    monkeypatch.setattr(
        reference,
        "_open_gymnasium_mujoco_env",
        lambda env_id: (new_env, _open_metadata(env_id)),
    )
    fixed_versions = {
        "python": "test",
        "numpy": np.__version__,
        "packages": {},
        "legacy_modules_imported": {
            "d4rl": False,
            "mujoco_py": False,
        },
    }
    monkeypatch.setattr(
        legacy,
        "_rollout_environment_versions",
        lambda: fixed_versions,
    )
    monkeypatch.setattr(
        reference,
        "_rollout_environment_versions",
        lambda: fixed_versions,
    )
    monkeypatch.setattr(legacy, "utc_now", lambda: "fixed-time")
    monkeypatch.setattr(reference, "utc_now", lambda: "fixed-time")

    expected = legacy.preflight_rollout_environment(**_preflight_kwargs(tmp_path / "legacy"))
    actual = reference.preflight_rollout_environment(**_preflight_kwargs(tmp_path / "reference"))
    assert actual == expected
    assert old_env.reset_seeds == new_env.reset_seeds == [7, 8]
    assert len(old_env.actions) == len(new_env.actions) == 3
    for old_action, new_action in zip(old_env.actions, new_env.actions):
        np.testing.assert_array_equal(new_action, old_action)
    assert old_env.closed is True
    assert new_env.closed is True
    saved = json.loads((tmp_path / "reference" / "rollout_preflight.json").read_text())
    assert saved == actual


def test_preflight_failure_persists_before_required_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_open(env_id: str):
        raise RuntimeError(f"synthetic failure for {env_id}")

    monkeypatch.setattr(
        reference,
        "_open_gymnasium_mujoco_env",
        fail_open,
    )
    output_dir = tmp_path / "failure"
    with pytest.raises(
        RuntimeError,
        match="Gymnasium rollout preflight failed",
    ):
        reference.preflight_rollout_environment(**_preflight_kwargs(output_dir))
    report = json.loads((output_dir / "rollout_preflight.json").read_text())
    assert report["status"] == "failed"
    assert report["error_type"] == "RuntimeError"
    assert "synthetic failure" in report["error"]
    assert "Traceback" in report["traceback"]
    assert report["legacy_d4rl_fallback"] == "forbidden"


def test_optional_preflight_failure_is_not_claimed_as_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reference,
        "_open_gymnasium_mujoco_env",
        lambda env_id: (_ for _ in ()).throw(RuntimeError(f"unavailable {env_id}")),
    )
    kwargs = _preflight_kwargs(tmp_path / "optional")
    kwargs["required"] = False
    report = reference.preflight_rollout_environment(**kwargs)
    assert report["status"] == "failed"
    assert report["required"] is False
    assert report["interaction_verified"] is False
    assert report["normalized_score_verified"] is False


def test_process_isolated_preflight_captures_sigsegv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reference.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=-11,
            stdout="",
            stderr="native crash",
        ),
    )
    kwargs = _preflight_kwargs(tmp_path / "segfault")
    kwargs.update(
        {
            "expected_observation_dim": 11,
            "expected_action_dim": 3,
            "process_isolated": True,
            "required": False,
        }
    )
    report = reference.preflight_rollout_environment(**kwargs)
    assert report["status"] == "failed"
    assert report["error_type"] == "NativeProcessSignal"
    assert report["subprocess_isolation"]["signal_name"] == "SIGSEGV"
    assert report["legacy_d4rl_fallback"] == "forbidden"
    command = report["subprocess_isolation"]["command"]
    assert "drpo_reference.external.hopper_rollout" in command
    assert "d4rl" not in command


def test_process_isolated_preflight_captures_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            args[0],
            kwargs["timeout"],
            output=b"partial",
            stderr=b"stuck",
        )

    monkeypatch.setattr(reference.subprocess, "run", timeout)
    kwargs = _preflight_kwargs(tmp_path / "timeout")
    kwargs.update(
        {
            "process_isolated": True,
            "required": False,
            "timeout_seconds": 9,
        }
    )
    report = reference.preflight_rollout_environment(**kwargs)
    assert report["status"] == "failed"
    assert report["error_type"] == "RolloutPreflightTimeout"
    assert report["subprocess_isolation"]["stdout"] == "partial"
    assert report["subprocess_isolation"]["stderr"] == "stuck"
    assert report["subprocess_isolation"]["timeout_seconds"] == 9


def test_rollout_evaluation_matches_authoritative_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_env = FakeEvaluationEnv()
    new_env = FakeEvaluationEnv()
    monkeypatch.setattr(
        legacy,
        "_open_gymnasium_mujoco_env",
        lambda env_id: (old_env, _open_metadata(env_id)),
    )
    monkeypatch.setattr(
        reference,
        "_open_gymnasium_mujoco_env",
        lambda env_id: (new_env, _open_metadata(env_id)),
    )
    old_policy, new_policy = _policy_pair()
    old_normalizer = legacy.Normalizer(
        mean=np.asarray([0.5, 1.0, 1.5], dtype=np.float32),
        std=np.asarray([0.5, 1.0, 1.5], dtype=np.float32),
    )
    new_normalizer = Normalizer(
        mean=old_normalizer.mean.copy(),
        std=old_normalizer.std.copy(),
    )
    expected = legacy.evaluate_d4rl_rollouts(
        policy=old_policy,
        obs_norm=old_normalizer,
        backend="gymnasium_mujoco",
        dataset_id="hopper-medium-replay-v2",
        env_id="Hopper-v4",
        episodes=2,
        seed=10,
        device=torch.device("cpu"),
        normalized_score_percent=True,
        reference_min_score=0.0,
        reference_max_score=8.0,
        required=True,
    )
    diagnostics = tmp_path / "rollout.json"
    actual = reference.evaluate_d4rl_rollouts(
        policy=new_policy,
        obs_norm=new_normalizer,
        backend="gymnasium_mujoco",
        dataset_id="hopper-medium-replay-v2",
        env_id="Hopper-v4",
        episodes=2,
        seed=10,
        device="cpu",
        normalized_score_percent=True,
        reference_min_score=0.0,
        reference_max_score=8.0,
        required=True,
        diagnostics_path=diagnostics,
    )
    assert actual == expected
    assert old_env.reset_seeds == new_env.reset_seeds == [10, 11]
    assert len(old_env.actions) == len(new_env.actions) == 4
    for old_action, new_action in zip(old_env.actions, new_env.actions):
        np.testing.assert_allclose(new_action, old_action)
        assert np.max(np.abs(new_action)) <= 0.2
    assert old_env.closed is True
    assert new_env.closed is True
    assert json.loads(diagnostics.read_text()) == actual


def test_disabled_rollout_matches_authoritative_runner() -> None:
    old_policy, new_policy = _policy_pair()
    old_normalizer = legacy.Normalizer(
        mean=np.zeros(3, dtype=np.float32),
        std=np.ones(3, dtype=np.float32),
    )
    new_normalizer = Normalizer(
        mean=old_normalizer.mean.copy(),
        std=old_normalizer.std.copy(),
    )
    common = {
        "backend": "gymnasium_mujoco",
        "dataset_id": "hopper-medium-replay-v2",
        "env_id": "Hopper-v4",
        "episodes": 0,
        "seed": 10,
        "device": torch.device("cpu"),
        "normalized_score_percent": True,
        "reference_min_score": 0.0,
        "reference_max_score": 8.0,
        "required": True,
    }
    expected = legacy.evaluate_d4rl_rollouts(
        policy=old_policy,
        obs_norm=old_normalizer,
        **common,
    )
    actual = reference.evaluate_d4rl_rollouts(
        policy=new_policy,
        obs_norm=new_normalizer,
        **common,
    )
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], float) and np.isnan(actual[key]):
            assert np.isnan(expected[key])
        else:
            assert actual[key] == expected[key]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "legacy_d4rl"),
        ("dataset_id", "hopper-medium-v2"),
        ("env_id", "Hopper-v3"),
    ],
)
def test_rollout_identity_mismatch_fails_closed(
    field: str,
    value: str,
) -> None:
    identity = {
        "backend": "gymnasium_mujoco",
        "dataset_id": "hopper-medium-replay-v2",
        "env_id": "Hopper-v4",
    }
    identity[field] = value
    with pytest.raises(ValueError, match="frozen protocol"):
        reference.validate_rollout_identity(**identity)


def test_rollout_open_path_never_imports_legacy_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    class FakeGymnasium:
        __name__ = "gymnasium"

        @staticmethod
        def make(env_id: str):
            assert env_id == "Hopper-v4"
            return SimpleNamespace(close=lambda: None)

    def fake_import(name: str):
        imported.append(name)
        if name == "gymnasium":
            return FakeGymnasium()
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(reference.importlib, "import_module", fake_import)
    env, metadata = reference._open_gymnasium_mujoco_env("Hopper-v4")
    env.close()
    assert imported == ["gymnasium"]
    assert metadata["legacy_d4rl_fallback"] == "forbidden"


def test_reset_and_step_compatibility_match_legacy() -> None:
    class LegacyResetEnv:
        def __init__(self) -> None:
            self.seed_value: int | None = None

        def seed(self, value: int) -> None:
            self.seed_value = value

        def reset(self) -> np.ndarray:
            return np.asarray([1.0, 2.0], dtype=np.float32)

        def step(self, action: np.ndarray):
            return (
                np.asarray([3.0, 4.0], dtype=np.float32),
                2.0,
                False,
                True,
                {"truncated": True},
            )

    old_env = LegacyResetEnv()
    new_env = LegacyResetEnv()
    old_reset = legacy._reset_env(old_env, 9)
    new_reset = reference._reset_env(new_env, 9)
    np.testing.assert_array_equal(new_reset[0], old_reset[0])
    assert new_reset[1] == old_reset[1]
    assert new_env.seed_value == old_env.seed_value == 9
    action = np.asarray([0.0], dtype=np.float32)
    old_step = legacy._step_env(old_env, action)
    new_step = reference._step_env(new_env, action)
    np.testing.assert_array_equal(new_step[0], old_step[0])
    assert new_step[1:] == old_step[1:]


def test_protocol_wrapper_passes_exact_frozen_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_preflight(**kwargs):
        captured.update(kwargs)
        return {"status": "passed"}

    monkeypatch.setattr(
        reference,
        "preflight_rollout_environment",
        fake_preflight,
    )
    protocol = HopperProtocol()
    result = reference.preflight_from_protocol(
        protocol=protocol,
        expected_observation_dim=11,
        expected_action_dim=3,
        seed=100,
        output_dir=tmp_path,
    )
    assert result == {"status": "passed"}
    assert captured == {
        "backend": protocol.rollout_backend,
        "dataset_id": protocol.rollout_dataset_id,
        "env_id": protocol.env_id,
        "expected_observation_dim": 11,
        "expected_action_dim": 3,
        "seed": 100,
        "max_steps": protocol.rollout_preflight_max_steps,
        "normalized_score_percent": protocol.normalized_score_percent,
        "reference_min_score": protocol.normalized_score_reference_min,
        "reference_max_score": protocol.normalized_score_reference_max,
        "output_dir": tmp_path,
        "required": protocol.rollout_required,
        "process_isolated": protocol.process_isolated_preflight,
        "timeout_seconds": protocol.rollout_preflight_timeout_seconds,
    }
