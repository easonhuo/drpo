"""Hopper E7-Q2 Gymnasium rollout and isolated preflight boundary."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drpo_reference.common.io import atomic_json
from drpo_reference.external.hopper_data import Normalizer
from drpo_reference.external.hopper_models import SquashedGaussianPolicy
from drpo_reference.external.hopper_protocol import HopperProtocol


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def normalize_d4rl_reference_score(
    raw_return: float,
    reference_min_score: float,
    reference_max_score: float,
    *,
    percent: bool,
) -> float:
    """Apply the frozen D4RL-v2 reference-score convention."""

    raw = float(raw_return)
    minimum = float(reference_min_score)
    maximum = float(reference_max_score)
    if not all(
        math.isfinite(value)
        for value in (raw, minimum, maximum)
    ):
        raise ValueError(
            "Raw return and D4RL reference scores must be finite"
        )
    if maximum <= minimum:
        raise ValueError(
            "D4RL reference_max_score must exceed reference_min_score"
        )
    normalized = (raw - minimum) / (maximum - minimum)
    return normalized * 100.0 if percent else normalized


def validate_rollout_identity(
    *,
    backend: str,
    dataset_id: str,
    env_id: str,
) -> dict[str, str]:
    """Fail closed unless the registered E7-Q2 simulator identity is used."""

    protocol = HopperProtocol()
    expected = {
        "backend": protocol.rollout_backend,
        "dataset_id": protocol.rollout_dataset_id,
        "env_id": protocol.env_id,
    }
    actual = {
        "backend": str(backend),
        "dataset_id": str(dataset_id),
        "env_id": str(env_id),
    }
    mismatches = {
        key: {
            "expected": expected[key],
            "actual": actual[key],
        }
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatches:
        raise ValueError(
            "Hopper rollout identity does not match the frozen protocol: "
            f"{mismatches}"
        )
    return expected


def _open_gymnasium_mujoco_env(
    env_id: str,
) -> tuple[Any, dict[str, Any]]:
    """Open Gymnasium MuJoCo without importing legacy D4RL/mujoco-py."""

    legacy_modules_before = {
        name: name in sys.modules
        for name in ("d4rl", "mujoco_py")
    }
    gymnasium = importlib.import_module("gymnasium")
    env = gymnasium.make(env_id)
    legacy_modules_after = {
        name: name in sys.modules
        for name in ("d4rl", "mujoco_py")
    }
    if any(legacy_modules_after.values()):
        env.close()
        raise RuntimeError(
            "Gymnasium rollout process contains a legacy "
            "D4RL/mujoco_py module"
        )
    metadata = {
        "backend": "gymnasium_mujoco",
        "gym_backend": "gymnasium",
        "gym_module": getattr(
            gymnasium,
            "__name__",
            "gymnasium",
        ),
        "evaluation_env_id": env_id,
        "legacy_d4rl_fallback": "forbidden",
        "legacy_modules_before": legacy_modules_before,
        "legacy_modules_after": legacy_modules_after,
    }
    return env, metadata


def _reset_env(
    env: Any,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        result = env.reset(seed=int(seed))
        reset_mode = "reset_seed_kwarg"
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(int(seed))
        result = env.reset()
        reset_mode = "legacy_env_seed"
    if isinstance(result, tuple):
        observation = result[0]
        info = (
            result[1]
            if len(result) > 1 and isinstance(result[1], dict)
            else {}
        )
    else:
        observation = result
        info = {}
    return np.asarray(observation, dtype=np.float32), {
        "reset_mode": reset_mode,
        "info_keys": sorted(str(key) for key in info),
    }


def _step_env(
    env: Any,
    action: np.ndarray,
) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
    result = env.step(action)
    if not isinstance(result, tuple):
        raise RuntimeError(
            f"env.step returned {type(result).__name__}, expected tuple"
        )
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
        api = "new_five_tuple"
    elif len(result) == 4:
        observation, reward, done, info = result
        done = bool(done)
        api = "legacy_four_tuple"
    else:
        raise RuntimeError(
            f"env.step returned {len(result)} values, expected 4 or 5"
        )
    return (
        np.asarray(observation, dtype=np.float32),
        float(reward),
        done,
        {
            "step_api": api,
            "info_keys": (
                sorted(str(key) for key in info)
                if isinstance(info, dict)
                else []
            ),
        },
    )


def _clip_action_to_space(
    env: Any,
    action: np.ndarray,
) -> np.ndarray:
    clipped = np.asarray(action, dtype=np.float32)
    space = getattr(env, "action_space", None)
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if low is not None and high is not None:
        clipped = np.clip(
            clipped,
            np.asarray(low),
            np.asarray(high),
        )
    return clipped.astype(np.float32, copy=False)


def _max_episode_steps(env: Any, fallback: int) -> int:
    spec = getattr(env, "spec", None)
    value = getattr(spec, "max_episode_steps", None)
    if isinstance(value, int) and value > 0:
        return min(value, fallback)
    value = getattr(env, "_max_episode_steps", None)
    if isinstance(value, int) and value > 0:
        return min(value, fallback)
    return fallback


def _rollout_environment_versions() -> dict[str, Any]:
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "packages": {
            name: _package_version(name)
            for name in (
                "gymnasium",
                "mujoco",
                "gym",
                "d4rl",
                "mujoco-py",
            )
        },
        "legacy_modules_imported": {
            name: name in sys.modules
            for name in ("d4rl", "mujoco_py")
        },
    }


def _run_rollout_preflight_worker(
    *,
    backend: str,
    dataset_id: str,
    env_id: str,
    expected_observation_dim: int,
    expected_action_dim: int,
    seed: int,
    max_steps: int,
    normalized_score_percent: bool,
    reference_min_score: float,
    reference_max_score: float,
    output_report: Path,
) -> dict[str, Any]:
    """Run one environment preflight in the current process."""

    output_report.parent.mkdir(parents=True, exist_ok=True)
    versions = _rollout_environment_versions()
    atomic_json(
        output_report.parent / "environment_versions.json",
        versions,
    )
    report: dict[str, Any] = {
        "status": "running",
        "started_utc": utc_now(),
        "backend": backend,
        "dataset_id": dataset_id,
        "evaluation_env_id": env_id,
        "expected_observation_dim": expected_observation_dim,
        "expected_action_dim": expected_action_dim,
        "max_steps": max_steps,
        "versions": versions,
        "normalization": {
            "protocol": "d4rl_v2_reference",
            "reference_dataset_id": dataset_id,
            "reference_min_score": float(reference_min_score),
            "reference_max_score": float(reference_max_score),
            "percent": bool(normalized_score_percent),
        },
    }
    env = None
    try:
        validate_rollout_identity(
            backend=backend,
            dataset_id=dataset_id,
            env_id=env_id,
        )
        env, open_metadata = _open_gymnasium_mujoco_env(env_id)
        report["open"] = open_metadata
        observation, reset_metadata = _reset_env(env, seed)
        report["reset"] = {
            **reset_metadata,
            "observation_shape": list(observation.shape),
            "observation_finite": bool(
                np.all(np.isfinite(observation))
            ),
        }
        if observation.size != expected_observation_dim:
            raise RuntimeError(
                f"Environment observation size {observation.size} "
                "does not match dataset observation dimension "
                f"{expected_observation_dim}"
            )
        action_space = getattr(env, "action_space", None)
        if action_space is None or not hasattr(
            action_space,
            "sample",
        ):
            raise RuntimeError(
                "Environment does not expose a sampleable action_space"
            )
        if hasattr(action_space, "seed"):
            action_space.seed(int(seed))
        sample_action = _clip_action_to_space(
            env,
            action_space.sample(),
        )
        if sample_action.size != expected_action_dim:
            raise RuntimeError(
                f"Environment action size {sample_action.size} "
                "does not match dataset action dimension "
                f"{expected_action_dim}"
            )
        (
            next_observation,
            reward,
            done,
            step_metadata,
        ) = _step_env(env, sample_action)
        report["single_step"] = {
            **step_metadata,
            "action_shape": list(sample_action.shape),
            "action_finite": bool(
                np.all(np.isfinite(sample_action))
            ),
            "next_observation_shape": list(
                next_observation.shape
            ),
            "next_observation_finite": bool(
                np.all(np.isfinite(next_observation))
            ),
            "reward": reward,
            "reward_finite": math.isfinite(reward),
            "done": done,
        }
        _observation, _ = _reset_env(env, seed + 1)
        total = 0.0
        limit = _max_episode_steps(env, max_steps)
        episode_steps = 0
        done = False
        last_api = None
        while not done and episode_steps < limit:
            action = _clip_action_to_space(
                env,
                action_space.sample(),
            )
            _observation, reward, done, metadata = _step_env(
                env,
                action,
            )
            total += reward
            episode_steps += 1
            last_api = metadata["step_api"]
        normalized = normalize_d4rl_reference_score(
            total,
            reference_min_score,
            reference_max_score,
            percent=normalized_score_percent,
        )
        report["random_episode"] = {
            "steps": episode_steps,
            "step_limit": limit,
            "terminated_or_truncated": done,
            "return": total,
            "normalized_return": normalized,
            "normalized_return_available": math.isfinite(
                normalized
            ),
            "last_step_api": last_api,
        }
        if episode_steps <= 0:
            raise RuntimeError(
                "Random rollout completed zero steps"
            )
        if not math.isfinite(total):
            raise RuntimeError(
                "Random rollout return is non-finite"
            )
        if not math.isfinite(normalized):
            raise RuntimeError(
                "D4RL-reference normalized score is non-finite"
            )
        report.update(
            {
                "status": "passed",
                "completed_utc": utc_now(),
                "interaction_verified": True,
                "normalized_score_verified": True,
            }
        )
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "interaction_verified": False,
                "normalized_score_verified": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        atomic_json(output_report, report)
    return report


def _signal_name(returncode: int) -> str | None:
    if returncode >= 0:
        return None
    try:
        import signal

        return signal.Signals(-returncode).name
    except (ValueError, OSError):
        return f"SIGNAL_{-returncode}"


def _diagnostic_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _preflight_worker_command(
    *,
    backend: str,
    dataset_id: str,
    env_id: str,
    expected_observation_dim: int,
    expected_action_dim: int,
    seed: int,
    max_steps: int,
    normalized_score_percent: bool,
    reference_min_score: float,
    reference_max_score: float,
    output_report: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "drpo_reference.external.hopper_rollout",
        "rollout-preflight-worker",
        "--backend",
        backend,
        "--dataset-id",
        dataset_id,
        "--env-id",
        env_id,
        "--expected-observation-dim",
        str(expected_observation_dim),
        "--expected-action-dim",
        str(expected_action_dim),
        "--seed",
        str(seed),
        "--max-steps",
        str(max_steps),
        "--reference-min-score",
        repr(float(reference_min_score)),
        "--reference-max-score",
        repr(float(reference_max_score)),
        "--output-report",
        str(output_report),
    ]
    if normalized_score_percent:
        command.append("--normalized-score-percent")
    return command


def preflight_rollout_environment(
    *,
    backend: str,
    dataset_id: str,
    env_id: str,
    expected_observation_dim: int,
    expected_action_dim: int,
    seed: int,
    max_steps: int,
    normalized_score_percent: bool,
    reference_min_score: float,
    reference_max_score: float,
    output_dir: Path,
    required: bool,
    process_isolated: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Verify the registered environment, isolating native crashes."""

    output_dir.mkdir(parents=True, exist_ok=True)
    worker_report = (
        output_dir / "rollout_preflight_worker.json"
    )
    canonical_report = output_dir / "rollout_preflight.json"

    if process_isolated:
        command = _preflight_worker_command(
            backend=backend,
            dataset_id=dataset_id,
            env_id=env_id,
            expected_observation_dim=expected_observation_dim,
            expected_action_dim=expected_action_dim,
            seed=seed,
            max_steps=max_steps,
            normalized_score_percent=normalized_score_percent,
            reference_min_score=reference_min_score,
            reference_max_score=reference_max_score,
            output_report=worker_report,
        )
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=int(timeout_seconds),
                check=False,
            )
            if worker_report.is_file():
                report = json.loads(
                    worker_report.read_text(encoding="utf-8")
                )
            else:
                report = {
                    "status": "failed",
                    "interaction_verified": False,
                    "normalized_score_verified": False,
                    "error_type": (
                        "NativeProcessSignal"
                        if completed.returncode < 0
                        else "WorkerReportMissing"
                    ),
                    "error": (
                        "rollout worker terminated by "
                        f"{_signal_name(completed.returncode)}"
                        if completed.returncode < 0
                        else (
                            "rollout worker exited without "
                            "writing its report"
                        )
                    ),
                }
            report["subprocess_isolation"] = {
                "enabled": True,
                "command": command,
                "returncode": completed.returncode,
                "signal_name": _signal_name(
                    completed.returncode
                ),
                "stdout": _diagnostic_text(completed.stdout),
                "stderr": _diagnostic_text(completed.stderr),
                "timeout_seconds": int(timeout_seconds),
            }
            if completed.returncode != 0:
                report["status"] = "failed"
                report["interaction_verified"] = False
                report["normalized_score_verified"] = False
                if completed.returncode < 0:
                    report["error_type"] = (
                        "NativeProcessSignal"
                    )
                    report["error"] = (
                        "rollout worker terminated by "
                        f"{_signal_name(completed.returncode)}"
                    )
        except subprocess.TimeoutExpired as exc:
            report = {
                "status": "failed",
                "interaction_verified": False,
                "normalized_score_verified": False,
                "error_type": "RolloutPreflightTimeout",
                "error": (
                    "rollout worker exceeded "
                    f"{timeout_seconds} seconds"
                ),
                "subprocess_isolation": {
                    "enabled": True,
                    "command": command,
                    "timeout_seconds": int(timeout_seconds),
                    "stdout": _diagnostic_text(exc.stdout),
                    "stderr": _diagnostic_text(exc.stderr),
                },
            }
        except Exception as exc:
            report = {
                "status": "failed",
                "interaction_verified": False,
                "normalized_score_verified": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "subprocess_isolation": {
                    "enabled": True,
                    "command": command,
                },
            }
    else:
        report = _run_rollout_preflight_worker(
            backend=backend,
            dataset_id=dataset_id,
            env_id=env_id,
            expected_observation_dim=expected_observation_dim,
            expected_action_dim=expected_action_dim,
            seed=seed,
            max_steps=max_steps,
            normalized_score_percent=normalized_score_percent,
            reference_min_score=reference_min_score,
            reference_max_score=reference_max_score,
            output_report=worker_report,
        )
        report["subprocess_isolation"] = {
            "enabled": False
        }

    report["required"] = bool(required)
    report["legacy_d4rl_fallback"] = "forbidden"
    atomic_json(canonical_report, report)
    if required and report.get("status") != "passed":
        raise RuntimeError(
            "Gymnasium rollout preflight failed for "
            f"{env_id!r}: "
            f"{report.get('error', 'unknown failure')}. "
            f"See {canonical_report}"
        )
    return report


def _tensor(
    array: np.ndarray,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.as_tensor(
        array,
        dtype=torch.float32,
        device=device,
    )


def evaluate_d4rl_rollouts(
    *,
    policy: SquashedGaussianPolicy,
    obs_norm: Normalizer,
    backend: str,
    dataset_id: str,
    env_id: str,
    episodes: int,
    seed: int,
    device: torch.device | str,
    normalized_score_percent: bool,
    reference_min_score: float,
    reference_max_score: float,
    required: bool,
    diagnostics_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate the deterministic actor in the registered Hopper simulator."""

    if episodes <= 0:
        return {
            "rollout_status": "disabled",
            "rollout_return_mean": float("nan"),
            "rollout_return_std": float("nan"),
            "normalized_return": float("nan"),
            "normalized_return_available": False,
            "rollout_episodes": 0,
        }
    env = None
    try:
        validate_rollout_identity(
            backend=backend,
            dataset_id=dataset_id,
            env_id=env_id,
        )
        env, open_metadata = _open_gymnasium_mujoco_env(
            env_id
        )
        returns: list[float] = []
        episode_steps: list[int] = []
        for episode in range(episodes):
            episode_seed = int(seed + episode)
            observation, _ = _reset_env(
                env,
                episode_seed,
            )
            total = 0.0
            done = False
            steps = 0
            limit = _max_episode_steps(env, 10_000)
            while not done and steps < limit:
                normalized_observation = obs_norm.transform(
                    observation.reshape(1, -1)
                )
                with torch.no_grad():
                    action = (
                        policy.action_mean(
                            _tensor(
                                normalized_observation,
                                device,
                            )
                        )[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                action = _clip_action_to_space(env, action)
                observation, reward, done, _ = _step_env(
                    env,
                    action,
                )
                total += reward
                steps += 1
            if steps >= limit and not done:
                raise RuntimeError(
                    "Episode reached safety limit "
                    f"{limit} without termination"
                )
            returns.append(total)
            episode_steps.append(steps)
        mean_return = float(np.mean(returns))
        normalized = normalize_d4rl_reference_score(
            mean_return,
            reference_min_score,
            reference_max_score,
            percent=normalized_score_percent,
        )
        result: dict[str, Any] = {
            "rollout_status": "available",
            "rollout_return_mean": mean_return,
            "rollout_return_std": float(np.std(returns)),
            "normalized_return": normalized,
            "normalized_return_available": math.isfinite(
                normalized
            ),
            "rollout_episodes": int(episodes),
            "rollout_episode_steps_mean": float(
                np.mean(episode_steps)
            ),
            "rollout_open_metadata": open_metadata,
            "rollout_backend": backend,
            "evaluation_env_id": env_id,
            "offline_dataset_id": dataset_id,
            "normalization": {
                "protocol": "d4rl_v2_reference",
                "reference_min_score": float(
                    reference_min_score
                ),
                "reference_max_score": float(
                    reference_max_score
                ),
                "percent": bool(normalized_score_percent),
            },
        }
        if required and not math.isfinite(normalized):
            raise RuntimeError(
                "Required normalized return is unavailable "
                "or non-finite"
            )
        if diagnostics_path is not None:
            atomic_json(diagnostics_path, result)
        return result
    except Exception as exc:
        failure: dict[str, Any] = {
            "rollout_status": "unavailable",
            "rollout_return_mean": float("nan"),
            "rollout_return_std": float("nan"),
            "normalized_return": float("nan"),
            "normalized_return_available": False,
            "rollout_episodes": 0,
            "rollout_backend": backend,
            "evaluation_env_id": env_id,
            "offline_dataset_id": dataset_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if diagnostics_path is not None:
            atomic_json(diagnostics_path, failure)
        if required:
            raise RuntimeError(
                "Required rollout evaluation failed for "
                f"{env_id!r}: {exc}"
            ) from exc
        return failure
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def preflight_from_protocol(
    *,
    protocol: HopperProtocol,
    expected_observation_dim: int,
    expected_action_dim: int,
    seed: int,
    output_dir: Path,
    required: bool | None = None,
) -> dict[str, Any]:
    """Run preflight from one frozen protocol object."""

    return preflight_rollout_environment(
        backend=protocol.rollout_backend,
        dataset_id=protocol.rollout_dataset_id,
        env_id=protocol.env_id,
        expected_observation_dim=expected_observation_dim,
        expected_action_dim=expected_action_dim,
        seed=seed,
        max_steps=protocol.rollout_preflight_max_steps,
        normalized_score_percent=(
            protocol.normalized_score_percent
        ),
        reference_min_score=(
            protocol.normalized_score_reference_min
        ),
        reference_max_score=(
            protocol.normalized_score_reference_max
        ),
        output_dir=output_dir,
        required=(
            protocol.rollout_required
            if required is None
            else bool(required)
        ),
        process_isolated=(
            protocol.process_isolated_preflight
        ),
        timeout_seconds=(
            protocol.rollout_preflight_timeout_seconds
        ),
    )


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Internal Hopper rollout preflight worker; "
            "not a public experiment runner."
        )
    )
    parser.add_argument(
        "command",
        choices=("rollout-preflight-worker",),
    )
    parser.add_argument("--backend", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument(
        "--expected-observation-dim",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--expected-action-dim",
        type=int,
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument(
        "--normalized-score-percent",
        action="store_true",
    )
    parser.add_argument(
        "--reference-min-score",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--reference-max-score",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _worker_parser().parse_args(argv)
    report = _run_rollout_preflight_worker(
        backend=args.backend,
        dataset_id=args.dataset_id,
        env_id=args.env_id,
        expected_observation_dim=(
            args.expected_observation_dim
        ),
        expected_action_dim=args.expected_action_dim,
        seed=args.seed,
        max_steps=args.max_steps,
        normalized_score_percent=(
            args.normalized_score_percent
        ),
        reference_min_score=args.reference_min_score,
        reference_max_score=args.reference_max_score,
        output_report=args.output_report,
    )
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":  # pragma: no cover - subprocess entry
    raise SystemExit(main())
