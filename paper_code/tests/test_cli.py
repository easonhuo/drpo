from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import drpo_reference.experiments as public_experiments
from drpo_reference import cli
from drpo_reference.external.d4rl_tasks import resolve_d4rl_task


def test_cli_dispatches_cu1_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {}

    monkeypatch.setattr(cli, "run_cu1_stage", fake_run)
    assert (
        cli.main(
            [
                "cu1",
                "--stage",
                "source",
                "--output",
                str(tmp_path),
                "--seeds",
                "10,11",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert observed == {
        "stage": "source",
        "output_root": tmp_path,
        "seeds": (10, 11),
        "smoke": False,
        "device": "cpu",
    }


def test_cli_rejects_seed_override_for_all(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="not valid"):
        cli.main(
            [
                "cu1",
                "--stage",
                "all",
                "--output",
                str(tmp_path),
                "--seeds",
                "10",
            ]
        )


def test_cli_dispatches_d4rl_public_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {}

    monkeypatch.setattr(cli, "run_d4rl", fake_run)
    dataset_root = tmp_path / "datasets"
    output = tmp_path / "output"
    assert (
        cli.main(
            [
                "d4rl",
                "--dataset-root",
                str(dataset_root),
                "--output",
                str(output),
                "--tasks",
                "halfcheetah-medium-v2,walker2d-medium-v2",
                "--seeds",
                "7,8",
                "--steps",
                "100",
                "--batch-size",
                "32",
                "--device",
                "cpu",
                "--eval-episodes",
                "5",
                "--eval-max-steps",
                "900",
            ]
        )
        == 0
    )
    assert observed == {
        "dataset_root": dataset_root,
        "output_root": output,
        "task_ids": (
            "halfcheetah-medium-v2",
            "walker2d-medium-v2",
        ),
        "seeds": (7, 8),
        "steps": 100,
        "batch_size": 32,
        "device": "cpu",
        "smoke": False,
        "eval_episodes": 5,
        "eval_max_steps": 900,
    }


def test_cli_dispatches_countdown_public_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {}

    monkeypatch.setattr(cli, "run_countdown", fake_run)
    config = tmp_path / "countdown.json"
    output = tmp_path / "countdown-output"
    assert (
        cli.main(
            [
                "countdown",
                "--config",
                str(config),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert observed == {
        "config_path": config,
        "output_root": output,
    }


def test_cli_requires_d4rl_steps_outside_smoke(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--steps is required"):
        cli.main(
            [
                "d4rl",
                "--dataset-root",
                str(tmp_path / "datasets"),
                "--output",
                str(tmp_path / "output"),
                "--seeds",
                "7",
            ]
        )


def _patch_public_runner_inputs(
    monkeypatch: pytest.MonkeyPatch,
    task: object,
    dataset_path: Path,
    *,
    with_agent: bool = False,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_validate(
        path: Path,
        task_spec: object,
        *,
        require_verified_sha: bool,
    ) -> dict[str, object]:
        assert path == dataset_path
        assert task_spec is task
        assert require_verified_sha is False
        return {
            "task_id": getattr(task, "task_id"),
            "path": str(path),
            "identity_verified": False,
        }

    def fake_train(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        run_root = Path(kwargs["output_root"])
        checkpoint = run_root / "ckpts" / "step_0000002.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        result: dict[str, object] = {
            "loss_records": [
                {"step": 1, "loss": 1.0},
                {"step": 2, "loss": 0.5},
            ],
            "checkpoints": [str(checkpoint)],
        }
        if with_agent:
            result["agent"] = object()
        return result

    monkeypatch.setattr(
        public_experiments,
        "validate_dataset_path",
        fake_validate,
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
    monkeypatch.setattr(
        public_experiments,
        "train_canonical_exprank",
        fake_train,
    )
    return calls


def test_d4rl_public_runner_writes_lightweight_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = resolve_d4rl_task("halfcheetah-medium-v2")
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    dataset_path = dataset_root / task.dataset_basename
    dataset_path.write_bytes(b"test-dataset")
    calls = _patch_public_runner_inputs(
        monkeypatch,
        task,
        dataset_path,
    )

    output = tmp_path / "output"
    summary = public_experiments.run_d4rl(
        dataset_root=dataset_root,
        output_root=output,
        task_ids=(task.task_id,),
        seeds=(7, 8),
        steps=2,
        batch_size=4,
        device="cpu",
        smoke=False,
    )

    assert len(calls) == 2
    assert summary["expected_runs"] == 2
    assert summary["completed_runs"] == 2
    assert summary["completed_evaluations"] == 0
    assert summary["training_completed"] is True
    assert summary["evaluation_completed"] is False
    completed = json.loads((output / "COMPLETED.json").read_text())
    assert completed == {
        "completed_evaluations": 0,
        "completed_runs": 2,
        "evaluation_completed": False,
        "evaluation_configured": False,
        "expected_runs": 2,
        "formal_result_claim": False,
        "method_ranking_claim_allowed": False,
        "status": "training_completed_non_formal",
        "training_completed": True,
    }
    manifest = json.loads((output / "RUN_MANIFEST.json").read_text())
    assert manifest["internal_formal_audit_included"] is False
    assert manifest["paper_artifact_binding_included"] is False
    assert manifest["evaluation"]["configured"] is False


def test_evaluate_d4rl_agent_uses_real_task_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = resolve_d4rl_task("halfcheetah-medium-v2")

    class FakeActionSpace:
        low = np.array([-1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)

    class FakeEnv:
        action_space = FakeActionSpace()
        spec = SimpleNamespace(max_episode_steps=3)

        def __init__(self) -> None:
            self.steps = 0
            self.closed = False

        def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
            self.steps = 0
            return np.zeros(3, dtype=np.float32), {"seed": seed}

        def step(
            self,
            action: np.ndarray,
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
            assert action.shape == (2,)
            self.steps += 1
            return (
                np.full(3, self.steps, dtype=np.float32),
                1.0,
                self.steps >= 2,
                False,
                {},
            )

        def close(self) -> None:
            self.closed = True

    class FakeAgent:
        def get_action(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
            assert observation.shape == (3,)
            return np.array([2.0, -2.0], dtype=np.float32), 0.0

    environments: list[FakeEnv] = []

    def fake_import(name: str) -> object:
        assert name == "gymnasium"

        def make(env_id: str) -> FakeEnv:
            assert env_id == task.env_id
            env = FakeEnv()
            environments.append(env)
            return env

        return SimpleNamespace(make=make)

    monkeypatch.setattr(
        public_experiments.importlib,
        "import_module",
        fake_import,
    )
    result = public_experiments.evaluate_d4rl_agent(
        agent=FakeAgent(),
        task=task,
        observation_dim=3,
        action_dim=2,
        episodes=2,
        seed=11,
        max_steps=3,
    )

    assert result["status"] == "completed"
    assert result["raw_returns"] == [2.0, 2.0]
    assert result["raw_return_mean"] == pytest.approx(2.0)
    assert result["raw_return_std"] == pytest.approx(0.0)
    assert result["normalized_score_std"] == pytest.approx(0.0)
    assert environments and environments[0].closed is True


def test_d4rl_public_runner_aggregates_rollout_scores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = resolve_d4rl_task("walker2d-medium-v2")
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    dataset_path = dataset_root / task.dataset_basename
    dataset_path.write_bytes(b"test-dataset")
    _patch_public_runner_inputs(
        monkeypatch,
        task,
        dataset_path,
        with_agent=True,
    )

    def fake_evaluate(**kwargs: object) -> dict[str, object]:
        seed = int(kwargs["seed"])
        return {
            "status": "completed",
            "raw_return_mean": float(seed),
            "raw_return_std": 0.0,
            "normalized_score_mean": float(seed * 2),
            "normalized_score_std": 0.0,
        }

    monkeypatch.setattr(
        public_experiments,
        "evaluate_d4rl_agent",
        fake_evaluate,
    )
    output = tmp_path / "output"
    summary = public_experiments.run_d4rl(
        dataset_root=dataset_root,
        output_root=output,
        task_ids=(task.task_id,),
        seeds=(7, 9),
        steps=2,
        batch_size=4,
        device="cpu",
        smoke=False,
        eval_episodes=2,
        eval_max_steps=100,
    )

    task_summary = summary["tasks"][task.task_id]["evaluation_summary"]
    assert summary["evaluation_completed"] is True
    assert summary["completed_evaluations"] == 2
    assert task_summary == {
        "seed_count": 2,
        "raw_return_mean_across_seeds": pytest.approx(8.0),
        "raw_return_std_across_seeds": pytest.approx(1.0),
        "normalized_score_mean_across_seeds": pytest.approx(16.0),
        "normalized_score_std_across_seeds": pytest.approx(2.0),
    }
    assert (output / task.task_id / "seed_7" / "EVALUATION.json").is_file()


def test_extracted_package_installs_and_exposes_all_entrypoints(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    package_root = tmp_path / "package"
    package_root.mkdir()
    for name in ("README.md", "pyproject.toml", "configs", "src"):
        source = source_root / name
        destination = package_root / name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)

    target = tmp_path / "site"
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(target),
            str(package_root),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert install.returncode == 0, install.stdout + install.stderr

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(target)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pathlib, drpo_reference; "
            "print(pathlib.Path(drpo_reference.__file__).resolve())",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    installed_module = Path(probe.stdout.strip())
    assert installed_module.is_relative_to(target.resolve())

    for arguments in (
        ("--help",),
        ("cu1", "--help"),
        ("du1", "--help"),
        ("hopper", "--help"),
        ("d4rl", "--help"),
        ("countdown", "--help"),
    ):
        result = subprocess.run(
            [sys.executable, "-m", "drpo_reference", *arguments],
            cwd=tmp_path,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"entrypoint failed for {arguments}:\n{result.stdout}\n{result.stderr}"
        )
