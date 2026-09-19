from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import drpo_reference.experiments as public_experiments
from drpo_reference import cli
from drpo_reference.experiments.d4rl import resolve_d4rl_task

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
    assert observed["seeds"] == (10, 11)
    assert observed["stage"] == "source"

def test_cli_dispatches_d4rl_public_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {}

    monkeypatch.setattr(cli, "run_d4rl", fake_run)
    assert (
        cli.main(
            [
                "d4rl",
                "--dataset-root",
                str(tmp_path / "datasets"),
                "--output",
                str(tmp_path / "output"),
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
            ]
        )
        == 0
    )
    assert observed["task_ids"] == (
        "halfcheetah-medium-v2",
        "walker2d-medium-v2",
    )
    assert observed["seeds"] == (7, 8)
    assert observed["steps"] == 100

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
    assert cli.main(["countdown", "--config", str(config), "--output", str(output)]) == 0
    assert observed == {"config_path": config, "output_root": output}

def test_evaluate_d4rl_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = resolve_d4rl_task("halfcheetah-medium-v2")

    class FakeEnv:
        action_space = SimpleNamespace(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
        spec = SimpleNamespace(max_episode_steps=3)

        def __init__(self) -> None:
            self.steps = 0

        def reset(self, *, seed: int):
            self.steps = 0
            return np.zeros(3, dtype=np.float32), {"seed": seed}

        def step(self, action: np.ndarray):
            self.steps += 1
            return (
                np.full(3, self.steps, dtype=np.float32),
                1.0,
                self.steps >= 2,
                False,
                {},
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        public_experiments.importlib,
        "import_module",
        lambda name: SimpleNamespace(make=lambda env_id: FakeEnv()),
    )

    class FakeAgent:
        def get_action(self, observation: np.ndarray):
            return np.array([2.0, -2.0], dtype=np.float32)

    result = public_experiments.evaluate_d4rl_agent(
        agent=FakeAgent(),
        task=task,
        episodes=2,
        seed=11,
        max_steps=3,
    )
    assert result["raw_returns"] == [2.0, 2.0]
    assert result["raw_return_mean"] == pytest.approx(2.0)

def test_d4rl_runner_aggregates_scores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = resolve_d4rl_task("walker2d-medium-v2")

    monkeypatch.setattr(public_experiments, "load_d4rl_hdf5", lambda *a, **k: object())
    monkeypatch.setattr(
        public_experiments,
        "prepare_canonical_locomotion_dataset",
        lambda data: SimpleNamespace(size=32, observation_dim=3, action_dim=2),
    )
    monkeypatch.setattr(
        public_experiments,
        "train_canonical_method",
        lambda **kwargs: object(),
    )

    def fake_evaluate(**kwargs: object) -> dict[str, object]:
        seed = int(kwargs["seed"])
        return {
            "raw_return_mean": float(seed),
            "raw_return_std": 0.0,
            "normalized_score_mean": float(seed * 2),
            "normalized_score_std": 0.0,
        }

    monkeypatch.setattr(public_experiments, "evaluate_d4rl_agent", fake_evaluate)
    result = public_experiments.run_d4rl(
        dataset_root=tmp_path / "datasets",
        output_root=tmp_path / "output",
        task_ids=(task.task_id,),
        seeds=(7, 9),
        steps=2,
        batch_size=4,
        device="cpu",
        eval_episodes=2,
    )
    summary = result["tasks"][task.task_id]["methods"]["exprank"]["evaluation_summary"]
    assert summary["raw_return_mean_across_seeds"] == pytest.approx(8.0)
    assert summary["normalized_score_mean_across_seeds"] == pytest.approx(16.0)
    assert (tmp_path / "output" / "results.json").is_file()
