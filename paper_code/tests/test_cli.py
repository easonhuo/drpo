from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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


def test_d4rl_public_runner_writes_lightweight_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = resolve_d4rl_task("halfcheetah-medium-v2")
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    dataset_path = dataset_root / task.dataset_basename
    dataset_path.write_bytes(b"test-dataset")
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
            "task_id": task.task_id,
            "path": str(path),
            "identity_verified": False,
        }

    def fake_train(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        run_root = Path(kwargs["output_root"])
        checkpoint = run_root / "ckpts" / "step_0000002.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        return {
            "loss_records": [
                {"step": 1, "loss": 1.0},
                {"step": 2, "loss": 0.5},
            ],
            "checkpoints": [str(checkpoint)],
        }

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
        lambda data: SimpleNamespace(size=32),
    )
    monkeypatch.setattr(
        public_experiments,
        "train_canonical_exprank",
        fake_train,
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
    assert summary["training_completed"] is True
    assert summary["evaluation_completed"] is False
    completed = json.loads((output / "COMPLETED.json").read_text())
    assert completed == {
        "completed_runs": 2,
        "evaluation_completed": False,
        "expected_runs": 2,
        "formal_result_claim": False,
        "method_ranking_claim_allowed": False,
        "status": "training_completed_non_formal",
        "training_completed": True,
    }
    manifest = json.loads((output / "RUN_MANIFEST.json").read_text())
    assert manifest["internal_formal_audit_included"] is False
    assert manifest["paper_artifact_binding_included"] is False
    assert manifest["rollout_evaluation_configured"] is False
