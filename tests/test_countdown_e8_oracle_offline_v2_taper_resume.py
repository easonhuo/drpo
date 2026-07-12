from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "resume_countdown_e8_oracle_offline_v2_taper_sweep.py"
)
SPEC = importlib.util.spec_from_file_location("e8_taper_resume", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


@dataclass(frozen=True)
class FakeCell:
    name: str
    method: str
    rho: float
    seed_offset: int
    coefficient: float = 1.0


class FakeCore:
    VERSION = "0.1.0"
    METHODS = ("reciprocal_linear", "reciprocal_quadratic", "exponential")

    @staticmethod
    def _run_identity(**kwargs):
        cell = kwargs["cell"]
        return {"cell": cell.name, "source": "frozen"}

    @staticmethod
    def _identity_equal(left, right):
        return left == right


class FakeRuntime:
    VERSION = "0.2.0-runtime"


def test_child_argv_removes_detach_and_replaces_resume_id() -> None:
    actual = resume.child_argv(
        [
            "--repo-root",
            "/root/drpo",
            "--detach",
            "--resume-id",
            "old",
            "--dry-run",
        ],
        "new",
    )
    assert actual == ["--repo-root", "/root/drpo", "--resume-id", "new"]


def test_runtime_command_reuses_original_work_dir_and_gpu_pool(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    args = SimpleNamespace(
        work_dir=str(tmp_path / "work"),
        calibration_gpu="0",
        gpus="0,1,2,3,4,5,6,7",
    )
    paths = {
        "model_path": tmp_path / "model",
        "bank": tmp_path / "bank.jsonl",
        "val": tmp_path / "val.jsonl",
        "test": tmp_path / "test.jsonl",
        "global_calibration": tmp_path / "global.json",
        "base_config": tmp_path / "base.yaml",
        "sweep_config": tmp_path / "sweep.yaml",
    }
    command = resume.runtime_command(args, repo, paths)
    assert command[2].endswith("countdown_e8_oracle_offline_v2_taper_runtime.py")
    assert command[3] == "run"
    assert command[command.index("--work_dir") + 1] == str((tmp_path / "work").resolve())
    assert command[command.index("--gpus") + 1] == "0,1,2,3,4,5,6,7"


def test_final_audit_requires_clean_72_of_72(tmp_path: Path) -> None:
    good = {
        "expected_cells": 72,
        "summary_count": 72,
        "failed_cells": [],
        "all_expected_cells_present": True,
    }
    (tmp_path / "SWEEP_COMPLETE.json").write_text(json.dumps(good))
    assert resume.final_audit(tmp_path) == good

    bad = dict(good, summary_count=71)
    (tmp_path / "SWEEP_COMPLETE.json").write_text(json.dumps(bad))
    with pytest.raises(resume.ResumeError):
        resume.final_audit(tmp_path)


def test_validate_summaries_preserves_complete_and_resumes_only_missing_or_deferred(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    runtime_file = repo / resume.RUNTIME_REL
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_text("runtime")
    methods = tmp_path / "methods"
    complete = FakeCell("complete", "reciprocal_linear", 0.5, 0)
    missing = FakeCell("missing", "reciprocal_quadratic", 0.5, 1000)
    deferred = FakeCell("deferred", "exponential", 0.5, 2000)
    paths = {
        "model_path": tmp_path / "model",
        "bank": tmp_path / "bank",
        "val": tmp_path / "val",
        "test": tmp_path / "test",
        "base_config": tmp_path / "base",
        "sweep_config": tmp_path / "sweep",
        "calibration": tmp_path / "calibration",
    }

    def write_summary(cell: FakeCell, *, is_deferred: bool) -> None:
        target = methods / cell.name / "summary.json"
        target.parent.mkdir(parents=True)
        target.write_text(
            json.dumps(
                {
                    "experiment_id": resume.EXPERIMENT_ID,
                    "cell": cell.name,
                    "method": cell.method,
                    "rho": cell.rho,
                    "seed_offset": cell.seed_offset,
                    "run_identity": {"cell": cell.name, "source": "frozen"},
                    "runtime_version": FakeRuntime.VERSION,
                    "runtime_source_sha256": resume.sha256(runtime_file),
                    "best_terminal_same_generation_seed": True,
                    "best_evaluation": (
                        {"deferred_posthoc_evaluation": True} if is_deferred else {"ok": True}
                    ),
                    "terminal_evaluation": {"ok": True},
                }
            )
        )

    write_summary(complete, is_deferred=False)
    write_summary(deferred, is_deferred=True)
    completed, pending, deferred_cells = resume.validate_summaries(
        core=FakeCore,
        runtime=FakeRuntime,
        cells=[complete, missing, deferred],
        paths=paths,
        methods_dir=methods,
        repo=repo,
    )
    assert completed == [complete]
    assert pending == [missing, deferred]
    assert deferred_cells == [deferred]


def test_validate_summaries_fails_closed_on_identity_drift(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    runtime_file = repo / resume.RUNTIME_REL
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_text("runtime")
    cell = FakeCell("cell", "exponential", 0.25, 0)
    target = tmp_path / "methods" / cell.name / "summary.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "experiment_id": resume.EXPERIMENT_ID,
                "cell": cell.name,
                "method": cell.method,
                "rho": cell.rho,
                "seed_offset": cell.seed_offset,
                "run_identity": {"cell": "wrong"},
                "runtime_version": FakeRuntime.VERSION,
                "runtime_source_sha256": resume.sha256(runtime_file),
                "best_terminal_same_generation_seed": True,
                "best_evaluation": {"ok": True},
                "terminal_evaluation": {"ok": True},
            }
        )
    )
    paths = {name: tmp_path / name for name in (
        "model_path",
        "bank",
        "val",
        "test",
        "base_config",
        "sweep_config",
        "calibration",
    )}
    with pytest.raises(resume.ResumeError, match="Run identity mismatch"):
        resume.validate_summaries(
            core=FakeCore,
            runtime=FakeRuntime,
            cells=[cell],
            paths=paths,
            methods_dir=tmp_path / "methods",
            repo=repo,
        )


def test_preflight_rejects_head_drift_before_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    args = SimpleNamespace(repo_root=str(repo), work_dir=str(work))
    fake_core = SimpleNamespace(
        load_yaml=lambda _path: {},
        validate_sweep_config=lambda _config: None,
        build_cells=lambda _config: [object()] * 72,
    )
    monkeypatch.setattr(resume, "import_runner", lambda _repo: (FakeRuntime, fake_core))
    monkeypatch.setattr(resume, "resolved_inputs", lambda *_args: {"sweep_config": tmp_path})
    monkeypatch.setattr(
        resume,
        "git_state",
        lambda _repo: {"commit": "a" * 40, "branch": "main", "dirty": False, "status": []},
    )
    monkeypatch.setattr(
        resume,
        "read_json",
        lambda _path, _label: {
            "experiment_id": resume.EXPERIMENT_ID,
            "base_commit": "b" * 40,
            "git_status_at_launch": [],
        },
    )
    with pytest.raises(resume.ResumeError, match="Do not git pull"):
        resume.preflight(args)


def test_parser_keeps_original_eight_gpu_pool() -> None:
    args = resume.parser().parse_args([])
    assert args.gpus == "0,1,2,3,4,5,6,7"
    assert args.work_dir == "/root/experiment_output/e8_v2_taper_sweep"
