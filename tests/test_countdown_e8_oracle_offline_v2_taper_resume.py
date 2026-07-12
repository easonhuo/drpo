from __future__ import annotations

import importlib.util
import json
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


def test_child_args_removes_detach_and_replaces_resume_id() -> None:
    actual = resume.child_args(
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


def test_final_audit_requires_clean_72_of_72(tmp_path: Path) -> None:
    good = {
        "expected_cells": 72,
        "summary_count": 72,
        "failed_cells": [],
        "all_expected_cells_present": True,
    }
    (tmp_path / "SWEEP_COMPLETE.json").write_text(json.dumps(good))
    assert resume.final_audit(tmp_path) == good

    (tmp_path / "SWEEP_COMPLETE.json").write_text(
        json.dumps({**good, "summary_count": 71})
    )
    with pytest.raises(resume.ResumeError, match="72/72"):
        resume.final_audit(tmp_path)


def test_run_finalizes_without_launch_when_all_cells_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    complete = {
        "expected_cells": 72,
        "summary_count": 72,
        "failed_cells": [],
        "all_expected_cells_present": True,
    }
    (work / "SWEEP_COMPLETE.json").write_text(json.dumps(complete))
    args = SimpleNamespace(
        work_dir=str(work),
        repo_root=str(tmp_path),
        resume_id="resume-test",
    )
    monkeypatch.setattr(
        resume,
        "preflight",
        lambda _args: (
            {
                "pending_cells": 0,
                "completed_cells": 72,
                "source": {"commit": "a" * 40, "branch": "main", "dirty": False},
            },
            ["unused"],
        ),
    )
    assert resume.run(args) == 0
    payload = json.loads(
        (work / "resume_runs" / "resume-test" / "RESUME_COMPLETE.json").read_text()
    )
    assert payload["sweep_complete"] == complete


def test_run_preserves_failure_and_records_nonzero_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    args = SimpleNamespace(
        work_dir=str(work),
        repo_root=str(repo),
        resume_id="resume-fail",
    )
    monkeypatch.setattr(
        resume,
        "preflight",
        lambda _args: (
            {
                "pending_cells": 1,
                "completed_cells": 71,
                "source": {"commit": "a" * 40, "branch": "main", "dirty": False},
            },
            ["python", "runtime.py"],
        ),
    )
    monkeypatch.setattr(
        resume.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=9),
    )
    assert resume.run(args) == 9
    failed = json.loads(
        (work / "resume_runs" / "resume-fail" / "RESUME_FAILED.json").read_text()
    )
    assert failed["runtime_returncode"] == 9


def test_preflight_rejects_head_drift_before_reading_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    args = SimpleNamespace(repo_root=str(repo), work_dir=str(work))
    fake_runtime = SimpleNamespace(VERSION="0.2.0-runtime")
    fake_core = SimpleNamespace()
    monkeypatch.setattr(resume, "import_runner", lambda _repo: (fake_runtime, fake_core))
    monkeypatch.setattr(resume, "resolve_paths", lambda *_args: {})
    monkeypatch.setattr(
        resume,
        "git_state",
        lambda _repo: {
            "commit": "a" * 40,
            "branch": "main",
            "dirty": False,
            "status": [],
        },
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
    with pytest.raises(resume.ResumeError, match="do not git pull"):
        resume.preflight(args)


def test_detach_launches_controller_in_new_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    args = SimpleNamespace(
        work_dir=str(work),
        repo_root=str(repo),
        resume_id="resume-detach",
    )
    monkeypatch.setattr(
        resume,
        "preflight",
        lambda _args: (
            {
                "pending_cells": 21,
                "completed_cells": 51,
                "source": {"commit": "a" * 40, "branch": "main", "dirty": False},
            },
            ["unused"],
        ),
    )
    captured = {}

    class FakeProcess:
        pid = 1234

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(resume.subprocess, "Popen", fake_popen)
    assert resume.detach(args, ["--detach"]) == 0
    assert "--detach" not in captured["command"]
    assert captured["kwargs"]["start_new_session"] is True
    launch = json.loads(
        (work / "resume_runs" / "resume-detach" / "RESUME_LAUNCH.json").read_text()
    )
    assert launch["controller_pid"] == 1234


def test_parser_keeps_frozen_eight_gpu_pool() -> None:
    args = resume.parser().parse_args([])
    assert args.gpus == "0,1,2,3,4,5,6,7"
    assert args.work_dir == "/root/experiment_output/e8_v2_taper_sweep"
