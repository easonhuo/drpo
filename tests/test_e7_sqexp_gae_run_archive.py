from __future__ import annotations

import json
from pathlib import Path

from drpo import e7_sqexp_gae as runner


def _write_failed_run(work_dir: Path, marker: str) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "PREPARE_SUMMARY.json").write_text('{"prepared_pairs": 12}')
    (work_dir / "EXECUTION_PLAN.json").write_text('{"expected_branches": 192}')
    (work_dir / "RUN_IDENTITY.json").write_text('{"marker": "stable"}')
    (work_dir / "RUN_SUMMARY.json").write_text(
        json.dumps({"failed": 192, "marker": marker})
    )
    aggregate = work_dir / "aggregate"
    aggregate.mkdir(exist_ok=True)
    (aggregate / "terminal_audit.json").write_text(
        json.dumps({"status": "FAIL", "marker": marker})
    )
    (aggregate / "failed_branches.csv").write_text(f"branch,status\n{marker},failed\n")


def test_failed_run_summary_and_partial_aggregate_are_archived(tmp_path: Path) -> None:
    work_dir = tmp_path / "sqexp_gae_002"
    _write_failed_run(work_dir, "first")

    archive = runner._archive_stale_run_failure(work_dir)  # noqa: SLF001

    assert archive == work_dir / "failed_run_attempts" / "attempt-001"
    assert not (work_dir / "RUN_SUMMARY.json").exists()
    assert not (work_dir / "aggregate").exists()
    assert json.loads((archive / "RUN_SUMMARY.json").read_text())["marker"] == "first"
    assert json.loads(
        (archive / "aggregate" / "terminal_audit.json").read_text()
    )["marker"] == "first"
    assert (work_dir / "PREPARE_SUMMARY.json").is_file()
    assert (work_dir / "EXECUTION_PLAN.json").is_file()
    assert (work_dir / "RUN_IDENTITY.json").is_file()
    manifest = json.loads((archive / "ARCHIVE_MANIFEST.json").read_text())
    assert manifest["attempt_index"] == 1
    assert manifest["reason"] == "preserve_failed_run_evidence_before_resume"
    assert "RUN_SUMMARY.json" in manifest["files_sha256"]
    assert "aggregate/terminal_audit.json" in manifest["files_sha256"]


def test_failed_run_archives_use_monotonic_attempt_indices(tmp_path: Path) -> None:
    work_dir = tmp_path / "sqexp_gae_002"
    _write_failed_run(work_dir, "first")
    runner._archive_stale_run_failure(work_dir)  # noqa: SLF001

    _write_failed_run(work_dir, "second")
    second = runner._archive_stale_run_failure(work_dir)  # noqa: SLF001

    first = work_dir / "failed_run_attempts" / "attempt-001"
    assert second == work_dir / "failed_run_attempts" / "attempt-002"
    assert json.loads((first / "RUN_SUMMARY.json").read_text())["marker"] == "first"
    assert json.loads((second / "RUN_SUMMARY.json").read_text())["marker"] == "second"


def test_successful_run_summary_is_not_archived(tmp_path: Path) -> None:
    work_dir = tmp_path / "sqexp_gae_002"
    work_dir.mkdir()
    summary = work_dir / "RUN_SUMMARY.json"
    summary.write_text('{"failed": 0, "completed": 192}')

    assert runner._archive_stale_run_failure(work_dir) is None  # noqa: SLF001
    assert summary.is_file()
