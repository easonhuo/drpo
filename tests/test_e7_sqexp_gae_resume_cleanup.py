from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from drpo import e7_canonical_sweep as base
from drpo import e7_sqexp_gae_contract as protocol


def _branch() -> base.Branch:
    return base.Branch(
        branch_id="hopper-medium-expert-v2__seed200__td__positive_only__w0_0__a2c__steps1m",
        branch_kind="injected",
        dataset=base.DatasetSpec(
            id="hopper-medium-expert-v2",
            path="/tmp/hopper-medium-expert-v2.hdf5",
            sha256="0" * 64,
        ),
        seed=200,
        template_values={
            "steps": "1000000",
            "diagnostics_interval": "1000",
            "sampled_values_per_update": "16",
            "advantage_estimator": "td",
            "actor_update_mode": "a2c",
            "weight_method": "positive_only",
            "weight_at_zero": "0",
            "exp_coefficient": "0",
            "reference_distance": "2",
        },
        negative_control=None,
    )


def _write_failed_attempt(branch_dir: Path, marker: str) -> None:
    (branch_dir / "FAILED.json").write_text('{"return_code": 2}')
    (branch_dir / "LAUNCH.json").write_text(f'{{"marker": "{marker}"}}')
    (branch_dir / "branch_manifest.json").write_text('{"status": "failed"}')
    (branch_dir / "branch_config.json").write_text('{"old": true}')
    (branch_dir / "stdout_stderr.log").write_text(f"failure-{marker}\n")


def _build_command(tmp_path: Path, branch_dir: Path) -> dict[str, object]:
    _, config = protocol.branch_command(
        contract_path=tmp_path / "contract.json",
        contract=SimpleNamespace(source_root=tmp_path),
        branch=_branch(),
        branch_dir=branch_dir,
        trainer_argv_template=["--seed", "{seed}", "--steps", "{steps}"],
    )
    return config


def test_branch_command_archives_stale_failure_before_resume(tmp_path: Path) -> None:
    branch_dir = tmp_path / "work" / "branches" / _branch().branch_id
    branch_dir.mkdir(parents=True)
    _write_failed_attempt(branch_dir, "first")

    config = _build_command(tmp_path, branch_dir)

    archive = branch_dir / "failed_attempts" / "attempt-001"
    assert not (branch_dir / "FAILED.json").exists()
    assert (archive / "FAILED.json").is_file()
    assert (archive / "LAUNCH.json").read_text() == '{"marker": "first"}'
    assert (archive / "branch_manifest.json").is_file()
    assert (archive / "branch_config.json").read_text() == '{"old": true}'
    assert (archive / "stdout_stderr.log").read_text() == "failure-first\n"
    archive_manifest = json.loads((archive / "ARCHIVE_MANIFEST.json").read_text())
    assert archive_manifest["attempt_index"] == 1
    assert archive_manifest["reason"] == "preserve_failed_branch_evidence_before_resume"
    assert config["resumed_from_failed_attempt"] == str(archive)
    assert (branch_dir / "branch_config.json").is_file()


def test_repeated_failed_resume_uses_append_only_attempt_indices(tmp_path: Path) -> None:
    branch_dir = tmp_path / "work" / "branches" / _branch().branch_id
    branch_dir.mkdir(parents=True)
    _write_failed_attempt(branch_dir, "first")
    _build_command(tmp_path, branch_dir)

    _write_failed_attempt(branch_dir, "second")
    config = _build_command(tmp_path, branch_dir)

    first = branch_dir / "failed_attempts" / "attempt-001"
    second = branch_dir / "failed_attempts" / "attempt-002"
    assert (first / "stdout_stderr.log").read_text() == "failure-first\n"
    assert (second / "stdout_stderr.log").read_text() == "failure-second\n"
    assert config["resumed_from_failed_attempt"] == str(second)
