from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "scripts" / "validate_governance_pipeline_stage_status.py"
LEDGER = REPO_ROOT / "docs" / "governance_pipeline_stage_status.yaml"


def run_validator(repo: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "validate_governance_pipeline_stage_status.py"),
            "--repo-root",
            str(repo),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"validator failed\nstdout={proc.stdout}\nstderr={proc.stderr}")
    return proc


def copy_repository(tmp_path: Path) -> Path:
    destination = tmp_path / "repo"
    shutil.copytree(
        REPO_ROOT,
        destination,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc"),
    )
    return destination


def test_current_repository_stage_closure_is_valid() -> None:
    proc = run_validator(REPO_ROOT)
    assert "Stage 1=closed_maintenance_only" in proc.stdout
    assert "Stage 2=closed_maintenance_only" in proc.stdout
    assert "Stage 3=shadow_active" in proc.stdout


def test_protected_file_tamper_is_rejected(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    ledger = yaml.safe_load((repo / "docs" / "governance_pipeline_stage_status.yaml").read_text())
    protected = ledger["stages"]["stage_1"]["protected_files"][0]["path"]
    path = repo / protected
    path.write_bytes(path.read_bytes() + b"\n# unauthorized change\n")

    proc = run_validator(repo, check=False)
    assert proc.returncode == 2
    assert "protected file hash changed without authorization" in proc.stderr


def test_protected_file_symlink_is_rejected_even_when_target_bytes_match(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    ledger = yaml.safe_load((repo / "docs" / "governance_pipeline_stage_status.yaml").read_text())
    protected = ledger["stages"]["stage_1"]["protected_files"][0]["path"]
    path = repo / protected
    target = path.with_name(path.name + ".same-bytes")
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target.name)

    proc = run_validator(repo, check=False)
    assert proc.returncode == 2
    assert "may not contain a symlink" in proc.stderr


def test_closed_stage_cannot_be_reopened_by_editing_only_the_ledger(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    ledger_path = repo / "docs" / "governance_pipeline_stage_status.yaml"
    ledger = yaml.safe_load(ledger_path.read_text())
    ledger["stages"]["stage_1"]["status"] = "reopened"
    ledger_path.write_text(yaml.safe_dump(ledger, sort_keys=False))

    proc = run_validator(repo, check=False)
    assert proc.returncode == 2
    assert "does not authorize status reopened" in proc.stderr


def test_ledger_hash_edit_without_matching_authorization_is_rejected(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    ledger_path = repo / "docs" / "governance_pipeline_stage_status.yaml"
    ledger = yaml.safe_load(ledger_path.read_text())
    entry = ledger["stages"]["stage_2"]["protected_files"][0]
    entry["sha256"] = "0" * 64
    ledger_path.write_text(yaml.safe_dump(ledger, sort_keys=False))

    proc = run_validator(repo, check=False)
    assert proc.returncode == 2
    assert "ledger hash is not bound by authorization" in proc.stderr


def test_shadow_active_stage_keeps_manual_authority_and_forbids_cutover(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    ledger_path = repo / "docs" / "governance_pipeline_stage_status.yaml"
    ledger = yaml.safe_load(ledger_path.read_text())
    ledger["stages"]["stage_3"]["authority_cutover_allowed"] = True
    ledger_path.write_text(yaml.safe_dump(ledger, sort_keys=False))

    proc = run_validator(repo, check=False)
    assert proc.returncode == 2
    assert "must forbid authority cutover" in proc.stderr


def test_shadow_active_stage_requires_derived_observation_ledger(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    ledger_path = repo / "docs" / "governance_pipeline_stage_status.yaml"
    ledger = yaml.safe_load(ledger_path.read_text())
    ledger["stages"]["stage_3"]["observation_ledger_mode"] = "manual_counter"
    ledger_path.write_text(yaml.safe_dump(ledger, sort_keys=False))

    proc = run_validator(repo, check=False)
    assert proc.returncode == 2
    assert "observation_ledger_mode" in proc.stderr


def test_stage3_feature_freeze_cannot_be_silently_relaxed(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    ledger_path = repo / "docs" / "governance_pipeline_stage_status.yaml"
    ledger = yaml.safe_load(ledger_path.read_text())
    ledger["stages"]["stage_3"]["feature_state"] = "feature_development_open"
    ledger_path.write_text(yaml.safe_dump(ledger, sort_keys=False))

    proc = run_validator(repo, check=False)
    assert proc.returncode == 2
    assert "feature_state must be feature_frozen_bugfix_only" in proc.stderr


def test_stage4_design_cannot_enable_implementation_early(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    ledger_path = repo / "docs" / "governance_pipeline_stage_status.yaml"
    ledger = yaml.safe_load(ledger_path.read_text())
    ledger["stages"]["stage_4"]["implementation_allowed"] = True
    ledger_path.write_text(yaml.safe_dump(ledger, sort_keys=False))

    proc = run_validator(repo, check=False)
    assert proc.returncode == 2
    assert "implementation must not be allowed" in proc.stderr


def test_stage4_design_cannot_switch_authority(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    ledger_path = repo / "docs" / "governance_pipeline_stage_status.yaml"
    ledger = yaml.safe_load(ledger_path.read_text())
    ledger["stages"]["stage_4"]["authority_cutover_allowed"] = True
    ledger_path.write_text(yaml.safe_dump(ledger, sort_keys=False))

    proc = run_validator(repo, check=False)
    assert proc.returncode == 2
    assert "stage_4 design must forbid authority cutover" in proc.stderr


def test_stage4_phase_order_cannot_be_collapsed(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path)
    ledger_path = repo / "docs" / "governance_pipeline_stage_status.yaml"
    ledger = yaml.safe_load(ledger_path.read_text())
    ledger["stages"]["stage_4"]["phase_plan"] = ["stage_4_all_at_once"]
    ledger_path.write_text(yaml.safe_dump(ledger, sort_keys=False))

    proc = run_validator(repo, check=False)
    assert proc.returncode == 2
    assert "4A/4B/4C sequence" in proc.stderr
