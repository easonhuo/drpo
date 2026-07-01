from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_governance_pipeline_stage_status.py"
LEDGER = REPO_ROOT / "docs" / "governance_pipeline_stage_status.yaml"

_SPEC = importlib.util.spec_from_file_location("stage_status_validator", VALIDATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(VALIDATOR)


def run_validator(repo: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / VALIDATOR_PATH.name), "--repo-root", str(repo)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"validator failed\nstdout={proc.stdout}\nstderr={proc.stderr}")
    return proc


def write_mutated_ledger(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    payload = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    mutate(payload)
    output = tmp_path / "ledger.yaml"
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output


def assert_ledger_invalid(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    path = write_mutated_ledger(tmp_path, mutate)
    with pytest.raises(VALIDATOR.StageStatusError, match=message):
        VALIDATOR.validate(REPO_ROOT, path)


def hardlink_repository() -> Path:
    destination = Path(tempfile.gettempdir()) / f"drpo-stage-status-{uuid.uuid4().hex}"
    shutil.copytree(
        REPO_ROOT,
        destination,
        copy_function=os.link,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc"),
    )
    return destination


def replace_bytes(path: Path, payload: bytes) -> None:
    replacement = path.with_name(path.name + ".replacement")
    replacement.write_bytes(payload)
    os.replace(replacement, path)


def after_image_fixture(tmp_path: Path, relative: str) -> tuple[Path, dict[str, Any]]:
    source = REPO_ROOT / relative
    payload = json.loads(source.read_text(encoding="utf-8"))
    repo = tmp_path / "repo"
    target_manifest = repo / relative
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target_manifest)
    for entry in payload["files"]:
        src = REPO_ROOT / entry["path"]
        dst = repo / entry["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return repo, payload


def test_current_repository_stage_closure_is_valid() -> None:
    proc = run_validator(REPO_ROOT)
    assert "Stage 1=closed_maintenance_only" in proc.stdout
    assert "Stage 2=closed_maintenance_only" in proc.stdout
    assert "Stage 3=shadow_active" in proc.stdout
    assert "Stage 4=active" in proc.stdout
    assert "Stage 5=blocked_by_predecessor" in proc.stdout


def test_protected_file_tamper_is_rejected() -> None:
    repo = hardlink_repository()
    try:
        ledger = yaml.safe_load((repo / "docs/governance_pipeline_stage_status.yaml").read_text())
        protected = ledger["stages"]["stage_1"]["protected_files"][0]["path"]
        path = repo / protected
        replace_bytes(path, path.read_bytes() + b"\n# unauthorized change\n")
        proc = run_validator(repo, check=False)
        assert proc.returncode == 2
        assert "protected file hash changed without authorization" in proc.stderr
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_protected_file_symlink_is_rejected_even_when_target_bytes_match() -> None:
    repo = hardlink_repository()
    try:
        ledger = yaml.safe_load((repo / "docs/governance_pipeline_stage_status.yaml").read_text())
        protected = ledger["stages"]["stage_1"]["protected_files"][0]["path"]
        path = repo / protected
        target = path.with_name(path.name + ".same-bytes")
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target.name)
        proc = run_validator(repo, check=False)
        assert proc.returncode == 2
        assert "may not contain a symlink" in proc.stderr
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_closed_stage_cannot_be_reopened_by_editing_only_the_ledger(tmp_path: Path) -> None:
    assert_ledger_invalid(
        tmp_path,
        lambda x: x["stages"]["stage_1"].__setitem__("status", "reopened"),
        "does not authorize status reopened",
    )


def test_ledger_hash_edit_without_matching_authorization_is_rejected(tmp_path: Path) -> None:
    def mutate(x: dict[str, Any]) -> None:
        x["stages"]["stage_2"]["protected_files"][0]["sha256"] = "0" * 64

    assert_ledger_invalid(tmp_path, mutate, "ledger hash is not bound by authorization")


def test_shadow_active_stage_keeps_manual_authority_and_forbids_cutover(tmp_path: Path) -> None:
    assert_ledger_invalid(
        tmp_path,
        lambda x: x["stages"]["stage_3"].__setitem__("authority_cutover_allowed", True),
        "must forbid authority cutover",
    )


def test_shadow_active_stage_requires_derived_observation_ledger(tmp_path: Path) -> None:
    assert_ledger_invalid(
        tmp_path,
        lambda x: x["stages"]["stage_3"].__setitem__("observation_ledger_mode", "manual_counter"),
        "observation_ledger_mode",
    )


def test_stage3_feature_freeze_cannot_be_silently_relaxed(tmp_path: Path) -> None:
    assert_ledger_invalid(
        tmp_path,
        lambda x: x["stages"]["stage_3"].__setitem__("feature_state", "feature_development_open"),
        "feature_state must be feature_frozen_bugfix_only",
    )


def test_stage4b_final_acceptance_is_recorded_without_starting_stage4c() -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    stage_4 = ledger["stages"]["stage_4"]
    assert stage_4["status"] == "active"
    assert stage_4["implementation_state"] == "stage_4b_accepted"
    assert stage_4["implementation_allowed"] is False
    assert stage_4["active_phase"] == "stage_4b_accepted_waiting_stage_4c_authorization"
    assert stage_4["phase_states"] == {
        "stage_4a_schema_inventory": "accepted",
        "stage_4b_lossless_candidate": "accepted",
        "stage_4c_context_assembly_shadow_validation": "ready_for_authorization",
    }
    assert stage_4["depends_on"] == ["stage_3_feature_frozen"]
    assert stage_4["manual_handoff_remains_authoritative"] is True
    assert stage_4["authority_cutover_allowed"] is False


def test_stage4a_cannot_revert_to_stage3_shadow_closure_dependency(tmp_path: Path) -> None:
    assert_ledger_invalid(
        tmp_path,
        lambda x: x["stages"]["stage_4"].__setitem__("depends_on", ["stage_3_shadow_validation"]),
        "must depend on Stage 3 feature freeze",
    )


def test_stage4b_acceptance_cannot_be_reverted_by_editing_only_the_ledger(tmp_path: Path) -> None:
    def mutate(x: dict[str, Any]) -> None:
        x["stages"]["stage_4"]["phase_states"]["stage_4b_lossless_candidate"] = (
            "ready_for_authorization"
        )

    assert_ledger_invalid(tmp_path, mutate, "phase states must accept 4A/4B")


def test_stage4c_cannot_start_without_separate_authorization(tmp_path: Path) -> None:
    def mutate(x: dict[str, Any]) -> None:
        x["stages"]["stage_4"]["phase_states"]["stage_4c_context_assembly_shadow_validation"] = (
            "authorized"
        )

    assert_ledger_invalid(tmp_path, mutate, "phase states must accept 4A/4B")


def test_stage4_design_cannot_switch_authority(tmp_path: Path) -> None:
    assert_ledger_invalid(
        tmp_path,
        lambda x: x["stages"]["stage_4"].__setitem__("authority_cutover_allowed", True),
        "shadow authority boundary changed",
    )


def test_stage4_phase_order_cannot_be_collapsed(tmp_path: Path) -> None:
    assert_ledger_invalid(
        tmp_path,
        lambda x: x["stages"]["stage_4"].__setitem__("phase_plan", ["stage_4_all_at_once"]),
        "4A/4B/4C sequence",
    )


def test_stage4a_after_image_tamper_is_rejected(tmp_path: Path) -> None:
    relative = "docs/governance_stage4a_acceptance/AFTER_IMAGE.json"
    repo, payload = after_image_fixture(tmp_path, relative)
    target = repo / payload["files"][0]["path"]
    target.write_bytes(target.read_bytes() + b"\nunauthorized drift\n")
    with pytest.raises(VALIDATOR.StageStatusError, match="Stage 4A after-image drift"):
        VALIDATOR.validate_stage4a_after_image(repo, relative)


def test_stage4b_after_image_tamper_is_rejected(tmp_path: Path) -> None:
    relative = "docs/governance_stage4b_acceptance/AFTER_IMAGE.json"
    repo, payload = after_image_fixture(tmp_path, relative)
    target = repo / payload["files"][0]["path"]
    target.write_bytes(target.read_bytes() + b"\nunauthorized drift\n")
    with pytest.raises(VALIDATOR.StageStatusError, match="Stage 4B after-image drift"):
        VALIDATOR.validate_stage4b_after_image(repo, relative)


def test_stage5_waits_for_both_stage3_and_stage4_validation() -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    stage_5 = ledger["stages"]["stage_5"]
    assert stage_5["status"] == "blocked_by_predecessor"
    assert stage_5["implementation_allowed"] is False
    assert stage_5["authority_cutover_allowed"] is False
    assert stage_5["depends_on"] == ["stage_3_shadow_validation", "stage_4_lossless_validation"]


def test_stage5_cannot_drop_stage3_shadow_validation_dependency(tmp_path: Path) -> None:
    assert_ledger_invalid(
        tmp_path,
        lambda x: x["stages"]["stage_5"].__setitem__("depends_on", ["stage_4_lossless_validation"]),
        "wait for both Stage 3 and Stage 4 validation",
    )


def test_stage5_cannot_enable_authority_cutover(tmp_path: Path) -> None:
    assert_ledger_invalid(
        tmp_path,
        lambda x: x["stages"]["stage_5"].__setitem__("authority_cutover_allowed", True),
        "must forbid authority cutover",
    )
