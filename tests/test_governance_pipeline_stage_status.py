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
AUTHORITY = REPO_ROOT / "docs" / "handoff_versions" / "AUTHORITY.yaml"

_SPEC = importlib.util.spec_from_file_location("stage_status_validator", VALIDATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(VALIDATOR)


def run_validator(repo: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / VALIDATOR_PATH.name),
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
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    assert f"Stage 5={ledger['stages']['stage_5']['status']}" in proc.stdout


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


def test_stage5_hardened_control_plane_tamper_is_rejected() -> None:
    repo = hardlink_repository()
    try:
        ledger = yaml.safe_load((repo / "docs/governance_pipeline_stage_status.yaml").read_text())
        protected = ledger["stages"]["stage_5"]["protected_files"][0]["path"]
        path = repo / protected
        replace_bytes(path, path.read_bytes() + b"\n# unauthorized Stage 5 drift\n")
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


def test_closed_stage_cannot_be_reopened_by_editing_only_the_ledger(
    tmp_path: Path,
) -> None:
    assert_ledger_invalid(
        tmp_path,
        lambda x: x["stages"]["stage_1"].__setitem__("status", "reopened"),
        "does not authorize status reopened",
    )


def test_ledger_hash_edit_without_matching_authorization_is_rejected(
    tmp_path: Path,
) -> None:
    def mutate(x: dict[str, Any]) -> None:
        x["stages"]["stage_2"]["protected_files"][0]["sha256"] = "0" * 64

    assert_ledger_invalid(tmp_path, mutate, "ledger hash is not bound by authorization")


def test_shadow_active_stage_keeps_manual_authority_and_forbids_cutover(
    tmp_path: Path,
) -> None:
    assert_ledger_invalid(
        tmp_path,
        lambda x: x["stages"]["stage_3"].__setitem__("authority_cutover_allowed", True),
        "must forbid authority cutover",
    )


def test_shadow_active_stage_requires_derived_observation_ledger(
    tmp_path: Path,
) -> None:
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


def test_stage4a_cannot_revert_to_stage3_shadow_closure_dependency(
    tmp_path: Path,
) -> None:
    assert_ledger_invalid(
        tmp_path,
        lambda x: x["stages"]["stage_4"].__setitem__("depends_on", ["stage_3_shadow_validation"]),
        "must depend on Stage 3 feature freeze",
    )


def test_stage4b_acceptance_cannot_be_reverted_by_editing_only_the_ledger(
    tmp_path: Path,
) -> None:
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


def test_stage4a_dynamic_outputs_use_current_source_determinism_not_frozen_hash(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(
        REPO_ROOT,
        repo,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc"),
    )
    handoff = repo / "docs/handoff.md"
    handoff.write_text(
        handoff.read_text(encoding="utf-8")
        + "\n<!-- stage4a-current-source-determinism-probe -->\n",
        encoding="utf-8",
    )
    build = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/build_stage4_context.py"),
            "--repo-root",
            str(repo),
            "--json",
            "build",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert build.returncode == 0, build.stderr or build.stdout
    result = VALIDATOR.validate_stage4a_after_image(
        repo,
        "docs/governance_stage4a_acceptance/AFTER_IMAGE.json",
    )
    assert result["dynamic_file_count"] > 0

    dynamic = repo / "docs/handoff_shadow/stage4/minimal/generated/MODULE_INDEX.json"
    dynamic.write_text(dynamic.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(
        VALIDATOR.StageStatusError,
        match="dynamic outputs are not deterministic current-source materializations",
    ):
        VALIDATOR.validate_stage4a_after_image(
            repo,
            "docs/governance_stage4a_acceptance/AFTER_IMAGE.json",
        )


def test_stage4b_after_image_tamper_is_rejected(tmp_path: Path) -> None:
    relative = "docs/governance_stage4b_acceptance/AFTER_IMAGE.json"
    repo, payload = after_image_fixture(tmp_path, relative)
    target = repo / payload["files"][0]["path"]
    target.write_bytes(target.read_bytes() + b"\nunauthorized drift\n")
    with pytest.raises(VALIDATOR.StageStatusError, match="Stage 4B after-image drift"):
        VALIDATOR.validate_stage4b_after_image(repo, relative)


def _current_stage5_mode() -> str:
    payload = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    return payload["mode"]


def test_stage5_authority_state_matches_mode_and_dependencies() -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    stage_5 = ledger["stages"]["stage_5"]
    mode = _current_stage5_mode()

    assert stage_5["implementation_allowed"] is False
    assert stage_5["repository_pre_cutover_closure"] == "complete"
    assert stage_5["accepted_candidate_commit"] == ("65fc7539e89d6ff4405dde09174224f8ef69228e")
    assert stage_5["confirmed_blockers_open"] == []
    assert stage_5["code_only_normalization"] == "verified_no_op"
    assert stage_5["stage4a_generated_validation"] == "current_source_deterministic"
    assert stage_5["checkpoint_manifest_schema_version"] == 2
    assert stage_5["stage3_full_acceptance_current_at_cutover_required"] is True
    assert stage_5["depends_on"] == [
        "stage_3_shadow_validation",
        "stage_4_lossless_validation",
    ]

    if mode == "manual":
        assert stage_5["status"] == "active"
        assert stage_5["cutover_lifecycle_state"] == "implemented_not_executed"
        assert stage_5["candidate_only"] is True
        assert stage_5["current_write_authority"] == "manual_handoff"
        assert stage_5["implementation_state"] == "candidate_hardened_pre_cutover_accepted"
        assert stage_5["pre_cutover_acceptance_state"] == "independently_accepted"
        assert stage_5["authority_cutover_allowed"] is False
        assert stage_5["production_cutover_executed"] is False
    elif mode == "delta":
        authority = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
        assert stage_5["status"] in {"active", "closed_maintenance_only"}
        if stage_5["status"] == "active":
            assert stage_5["cutover_lifecycle_state"] == "implemented_not_executed"
            assert stage_5["status_authorization"] == stage_5["cutover_authorization"]
        else:
            assert stage_5["cutover_lifecycle_state"] == "executed_verified"
            closure_path = (
                REPO_ROOT
                / "docs/governance_stage_authorizations"
                / f"{stage_5['status_authorization']}.yaml"
            )
            closure = yaml.safe_load(closure_path.read_text(encoding="utf-8"))
            assert closure["kind"] == "closure"
            assert closure["authorized_stage_statuses"]["stage_5"] == "closed_maintenance_only"
            assert closure["closure_evidence"]["activation_commit"] == (
                "e33a3d1ce8de8ebaf0969a2ec9830a031f7a6c04"
            )
        assert stage_5["candidate_only"] is False
        assert stage_5["current_write_authority"] == "schema_v3_delta"
        assert stage_5["implementation_state"] == "production_delta_authority_active"
        assert stage_5["pre_cutover_acceptance_state"] == "accepted_by_cutover_authorization"
        assert stage_5["authority_cutover_allowed"] is True
        assert stage_5["production_cutover_executed"] is True
        assert (
            stage_5["cutover_checkpoint_manifest"]
            == authority["delta_authority"]["checkpoint_manifest"]
        )
    else:  # pragma: no cover - the validator rejects this first
        raise AssertionError(f"unsupported authority mode: {mode}")


def test_stage5_closed_delta_rejects_unverified_lifecycle(tmp_path: Path) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    if (
        _current_stage5_mode() != "delta"
        or ledger["stages"]["stage_5"]["status"] != "closed_maintenance_only"
    ):
        pytest.skip("current repository is not in closed Stage 5 delta mode")
    assert_ledger_invalid(
        tmp_path,
        lambda x: x["stages"]["stage_5"].__setitem__(
            "cutover_lifecycle_state", "implemented_not_executed"
        ),
        "verified closure authorization and activation evidence",
    )


def test_stage5_closed_delta_rejects_cutover_authorization_as_status_authorization(
    tmp_path: Path,
) -> None:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    if (
        _current_stage5_mode() != "delta"
        or ledger["stages"]["stage_5"]["status"] != "closed_maintenance_only"
    ):
        pytest.skip("current repository is not in closed Stage 5 delta mode")
    assert_ledger_invalid(
        tmp_path,
        lambda x: x["stages"]["stage_5"].__setitem__(
            "status_authorization",
            x["stages"]["stage_5"]["cutover_authorization"],
        ),
        "does not authorize status closed_maintenance_only",
    )


def test_stage5_acceptance_state_cannot_desynchronize_from_authority_mode(
    tmp_path: Path,
) -> None:
    mode = _current_stage5_mode()
    replacement = (
        "ready_for_independent_acceptance" if mode == "manual" else "independently_accepted"
    )
    message = (
        "independently accepted manual boundary"
        if mode == "manual"
        else "complete authorized delta cutover"
    )
    assert_ledger_invalid(
        tmp_path,
        lambda x: x["stages"]["stage_5"].__setitem__("pre_cutover_acceptance_state", replacement),
        message,
    )


def test_safe_repo_path_accepts_canonical_repo_root_alias(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    report = repo / "docs" / "report.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}\n", encoding="utf-8")
    alias = tmp_path / "repo-alias"
    alias.symlink_to(repo, target_is_directory=True)

    resolved = VALIDATOR.safe_repo_path(alias, "docs/report.json", "alias probe")

    assert resolved == report.resolve()


def test_stage5_historical_acceptance_does_not_freeze_current_bytes() -> None:
    repo = hardlink_repository()
    try:
        accepted = repo / "scripts/handoff_authority.py"
        replace_bytes(accepted, accepted.read_bytes() + b"\n# authorized maintenance probe\n")
        ledger = yaml.safe_load(
            (repo / "docs/governance_pipeline_stage_status.yaml").read_text(encoding="utf-8")
        )
        stage_5 = ledger["stages"]["stage_5"]
        result = VALIDATOR.validate_stage5_pre_cutover_acceptance(
            repo,
            report_path_value=stage_5["pre_cutover_acceptance_report"],
            stage3_report_path_value=stage_5["stage3_pre_stage5_full_acceptance_report"],
            accepted_commit=stage_5["accepted_candidate_commit"],
        )
        assert result["accepted_file_count"] > 0
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_stage5_acceptance_report_tamper_is_rejected(tmp_path: Path) -> None:
    repo = hardlink_repository()
    try:
        path = repo / "docs/governance_stage5_acceptance/ACCEPTANCE_REPORT.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["repository_pre_cutover_closure"] = "FAIL"
        replace_bytes(
            path,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        with pytest.raises(
            VALIDATOR.StageStatusError,
            match="independent pre-cutover acceptance report invalid",
        ):
            VALIDATOR.validate(
                repo,
                repo / "docs/governance_pipeline_stage_status.yaml",
            )
    finally:
        shutil.rmtree(repo)


def test_stage5_cannot_drop_stage3_shadow_validation_dependency(tmp_path: Path) -> None:
    assert_ledger_invalid(
        tmp_path,
        lambda x: x["stages"]["stage_5"].__setitem__("depends_on", ["stage_4_lossless_validation"]),
        "preserve Stage 3/4 validation dependencies",
    )


def test_stage5_authority_cutover_flag_cannot_desynchronize(
    tmp_path: Path,
) -> None:
    mode = _current_stage5_mode()
    replacement = mode == "manual"
    message = (
        "must forbid authority cutover" if mode == "manual" else "complete authorized delta cutover"
    )
    assert_ledger_invalid(
        tmp_path,
        lambda x: x["stages"]["stage_5"].__setitem__("authority_cutover_allowed", replacement),
        message,
    )


def test_stage5_authority_config_cannot_change_mode_without_complete_transaction() -> None:
    repo = hardlink_repository()
    try:
        path = repo / "docs/handoff_versions/AUTHORITY.yaml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if payload["mode"] == "manual":
            payload["mode"] = "delta"
            expected = "must remain manual candidate mode"
        else:
            payload["mode"] = "manual"
            ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
            expected = (
                "manual candidate mode cannot be closed before cutover"
                if ledger["stages"]["stage_5"]["status"] == "closed_maintenance_only"
                else "candidate must forbid authority cutover"
            )
        replace_bytes(path, yaml.safe_dump(payload, sort_keys=False).encode("utf-8"))
        proc = run_validator(repo, check=False)
        assert proc.returncode == 2
        assert expected in proc.stderr
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_stage5_dynamic_refresh_must_match_authority_mode() -> None:
    repo = hardlink_repository()
    try:
        path = repo / "docs/handoff_versions/AUTHORITY.yaml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        mode = payload["mode"]
        payload["generated_views"]["stage4a_minimal_refresh"] = mode == "manual"
        replace_bytes(path, yaml.safe_dump(payload, sort_keys=False).encode("utf-8"))
        proc = run_validator(repo, check=False)
        assert proc.returncode == 2
        expected = (
            "illegally activates cutover behavior"
            if mode == "manual"
            else "complete authorized delta cutover"
        )
        assert expected in proc.stderr
    finally:
        shutil.rmtree(repo, ignore_errors=True)
