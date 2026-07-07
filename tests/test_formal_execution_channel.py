from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_formal_execution_channel.py"
SPEC = importlib.util.spec_from_file_location("formal_channel_validator", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)


def write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | 0o100)


def base_channel() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_id": "GOV-FORMAL-ENTRYPOINT-01",
        "channel_id": "hardened-v1",
        "registration_base_commit": "1" * 40,
        "guard_entrypoint": "scripts/run_experiment_guard_hardened.py",
        "package_entrypoint": "scripts/package_experiment_hardened.py",
        "verify_entrypoint": "scripts/verify_experiment_package_hardened.py",
        "hardened_core": "scripts/artifact_protocol_hardened.py",
        "artifact_protocol": "docs/formal_experiment_artifact_protocol.md",
        "validator_entrypoint": "scripts/validate_formal_execution_channel.py",
        "allowed_execution_classes": [
            "formal",
            "pilot",
            "historical_formal",
            "superseded",
        ],
        "default_artifact_budget": {
            "main_package_hard_limit_mib": 25,
            "single_file_main_limit_mib": 10,
            "large_file_storage": "persistent_local_index",
            "sidecar_default_enabled": False,
        },
        "formal_requirements": {
            "require_explicit_execution_class": True,
            "require_canonical_channel_ref": True,
            "require_guarded_launch": True,
            "require_canonical_artifact_owner": True,
            "forbid_new_runner_archive_writes": True,
            "fail_closed_on_missing_core": True,
        },
        "legacy_runner_archive_exceptions": [],
    }


def formal_experiment(entrypoint: str = "src/drpo/formal_runner.py") -> dict[str, Any]:
    return {
        "id": "FORMAL-TEST-01",
        "status": "not_run",
        "execution_class": "formal",
        "code_entrypoint": entrypoint,
        "formal_launch_template": (
            "python3 scripts/run_experiment_guard_hardened.py --run-class formal "
            "--experiment-id FORMAL-TEST-01 -- python3 " + entrypoint
        ),
        "formal_execution": {
            "channel_ref": "hardened-v1",
            "activation_state": "active",
            "entrypoint_status": "implemented",
            "entrypoint": entrypoint,
            "launch_mode": "canonical_guard",
            "artifact_owner": "canonical_channel",
            "guard_entrypoint": "scripts/run_experiment_guard_hardened.py",
            "package_entrypoint": "scripts/package_experiment_hardened.py",
            "verify_entrypoint": "scripts/verify_experiment_package_hardened.py",
            "hardened_core": "scripts/artifact_protocol_hardened.py",
            "artifact_protocol": "docs/formal_experiment_artifact_protocol.md",
            "inherit_default_artifact_budget": True,
            "runner_archive_policy": {"mode": "forbid"},
        },
    }


def make_repo(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    repo = tmp_path / "repo"
    write(repo / "scripts/artifact_protocol_hardened.py", "# shared core\n")
    wrapper_specs = {
        "run_experiment_guard_hardened.py": "guard_main",
        "package_experiment_hardened.py": "package_main",
        "verify_experiment_package_hardened.py": "verify_main",
    }
    for filename, symbol in wrapper_specs.items():
        write(
            repo / "scripts" / filename,
            (
                "try:\n"
                f"    from artifact_protocol_hardened import {symbol}\n"
                "except ModuleNotFoundError:\n"
                "    raise SystemExit(2)\n"
            ),
            executable=True,
        )
    write(repo / "scripts/validate_formal_execution_channel.py", "# validator\n")
    write(repo / "docs/formal_experiment_artifact_protocol.md", "# protocol\n")
    write(repo / "src/drpo/formal_runner.py", "print('formal')\n")
    registry = {
        "formal_execution_channel": base_channel(),
        "experiments": [formal_experiment()],
    }
    return repo, registry


def validate(repo: Path, registry: dict[str, Any]) -> dict[str, Any]:
    path = repo / "experiments/registry.yaml"
    write(path, yaml.safe_dump(registry, sort_keys=False))
    return VALIDATOR.validate_registry(repo, path)


def test_current_registry_uses_canonical_channel() -> None:
    report = VALIDATOR.validate_registry(REPO_ROOT, REPO_ROOT / "experiments/registry.yaml")
    assert report["matched"] is True
    assert report["channel_id"] == "hardened-v1"
    assert report["execution_class_counts"] == {
        "formal": 14,
        "historical_formal": 2,
        "pilot": 8,
        "superseded": 2,
    }
    assert report["formal_experiments"] == [
        "C-U1-E4-CONV-01",
        "C-U1-E4-TAPER-01",
        "C-U1-E4-TAPER-NEAR-RETENTION-01",
        "C-U1-E4-TAPER-BUDGET-MATCH-01",
        "C-U1-E4-TAPER-CONV-01",
        "C-U1-E4-TAPER-CONFIRM-01",
        "EXT-H-E7-Q2",
        "D-U1-E5-LONGRUN-RERUN",
        "EXT-H-E7-BENCH-01",
        "EXT-C-E8-SCALE-01",
        "D-U1-E6-SEMANTIC-LONGRUN-01",
        "D-U1-E6-SEMANTIC-GAP-LONGRUN-01",
        "D-U1-E6-CONDITIONAL-GAP-01",
        "D-U1-E6-CARTESIAN-TAPER-01",
    ]


def test_formal_experiment_rejects_noncanonical_guard(tmp_path: Path) -> None:
    repo, registry = make_repo(tmp_path)
    registry["experiments"][0]["formal_execution"]["guard_entrypoint"] = "scripts/custom_guard.py"
    with pytest.raises(VALIDATOR.ChannelError, match="guard_entrypoint"):
        validate(repo, registry)


def test_new_formal_entrypoint_direct_zip_is_rejected(tmp_path: Path) -> None:
    repo, registry = make_repo(tmp_path)
    write(
        repo / "src/drpo/formal_runner.py",
        "import zipfile\nzipfile.ZipFile('result.zip', 'w')\n",
    )
    with pytest.raises(VALIDATOR.ChannelError, match="creates archives directly"):
        validate(repo, registry)


def test_canonical_wrappers_must_share_hardened_core(tmp_path: Path) -> None:
    repo, registry = make_repo(tmp_path)
    write(
        repo / "scripts/package_experiment_hardened.py",
        "raise SystemExit(0)\n",
        executable=True,
    )
    with pytest.raises(VALIDATOR.ChannelError, match="shared hardened core"):
        validate(repo, registry)


def test_formal_experiment_rejects_noncanonical_packager(tmp_path: Path) -> None:
    repo, registry = make_repo(tmp_path)
    registry["experiments"][0]["formal_execution"]["package_entrypoint"] = (
        "scripts/custom_packager.py"
    )
    with pytest.raises(VALIDATOR.ChannelError, match="package_entrypoint"):
        validate(repo, registry)


def test_every_registry_experiment_requires_execution_class(tmp_path: Path) -> None:
    repo, registry = make_repo(tmp_path)
    del registry["experiments"][0]["execution_class"]
    with pytest.raises(VALIDATOR.ChannelError, match="execution_class"):
        validate(repo, registry)


def test_planned_formal_entrypoint_must_remain_blocked(tmp_path: Path) -> None:
    repo, registry = make_repo(tmp_path)
    experiment = registry["experiments"][0]
    experiment["implementation_state"] = "not_implemented"
    experiment["formal_execution"].update(
        {
            "activation_state": "blocked",
            "entrypoint_status": "planned",
            "entrypoint": None,
        }
    )
    experiment["execution_gate"] = {
        "state": "blocked",
        "blocking_reason": "runner implementation is not complete",
    }
    experiment.pop("formal_launch_template")
    report = validate(repo, registry)
    assert report["experiment_reports"][0]["state"] == "planned_blocked"


def test_active_planned_entrypoint_is_rejected(tmp_path: Path) -> None:
    repo, registry = make_repo(tmp_path)
    experiment = registry["experiments"][0]
    experiment["implementation_state"] = "not_implemented"
    experiment["formal_execution"].update(
        {
            "entrypoint_status": "planned",
            "entrypoint": None,
        }
    )
    with pytest.raises(VALIDATOR.ChannelError, match="must remain blocked"):
        validate(repo, registry)


def test_ready_implemented_experiment_cannot_remain_blocked(tmp_path: Path) -> None:
    repo, registry = make_repo(tmp_path)
    experiment = registry["experiments"][0]
    experiment["execution_gate"] = {"state": "ready", "reason": "approved"}
    experiment["formal_execution"]["activation_state"] = "blocked"
    with pytest.raises(VALIDATOR.ChannelError, match="requires activation_state=active"):
        validate(repo, registry)


def test_active_experiment_cannot_have_blocked_gate(tmp_path: Path) -> None:
    repo, registry = make_repo(tmp_path)
    registry["experiments"][0]["execution_gate"] = {
        "state": "blocked",
        "blocking_reason": "dependency missing",
    }
    with pytest.raises(VALIDATOR.ChannelError, match="cannot sit behind"):
        validate(repo, registry)


def test_blocked_experiment_requires_reason_or_dependency(tmp_path: Path) -> None:
    repo, registry = make_repo(tmp_path)
    experiment = registry["experiments"][0]
    experiment["formal_execution"]["activation_state"] = "blocked"
    with pytest.raises(VALIDATOR.ChannelError, match="requires a dependency"):
        validate(repo, registry)


def test_planned_blocked_experiment_with_reason_is_allowed(tmp_path: Path) -> None:
    repo, registry = make_repo(tmp_path)
    experiment = registry["experiments"][0]
    experiment["implementation_state"] = "not_implemented"
    experiment["formal_execution"].update(
        {
            "activation_state": "blocked",
            "entrypoint_status": "planned",
            "entrypoint": None,
        }
    )
    experiment["execution_gate"] = {
        "state": "blocked",
        "blocked_by": ["FORMAL-DEPENDENCY-01"],
    }
    experiment.pop("formal_launch_template")
    report = validate(repo, registry)
    assert report["experiment_reports"][0]["state"] == "planned_blocked"


def test_repository_development_formal_registrations_are_fail_closed() -> None:
    report = VALIDATOR.validate_registry(REPO_ROOT, REPO_ROOT / "experiments" / "registry.yaml")
    experiments = {item["id"]: item for item in report["experiment_reports"]}
    development = {item["id"]: item for item in report["development_registration_reports"]}
    assert development["D-U1-E6-SEMANTIC-PILOT-01"]["state"] == "development_nonformal"
    assert experiments["D-U1-E6-SEMANTIC-LONGRUN-01"]["state"] == "blocked"
    assert experiments["EXT-H-E7-Q2"]["state"] == "blocked"
    assert development["D-U1-E6-TAPER-01"]["state"] == "planned_blocked"


def test_blocked_development_formal_registration_requires_metadata(tmp_path: Path) -> None:
    repo, registry = make_repo(tmp_path)
    registration = copy.deepcopy(registry["experiments"][0])
    registration["id"] = "DEV-FORMAL-01"
    registration["implementation_state"] = "not_implemented"
    registration["formal_execution"].update(
        {
            "activation_state": "blocked",
            "entrypoint_status": "planned",
            "entrypoint": None,
        }
    )
    registration.pop("formal_launch_template")
    registry["development_experiment_registrations"] = [registration]
    with pytest.raises(VALIDATOR.ChannelError, match="requires a dependency"):
        validate(repo, registry)


def test_legacy_archive_exception_cannot_be_copied_to_new_entrypoint(
    tmp_path: Path,
) -> None:
    repo, registry = make_repo(tmp_path)
    write(
        repo / "src/drpo/legacy_runner.py",
        "import zipfile\nzipfile.ZipFile('checkpoint.zip', 'w')\n",
    )
    registry["formal_execution_channel"]["legacy_runner_archive_exceptions"] = [
        {
            "exception_id": "LEGACY-ONLY",
            "entrypoint": "src/drpo/legacy_runner.py",
            "allowed_experiment_ids": ["OLD-FORMAL-01"],
            "scope": "recovery_checkpoint_only",
            "no_new_entrypoints": True,
        }
    ]
    registry["experiments"][0]["formal_execution"]["runner_archive_policy"] = {
        "mode": "legacy_exception",
        "exception_id": "LEGACY-ONLY",
    }
    with pytest.raises(VALIDATOR.ChannelError, match="entrypoint mismatch"):
        validate(repo, registry)


def test_formal_requirements_cannot_be_weakened(tmp_path: Path) -> None:
    repo, registry = make_repo(tmp_path)
    registry["formal_execution_channel"]["formal_requirements"][
        "forbid_new_runner_archive_writes"
    ] = False
    with pytest.raises(VALIDATOR.ChannelError, match="fail-closed requirements"):
        validate(repo, registry)


def test_guarded_pilot_operator_must_exist(tmp_path: Path) -> None:
    repo, registry = make_repo(tmp_path)
    pilot = {
        "id": "PILOT-01",
        "status": "pilot",
        "execution_class": "pilot",
        "pilot_execution": {
            "channel_ref": "hardened-v1",
            "launch_mode": "guarded_orchestrator",
            "operator_entrypoint": "scripts/missing_pilot.py",
            "guard_required": True,
        },
    }
    registry["experiments"] = [pilot]
    with pytest.raises(VALIDATOR.ChannelError, match="pilot operator"):
        validate(repo, registry)


def test_cli_default_output_is_compact(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo, registry = make_repo(tmp_path)
    registry_path = repo / "experiments/registry.yaml"
    write(registry_path, yaml.safe_dump(registry, sort_keys=False))
    report_path = repo / "formal_channel_report.json"
    rc = VALIDATOR.main(
        [
            "--repo-root",
            os.fspath(repo),
            "--report-out",
            os.fspath(report_path),
        ]
    )
    assert rc == 0
    output = capsys.readouterr().out
    assert "Formal execution channel validation: PASS" in output
    assert "experiment_reports" not in output
    assert "experiment_reports" in report_path.read_text(encoding="utf-8")
