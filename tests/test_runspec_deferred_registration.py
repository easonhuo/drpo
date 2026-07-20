from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "scripts" / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import runspec_safety as safety  # noqa: E402
from runspec_lib import RunSpecError, read_yaml  # noqa: E402
from runspec_registration import (  # noqa: E402
    DEFERRED,
    PRE_REGISTERED,
    validate_registration_block,
)


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr or proc.stdout)
    return proc.stdout.strip()


def init_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Deferred RunSpec Test")
    git(repo, "config", "user.email", "deferred@example.invalid")
    script = repo / "scripts" / "run.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        "#!/bin/sh\nset -eu\nmkdir -p outputs\necho '{}' > outputs/result.json\n",
        encoding="utf-8",
    )
    registry = repo / "experiments" / "registry.yaml"
    registry.parent.mkdir(parents=True)
    registry.write_text("experiments:\n- id: REGISTERED-ONLY\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    return repo, git(repo, "rev-parse", "HEAD")


def write_spec(
    repo: Path,
    commit: str,
    *,
    run_id: str,
    experiment_id: str,
    registration: dict[str, object] | None,
) -> Path:
    payload: dict[str, object] = {
        "version": 1,
        "run_id": run_id,
        "lane": "e7",
        "experiment_id": experiment_id,
        "repo_commit": commit,
        "provenance": {"commit_policy": "exact_head"},
        "entrypoint": {"command": "bash scripts/run.sh", "cwd": "repo_root"},
        "policy": {
            "formal_evidence_allowed": False,
            "existing_script_required": True,
            "forbid_new_launcher": True,
            "forbid_hparam_change": True,
            "forbid_cross_lane": True,
        },
        "outputs": {"run_dir": "outputs"},
        "artifacts": {
            "package_policy": "manifest_only",
            "include": ["outputs/result.json"],
            "exclude": [],
            "max_package_size_mb": 10,
        },
        "publish": {"enabled": False, "auto": False},
        "recovery": {"enabled": False},
    }
    if registration is not None:
        payload["registration"] = registration
    path = repo / "runspecs" / "ready" / f"{run_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_missing_registration_preserves_pre_registered_behavior(tmp_path: Path) -> None:
    repo, commit = init_repo(tmp_path)
    spec_path = write_spec(
        repo,
        commit,
        run_id="PRE-1",
        experiment_id="UNREGISTERED-EXP",
        registration=None,
    )
    with pytest.raises(RunSpecError, match="experiment_id not found"):
        safety.validate_runspec_safe(repo, spec_path, lane_config={"lane": "e7"})


def test_deferred_registration_is_claimable_without_registry_entry(tmp_path: Path) -> None:
    repo, commit = init_repo(tmp_path)
    spec_path = write_spec(
        repo,
        commit,
        run_id="DEFERRED-1",
        experiment_id="CODE-FIRST-EXP",
        registration={"mode": DEFERRED},
    )
    spec = safety.validate_runspec_safe(repo, spec_path, lane_config={"lane": "e7"})
    assert spec["registration"] == {"mode": DEFERRED, "closure_required": True}

    claimed = safety.claim_next_runspec_safe(
        repo,
        lane_config={"lane": "e7"},
        run_id="DEFERRED-1",
    )
    claimed_spec = read_yaml(claimed)
    assert claimed_spec["registration"]["mode"] == DEFERRED
    assert claimed_spec["registration"]["closure_required"] is True


def test_deferred_packaging_records_registration_timing(tmp_path: Path) -> None:
    repo, commit = init_repo(tmp_path)
    spec_path = write_spec(
        repo,
        commit,
        run_id="DEFERRED-PACKAGE-1",
        experiment_id="CODE-FIRST-EXP",
        registration={"mode": DEFERRED},
    )
    output = repo / "outputs" / "result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("{}\n", encoding="utf-8")

    manifest = safety.package_artifacts_safe(repo, spec_path)
    assert manifest["registration_mode"] == DEFERRED
    assert manifest["registration_closure_required"] is True
    manifest_path = repo / "runspec_artifacts" / "DEFERRED-PACKAGE-1_manifest.json"
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["registration_mode"] == DEFERRED
    assert persisted["registration_closure_required"] is True


def test_deferred_registration_requires_full_commit_sha() -> None:
    with pytest.raises(RunSpecError, match="full 40-character Git SHA"):
        validate_registration_block(
            {
                "registration": {"mode": DEFERRED},
                "repo_commit": "abc123",
            }
        )


def test_explicit_registry_override_still_fails_closed(tmp_path: Path) -> None:
    repo, commit = init_repo(tmp_path)
    spec_path = write_spec(
        repo,
        commit,
        run_id="DEFERRED-OVERRIDE-1",
        experiment_id="CODE-FIRST-EXP",
        registration={"mode": DEFERRED, "closure_required": True},
    )
    with pytest.raises(RunSpecError, match="experiment_id not found"):
        safety.validate_runspec_safe(
            repo,
            spec_path,
            lane_config={"lane": "e7"},
            require_registry=True,
        )


def test_registration_mode_validation_is_closed() -> None:
    with pytest.raises(RunSpecError, match="registration must be a mapping"):
        validate_registration_block({"registration": []})
    with pytest.raises(RunSpecError, match="pre_registered or deferred"):
        validate_registration_block(
            {
                "registration": {"mode": "after_results"},
                "repo_commit": "0" * 40,
            }
        )
    assert validate_registration_block({}) == {
        "mode": PRE_REGISTERED,
        "closure_required": False,
    }
