from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENT_SCRIPTS = ROOT / "scripts" / "agent"
if str(AGENT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(AGENT_SCRIPTS))

from runspec_lib import validate_runspec  # noqa: E402
from runspec_recovery import validate_recovery_policy  # noqa: E402


PINNED_IMPLEMENTATION = "d4cdc176562db1fa439afa9f9a0cbd9a9c8d0259"
PILOT_RUNSPEC = (
    ROOT
    / "runspecs"
    / "templates"
    / "E7_SQUARED_EXP_NIGHT_PILOT_20260714_01.yaml"
)
LIVENESS_RUNSPEC = (
    ROOT
    / "runspecs"
    / "templates"
    / "E7_SQUARED_EXP_NIGHT_LIVENESS_20260714_01.yaml"
)
AUTO_SCRIPT = ROOT / "scripts" / "run_e7_squared_exp_night_auto.py"
RUN_SCRIPT = ROOT / "scripts" / "run_e7_squared_exp_night_one_click.sh"
RESUME_SCRIPT = ROOT / "scripts" / "run_e7_squared_exp_night_resume_one_click.sh"
LIVENESS_SCRIPT = ROOT / "scripts" / "run_e7_squared_exp_night_liveness_one_click.sh"
CAP_VALIDATOR = ROOT / "scripts" / "validate_user_approved_worker_cap.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def test_pilot_runspec_is_valid_and_recovery_is_bounded() -> None:
    spec = validate_runspec(ROOT, PILOT_RUNSPEC, require_registry=False)
    assert spec["experiment_id"] == "EXT-H-E7-SQUARED-EXP-NIGHT-01"
    assert spec["repo_commit"] == PINNED_IMPLEMENTATION
    assert spec["policy"]["formal_evidence_allowed"] is False
    criteria = " ".join(spec["success_criteria"])
    assert "126" in criteria
    assert "1000000" in criteria
    assert "500000" in criteria
    assert "KL-refresh diagnostics" in criteria
    assert "GAE_STAGE_STATUS.json" in criteria
    assert "LAUNCH_REGISTRATION_STATUS.json" in criteria
    assert spec["outputs"]["run_dir"].endswith("squared_exp_night_1m_001")

    recovery = validate_recovery_policy(ROOT, spec)
    assert recovery is not None
    assert recovery["max_attempts"] == 2
    assert recovery["retryable_exit_codes"] == [137, 143]
    assert recovery["resume_command"] == (
        "bash scripts/run_e7_squared_exp_night_resume_one_click.sh"
    )


def test_liveness_runspec_is_valid_and_non_scientific() -> None:
    spec = validate_runspec(ROOT, LIVENESS_RUNSPEC, require_registry=False)
    assert spec["experiment_id"] == "EXT-H-E7-SQUARED-EXP-NIGHT-01"
    assert spec["repo_commit"] == PINNED_IMPLEMENTATION
    assert spec["policy"]["scientific_aggregation_allowed"] is False
    assert spec["policy"]["formal_evidence_allowed"] is False
    assert spec["entrypoint"]["command"] == (
        "bash scripts/run_e7_squared_exp_night_liveness_one_click.sh"
    )
    script = LIVENESS_SCRIPT.read_text()
    assert 'PROBE_STEPS="${E7_SQUARED_EXP_LIVENESS_PROBE_STEPS:-500}"' in script
    assert 'MAX_WORKERS="${E7_SQUARED_EXP_LIVENESS_MAX_WORKERS:-2}"' in script
    assert validate_recovery_policy(ROOT, spec) is None


def test_code_first_launch_is_not_blocked_by_registration() -> None:
    forbidden_messages = (
        "experiment is not authoritatively registered",
        "experiment is absent from experiments/registry.yaml",
    )
    for path in (RUN_SCRIPT, RESUME_SCRIPT, LIVENESS_SCRIPT):
        text = path.read_text()
        for message in forbidden_messages:
            assert message not in text
    auto = AUTO_SCRIPT.read_text()
    assert "LAUNCH_REGISTRATION_STATUS.json" in auto
    assert '"launch_blocked_by_registration": False' in auto
    assert '"code_first_pre_registration"' in auto


def test_templates_pin_all_scientific_and_execution_paths() -> None:
    pilot = yaml.safe_load(PILOT_RUNSPEC.read_text())
    liveness = yaml.safe_load(LIVENESS_RUNSPEC.read_text())
    assert pilot["repo_commit"] == PINNED_IMPLEMENTATION
    assert liveness["repo_commit"] == PINNED_IMPLEMENTATION
    assert pilot["provenance"]["commit_policy"] == "protected_paths_unchanged"
    assert liveness["provenance"]["commit_policy"] == "protected_paths_unchanged"

    required_common = {
        "configs/e7_squared_exp_night_v1.json",
        "src/drpo/e7_squared_exp_kernel.py",
        "src/drpo/e7_ppo_kl_refresh.py",
        "src/drpo/e7_squared_exp_night.py",
        "src/drpo/e7_squared_exp_night_bootstrap.py",
        "src/drpo/e7_squared_exp_night_aggregate.py",
        "src/drpo/e7_squared_exp_night_runtime_autotune.py",
        "scripts/run_e7_squared_exp_night_auto.py",
    }
    pilot_protected = set(pilot["provenance"]["protected_paths"])
    liveness_protected = set(liveness["provenance"]["protected_paths"])
    assert required_common <= pilot_protected
    assert required_common <= liveness_protected
    assert "scripts/run_e7_squared_exp_night_one_click.sh" in pilot_protected
    assert "scripts/run_e7_squared_exp_night_resume_one_click.sh" in pilot_protected
    assert (
        "scripts/run_e7_squared_exp_night_liveness_one_click.sh"
        in liveness_protected
    )


def test_one_click_resume_contract() -> None:
    run_text = RUN_SCRIPT.read_text()
    resume_text = RESUME_SCRIPT.read_text()
    for text in (run_text, resume_text):
        assert "DRPO_RUNTIME_MAX_WORKERS" in text
        assert "DRPO_RUNTIME_MAX_WORKERS_APPROVAL_FILE" in text
        assert "E7_SQUARED_EXP_MAX_WORKERS_APPROVAL_FILE" in text
        assert "validate_user_approved_worker_cap.sh" in text
        assert "USER_APPROVED_WORKER_CAP.json" not in text
    assert "RUNTIME_SELECTION.json" in run_text
    assert "RUN_IDENTITY.json" in run_text
    assert "partial runtime identity" in run_text
    assert "--resume" in run_text


def test_worker_cap_gate_defaults_unset_and_requires_exact_approval(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Worker Cap Test")

    copied_paths = (
        "scripts/validate_user_approved_worker_cap.sh",
        "scripts/run_e7_squared_exp_night_one_click.sh",
        "scripts/run_e7_squared_exp_night_resume_one_click.sh",
        "scripts/run_e7_squared_exp_night_auto.py",
        "src/drpo/e7_squared_exp_night_runtime_autotune.py",
        "src/drpo/e7_squared_exp_night.py",
    )
    for relative in copied_paths:
        source = ROOT / relative
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    contract = repo / "fixtures" / "contract.json"
    run_spec = repo / "fixtures" / "run_spec.json"
    grid = repo / "configs" / "cap_test_grid.json"
    contract.parent.mkdir(parents=True, exist_ok=True)
    grid.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text('{"contract": 1}\n', encoding="utf-8")
    run_spec.write_text('{"run_spec": 1}\n', encoding="utf-8")
    grid.write_text(
        json.dumps({"experiment_id": "EXT-H-E7-CAP-TEST-01"}) + "\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base runtime code")
    approved_code_commit = _git(repo, "rev-parse", "HEAD")

    validator = repo / "scripts" / "validate_user_approved_worker_cap.sh"
    unset_work = tmp_path / "work-unset"
    unset = subprocess.run(
        [
            "bash",
            str(validator),
            str(repo),
            str(unset_work),
            "",
            "",
            str(contract),
            str(run_spec),
            str(grid),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert unset.returncode == 0, unset.stderr
    unset_identity = json.loads(
        (unset_work / "USER_APPROVED_WORKER_CAP.json").read_text(encoding="utf-8")
    )
    assert unset_identity["mode"] == "unset_autotune_controls_concurrency"
    assert unset_identity["max_workers"] is None

    cap_work = tmp_path / "work-cap"
    missing_approval = subprocess.run(
        [
            "bash",
            str(validator),
            str(repo),
            str(cap_work),
            "4",
            "",
            str(contract),
            str(run_spec),
            str(grid),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing_approval.returncode == 2
    assert "requires an explicit approval file" in missing_approval.stderr

    approval = repo / "docs" / "runtime_worker_cap_authorizations" / "test.json"
    approval.parent.mkdir(parents=True, exist_ok=True)
    approval_payload = {
        "schema_version": 1,
        "authorization_id": "E7-WORKER-CAP-TEST-01",
        "status": "approved",
        "approved_by": "repository_owner",
        "approval_reference": "test-only explicit owner approval",
        "reason": "exercise the fail-closed cap gate",
        "scope": {
            "experiment_id": "EXT-H-E7-CAP-TEST-01",
            "work_dir": str(cap_work.resolve()),
            "max_workers": 4,
            "affinity_cpu_ids": sorted(os.sched_getaffinity(0)),
            "contract_sha256": _sha256(contract),
            "run_spec_sha256": _sha256(run_spec),
            "grid_sha256": _sha256(grid),
            "approved_code_commit": approved_code_commit,
        },
    }
    approval.write_text(
        json.dumps(approval_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _git(repo, "add", str(approval.relative_to(repo)))
    _git(repo, "commit", "-m", "record exact cap approval")

    approved = subprocess.run(
        [
            "bash",
            str(validator),
            str(repo),
            str(cap_work),
            "4",
            str(approval),
            str(contract),
            str(run_spec),
            str(grid),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert approved.returncode == 0, approved.stderr
    cap_identity = json.loads(
        (cap_work / "USER_APPROVED_WORKER_CAP.json").read_text(encoding="utf-8")
    )
    assert cap_identity["mode"] == "user_approved_hard_cap"
    assert cap_identity["max_workers"] == 4
    assert cap_identity["authorization"]["authorization_id"] == (
        "E7-WORKER-CAP-TEST-01"
    )

    changed = subprocess.run(
        [
            "bash",
            str(validator),
            str(repo),
            str(cap_work),
            "5",
            str(approval),
            str(contract),
            str(run_spec),
            str(grid),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert changed.returncode == 2

    removed = subprocess.run(
        [
            "bash",
            str(validator),
            str(repo),
            str(cap_work),
            "",
            "",
            str(contract),
            str(run_spec),
            str(grid),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert removed.returncode == 2
    assert "worker-cap policy changed" in removed.stderr
