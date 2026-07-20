from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import handoff_authority as authority  # noqa: E402
import handoff_delta_shadow as shadow  # noqa: E402


def run(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed\n{proc.stdout}\n{proc.stderr}")
    return proc


def git_text(repo: Path, *args: str) -> str:
    return run(repo, *args).stdout.strip()


def assert_governance_valid(repo: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/validate_governance_pipeline_stage_status.py"),
            "--repo-root",
            str(repo),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def copy_repository(destination: Path) -> Path:
    """Create a history-preserving fixture with the current worktree overlaid.

    Stage 5 pre-cutover acceptance is historical evidence bound to an older
    commit.  Re-initializing a one-commit synthetic repository discards that
    evidence and no longer models the real lifecycle.  Clone the local history,
    then apply the current uncommitted candidate diff and untracked files as one
    fixture-only maintenance commit.
    """

    source_head = git_text(REPO_ROOT, "rev-parse", "HEAD")
    committed_authority = yaml.safe_load(
        run(REPO_ROOT, "show", f"{source_head}:docs/handoff_versions/AUTHORITY.yaml").stdout
    )
    fixture_source = source_head
    maintenance_candidate = (
        os.environ.get("DRPO_STAGE5_MAINTENANCE_CANDIDATE") == "1"
    )
    if committed_authority["mode"] == "delta" and not maintenance_candidate:
        fixture_source = committed_authority["delta_authority"][
            "activation_parent_commit"
        ]

    clone = subprocess.run(
        ["git", "clone", "-q", "--no-hardlinks", str(REPO_ROOT), str(destination)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    assert clone.returncode == 0, clone.stderr or clone.stdout
    run(destination, "checkout", "-q", "-B", "main", fixture_source)
    run(destination, "remote", "remove", "origin")
    run(destination, "config", "user.name", "Stage5 Test")
    run(destination, "config", "user.email", "stage5@test.invalid")

    patch = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--binary", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    assert patch.returncode == 0, patch.stderr.decode(errors="replace")
    untracked = git_text(REPO_ROOT, "ls-files", "--others", "--exclude-standard")
    if fixture_source != source_head and (patch.stdout or untracked):
        raise AssertionError(
            "delta-mode Stage 5 integration fixtures require a clean committed source"
        )
    if patch.stdout:
        applied = subprocess.run(
            ["git", "-C", str(destination), "apply", "--binary", "-"],
            input=patch.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
        )
        assert applied.returncode == 0, applied.stderr.decode(errors="replace")

    for relative in untracked.splitlines():
        if not relative:
            continue
        source = REPO_ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)

    run(destination, "add", "-A")
    staged = run(destination, "diff", "--cached", "--quiet", check=False)
    if staged.returncode == 1:
        run(destination, "commit", "-q", "-m", "fixture current maintenance candidate")
    elif staged.returncode != 0:
        raise AssertionError(staged.stderr or staged.stdout)
    return destination


def add_test_current_full_acceptance(repo: Path) -> str:
    observations = shadow.observation_records(repo, replay=False)
    real_ids = sorted(str(item["update_id"]) for item in observations if item.get("kind") == "real")
    relative = "docs/handoff_deltas/STAGE5-TEST-PRE-CUTOVER-FULL/FULL_ACCEPTANCE_REPORT.json"
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    policy = yaml.safe_load((repo / "docs/handoff_delta_policy.yaml").read_text())
    payload = {
        "schema_version": 1,
        "report_schema_version": 2,
        "policy_id": policy["policy_id"],
        "tier": "full",
        "status": "PASS",
        "validation_worktree_head": git_text(repo, "rev-parse", "HEAD"),
        "reasons": ["stage5_test_pre_cutover_currentness"],
        "elapsed_seconds": 0.0,
        "target_seconds": float(policy["full_acceptance"]["target_seconds"]),
        "outcomes": [
            {
                "command": ["stage5-test-fixture-full-acceptance"],
                "returncode": 0,
                "timed_out": False,
                "elapsed_seconds": 0.0,
                "stdout": "test fixture currentness evidence",
                "stderr": "",
            }
        ],
        "coverage": {
            "bootstrap_observation_count": sum(
                item.get("kind") == "bootstrap" for item in observations
            ),
            "successful_real_observation_count": len(real_ids),
            "covered_update_ids": real_ids,
            "observation_fingerprint": shadow.observation_fingerprint(real_ids),
        },
        "corpus_audit": {
            "observation_count": len(observations),
            "all_stored_reports_revalidated": True,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run(repo, "add", relative)
    run(repo, "commit", "-q", "-m", "record current Stage 3 acceptance for test cutover")
    acceptance = shadow.acceptance_status(repo)
    assert acceptance["full_acceptance_due"] is False
    assert acceptance["uncovered_real_observation_count"] == 0
    assert acceptance["latest_full_acceptance"]["path"] == relative
    return relative


def add_test_cutover_authorization(repo: Path, *, checkpoint_id: str, authorization_id: str) -> str:
    base = git_text(repo, "rev-parse", "HEAD")
    relative = f"docs/governance_stage_authorizations/{authorization_id}.yaml"
    path = repo / relative
    payload = {
        "schema_version": 1,
        "authorization_id": authorization_id,
        "kind": "stage_transition",
        "change_class": "stage_transition",
        "claim_id": "GOV-HANDOFF-AUTHORITY-CUTOVER-01",
        "approval_record": "test_only_explicit_cutover_authorization",
        "base_commit": base,
        "cutover_checkpoint_id": checkpoint_id,
        "stage_ids": ["stage_5"],
        "authorized_stage_statuses": {"stage_5": "active"},
        "scope": [
            "activate_delta_handoff_authority",
            "create_cutover_checkpoint",
            "enable_manual_to_delta_cutover_transaction",
        ],
        "excluded_scope": [],
        "authorized_file_hashes": {},
        "rollback_plan": ["run_the_registered_delta_to_manual_rollback_transaction"],
        "remaining_uncertainties": ["test_only_authorization"],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    run(repo, "add", relative)
    run(repo, "commit", "-q", "-m", "authorize test cutover")
    return relative


def activate_delta_mode(repo: Path) -> tuple[str, str]:
    stage3_report = add_test_current_full_acceptance(repo)
    base = git_text(repo, "rev-parse", "HEAD")
    checkpoint_id = "STAGE5-TEST-CUTOVER"
    authorization_record = add_test_cutover_authorization(
        repo,
        checkpoint_id=checkpoint_id,
        authorization_id="GOV-STAGE5-TEST-CUTOVER-AUTH",
    )
    prepared = authority.prepare_cutover(
        repo,
        checkpoint_id=checkpoint_id,
        authorization_record=authorization_record,
        stage3_report=stage3_report,
        created_at_utc="2026-07-01T00:00:00+00:00",
    )
    assert prepared["mode"] == "cutover_prepared"
    prepared_report = authority.verify_prepared_cutover(repo)
    assert prepared_report["status"] == "PASS"
    assert prepared_report["mode"] == "cutover_prepared"
    assert prepared_report["source_parent_commit"] == git_text(repo, "rev-parse", "HEAD")
    run(repo, "add", "-A")
    run(repo, "commit", "-q", "-m", "activate delta authority")
    cutover = git_text(repo, "rev-parse", "HEAD")
    report = authority.verify_current_state(repo)
    assert report["mode"] == "delta"
    assert_governance_valid(repo)
    return base, cutover


def heading_path(repo: Path) -> list[str]:
    text = (repo / "docs/handoff.md").read_text()
    candidates = [h for h in shadow.parse_headings(text) if h.level == 2]
    assert candidates
    return list(candidates[0].path)


def make_source_delta(
    repo: Path,
    *,
    branch: str,
    base_commit: str,
    update_id: str,
    block_id: str,
    content: str,
    target_path: list[str],
) -> str:
    run(repo, "checkout", "-q", "-B", branch, base_commit)
    base_handoff = (repo / "docs/handoff.md").read_text()
    base_registry = (repo / "experiments/registry.yaml").read_text()
    operations = [
        {
            "operation_id": f"append-{block_id}",
            "op": "append_to_section",
            "heading_path": target_path,
            "block_id": block_id,
            "content": content,
        }
    ]
    candidate = shadow.render(base_handoff, operations).text
    delta = {
        "schema_version": 3,
        "update_id": update_id,
        "mode": "authoritative",
        "base": {
            "commit": base_commit,
            "handoff_sha256": shadow.sha256_text(base_handoff),
            "registry_sha256": shadow.sha256_text(base_registry),
        },
        "renderer_version": 1,
        "operations": operations,
        "registry": {
            "mode": "unchanged",
            "exact_base_after_sha256": None,
            "changes": [],
        },
        "expected": {
            "exact_base_candidate_sha256": shadow.sha256_text(candidate),
        },
    }
    path = repo / "docs/handoff_deltas" / update_id / "HANDOFF_DELTA.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(delta, sort_keys=False, allow_unicode=True))
    run(repo, "add", path.relative_to(repo).as_posix())
    run(repo, "commit", "-q", "-m", f"source {update_id}")
    return git_text(repo, "rev-parse", "HEAD")


def amend_source_file(repo: Path, source_commit: str, relative: str, text: str) -> str:
    run(repo, "checkout", "-q", source_commit)
    path = repo / relative
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")
    run(repo, "add", relative)
    run(repo, "commit", "--amend", "--no-edit", "-q")
    return git_text(repo, "rev-parse", "HEAD")


def add_worktree(repo: Path, destination: Path, commit: str, branch: str) -> Path:
    run(repo, "worktree", "add", "-q", "-b", branch, str(destination), commit)
    run(destination, "config", "user.name", "Stage5 Test")
    run(destination, "config", "user.email", "stage5@test.invalid")
    return destination


def normalize_source(
    central: Path,
    *,
    current: str,
    source_base: str,
    source_commit: str,
    name: str,
    tmp_path: Path,
) -> tuple[str, bytes, dict[str, bytes]]:
    trusted = add_worktree(central, tmp_path / f"trusted-{name}", current, f"trusted-{name}")
    target = add_worktree(central, tmp_path / f"target-{name}", current, f"target-{name}")
    cherry = run(target, "cherry-pick", source_commit, check=False)
    assert cherry.returncode == 0, cherry.stderr
    normalize = subprocess.run(
        [
            sys.executable,
            str(trusted / "scripts/handoff_authority.py"),
            "normalize",
            "--repo-root",
            str(target),
            "--trusted-repo-root",
            str(trusted),
            "--current-before",
            current,
            "--source-base",
            source_base,
            "--source-patch-commit",
            source_commit,
            "--json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    assert normalize.returncode == 0, normalize.stderr or normalize.stdout
    payload = json.loads(normalize.stdout)
    assert payload["status"] == "PASS"
    run(target, "add", "-A")
    run(target, "commit", "--amend", "--no-edit", "-q")
    normalized = git_text(target, "rev-parse", "HEAD")
    verify = subprocess.run(
        [
            sys.executable,
            str(target / "scripts/handoff_authority.py"),
            "verify",
            "--repo-root",
            str(target),
            "--json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    assert verify.returncode == 0, verify.stderr or verify.stdout
    handoff = (target / "docs/handoff.md").read_bytes()
    generated_root = target / "docs/handoff_shadow/stage4/minimal/generated"
    generated = {
        p.relative_to(generated_root).as_posix(): p.read_bytes()
        for p in sorted(generated_root.rglob("*"))
        if p.is_file()
    }
    return normalized, handoff, generated


def run_stale_independent_updates_commute_and_same_block_conflicts(tmp_path: Path) -> None:
    central = copy_repository(tmp_path / "repo")
    _, cutover = activate_delta_mode(central)
    target = heading_path(central)
    source_a = make_source_delta(
        central,
        branch="source-a",
        base_commit=cutover,
        update_id="STAGE5-TEST-A",
        block_id="stage5-test-a",
        content="A independent update.",
        target_path=target,
    )
    source_b = make_source_delta(
        central,
        branch="source-b",
        base_commit=cutover,
        update_id="STAGE5-TEST-B",
        block_id="stage5-test-b",
        content="B independent update.",
        target_path=target,
    )
    conflict = make_source_delta(
        central,
        branch="source-conflict",
        base_commit=cutover,
        update_id="STAGE5-TEST-CONFLICT",
        block_id="stage5-test-a",
        content="conflicting second meaning.",
        target_path=target,
    )

    direct_edit = make_source_delta(
        central,
        branch="source-direct-edit",
        base_commit=cutover,
        update_id="STAGE5-TEST-DIRECT-EDIT",
        block_id="stage5-test-direct-edit",
        content="Direct-edit rejection probe.",
        target_path=target,
    )
    direct_edit = amend_source_file(
        central, direct_edit, "docs/handoff.md", "\nunauthorized direct handoff edit\n"
    )
    control_edit = make_source_delta(
        central,
        branch="source-control-edit",
        base_commit=cutover,
        update_id="STAGE5-TEST-CONTROL-EDIT",
        block_id="stage5-test-control-edit",
        content="Control-plane rejection probe.",
        target_path=target,
    )
    control_edit = amend_source_file(
        central,
        control_edit,
        "docs/governance_stage5_versioned_handoff_spec.md",
        "\nunauthorized control-plane edit\n",
    )

    a_commit, _, _ = normalize_source(
        central,
        current=cutover,
        source_base=cutover,
        source_commit=source_a,
        name="a-first",
        tmp_path=tmp_path,
    )
    ab_commit, ab_handoff, ab_generated = normalize_source(
        central,
        current=a_commit,
        source_base=cutover,
        source_commit=source_b,
        name="ab",
        tmp_path=tmp_path,
    )
    assert ab_commit

    b_commit, _, _ = normalize_source(
        central,
        current=cutover,
        source_base=cutover,
        source_commit=source_b,
        name="b-first",
        tmp_path=tmp_path,
    )
    ba_commit, ba_handoff, ba_generated = normalize_source(
        central,
        current=b_commit,
        source_base=cutover,
        source_commit=source_a,
        name="ba",
        tmp_path=tmp_path,
    )
    assert ba_commit
    assert ab_handoff == ba_handoff
    assert ab_generated == ba_generated

    trusted = add_worktree(central, tmp_path / "trusted-conflict", a_commit, "trusted-conflict")
    target_repo = add_worktree(central, tmp_path / "target-conflict", a_commit, "target-conflict")
    cherry = run(target_repo, "cherry-pick", conflict, check=False)
    assert cherry.returncode == 0
    before = git_text(target_repo, "rev-parse", "HEAD")
    rejected = subprocess.run(
        [
            sys.executable,
            str(trusted / "scripts/handoff_authority.py"),
            "normalize",
            "--repo-root",
            str(target_repo),
            "--trusted-repo-root",
            str(trusted),
            "--current-before",
            a_commit,
            "--source-base",
            cutover,
            "--source-patch-commit",
            conflict,
            "--json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert rejected.returncode == 2
    assert "already exists" in rejected.stderr
    assert git_text(target_repo, "rev-parse", "HEAD") == before

    for label, source_commit, expected in (
        ("direct", direct_edit, "may not directly modify docs/handoff.md"),
        ("control", control_edit, "trusted control-plane paths"),
    ):
        trusted_reject = add_worktree(
            central, tmp_path / f"trusted-{label}", cutover, f"trusted-{label}"
        )
        target_reject = add_worktree(
            central, tmp_path / f"target-{label}", cutover, f"target-{label}"
        )
        cherry = run(target_reject, "cherry-pick", source_commit, check=False)
        assert cherry.returncode == 0
        before_reject = git_text(target_reject, "rev-parse", "HEAD")
        proc = subprocess.run(
            [
                sys.executable,
                str(trusted_reject / "scripts/handoff_authority.py"),
                "normalize",
                "--repo-root",
                str(target_reject),
                "--trusted-repo-root",
                str(trusted_reject),
                "--current-before",
                cutover,
                "--source-base",
                cutover,
                "--source-patch-commit",
                source_commit,
                "--json",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert proc.returncode == 2
        assert expected in proc.stderr
        assert git_text(target_reject, "rev-parse", "HEAD") == before_reject


def run_preintegration_report_revision_and_postintegration_tamper(
    tmp_path: Path,
) -> None:
    central = copy_repository(tmp_path / "preintegration-report-revision")
    current_authority = yaml.safe_load(
        (central / "docs/handoff_versions/AUTHORITY.yaml").read_text(encoding="utf-8")
    )
    if current_authority["mode"] == "delta":
        cutover = git_text(central, "rev-parse", "HEAD")
    else:
        _, cutover = activate_delta_mode(central)
    source = make_source_delta(
        central,
        branch="source-preintegration-report",
        base_commit=cutover,
        update_id="STAGE5-TEST-PREINTEGRATION-REPORT",
        block_id="stage5-test-preintegration-report",
        content="Pre-integration materialization-report history probe.",
        target_path=heading_path(central),
    )
    normalized, _, _ = normalize_source(
        central,
        current=cutover,
        source_base=cutover,
        source_commit=source,
        name="preintegration-report",
        tmp_path=tmp_path,
    )

    run(central, "checkout", "-q", "-B", "revised-report", normalized)
    relative = (
        "docs/handoff_deltas/STAGE5-TEST-PREINTEGRATION-REPORT/"
        "MATERIALIZATION_REPORT.json"
    )
    report_path = central / relative
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    run(central, "add", relative)
    run(central, "commit", "-q", "-m", "revise report before integration")
    revised = git_text(central, "rev-parse", "HEAD")

    run(central, "checkout", "-q", "-B", "main", cutover)
    run(
        central,
        "merge",
        "--no-ff",
        "-q",
        revised,
        "-m",
        "integrate report revision",
    )
    accepted = authority.verify_current_state(central)
    assert accepted["status"] == "PASS"
    assert "STAGE5-TEST-PREINTEGRATION-REPORT" in accepted[
        "authoritative_update_ids"
    ]

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_path.write_text(
        json.dumps(payload, indent=4, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    run(central, "add", relative)
    run(central, "commit", "-q", "-m", "tamper report after integration")
    try:
        authority.verify_current_state(central)
    except authority.HandoffAuthorityError as exc:
        assert "not immutable after integration" in str(exc)
    else:  # pragma: no cover - fail closed assertion
        raise AssertionError("post-integration report tamper was accepted")


def run_code_only_noop_and_rollback(tmp_path: Path) -> None:
    central = copy_repository(tmp_path / "noop-repo")
    _, cutover = activate_delta_mode(central)
    handoff_before = (central / "docs/handoff.md").read_bytes()

    run(central, "checkout", "-q", "-B", "source-code-only", cutover)
    readme = central / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\nStage 5 code-only probe.\n")
    run(central, "add", "README.md")
    run(central, "commit", "-q", "-m", "code-only update")
    source_commit = git_text(central, "rev-parse", "HEAD")

    trusted = add_worktree(central, tmp_path / "noop-trusted", cutover, "noop-trusted")
    target = add_worktree(central, tmp_path / "noop-target", cutover, "noop-target")
    cherry = run(target, "cherry-pick", source_commit, check=False)
    assert cherry.returncode == 0, cherry.stderr
    payload = authority.normalize_update(
        target,
        trusted,
        current_before=cutover,
        source_base=cutover,
        source_patch_commit=source_commit,
    )
    assert payload["normalization"] == "no_op"
    assert (target / "docs/handoff.md").read_bytes() == handoff_before
    assert not list((target / "docs/handoff_deltas").glob("*/MATERIALIZATION_REPORT.json"))

    rollback_repo = add_worktree(
        central,
        tmp_path / "rollback-target",
        cutover,
        "rollback-target",
    )
    rollback = authority.prepare_rollback(
        rollback_repo,
        rollback_id="STAGE5-TEST-ROLLBACK",
        reason="integration rollback simulation",
        created_at_utc="2026-07-01T01:00:00+00:00",
    )
    assert rollback["mode"] == "rollback_prepared"
    assert (rollback_repo / "docs/handoff.md").read_bytes() == handoff_before
    run(rollback_repo, "add", "-A")
    run(rollback_repo, "commit", "-q", "-m", "rollback to manual authority")
    verified = authority.verify_current_state(rollback_repo)
    assert verified["mode"] == "manual"
    ledger = yaml.safe_load(
        (rollback_repo / "docs/governance_pipeline_stage_status.yaml").read_text(encoding="utf-8")
    )
    stage5 = ledger["stages"]["stage_5"]
    assert stage5["implementation_state"] == "candidate_hardened_pre_cutover_accepted"
    assert stage5["pre_cutover_acceptance_state"] == "independently_accepted"
    assert stage5["repository_pre_cutover_closure"] == "complete"
    assert_governance_valid(rollback_repo)
    assert (rollback_repo / "docs/handoff.md").read_bytes() == handoff_before


def run_prepared_cutover_rejects_unexpected_worktree_file(
    tmp_path: Path,
) -> None:
    repo = copy_repository(tmp_path / "prepared-boundary")
    stage3_report = add_test_current_full_acceptance(repo)
    checkpoint_id = "STAGE5-TEST-PREPARED-BOUNDARY"
    authorization_record = add_test_cutover_authorization(
        repo,
        checkpoint_id=checkpoint_id,
        authorization_id="GOV-STAGE5-TEST-PREPARED-BOUNDARY-AUTH",
    )
    authority.prepare_cutover(
        repo,
        checkpoint_id=checkpoint_id,
        authorization_record=authorization_record,
        stage3_report=stage3_report,
        created_at_utc="2026-07-01T00:00:00+00:00",
    )
    (repo / "unexpected.txt").write_text("not part of cutover\n", encoding="utf-8")
    try:
        authority.verify_prepared_cutover(repo)
    except authority.HandoffAuthorityError as exc:
        assert "worktree boundary mismatch" in str(exc)
    else:
        raise AssertionError("prepared verification accepted an unexpected file")


def run_cutover_requires_independent_authorization(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path / "cutover-auth-required")
    try:
        authority.prepare_cutover(
            repo,
            checkpoint_id="STAGE5-TEST-MISSING-CUTOVER-AUTH",
            authorization_record=(
                "docs/governance_stage_authorizations/"
                "GOV-STAGE5-PRE-CUTOVER-HARDENING-2026-07-02.yaml"
            ),
            created_at_utc="2026-07-01T01:30:00+00:00",
        )
    except authority.HandoffAuthorityError as exc:
        assert "separate passing Stage 5 stage-transition authorization" in str(exc)
    else:  # pragma: no cover - fail closed assertion
        raise AssertionError("cutover accepted the hardening authorization")
    assert git_text(repo, "status", "--porcelain") == ""


def run_checkpoint_gate_rejections(tmp_path: Path) -> None:
    stale = copy_repository(tmp_path / "checkpoint-report-stale")
    stale_base = git_text(stale, "rev-parse", "HEAD")
    make_source_delta(
        stale,
        branch="main",
        base_commit=stale_base,
        update_id="STAGE5-TEST-UNCOVERED-OBSERVATION",
        block_id="stage5-test-uncovered-observation",
        content="Stage 5 test-only uncovered observation.",
        target_path=heading_path(stale),
    )
    stale_checkpoint_id = "STAGE5-TEST-CHECKPOINT-REPORT-STALE"
    stale_auth = add_test_cutover_authorization(
        stale,
        checkpoint_id=stale_checkpoint_id,
        authorization_id="GOV-STAGE5-TEST-CHECKPOINT-REPORT-STALE-AUTH",
    )
    try:
        authority.prepare_cutover(
            stale,
            checkpoint_id=stale_checkpoint_id,
            authorization_record=stale_auth,
            created_at_utc="2026-07-01T01:40:00+00:00",
        )
    except authority.HandoffAuthorityError as exc:
        assert "does not cover all real observations" in str(exc)
    else:  # pragma: no cover - fail closed assertion
        raise AssertionError("cutover accepted a stale Stage 3 Full Acceptance report")
    assert git_text(stale, "status", "--porcelain") == ""

    failing = copy_repository(tmp_path / "checkpoint-report-fail")
    report_path = (
        failing / "docs/handoff_deltas/GOV-STAGE5-PRE-CUTOVER-ACCEPTANCE-CLOSURE-2026-07-02/"
        "FULL_ACCEPTANCE_REPORT.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "FAIL"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    run(failing, "add", report_path.relative_to(failing).as_posix())
    run(failing, "commit", "-q", "-m", "inject failing Stage 3 report")
    checkpoint_id = "STAGE5-TEST-CHECKPOINT-REPORT-FAIL"
    auth_record = add_test_cutover_authorization(
        failing,
        checkpoint_id=checkpoint_id,
        authorization_id="GOV-STAGE5-TEST-CHECKPOINT-REPORT-FAIL-AUTH",
    )
    try:
        authority.prepare_cutover(
            failing,
            checkpoint_id=checkpoint_id,
            authorization_record=auth_record,
            created_at_utc="2026-07-01T01:45:00+00:00",
        )
    except authority.HandoffAuthorityError as exc:
        assert "not a passing full report" in str(exc)
    else:  # pragma: no cover - fail closed assertion
        raise AssertionError("cutover accepted a failing Stage 3 report")
    assert git_text(failing, "status", "--porcelain") == ""

    tampered = copy_repository(tmp_path / "checkpoint-registry-tamper")
    _, _ = activate_delta_mode(tampered)
    authority_payload = yaml.safe_load(
        (tampered / "docs/handoff_versions/AUTHORITY.yaml").read_text(encoding="utf-8")
    )
    manifest_path = tampered / authority_payload["delta_authority"]["checkpoint_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["registry_sha256_for_provenance"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    run(tampered, "add", manifest_path.relative_to(tampered).as_posix())
    run(tampered, "commit", "--amend", "--no-edit", "-q")
    try:
        authority.verify_current_state(tampered)
    except authority.HandoffAuthorityError as exc:
        assert "source-parent registry" in str(exc)
    else:  # pragma: no cover - fail closed assertion
        raise AssertionError("checkpoint accepted a forged registry provenance hash")


def run_cutover_commit_rejects_first_delta(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path / "cutover-delta-reject")
    stage3_report = add_test_current_full_acceptance(repo)
    base = git_text(repo, "rev-parse", "HEAD")
    checkpoint_id = "STAGE5-TEST-CUTOVER-DELTA-REJECT"
    authorization_record = add_test_cutover_authorization(
        repo,
        checkpoint_id=checkpoint_id,
        authorization_id="GOV-STAGE5-TEST-CUTOVER-DELTA-REJECT-AUTH",
    )
    authority.prepare_cutover(
        repo,
        checkpoint_id=checkpoint_id,
        authorization_record=authorization_record,
        stage3_report=stage3_report,
        created_at_utc="2026-07-01T02:00:00+00:00",
    )
    delta_path = repo / "docs/handoff_deltas/STAGE5-FIRST-DELTA-IN-CUTOVER/HANDOFF_DELTA.yaml"
    delta_path.parent.mkdir(parents=True)
    delta_path.write_text(
        "schema_version: 3\nupdate_id: STAGE5-FIRST-DELTA-IN-CUTOVER\n",
        encoding="utf-8",
    )
    run(repo, "add", "-A")
    run(repo, "commit", "-q", "-m", "invalid cutover with first delta")
    assert base != git_text(repo, "rev-parse", "HEAD")
    try:
        authority.verify_current_state(repo)
    except authority.HandoffAuthorityError as exc:
        assert "may not include the first production schema-v3 delta" in str(exc)
    else:  # pragma: no cover - fail closed assertion
        raise AssertionError("cutover commit with a production delta was accepted")


def make_bundle_package(
    repo: Path,
    *,
    base_commit: str,
    patch_commit: str,
    source_branch: str,
    destination: Path,
) -> Path:
    package_root = destination / "package"
    package_root.mkdir(parents=True)
    (package_root / "BASE_COMMIT.txt").write_text(base_commit + "\n", encoding="utf-8")
    (package_root / "PATCH_COMMIT.txt").write_text(patch_commit + "\n", encoding="utf-8")
    (package_root / "CHANGE_SUMMARY.md").write_text(
        "# Stage 5 real updater delta-mode integration probe\n",
        encoding="utf-8",
    )
    (package_root / "TEST_COMMANDS.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\npython3 scripts/handoff_authority.py verify --repo-root . --json >/dev/null\n",
        encoding="utf-8",
    )
    patch = subprocess.run(
        ["git", "-C", str(repo), "diff", "--binary", f"{base_commit}..{patch_commit}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert patch.returncode == 0, patch.stderr.decode("utf-8", errors="replace")
    (package_root / "update.patch").write_bytes(patch.stdout)
    bundle = run(
        repo,
        "bundle",
        "create",
        str(package_root / "change.bundle"),
        source_branch,
        check=False,
    )
    assert bundle.returncode == 0, bundle.stderr

    zip_path = destination / "stage5-real-updater.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_root.iterdir()):
            archive.write(path, arcname=path.name)
    return zip_path


def invoke_real_updater(
    repo: Path,
    package: Path,
    *,
    report_dir: Path,
    diagnostic_dir: Path,
) -> dict[str, object]:
    env = os.environ.copy()
    env.update(
        {
            "DRPO_UPDATE_ALLOW_ANY_REMOTE": "1",
            "DRPO_UPDATE_SKIP_FETCH": "1",
            "DRPO_UPDATE_REPORT_DIR": str(report_dir),
            "DRPO_UPDATE_DIAGNOSTIC_DIR": str(diagnostic_dir),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "DRPO_STAGE5_REAL_UPDATER_CHILD": "1",
        }
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "tools/drpo-update/drpo_update.py"),
            str(package),
            "--repo",
            str(repo),
            "--yes",
            "--no-push",
            "--test-mode",
            "fast",
        ],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    reports = sorted(report_dir.glob("*.json"))
    assert len(reports) == 1
    return json.loads(reports[0].read_text(encoding="utf-8"))


def run_real_drpo_update_delta_mode_stale_bundle(tmp_path: Path) -> None:
    repo = copy_repository(tmp_path / "updater-repo")
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    run(repo, "remote", "add", "origin", str(origin))
    _, cutover = activate_delta_mode(repo)

    run(repo, "checkout", "-q", "-B", "source-code-only-package", cutover)
    readme = repo / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8") + "\nStage 5 real delta-mode code-only updater probe.\n",
        encoding="utf-8",
    )
    run(repo, "add", "README.md")
    run(repo, "commit", "-q", "-m", "source code-only delta-mode package")
    code_source_commit = git_text(repo, "rev-parse", "HEAD")

    target_path = heading_path(repo)
    delta_source_commit = make_source_delta(
        repo,
        branch="source-v3-package",
        base_commit=cutover,
        update_id="STAGE5-REAL-UPDATER-V3",
        block_id="stage5-real-updater-v3",
        content="Real drpo-update delta-mode integration probe.",
        target_path=target_path,
    )

    run(repo, "checkout", "-q", "main")
    run(repo, "reset", "--hard", cutover)
    run(repo, "push", "-q", "-u", "origin", "main")

    code_package = make_bundle_package(
        repo,
        base_commit=cutover,
        patch_commit=code_source_commit,
        source_branch="source-code-only-package",
        destination=tmp_path / "code-updater-package",
    )
    code_report = invoke_real_updater(
        repo,
        code_package,
        report_dir=tmp_path / "code-reports",
        diagnostic_dir=tmp_path / "code-diagnostics",
    )
    code_normalization = code_report["handoff_normalization"]
    assert code_normalization["normalization"] == "no_op"
    assert "code-only updater probe" in (repo / "README.md").read_text(encoding="utf-8")
    stale_current = git_text(repo, "rev-parse", "HEAD")

    delta_package = make_bundle_package(
        repo,
        base_commit=cutover,
        patch_commit=delta_source_commit,
        source_branch="source-v3-package",
        destination=tmp_path / "delta-updater-package",
    )
    delta_report = invoke_real_updater(
        repo,
        delta_package,
        report_dir=tmp_path / "delta-reports",
        diagnostic_dir=tmp_path / "delta-diagnostics",
    )
    assert git_text(repo, "rev-parse", "HEAD") != stale_current
    handoff = (repo / "docs/handoff.md").read_text(encoding="utf-8")
    assert "stage5-real-updater-v3" in handoff
    materialization = (
        repo / "docs/handoff_deltas/STAGE5-REAL-UPDATER-V3/MATERIALIZATION_REPORT.json"
    )
    assert materialization.is_file()
    delta_normalization = delta_report["handoff_normalization"]
    assert delta_normalization["normalization"] == "materialized"
    assert delta_normalization["post_amend_verify"]["status"] == "PASS"
    assert_governance_valid(repo)


def run_checkpoint_path_alias_is_canonicalized(tmp_path: Path) -> None:
    real_repo = copy_repository(tmp_path / "real-repo")
    linked_repo = tmp_path / "repo-alias"
    try:
        linked_repo.symlink_to(real_repo, target_is_directory=True)
    except (OSError, NotImplementedError):
        print("Stage 5 checkpoint path-alias regression: SKIP (symlink unavailable)")
        return

    _, _ = activate_delta_mode(linked_repo)
    report = authority.verify_current_state(linked_repo)
    assert report["status"] == "PASS"
    assert report["mode"] == "delta"
    assert_governance_valid(linked_repo)


def main() -> int:
    import tempfile

    if os.environ.get("DRPO_STAGE5_REAL_UPDATER_CHILD") == "1":
        report = authority.verify_current_state(REPO_ROOT)
        assert report["status"] == "PASS"
        assert report["mode"] == "delta"
        print("Stage 5 real drpo-update child verification: PASS")
        return 0

    with tempfile.TemporaryDirectory(prefix="drpo-stage5-integration-") as directory:
        root = Path(directory)
        for name, callback in (
            ("stale", run_stale_independent_updates_commute_and_same_block_conflicts),
            (
                "preintegration-report",
                run_preintegration_report_revision_and_postintegration_tamper,
            ),
            ("noop", run_code_only_noop_and_rollback),
            ("cutover-auth", run_cutover_requires_independent_authorization),
            (
                "prepared-boundary",
                run_prepared_cutover_rejects_unexpected_worktree_file,
            ),
            ("checkpoint-gates", run_checkpoint_gate_rejections),
            ("path-alias", run_checkpoint_path_alias_is_canonicalized),
            ("cutover", run_cutover_commit_rejects_first_delta),
            ("updater", run_real_drpo_update_delta_mode_stale_bundle),
        ):
            work = root / name
            work.mkdir()
            callback(work)
    print("Stage 5 stale-base, lifecycle, and real drpo-update integration: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
