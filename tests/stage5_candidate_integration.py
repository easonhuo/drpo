from __future__ import annotations

import json
import shutil
import subprocess
import sys
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
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed\n{proc.stdout}\n{proc.stderr}")
    return proc


def git_text(repo: Path, *args: str) -> str:
    return run(repo, *args).stdout.strip()


def copy_repository(destination: Path) -> Path:
    shutil.copytree(
        REPO_ROOT,
        destination,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".pytest_cache", "*.pyc", ".ruff_cache"
        ),
    )
    run(destination, "init", "-q")
    run(destination, "config", "user.name", "Stage5 Test")
    run(destination, "config", "user.email", "stage5@test.invalid")
    run(destination, "add", "-A")
    run(destination, "commit", "-q", "-m", "base")
    return destination


def activate_delta_mode(repo: Path) -> tuple[str, str]:
    base = git_text(repo, "rev-parse", "HEAD")
    checkpoint_id = "STAGE5-TEST-CUTOVER"
    checkpoint_dir = repo / "docs/handoff_versions/checkpoints" / checkpoint_id
    checkpoint_dir.mkdir(parents=True)
    checkpoint_handoff = checkpoint_dir / "handoff.md"
    checkpoint_handoff.write_bytes((repo / "docs/handoff.md").read_bytes())
    manifest = {
        "schema_version": 1,
        "checkpoint_id": checkpoint_id,
        "source_parent_commit": base,
        "handoff_path": checkpoint_handoff.relative_to(repo).as_posix(),
        "handoff_sha256": shadow.sha256_file(checkpoint_handoff),
        "registry_sha256_for_provenance": shadow.sha256_file(
            repo / "experiments/registry.yaml"
        ),
        "stage3_full_acceptance_report": (
            "docs/handoff_deltas/GOV-STAGE3-PRE-STAGE5-FULL-CHECKPOINT-2026-07-01/"
            "FULL_ACCEPTANCE_REPORT.json"
        ),
        "stage4b_cutover_audit_report": (
            "docs/governance_stage4b_acceptance/ACCEPTANCE_REPORT.json"
        ),
        "created_at_utc": "2026-07-01T00:00:00+00:00",
    }
    manifest_path = checkpoint_dir / "CHECKPOINT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    config_path = repo / authority.AUTHORITY_PATH
    config = yaml.safe_load(config_path.read_text())
    config["mode"] = "delta"
    config["delta_authority"]["checkpoint_manifest"] = manifest_path.relative_to(repo).as_posix()
    config["delta_authority"]["activation_parent_commit"] = base
    config["generated_views"]["stage4a_minimal_refresh"] = True
    config["safety"]["direct_handoff_edit_forbidden"] = True
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    run(repo, "add", "-A")
    run(repo, "commit", "-q", "-m", "activate delta authority")
    cutover = git_text(repo, "rev-parse", "HEAD")
    report = authority.verify_current_state(repo)
    assert report["mode"] == "delta"
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


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="drpo-stage5-integration-") as directory:
        run_stale_independent_updates_commute_and_same_block_conflicts(Path(directory))
    print("Stage 5 stale-base integration: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
