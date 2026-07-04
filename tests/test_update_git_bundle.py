from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "scripts" / "create_update_git_bundle.py"
VERIFIER = REPO_ROOT / "scripts" / "verify_update_git_bundle.py"
HELPER = REPO_ROOT / "tools" / "drpo-update" / "drpo-update"


def run(
    cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = True
):
    proc = subprocess.run(
        cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return proc


def git(repo: Path, *args: str, check: bool = True):
    return run(["git", "-C", str(repo), *args], check=check)


def git_text(repo: Path, *args: str) -> str:
    return git(repo, *args).stdout.strip()


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        message,
    )
    return git_text(repo, "rev-parse", "HEAD")


@pytest.fixture
def git_fixture(tmp_path: Path):
    origin = tmp_path / "origin.git"
    run(["git", "init", "--bare", str(origin)])
    repo = tmp_path / "repo"
    run(["git", "clone", str(origin), str(repo)])
    git(repo, "checkout", "-b", "main")
    (repo / "tracked.txt").write_text("base\n")
    impact_map = repo / "tools" / "drpo-update" / "test_impact_map.json"
    impact_map.parent.mkdir(parents=True)
    impact_map.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "unknown_path_policy": "full",
                "full_commands": [["{python}", "-c", 'print("synthetic full gate")']],
                "control_plane_patterns": ["tools/drpo-update/test_impact_map.json"],
                "groups": [
                    {
                        "id": "fixture_tracked_file",
                        "risk": "low",
                        "patterns": ["tracked.txt"],
                        "pytest_targets": ["tests/test_smoke.py"],
                        "validators": [],
                    }
                ],
            }
        )
        + "\n"
    )
    smoke = repo / "tests" / "test_smoke.py"
    smoke.parent.mkdir(parents=True)
    smoke.write_text("def test_smoke():\n    assert True\n")
    base = commit_all(repo, "base")
    git(repo, "push", "-u", "origin", "main")
    git(origin, "symbolic-ref", "HEAD", "refs/heads/main")
    return origin, repo, base


def make_patch(repo: Path, package: Path, new_text: str, *, test_command: str = "true") -> None:
    package.mkdir(parents=True)
    (repo / "tracked.txt").write_text(new_text)
    patch = git(repo, "diff", "--binary").stdout
    (repo / "tracked.txt").write_text("base\n")
    (package / "update.patch").write_text(patch)
    (package / "CHANGE_SUMMARY.md").write_text("# Bundle test update\n")
    (package / "TEST_COMMANDS.sh").write_text(
        f"#!/usr/bin/env bash\nset -euo pipefail\n{test_command}\n"
    )
    os.chmod(package / "TEST_COMMANDS.sh", 0o755)
    (package / "BASE_COMMIT.txt").write_text(git_text(repo, "rev-parse", "HEAD") + "\n")


def build_bundle(repo: Path, package: Path) -> str:
    run([sys.executable, str(BUILDER), "--repo", str(repo), "--package-root", str(package)])
    return (package / "PATCH_COMMIT.txt").read_text().strip()


def helper_env(repo: Path, report_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "DRPO_UPDATE_REPO": str(repo),
            "DRPO_UPDATE_ALLOW_ANY_REMOTE": "1",
            "DRPO_UPDATE_SKIP_FETCH": "1",
            "DRPO_UPDATE_REPORT_DIR": str(report_dir),
            "DRPO_UPDATE_DIAGNOSTIC_DIR": str(report_dir.parent / "diagnostics"),
            "DRPO_PYTHON": sys.executable,
        }
    )
    return env


def test_bundle_verifier_proves_patch_tree_equivalence(git_fixture, tmp_path: Path):
    _, repo, _ = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "bundle\n")
    patch_commit = build_bundle(repo, package)
    proc = run(
        [sys.executable, str(VERIFIER), "--repo", str(repo), "--package", str(package), "--json"]
    )
    report = json.loads(proc.stdout)
    assert report["status"] == "PASS"
    assert report["patch_commit"] == patch_commit


def test_stale_ancestral_bundle_fails_preflight_without_applying(git_fixture, tmp_path: Path):
    origin, repo, base = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "bundle\n", test_command='test "$(cat tracked.txt)" = bundle')
    build_bundle(repo, package)

    (repo / "other.txt").write_text("new main work\n")
    current = commit_all(repo, "advance main")
    git(repo, "push", "origin", "main")
    assert current != base
    report_dir = tmp_path / "reports"
    proc = run(
        [str(HELPER), str(package), "--yes", "--no-push"],
        env=helper_env(repo, report_dir),
        check=False,
    )
    assert proc.returncode != 0
    assert "DRPO_UPDATE_PREFLIGHT_FAILED" in proc.stderr
    assert "Code: DRPO_UPDATE_PACKAGE_BASE_OUTDATED" in proc.stderr
    assert f"Package BASE_COMMIT: {base}" in proc.stderr
    assert f"Current main HEAD: {current}" in proc.stderr
    assert "regenerate this .drpoupdate package from current main" in proc.stderr
    assert (repo / "tracked.txt").read_text() == "base\n"
    assert (repo / "other.txt").read_text() == "new main work\n"
    assert git_text(repo, "status", "--porcelain") == ""
    assert git_text(origin, "rev-parse", "refs/heads/main") == current
    reports = list(report_dir.glob("*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["preflight_code"] == "DRPO_UPDATE_PACKAGE_BASE_OUTDATED"
    assert report["package_base_sha"] == base
    assert report["head_sha"] == current
    assert report["origin_main_sha"] == current
    assert report["test_commands"] == []
    diagnostic = next((tmp_path / "diagnostics").glob("DRPO_DIAGNOSTIC_*.zip"))
    with zipfile.ZipFile(diagnostic) as archive:
        diagnostic_report = json.loads(archive.read("apply_report.json"))
    assert diagnostic_report["preflight_code"] == "DRPO_UPDATE_PACKAGE_BASE_OUTDATED"
    assert diagnostic_report["package_base_sha"] == base


def test_non_main_origin_main_alias_fails_preflight_without_recovery(git_fixture, tmp_path: Path):
    _, repo, base = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "bundle\n", test_command='test "$(cat tracked.txt)" = bundle')
    build_bundle(repo, package)

    git(repo, "checkout", "-b", "codex/post-push-bundle-export")
    (repo / "other.txt").write_text("already pushed main work\n")
    current = commit_all(repo, "advance temporary branch")
    git(repo, "push", "origin", "HEAD:main")
    assert git_text(repo, "rev-parse", "refs/heads/main") == base
    assert git_text(repo, "rev-parse", "refs/remotes/origin/main") == current
    assert git_text(repo, "branch", "--show-current") == "codex/post-push-bundle-export"

    report_dir = tmp_path / "reports"
    proc = run(
        [str(HELPER), str(package), "--yes", "--no-push"],
        env=helper_env(repo, report_dir),
        check=False,
    )

    assert proc.returncode != 0
    assert "DRPO_UPDATE_PREFLIGHT_FAILED" in proc.stderr
    assert "Code: DRPO_UPDATE_NOT_ON_MAIN" in proc.stderr
    assert "Current branch: codex/post-push-bundle-export" in proc.stderr
    assert "Required branch: main" in proc.stderr
    assert git_text(repo, "branch", "--show-current") == "codex/post-push-bundle-export"
    assert git_text(repo, "rev-parse", "refs/heads/main") == base
    assert (repo / "tracked.txt").read_text() == "base\n"
    assert (repo / "other.txt").read_text() == "already pushed main work\n"
    assert git_text(repo, "status", "--porcelain") == ""
    report = json.loads(next(report_dir.glob("*.json")).read_text())
    assert report["preflight_code"] == "DRPO_UPDATE_NOT_ON_MAIN"
    assert report["current_branch"] == "codex/post-push-bundle-export"
    assert report["test_commands"] == []


def test_unrelated_non_main_branch_still_fails_closed(git_fixture, tmp_path: Path):
    _, repo, _ = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "bundle\n")
    build_bundle(repo, package)

    git(repo, "checkout", "-b", "feature/local-only")
    (repo / "other.txt").write_text("local only\n")
    before = commit_all(repo, "local only branch")

    proc = run(
        [str(HELPER), str(package), "--yes", "--no-push"],
        env=helper_env(repo, tmp_path / "reports"),
        check=False,
    )

    assert proc.returncode != 0
    assert "Code: DRPO_UPDATE_NOT_ON_MAIN" in proc.stderr
    assert "Current branch: feature/local-only" in proc.stderr
    assert git_text(repo, "branch", "--show-current") == "feature/local-only"
    assert git_text(repo, "rev-parse", "HEAD") == before


def test_dirty_worktree_lists_files_and_does_not_apply(git_fixture, tmp_path: Path):
    _, repo, before = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "bundle\n")
    build_bundle(repo, package)
    (repo / "local-notes.txt").write_text("keep me\n")

    report_dir = tmp_path / "reports"
    proc = run(
        [str(HELPER), str(package), "--yes", "--no-push"],
        env=helper_env(repo, report_dir),
        check=False,
    )

    assert proc.returncode != 0
    assert "Code: DRPO_UPDATE_DIRTY_WORKTREE" in proc.stderr
    assert "Reason: repository has uncommitted changes" in proc.stderr
    assert "?? local-notes.txt" in proc.stderr
    assert "commit them, stash them, or restore them manually" in proc.stderr
    assert git_text(repo, "rev-parse", "HEAD") == before
    assert (repo / "tracked.txt").read_text() == "base\n"
    report = json.loads(next(report_dir.glob("*.json")).read_text())
    assert report["preflight_code"] == "DRPO_UPDATE_DIRTY_WORKTREE"
    assert report["dirty_files"] == ["?? local-notes.txt"]
    assert report["test_commands"] == []


def test_local_main_ahead_of_origin_fails_preflight(git_fixture, tmp_path: Path):
    _, repo, base = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "bundle\n")
    build_bundle(repo, package)
    (repo / "local.txt").write_text("local ahead\n")
    local_head = commit_all(repo, "local ahead")

    proc = run(
        [str(HELPER), str(package), "--yes", "--no-push"],
        env=helper_env(repo, tmp_path / "reports"),
        check=False,
    )

    assert proc.returncode != 0
    assert "Code: DRPO_UPDATE_MAIN_NOT_SYNCED" in proc.stderr
    assert f"HEAD: {local_head}" in proc.stderr
    assert f"origin/main: {base}" in proc.stderr
    assert git_text(repo, "rev-parse", "HEAD") == local_head
    assert (repo / "tracked.txt").read_text() == "base\n"


def test_local_main_behind_origin_fails_preflight(git_fixture, tmp_path: Path):
    origin, repo, base = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "bundle\n")
    build_bundle(repo, package)
    peer = tmp_path / "peer"
    run(["git", "clone", str(origin), str(peer)])
    (peer / "remote.txt").write_text("remote ahead\n")
    remote_head = commit_all(peer, "remote ahead")
    git(peer, "push", "origin", "main")
    git(repo, "fetch", "origin", "main")

    proc = run(
        [str(HELPER), str(package), "--yes", "--no-push"],
        env=helper_env(repo, tmp_path / "reports"),
        check=False,
    )

    assert proc.returncode != 0
    assert "Code: DRPO_UPDATE_MAIN_NOT_SYNCED" in proc.stderr
    assert f"HEAD: {base}" in proc.stderr
    assert f"origin/main: {remote_head}" in proc.stderr
    assert git_text(repo, "rev-parse", "HEAD") == base


def test_local_main_diverged_from_origin_fails_preflight(git_fixture, tmp_path: Path):
    origin, repo, base = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "bundle\n")
    build_bundle(repo, package)
    peer = tmp_path / "peer"
    run(["git", "clone", str(origin), str(peer)])
    (repo / "local.txt").write_text("local side\n")
    local_head = commit_all(repo, "local side")
    (peer / "remote.txt").write_text("remote side\n")
    remote_head = commit_all(peer, "remote side")
    git(peer, "push", "origin", "main")
    git(repo, "fetch", "origin", "main")
    assert local_head != remote_head != base

    proc = run(
        [str(HELPER), str(package), "--yes", "--no-push"],
        env=helper_env(repo, tmp_path / "reports"),
        check=False,
    )

    assert proc.returncode != 0
    assert "Code: DRPO_UPDATE_MAIN_NOT_SYNCED" in proc.stderr
    assert f"HEAD: {local_head}" in proc.stderr
    assert f"origin/main: {remote_head}" in proc.stderr
    assert git_text(repo, "rev-parse", "HEAD") == local_head


def test_invalid_package_uses_structured_preflight_error(git_fixture, tmp_path: Path):
    _, repo, before = git_fixture
    package = tmp_path / "invalid-package"
    package.mkdir()

    report_dir = tmp_path / "reports"
    proc = run(
        [str(HELPER), str(package), "--yes", "--no-push"],
        env=helper_env(repo, report_dir),
        check=False,
    )

    assert proc.returncode != 0
    assert "Code: DRPO_UPDATE_PACKAGE_INVALID" in proc.stderr
    assert "Reason: update package is invalid" in proc.stderr
    report = json.loads(next(report_dir.glob("*.json")).read_text())
    assert report["preflight_code"] == "DRPO_UPDATE_PACKAGE_INVALID"
    assert report["test_commands"] == []
    assert git_text(repo, "rev-parse", "HEAD") == before


def test_conflicting_stale_bundle_fails_preflight_without_apply(git_fixture, tmp_path: Path):
    _, repo, _ = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "package side\n")
    build_bundle(repo, package)

    (repo / "tracked.txt").write_text("main side\n")
    before = commit_all(repo, "conflicting main")
    git(repo, "push", "origin", "main")
    proc = run(
        [str(HELPER), str(package), "--yes", "--no-push"],
        env=helper_env(repo, tmp_path / "reports"),
        check=False,
    )
    assert proc.returncode != 0
    assert "Code: DRPO_UPDATE_PACKAGE_BASE_OUTDATED" in proc.stderr
    assert git_text(repo, "rev-parse", "HEAD") == before
    assert (repo / "tracked.txt").read_text() == "main side\n"
    assert git_text(repo, "status", "--porcelain") == ""
    diagnostics = list((tmp_path / "diagnostics").glob("DRPO_DIAGNOSTIC_*.zip"))
    assert len(diagnostics) == 1
    with zipfile.ZipFile(diagnostics[0]) as archive:
        names = set(archive.namelist())
        assert "apply_report.json" in names
        assert "git/repository.bundle" in names
        assert not any(name.startswith("candidate/conflicts/") for name in names)
        report = json.loads(archive.read("apply_report.json"))
        assert report["failure_phase"] == "repository_preflight"
        assert report["preflight_code"] == "DRPO_UPDATE_PACKAGE_BASE_OUTDATED"
        assert report["conflicts"] == []


def test_patch_bundle_mismatch_is_rejected(git_fixture, tmp_path: Path):
    _, repo, _ = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "bundle\n")
    build_bundle(repo, package)
    text = (package / "update.patch").read_text().replace("+bundle", "+different")
    (package / "update.patch").write_text(text)
    proc = run(
        [sys.executable, str(VERIFIER), "--repo", str(repo), "--package", str(package)],
        check=False,
    )
    assert proc.returncode != 0
    assert "different repository trees" in proc.stderr


def test_stale_legacy_package_remains_rejected(git_fixture, tmp_path: Path):
    _, repo, _ = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "legacy\n")
    (repo / "other.txt").write_text("advance\n")
    before = commit_all(repo, "advance")
    git(repo, "push", "origin", "main")
    proc = run(
        [str(HELPER), str(package), "--yes", "--no-push"],
        env=helper_env(repo, tmp_path / "reports"),
        check=False,
    )
    assert proc.returncode != 0
    assert "Code: DRPO_UPDATE_PACKAGE_BASE_OUTDATED" in proc.stderr
    assert "regenerate this .drpoupdate package from current main" in proc.stderr
    assert git_text(repo, "rev-parse", "HEAD") == before


def test_failed_package_tests_leave_main_untouched(git_fixture, tmp_path: Path):
    _, repo, _ = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "bundle\n", test_command="exit 9")
    build_bundle(repo, package)
    before = git_text(repo, "rev-parse", "HEAD")
    proc = run(
        [str(HELPER), str(package), "--yes", "--no-push"],
        env=helper_env(repo, tmp_path / "reports"),
        check=False,
    )
    assert proc.returncode != 0
    assert git_text(repo, "rev-parse", "HEAD") == before
    assert (repo / "tracked.txt").read_text() == "base\n"
    assert git_text(repo, "status", "--porcelain") == ""
    diagnostics = list((tmp_path / "diagnostics").glob("DRPO_DIAGNOSTIC_*.zip"))
    assert len(diagnostics) == 1
    with zipfile.ZipFile(diagnostics[0]) as archive:
        names = set(archive.namelist())
        assert "logs/package-tests.log" in names
        assert any(name.startswith("logs/repository-gates/") for name in names)
        assert "inputs/original_package/update.patch" in names
        report = json.loads(archive.read("apply_report.json"))
        assert report["status"] == "failed"
        assert report["diagnostic_zip"].endswith(diagnostics[0].name)
        assert any(item["label"] == "TEST_COMMANDS.sh" for item in report["test_commands"])


def test_default_failure_diagnostic_is_written_to_downloads(git_fixture, tmp_path: Path):
    _, repo, _ = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "bundle\n", test_command="exit 11")
    build_bundle(repo, package)
    home = tmp_path / "home"
    home.mkdir()
    env = helper_env(repo, tmp_path / "reports")
    env.pop("DRPO_UPDATE_DIAGNOSTIC_DIR")
    env["HOME"] = str(home)
    proc = run(
        [str(HELPER), str(package), "--yes", "--no-push"],
        env=env,
        check=False,
    )
    assert proc.returncode != 0
    diagnostics = list((home / "Downloads").glob("DRPO_DIAGNOSTIC_*.zip"))
    assert len(diagnostics) == 1
    assert f"Diagnostic ZIP: {diagnostics[0]}" in proc.stderr


def test_exact_base_legacy_package_succeeds_transactionally(git_fixture, tmp_path: Path):
    _, repo, _ = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "legacy\n", test_command='test "$(cat tracked.txt)" = legacy')
    report_dir = tmp_path / "reports"
    proc = run(
        [str(HELPER), str(package), "--yes", "--no-push"],
        env=helper_env(repo, report_dir),
    )
    assert "legacy-patch-exact-base" in proc.stdout
    assert (repo / "tracked.txt").read_text() == "legacy\n"
    assert git_text(repo, "status", "--porcelain") == ""


def test_incomplete_bundle_pair_is_rejected(git_fixture, tmp_path: Path):
    _, repo, _ = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "legacy\n")
    (package / "PATCH_COMMIT.txt").write_text("0" * 40 + "\n")
    proc = run(
        [str(HELPER), str(package), "--yes", "--no-push"],
        env=helper_env(repo, tmp_path / "reports"),
        check=False,
    )
    assert proc.returncode != 0
    assert "must appear together" in proc.stderr
    assert (repo / "tracked.txt").read_text() == "base\n"


def test_fast_override_cannot_downgrade_unknown_path(git_fixture, tmp_path: Path):
    _, repo, _ = git_fixture
    package = tmp_path / "package"
    package.mkdir(parents=True)
    (repo / "unknown.py").write_text("VALUE = 2\n")
    git(repo, "add", "-N", "unknown.py")
    patch = git(repo, "diff", "--binary", "--", "unknown.py").stdout
    git(repo, "reset", "--", "unknown.py")
    (repo / "unknown.py").unlink()
    (package / "update.patch").write_text(patch)
    (package / "CHANGE_SUMMARY.md").write_text("# Unsafe fast override test\n")
    (package / "TEST_COMMANDS.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\ntrue\n")
    os.chmod(package / "TEST_COMMANDS.sh", 0o755)
    (package / "BASE_COMMIT.txt").write_text(git_text(repo, "rev-parse", "HEAD") + "\n")
    build_bundle(repo, package)
    before = git_text(repo, "rev-parse", "HEAD")
    proc = run(
        [
            str(HELPER),
            str(package),
            "--yes",
            "--no-push",
            "--test-mode",
            "fast",
        ],
        env=helper_env(repo, tmp_path / "reports"),
        check=False,
    )
    assert proc.returncode != 0
    assert "fast mode cannot override" in proc.stderr
    assert git_text(repo, "rev-parse", "HEAD") == before


def test_unknown_changed_path_escalates_to_full_suite(git_fixture, tmp_path: Path):
    _, repo, _ = git_fixture
    package = tmp_path / "package"
    package.mkdir(parents=True)
    (repo / "unknown.py").write_text("VALUE = 1\n")
    git(repo, "add", "-N", "unknown.py")
    patch = git(repo, "diff", "--binary", "--", "unknown.py").stdout
    git(repo, "reset", "--", "unknown.py")
    (repo / "unknown.py").unlink()
    (package / "update.patch").write_text(patch)
    (package / "CHANGE_SUMMARY.md").write_text("# Unknown-path full-suite test\n")
    (package / "TEST_COMMANDS.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\ntrue\n")
    os.chmod(package / "TEST_COMMANDS.sh", 0o755)
    (package / "BASE_COMMIT.txt").write_text(git_text(repo, "rev-parse", "HEAD") + "\n")
    build_bundle(repo, package)

    report_dir = tmp_path / "reports"
    proc = run(
        [str(HELPER), str(package), "--yes", "--no-push"],
        env=helper_env(repo, report_dir),
    )
    assert "mode=full" in proc.stdout
    report = json.loads(next(report_dir.glob("*.json")).read_text())
    assert report["selected_test_mode"] == "full"
    assert report["test_selection"]["unknown_paths"] == ["unknown.py"]
    assert '{python} -c print("synthetic full gate") [full]' in report["tests"]


def test_successful_push_defaults_versioned_and_latest_main_bundles_to_downloads(
    git_fixture, tmp_path: Path
):
    origin, repo, _ = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "bundle\n")
    build_bundle(repo, package)
    home = tmp_path / "home"
    home.mkdir()
    downloads = home / "Downloads"
    report_dir = tmp_path / "reports"
    env = helper_env(repo, report_dir)
    env["HOME"] = str(home)
    proc = run([str(HELPER), str(package), "--yes"], env=env)
    head = git_text(repo, "rev-parse", "HEAD")
    assert git_text(origin, "rev-parse", "refs/heads/main") == head
    versioned = downloads / f"DRPO_MAIN_{head[:12]}.bundle"
    latest = downloads / "DRPO_MAIN_LATEST.bundle"
    versioned_sha = downloads / f"{versioned.name}.sha256"
    latest_sha = downloads / f"{latest.name}.sha256"
    assert versioned.is_file()
    assert latest.is_file()
    assert versioned_sha.is_file()
    assert latest_sha.is_file()
    assert versioned.read_bytes() == latest.read_bytes()
    digest = versioned_sha.read_text().split()[0]
    assert digest == hashlib.sha256(versioned.read_bytes()).hexdigest()
    assert latest_sha.read_text() == f"{digest}  {latest.name}\n"
    assert versioned_sha.read_text() == f"{digest}  {versioned.name}\n"
    assert not list(downloads.glob("*.tmp.*"))
    assert f"Main bundle: {versioned}" in proc.stdout
    assert f"Main bundle SHA-256: {versioned_sha}" in proc.stdout
    assert f"Latest bundle: {latest}" in proc.stdout
    assert f"Latest bundle SHA-256: {latest_sha}" in proc.stdout
    listed = git(repo, "bundle", "list-heads", str(versioned)).stdout.splitlines()
    assert f"{head} refs/heads/main" in listed
    report = json.loads(next(report_dir.glob("*.json")).read_text())
    assert report["pushed"] is True
    assert report["remote_head_after_push"] == head
    assert report["main_bundle_exported"] is True
    assert report["main_bundle_path"] == str(versioned)
    assert report["main_bundle_latest_path"] == str(latest)
    assert report["main_bundle_checksum_path"] == str(versioned_sha)
    assert report["main_bundle_latest_checksum_path"] == str(latest_sha)
    assert report["main_bundle_sha256"] == digest


def test_no_push_never_exports_official_main_bundle(git_fixture, tmp_path: Path):
    _, repo, _ = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "bundle\n")
    build_bundle(repo, package)
    downloads = tmp_path / "Downloads"
    report_dir = tmp_path / "reports"
    env = helper_env(repo, report_dir)
    env["DRPO_UPDATE_MAIN_BUNDLE_DIR"] = str(downloads)
    run([str(HELPER), str(package), "--yes", "--no-push"], env=env)
    assert not list(downloads.glob("DRPO_MAIN_*.bundle"))
    report = json.loads(next(report_dir.glob("*.json")).read_text())
    assert report["main_bundle_exported"] is False
    assert report["main_bundle_export_skipped"] == "no_push"


def test_explicit_export_disable_is_recorded_after_push(git_fixture, tmp_path: Path):
    _, repo, _ = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "bundle\n")
    build_bundle(repo, package)
    downloads = tmp_path / "Downloads"
    report_dir = tmp_path / "reports"
    env = helper_env(repo, report_dir)
    env["DRPO_UPDATE_MAIN_BUNDLE_DIR"] = str(downloads)
    run(
        [
            str(HELPER),
            str(package),
            "--yes",
            "--no-export-main-bundle",
        ],
        env=env,
    )
    assert not list(downloads.glob("DRPO_MAIN_*.bundle"))
    report = json.loads(next(report_dir.glob("*.json")).read_text())
    assert report["pushed"] is True
    assert report["main_bundle_export_skipped"] == "disabled_by_flag"


def test_post_push_export_failure_generates_diagnostic_without_rolling_back_push(
    git_fixture, tmp_path: Path
):
    origin, repo, before = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "bundle\n")
    build_bundle(repo, package)
    downloads = tmp_path / "downloads"
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_shasum = fake_bin / "shasum"
    fake_shasum.write_text("#!/usr/bin/env bash\nexit 19\n")
    fake_shasum.chmod(0o755)
    report_dir = tmp_path / "reports"
    env = helper_env(repo, report_dir)
    env["DRPO_UPDATE_MAIN_BUNDLE_DIR"] = str(downloads)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    proc = run([str(HELPER), str(package), "--yes"], env=env, check=False)
    assert proc.returncode != 0
    local_head = git_text(repo, "rev-parse", "HEAD")
    remote_head = git_text(origin, "rev-parse", "refs/heads/main")
    assert local_head != before
    assert remote_head == local_head
    report = json.loads(next(report_dir.glob("*.json")).read_text())
    assert report["status"] == "pushed_main_bundle_export_failed"
    assert report["pushed"] is True
    assert report["remote_head_after_push"] == local_head
    assert report["main_bundle_exported"] is False
    assert "UPDATE_PUSHED_BUNDLE_FAILED" in proc.stderr
    assert not list(downloads.glob("DRPO_MAIN_*.bundle"))
    assert not list(downloads.glob("*.tmp.*"))
    diagnostics = list((tmp_path / "diagnostics").glob("DRPO_DIAGNOSTIC_*.zip"))
    assert len(diagnostics) == 1
    assert f"Diagnostic ZIP: {diagnostics[0]}" in proc.stderr
