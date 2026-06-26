from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "scripts" / "create_update_git_bundle.py"
VERIFIER = REPO_ROOT / "scripts" / "verify_update_git_bundle.py"
HELPER = REPO_ROOT / "tools" / "drpo-update" / "drpo-update"


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = True):
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode != 0:
        raise AssertionError(f"command failed: {' '.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}")
    return proc


def git(repo: Path, *args: str, check: bool = True):
    return run(["git", "-C", str(repo), *args], check=check)


def git_text(repo: Path, *args: str) -> str:
    return git(repo, *args).stdout.strip()


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", message)
    return git_text(repo, "rev-parse", "HEAD")


@pytest.fixture
def git_fixture(tmp_path: Path):
    origin = tmp_path / "origin.git"
    run(["git", "init", "--bare", str(origin)])
    repo = tmp_path / "repo"
    run(["git", "clone", str(origin), str(repo)])
    git(repo, "checkout", "-b", "main")
    (repo / "tracked.txt").write_text("base\n")
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
    (package / "TEST_COMMANDS.sh").write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{test_command}\n")
    os.chmod(package / "TEST_COMMANDS.sh", 0o755)
    (package / "BASE_COMMIT.txt").write_text(git_text(repo, "rev-parse", "HEAD") + "\n")


def build_bundle(repo: Path, package: Path) -> str:
    run(["python3", str(BUILDER), "--repo", str(repo), "--package-root", str(package)])
    return (package / "PATCH_COMMIT.txt").read_text().strip()


def helper_env(repo: Path, report_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "DRPO_UPDATE_REPO": str(repo),
            "DRPO_UPDATE_ALLOW_ANY_REMOTE": "1",
            "DRPO_UPDATE_SKIP_FETCH": "1",
            "DRPO_UPDATE_REPORT_DIR": str(report_dir),
        }
    )
    return env


def test_bundle_verifier_proves_patch_tree_equivalence(git_fixture, tmp_path: Path):
    _, repo, _ = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "bundle\n")
    patch_commit = build_bundle(repo, package)
    proc = run(["python3", str(VERIFIER), "--repo", str(repo), "--package", str(package), "--json"])
    report = json.loads(proc.stdout)
    assert report["status"] == "PASS"
    assert report["patch_commit"] == patch_commit


def test_stale_ancestral_bundle_merges_nonconflicting_main(git_fixture, tmp_path: Path):
    _, repo, base = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "bundle\n", test_command='test "$(cat tracked.txt)" = bundle')
    build_bundle(repo, package)

    (repo / "other.txt").write_text("new main work\n")
    current = commit_all(repo, "advance main")
    assert current != base
    report_dir = tmp_path / "reports"
    proc = run(
        [str(HELPER), str(package), "--yes", "--no-push"],
        env=helper_env(repo, report_dir),
    )
    assert "git-bundle-three-way" in proc.stdout
    assert (repo / "tracked.txt").read_text() == "bundle\n"
    assert (repo / "other.txt").read_text() == "new main work\n"
    assert git_text(repo, "status", "--porcelain") == ""
    reports = list(report_dir.glob("*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["package_base"] == base
    assert report["head_before"] == current
    assert report["integration_mode"] == "git-bundle-three-way"
    assert report["status"] == "success_no_push"


def test_bundle_conflict_fails_without_modifying_main(git_fixture, tmp_path: Path):
    _, repo, _ = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "package side\n")
    build_bundle(repo, package)

    (repo / "tracked.txt").write_text("main side\n")
    before = commit_all(repo, "conflicting main")
    proc = run(
        [str(HELPER), str(package), "--yes", "--no-push"],
        env=helper_env(repo, tmp_path / "reports"),
        check=False,
    )
    assert proc.returncode != 0
    assert git_text(repo, "rev-parse", "HEAD") == before
    assert (repo / "tracked.txt").read_text() == "main side\n"
    assert git_text(repo, "status", "--porcelain") == ""


def test_patch_bundle_mismatch_is_rejected(git_fixture, tmp_path: Path):
    _, repo, _ = git_fixture
    package = tmp_path / "package"
    make_patch(repo, package, "bundle\n")
    build_bundle(repo, package)
    text = (package / "update.patch").read_text().replace("+bundle", "+different")
    (package / "update.patch").write_text(text)
    proc = run(
        ["python3", str(VERIFIER), "--repo", str(repo), "--package", str(package)],
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
    proc = run(
        [str(HELPER), str(package), "--yes", "--no-push"],
        env=helper_env(repo, tmp_path / "reports"),
        check=False,
    )
    assert proc.returncode != 0
    assert "no Git bundle is present" in proc.stderr
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
