from __future__ import annotations

import json
import os
import stat
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGER = ROOT / "scripts" / "package_update.py"
VERIFIER = ROOT / "scripts" / "verify_update_package.py"


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return proc


def git(repo: Path, *args: str) -> str:
    return run(["git", "-C", str(repo), *args]).stdout.strip()


@pytest.fixture
def package_fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    run(["git", "init", "-q", str(repo)])
    git(repo, "config", "user.name", "Packager Test")
    git(repo, "config", "user.email", "packager@example.invalid")
    tracked = repo / "tracked.txt"
    tracked.write_text("before\n")
    executable = repo / "run.sh"
    executable.write_text("#!/usr/bin/env bash\necho before\n")
    executable.chmod(0o755)
    git(repo, "add", "tracked.txt", "run.sh")
    git(repo, "commit", "-m", "base")
    base = git(repo, "rev-parse", "HEAD")

    tracked.write_text("after\n")
    executable.write_text("#!/usr/bin/env bash\necho after\n")
    patch = git(repo, "diff", "--binary") + "\n"
    tracked.write_text("before\n")
    executable.write_text("#!/usr/bin/env bash\necho before\n")

    stage = tmp_path / "stage"
    (stage / "modified_files").mkdir(parents=True)
    (stage / "BASE_COMMIT.txt").write_text(base + "\n")
    (stage / "update.patch").write_text(patch)
    (stage / "CHANGE_SUMMARY.md").write_text(
        "# Canonical packager test\n\nClaim: GOV-UPDATE-BUNDLE-DEFAULT-01\n"
    )
    tests = stage / "TEST_COMMANDS.sh"
    tests.write_text("#!/usr/bin/env bash\nset -euo pipefail\ntest -f tracked.txt\n")
    tests.chmod(0o755)
    (stage / "modified_files" / "tracked.txt").write_text("after\n")
    supplied_script = stage / "modified_files" / "run.sh"
    supplied_script.write_text("#!/usr/bin/env bash\necho after\n")
    supplied_script.chmod(0o755)
    return repo, stage, base


def test_canonical_packager_always_emits_bundle_pair_and_manifest(
    package_fixture, tmp_path: Path
):
    repo, stage, base = package_fixture
    output = tmp_path / "update.zip"
    proc = run(
        [
            "python3",
            str(PACKAGER),
            "--repo",
            str(repo),
            "--package-root",
            str(stage),
            "--output",
            str(output),
        ]
    )
    assert "Created canonical bundle-backed package" in proc.stdout
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert {
            "BASE_COMMIT.txt",
            "update.patch",
            "CHANGE_SUMMARY.md",
            "TEST_COMMANDS.sh",
            "change.bundle",
            "PATCH_COMMIT.txt",
            "UPDATE_PACKAGE_MANIFEST.json",
            "modified_files/tracked.txt",
            "modified_files/run.sh",
        }.issubset(names)
        test_mode = archive.getinfo("TEST_COMMANDS.sh").external_attr >> 16
        script_mode = archive.getinfo("modified_files/run.sh").external_attr >> 16
        assert stat.S_IMODE(test_mode) & 0o111
        assert stat.S_IMODE(script_mode) & 0o111
        manifest = json.loads(archive.read("UPDATE_PACKAGE_MANIFEST.json"))
    assert manifest["package_format"] == "bundle-backed-v1"
    assert manifest["base_commit"] == base
    assert {item["path"] for item in manifest["changed_files"]} == {
        "run.sh",
        "tracked.txt",
    }
    verified = run(
        [
            "python3",
            str(VERIFIER),
            "--repo",
            str(repo),
            "--package",
            str(output),
            "--json",
        ]
    )
    assert json.loads(verified.stdout)["status"] == "PASS"


def test_packager_rejects_missing_modified_after_image(package_fixture, tmp_path: Path):
    repo, stage, _ = package_fixture
    (stage / "modified_files" / "tracked.txt").unlink()
    proc = run(
        [
            "python3",
            str(PACKAGER),
            "--repo",
            str(repo),
            "--package-root",
            str(stage),
            "--output",
            str(tmp_path / "bad.zip"),
        ],
        check=False,
    )
    assert proc.returncode != 0
    assert "modified_files inventory mismatch" in proc.stderr


def test_production_verifier_rejects_legacy_patch_only_package(
    package_fixture, tmp_path: Path
):
    repo, stage, _ = package_fixture
    output = tmp_path / "legacy.zip"
    with zipfile.ZipFile(output, "w") as archive:
        for path in stage.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(stage).as_posix())
    proc = run(
        [
            "python3",
            str(VERIFIER),
            "--repo",
            str(repo),
            "--package",
            str(output),
        ],
        check=False,
    )
    assert proc.returncode != 0
    assert "new production packages must be bundle-backed" in proc.stderr


def test_packager_rejects_nonexecutable_test_commands(package_fixture, tmp_path: Path):
    repo, stage, _ = package_fixture
    os.chmod(stage / "TEST_COMMANDS.sh", 0o644)
    proc = run(
        [
            "python3",
            str(PACKAGER),
            "--repo",
            str(repo),
            "--package-root",
            str(stage),
            "--output",
            str(tmp_path / "bad-mode.zip"),
        ],
        check=False,
    )
    assert proc.returncode != 0
    assert "TEST_COMMANDS.sh must be executable" in proc.stderr
