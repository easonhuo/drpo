from __future__ import annotations

import importlib.util
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "artifact_protocol_hardened.py"
SPEC = importlib.util.spec_from_file_location("artifact_protocol_hardened_contract", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _build_governance_package(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "governance-test@example.invalid")
    _git(repo, "config", "user.name", "Governance Test")
    tracked = repo / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "base")
    base_commit = _git(repo, "rev-parse", "HEAD")
    tracked.write_text("after\n", encoding="utf-8")

    package = tmp_path / "governance-update.zip"
    returncode = MODULE.package_main(
        [
            "--repo-root",
            str(repo),
            "--experiment-id",
            "GOV-CONTRACT",
            "--package-kind",
            "governance",
            "--output",
            str(package),
            "--base-commit",
            base_commit,
            "--test-command",
            "python3 -c \"print('contract-ok')\"",
        ]
    )
    assert returncode == 0
    assert package.is_file()
    return package, base_commit


def test_update_patch_is_required_and_nonempty(tmp_path: Path) -> None:
    package, _ = _build_governance_package(tmp_path)
    assert "update.patch" in MODULE.REQUIRED_TOP_LEVEL
    with zipfile.ZipFile(package) as archive:
        patch = archive.read("update.patch")
    assert patch.strip()
    assert b"tracked.txt" in patch

    empty_patch = tmp_path / "empty-patch.zip"
    stage = tmp_path / "empty-stage"
    stage.mkdir()
    with zipfile.ZipFile(package) as archive:
        archive.extractall(stage)
    (stage / "update.patch").write_bytes(b"")
    MODULE.write_checksums(stage)
    MODULE.write_zip_from_stage(stage, empty_patch)
    with pytest.raises(ValueError, match="requires a non-empty update.patch"):
        MODULE.verify_package(
            empty_patch,
            repo_root=None,
            skip_head_match=True,
            hard_limit_mib=25.0,
        )


def test_change_summary_is_required_and_generated_with_identity(tmp_path: Path) -> None:
    package, base_commit = _build_governance_package(tmp_path)
    assert "CHANGE_SUMMARY.md" in MODULE.REQUIRED_TOP_LEVEL
    with zipfile.ZipFile(package) as archive:
        summary = archive.read("CHANGE_SUMMARY.md").decode("utf-8")
    assert summary.strip()
    assert "GOV-CONTRACT" in summary
    assert base_commit in summary
    assert "## Modified files" in summary
    assert "tracked.txt" in summary

    missing_summary = tmp_path / "missing-summary.zip"
    stage = tmp_path / "missing-summary-stage"
    stage.mkdir()
    with zipfile.ZipFile(package) as archive:
        archive.extractall(stage)
    (stage / "CHANGE_SUMMARY.md").unlink()
    MODULE.write_checksums(stage)
    MODULE.write_zip_from_stage(stage, missing_summary)
    with pytest.raises(ValueError, match="Missing required top-level files"):
        MODULE.verify_package(
            missing_summary,
            repo_root=None,
            skip_head_match=True,
            hard_limit_mib=25.0,
        )


def test_test_commands_is_required_executable_and_rejects_placeholders(
    tmp_path: Path,
) -> None:
    package, _ = _build_governance_package(tmp_path)
    assert "TEST_COMMANDS.sh" in MODULE.REQUIRED_TOP_LEVEL
    with zipfile.ZipFile(package) as archive:
        commands = archive.read("TEST_COMMANDS.sh").decode("utf-8")
        mode = archive.getinfo("TEST_COMMANDS.sh").external_attr >> 16
    assert commands.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in commands
    assert "contract-ok" in commands
    assert stat.S_IMODE(mode) & 0o111
    MODULE.validate_test_commands(commands)

    with pytest.raises(ValueError, match="placeholder tokens"):
        MODULE.validate_test_commands(
            "#!/usr/bin/env bash\nset -euo pipefail\npython3 /abs/path/test.py\n"
        )
