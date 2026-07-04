from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_DIR = REPO_ROOT / "tools" / "drpo-update"


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
):
    proc = subprocess.run(
        cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"command failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    return proc


def test_helper_reports_version():
    proc = run([str(HELPER_DIR / "drpo-update"), "--version"])
    assert proc.stdout.strip() == "drpo-update 2.4.1"


def test_wrapper_prefers_repository_virtualenv_over_bare_python3(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    helper = repo / "tools/drpo-update"
    helper.mkdir(parents=True)
    for name in ("drpo-update", "drpo_update.py", "test_selection.py"):
        source = HELPER_DIR / name
        target = helper / name
        target.write_bytes(source.read_bytes())
        os.chmod(target, source.stat().st_mode)

    venv_bin = repo / ".venv/bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(Path(sys.executable))

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text("#!/usr/bin/env bash\nexit 97\n", encoding="utf-8")
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.pop("DRPO_PYTHON", None)
    env.pop("VIRTUAL_ENV", None)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    proc = run(["bash", str(helper / "drpo-update"), "--version"], env=env)
    assert proc.stdout.strip() == "drpo-update 2.4.1"


def test_installer_defaults_to_repository_symlink(tmp_path: Path):
    origin = tmp_path / "origin.git"
    repo = tmp_path / "repo"
    run(["git", "init", "--bare", str(origin)])
    run(["git", "init", str(repo)])
    run(["git", "-C", str(repo), "remote", "add", "origin", str(origin)])
    target_dir = repo / "tools" / "drpo-update"
    target_dir.mkdir(parents=True)
    for name in ("drpo-update", "drpo_update.py", "test_selection.py", "install.sh"):
        source = HELPER_DIR / name
        destination = target_dir / name
        destination.write_bytes(source.read_bytes())
        os.chmod(destination, source.stat().st_mode)

    home = tmp_path / "home"
    home.mkdir()
    env = os.environ.copy()
    env.update({"HOME": str(home), "DRPO_UPDATE_ALLOW_ANY_REMOTE": "1", "SHELL": "/bin/zsh"})
    proc = run(["bash", str(target_dir / "install.sh"), str(repo)], env=env)
    installed = home / "bin" / "drpo-update"
    assert installed.is_symlink()
    assert installed.resolve() == (target_dir / "drpo-update").resolve()
    assert (home / ".config" / "drpo-update" / "repo_path").read_text().strip() == str(
        repo.resolve()
    )
    assert "drpo-update 2.4.1" in proc.stdout


def test_installer_copy_mode_installs_runtime_siblings_and_runs(tmp_path: Path):
    origin = tmp_path / "origin.git"
    repo = tmp_path / "repo"
    run(["git", "init", "--bare", str(origin)])
    run(["git", "init", str(repo)])
    run(["git", "-C", str(repo), "remote", "add", "origin", str(origin)])
    target_dir = repo / "tools" / "drpo-update"
    target_dir.mkdir(parents=True)
    for name in ("drpo-update", "drpo_update.py", "test_selection.py", "install.sh"):
        source = HELPER_DIR / name
        destination = target_dir / name
        destination.write_bytes(source.read_bytes())
        os.chmod(destination, source.stat().st_mode)

    home = tmp_path / "home"
    home.mkdir()
    env = os.environ.copy()
    env.update({"HOME": str(home), "DRPO_UPDATE_ALLOW_ANY_REMOTE": "1", "SHELL": "/bin/zsh"})
    proc = run(["bash", str(target_dir / "install.sh"), str(repo), "--copy"], env=env)
    installed = home / "bin" / "drpo-update"
    assert installed.is_file() and not installed.is_symlink()
    assert (home / "bin" / "drpo_update.py").is_file()
    assert (home / "bin" / "test_selection.py").is_file()
    version = run([str(installed), "--version"], env=env)
    assert version.stdout.strip() == "drpo-update 2.4.1"
    assert "Mode:       copy" in proc.stdout


def test_provenance_records_user_verified_original_hash():
    text = (HELPER_DIR / "SOURCE_PROVENANCE.md").read_text()
    assert "f344b0cffc163ecdeb80ec8d07b564d00c3538ad22e03887f87bd1ce2a85f4f3" in text
    assert "byte-identical" in text


def test_doctor_runs_non_destructive_transaction_self_tests(tmp_path: Path):
    # The doctor invokes only synthetic temporary-repository tests.
    env = os.environ.copy()
    env["DRPO_PYTHON"] = sys.executable
    proc = run(
        [str(HELPER_DIR / "drpo-update"), "--doctor", "--repo", str(REPO_ROOT)],
        cwd=REPO_ROOT,
        env=env,
    )
    assert "DOCTOR PYTHON COMPILE: PASS" in proc.stdout
    assert "DOCTOR SHELL SYNTAX: PASS" in proc.stdout
    assert "DOCTOR TRANSACTION PATHS: PASS" in proc.stdout
    assert "DRPO UPDATE DOCTOR: PASS" in proc.stdout


def _load_update_module():
    path = HELPER_DIR / "drpo_update.py"
    spec = importlib.util.spec_from_file_location("drpo_update_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recovery_artifact_is_rejected_with_actionable_message(tmp_path: Path):
    package = tmp_path / "raw-complete.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BASE_COMMIT.txt", "0" * 40 + "\n")
        archive.writestr("CHANGE_SUMMARY.md", "raw evidence\n")
        archive.writestr("update.patch", b"")
        archive.writestr(
            "ARTIFACT_MANIFEST.json",
            json.dumps(
                {
                    "package_kind": "experiment-raw-complete",
                    "experiment_id": "TEST-RAW-COMPLETE",
                    "base_commit": "0" * 40,
                }
            ),
        )

    updater = _load_update_module()
    with pytest.raises(
        updater.UpdateError,
        match=r"recovery/evidence package.*Do not pass it to drpo-update",
    ):
        updater.extract_package(package, tmp_path / "extract")


def test_stage5_handoff_normalization_is_noop_in_manual_mode(tmp_path: Path):
    updater = _load_update_module()
    repo = tmp_path / "repo"
    import shutil

    shutil.copytree(
        REPO_ROOT,
        repo,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc"),
    )
    authority_path = repo / "docs/handoff_versions/AUTHORITY.yaml"
    authority_payload = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    authority_payload["mode"] = "manual"
    authority_payload["delta_authority"]["checkpoint_manifest"] = None
    authority_payload["delta_authority"]["activation_parent_commit"] = None
    authority_payload["generated_views"]["stage4a_minimal_refresh"] = False
    authority_payload["safety"]["direct_handoff_edit_forbidden"] = False
    authority_path.write_text(
        yaml.safe_dump(authority_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    run(["git", "init", "-q", str(repo)])
    run(["git", "-C", str(repo), "config", "user.name", "Stage5 Test"])
    run(["git", "-C", str(repo), "config", "user.email", "stage5@test.invalid"])
    run(["git", "-C", str(repo), "add", "-A"])
    run(["git", "-C", str(repo), "commit", "-q", "-m", "base"])
    current = run(["git", "-C", str(repo), "rev-parse", "HEAD"]).stdout.strip()
    report = updater.ApplyReport(package="test", repository=str(repo))
    normalized = updater.run_handoff_normalization(
        repo,
        repo,
        current=current,
        base=current,
        source_patch_commit=None,
        report=report,
        log_dir=tmp_path / "logs",
    )
    assert normalized == current
    assert report.handoff_normalization == {
        "status": "PASS",
        "mode": "manual",
        "normalization": "not_applicable",
        "authority_transitioned": False,
    }
