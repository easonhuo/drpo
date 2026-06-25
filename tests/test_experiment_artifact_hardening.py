from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import artifact_protocol_hardened as hardened


SCRIPT_NAMES = [
    "artifact_protocol_hardened.py",
    "package_experiment_hardened.py",
    "verify_experiment_package_hardened.py",
    "run_experiment_guard_hardened.py",
    "resolve_main_commit.py",
]


def run(cmd: list[str], cwd: Path, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in SCRIPT_NAMES:
        shutil.copy2(PROJECT_ROOT / "scripts" / name, repo / "scripts" / name)
    (repo / "README.md").write_text("base\n")
    assert run(["git", "init", "-b", "main"], repo).returncode == 0
    assert run(["git", "config", "user.email", "test@example.com"], repo).returncode == 0
    assert run(["git", "config", "user.name", "DRPO Test"], repo).returncode == 0
    assert run(["git", "add", "."], repo).returncode == 0
    commit = run(["git", "commit", "-m", "base"], repo)
    assert commit.returncode == 0, commit.stderr
    return repo


def head(repo: Path) -> str:
    result = run(["git", "rev-parse", "HEAD"], repo)
    assert result.returncode == 0
    return result.stdout.strip()


def failed_result(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "RUN_FAILED.json").write_text('{"execution_state":"failed"}\n')
    (root / "run_manifest.json").write_text('{"execution_state":"failed"}\n')
    (root / "logs").mkdir()
    (root / "logs" / "run.log").write_text("traceback\n")
    return root


def package_failed(
    repo: Path,
    result: Path,
    output: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return run(
        [
            sys.executable,
            "scripts/package_experiment_hardened.py",
            "--repo-root",
            str(repo),
            "--experiment-id",
            "TEST-FAIL",
            "--package-kind",
            "experiment-failed",
            "--result-dir",
            str(result),
            "--output",
            str(output),
            "--base-commit",
            head(repo),
            *extra,
        ],
        repo,
        timeout=60,
    )


def test_nonzero_child_still_creates_failed_recovery_package(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    output_root = tmp_path / "run"
    artifact = tmp_path / "failed.zip"
    result = run(
        [
            sys.executable,
            "scripts/run_experiment_guard_hardened.py",
            "--experiment-id",
            "TEST-NONZERO",
            "--repo-root",
            str(repo),
            "--output-root",
            str(output_root),
            "--artifact-output",
            str(artifact),
            "--heartbeat-seconds",
            "0.05",
            "--stale-seconds",
            "5",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ],
        repo,
        timeout=30,
    )
    assert result.returncode == 7, result.stderr
    assert artifact.is_file()
    assert (output_root / "RUN_FAILED.json").is_file()
    with zipfile.ZipFile(artifact) as zf:
        assert "results/TEST-NONZERO/RUN_FAILED.json" in zf.namelist()


def test_missing_required_output_forces_failed_package(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    output_root = tmp_path / "run"
    artifact = tmp_path / "missing.zip"
    result = run(
        [
            sys.executable,
            "scripts/run_experiment_guard_hardened.py",
            "--experiment-id",
            "TEST-MISSING",
            "--repo-root",
            str(repo),
            "--output-root",
            str(output_root),
            "--artifact-output",
            str(artifact),
            "--required-output",
            "aggregate.csv",
            "--heartbeat-seconds",
            "0.05",
            "--stale-seconds",
            "5",
            "--",
            sys.executable,
            "-c",
            "print('done')",
        ],
        repo,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    marker = json.loads((output_root / "RUN_FAILED.json").read_text())
    assert marker["missing_required_outputs"] == ["aggregate.csv"]
    assert artifact.is_file()


def test_stale_timeout_preserves_failed_package(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    output_root = tmp_path / "run"
    artifact = tmp_path / "stale.zip"
    result = run(
        [
            sys.executable,
            "scripts/run_experiment_guard_hardened.py",
            "--experiment-id",
            "TEST-STALE",
            "--repo-root",
            str(repo),
            "--output-root",
            str(output_root),
            "--artifact-output",
            str(artifact),
            "--heartbeat-seconds",
            "0.05",
            "--stale-seconds",
            "0.25",
            "--fail-on-stale",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(5)",
        ],
        repo,
        timeout=30,
    )
    assert result.returncode != 0
    marker = json.loads((output_root / "RUN_FAILED.json").read_text())
    assert marker["stale_detected"] is True
    assert artifact.is_file()


def test_dirty_formal_worktree_is_rejected_before_launch(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "README.md").write_text("dirty\n")
    artifact = tmp_path / "should_not_exist.zip"
    result = run(
        [
            sys.executable,
            "scripts/run_experiment_guard_hardened.py",
            "--experiment-id",
            "TEST-DIRTY",
            "--repo-root",
            str(repo),
            "--output-root",
            str(tmp_path / "run"),
            "--artifact-output",
            str(artifact),
            "--",
            sys.executable,
            "-c",
            "print('must not run')",
        ],
        repo,
    )
    assert result.returncode == 2
    assert "clean worktree" in result.stderr
    assert not artifact.exists()


def test_dirty_pilot_requires_flag_and_captures_launch_snapshot(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "README.md").write_text("dirty pilot\n")
    (repo / "new_config.yaml").write_text("x: 1\n")
    output_root = tmp_path / "pilot"
    artifact = tmp_path / "pilot.zip"
    result = run(
        [
            sys.executable,
            "scripts/run_experiment_guard_hardened.py",
            "--run-class",
            "pilot",
            "--allow-dirty",
            "--experiment-id",
            "TEST-PILOT",
            "--repo-root",
            str(repo),
            "--output-root",
            str(output_root),
            "--artifact-output",
            str(artifact),
            "--",
            sys.executable,
            "-c",
            "print('pilot')",
        ],
        repo,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    snapshot = output_root / "provenance_launch" / "LAUNCH_SNAPSHOT_MANIFEST.json"
    assert snapshot.is_file()
    payload = json.loads(snapshot.read_text())
    assert payload["status"]
    assert any(row["path"] == "new_config.yaml" for row in payload["untracked"])


def test_base_commit_mismatch_fails_without_publishing_zip(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result_dir = failed_result(tmp_path / "failed")
    output = tmp_path / "mismatch.zip"
    result = run(
        [
            sys.executable,
            "scripts/package_experiment_hardened.py",
            "--repo-root",
            str(repo),
            "--experiment-id",
            "TEST-FAIL",
            "--package-kind",
            "experiment-failed",
            "--result-dir",
            str(result_dir),
            "--output",
            str(output),
            "--base-commit",
            "0" * 40,
        ],
        repo,
    )
    assert result.returncode == 2
    assert "does not match expected commit" in result.stderr
    assert not output.exists()
    assert not output.with_suffix(".zip.candidate").exists()


def test_checksum_corruption_is_rejected(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result_dir = failed_result(tmp_path / "failed")
    package = tmp_path / "valid.zip"
    built = package_failed(repo, result_dir, package)
    assert built.returncode == 0, built.stderr
    with zipfile.ZipFile(package) as zf:
        members = {name: zf.read(name) for name in zf.namelist() if not name.endswith("/")}
    members["results/TEST-FAIL/run_manifest.json"] = b'{"corrupted":true}\n'
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    checked = run(
        [
            sys.executable,
            "scripts/verify_experiment_package_hardened.py",
            str(package),
            "--repo-root",
            str(repo),
        ],
        repo,
    )
    assert checked.returncode == 2
    assert "Checksum mismatch" in checked.stderr


def test_large_checkpoint_stays_out_of_lightweight_main_and_can_use_sidecar(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result_dir = failed_result(tmp_path / "failed")
    checkpoint = result_dir / "method" / "adapter_model.safetensors"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(os.urandom(2 * 1024 * 1024))
    package = tmp_path / "evidence.zip"
    sidecar = tmp_path / "checkpoint_sidecar.zip"
    built = package_failed(
        repo,
        result_dir,
        package,
        "--max-package-mib",
        "1",
        "--max-single-file-mib",
        "0.1",
        "--sidecar-output",
        str(sidecar),
    )
    assert built.returncode == 0, built.stderr
    assert package.stat().st_size < 1024 * 1024
    assert sidecar.is_file()
    with zipfile.ZipFile(package) as zf:
        names = set(zf.namelist())
        assert "results/TEST-FAIL/method/adapter_model.safetensors" not in names
        index = json.loads(zf.read("LARGE_FILE_INDEX.json"))
        row = next(x for x in index["entries"] if x["path"].endswith("adapter_model.safetensors"))
        assert row["include_main"] is False
        assert row["include_sidecar"] is True
        assert index["sidecar"]["sha256"]
    with zipfile.ZipFile(sidecar) as zf:
        assert "results/TEST-FAIL/method/adapter_model.safetensors" in zf.namelist()


def test_failed_package_without_sidecar_is_still_lightweight(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result_dir = failed_result(tmp_path / "failed")
    checkpoint = result_dir / "method" / "adapter_model.safetensors"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(os.urandom(2 * 1024 * 1024))
    package = tmp_path / "evidence_only.zip"
    built = package_failed(
        repo,
        result_dir,
        package,
        "--max-package-mib",
        "1",
        "--max-single-file-mib",
        "0.1",
    )
    assert built.returncode == 0, built.stderr
    assert package.stat().st_size < 1024 * 1024
    with zipfile.ZipFile(package) as zf:
        names = set(zf.namelist())
        assert "results/TEST-FAIL/method/adapter_model.safetensors" not in names
        index = json.loads(zf.read("LARGE_FILE_INDEX.json"))
        row = next(x for x in index["entries"] if x["path"].endswith("adapter_model.safetensors"))
        assert row["include_sidecar"] is False
        assert row["reason"] == "large_or_checkpoint_index_only"


def test_external_result_symlink_is_rejected(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result_dir = failed_result(tmp_path / "failed")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.safetensors").write_bytes(b"model")
    (result_dir / "best_adapter").symlink_to(outside, target_is_directory=True)
    output = tmp_path / "unsafe.zip"
    built = package_failed(repo, result_dir, output)
    assert built.returncode == 2
    assert "external symlink" in built.stderr
    assert not output.exists()


def test_internal_verification_failure_leaves_no_final_zip(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result_dir = failed_result(tmp_path / "failed")
    output = tmp_path / "too_small_limit.zip"
    built = package_failed(
        repo,
        result_dir,
        output,
        "--max-package-mib",
        "0.0001",
    )
    assert built.returncode == 2
    assert "exceeding hard limit" in built.stderr
    assert not output.exists()
    assert not output.with_suffix(".zip.candidate").exists()


def test_commit_resolver_rejects_wrong_expected_sha(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    checked = run(
        [
            sys.executable,
            "scripts/resolve_main_commit.py",
            "--repo-root",
            str(repo),
            "--expected-sha",
            "f" * 40,
        ],
        repo,
    )
    assert checked.returncode == 2
    assert "does not match expected commit" in checked.stderr


def test_code_change_during_formal_run_marks_provenance_compromised(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    output_root = tmp_path / "run"
    artifact = tmp_path / "compromised.zip"
    code = "from pathlib import Path; Path('README.md').write_text('changed during run\\n')"
    result = run(
        [
            sys.executable,
            "scripts/run_experiment_guard_hardened.py",
            "--experiment-id",
            "TEST-COMPROMISED",
            "--repo-root",
            str(repo),
            "--output-root",
            str(output_root),
            "--artifact-output",
            str(artifact),
            "--",
            sys.executable,
            "-c",
            code,
        ],
        repo,
        timeout=30,
    )
    assert result.returncode == 2
    marker = json.loads((output_root / "RUN_FAILED.json").read_text())
    assert marker["provenance_compromised"] is True
    assert artifact.is_file()


def test_public_entrypoints_are_executable() -> None:
    for name in (
        "package_experiment_hardened.py",
        "run_experiment_guard_hardened.py",
        "verify_experiment_package_hardened.py",
        "resolve_main_commit.py",
    ):
        mode = (PROJECT_ROOT / "scripts" / name).stat().st_mode
        assert mode & stat.S_IXUSR, f"{name} lost its executable bit"


@pytest.mark.parametrize(
    "wrapper_name, expected_message",
    [
        ("package_experiment_hardened.py", "refusing to fall back"),
        ("run_experiment_guard_hardened.py", "refusing to run"),
        ("verify_experiment_package_hardened.py", "refusing to use"),
    ],
)
def test_public_wrappers_fail_closed_without_core_module(
    tmp_path: Path, wrapper_name: str, expected_message: str
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(PROJECT_ROOT / "scripts" / wrapper_name, scripts / wrapper_name)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    checked = subprocess.run(
        [sys.executable, str(scripts / wrapper_name), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        env=env,
        timeout=10,
    )
    assert checked.returncode == 2
    assert "artifact_protocol_hardened.py is required" in checked.stderr
    assert expected_message in checked.stderr


def test_commit_resolver_timeout_uses_non_authoritative_local_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    expected = head(repo)
    add_remote = run(
        ["git", "remote", "add", "origin", "https://example.invalid/drpo.git"],
        repo,
    )
    assert add_remote.returncode == 0
    update_ref = run(
        ["git", "update-ref", "refs/remotes/origin/main", expected],
        repo,
    )
    assert update_ref.returncode == 0
    original_run = hardened.run

    def fake_run(
        cmd: list[str],
        cwd: Path,
        *,
        check: bool = True,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["git", "ls-remote", "--exit-code"]:
            raise subprocess.TimeoutExpired(cmd, timeout or 0.0)
        return original_run(cmd, cwd, check=check, timeout=timeout)

    monkeypatch.setattr(hardened, "run", fake_run)
    report = hardened.resolve_origin_main(repo, timeout=0.01)
    assert report["sha"] == expected
    assert report["source"] == "local_remote_tracking_ref"
    assert report["authoritative"] is False
    assert "timed out" in report["error"]
