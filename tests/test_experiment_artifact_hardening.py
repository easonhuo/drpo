from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import artifact_protocol_hardened as hardened  # noqa: E402


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
    # A local origin makes normal formal-guard tests authoritative without network.
    # Tests that exercise offline behavior remove or replace this remote explicitly.
    assert run(["git", "remote", "add", "origin", "."], repo).returncode == 0
    return repo


def head(repo: Path) -> str:
    result = run(["git", "rev-parse", "HEAD"], repo)
    assert result.returncode == 0
    return result.stdout.strip()


def failed_result(
    root: Path,
    experiment_id: str = "TEST-FAIL",
    base_commit: str = "PENDING",
) -> Path:
    root.mkdir(parents=True)
    identity = {"experiment_id": experiment_id, "base_commit": base_commit}
    (root / "RUN_FAILED.json").write_text(
        json.dumps({**identity, "execution_state": "failed"}) + "\n"
    )
    (root / "run_manifest.json").write_text(
        json.dumps({**identity, "execution_state": "failed"}) + "\n"
    )
    (root / "logs").mkdir()
    (root / "logs" / "run.log").write_text("traceback\n")
    return root


def package_failed(
    repo: Path,
    result: Path,
    output: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    base = head(repo)
    identity = {"experiment_id": "TEST-FAIL", "base_commit": base}
    for name in ("RUN_FAILED.json", "run_manifest.json"):
        payload = json.loads((result / name).read_text())
        payload.update(identity)
        (result / name).write_text(json.dumps(payload) + "\n")
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
            base,
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
        "--sidecar-purpose",
        "restart",
        "--sidecar-file",
        "method/adapter_model.safetensors",
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



def final_result(root: Path, experiment_id: str, base_commit: str) -> Path:
    root.mkdir(parents=True)
    identity = {"experiment_id": experiment_id, "base_commit": base_commit}
    (root / "RUN_COMPLETE.json").write_text(json.dumps({**identity, "execution_state": "complete"}) + "\n")
    (root / "TERMINAL_AUDIT.json").write_text(json.dumps({**identity, "status": "passed"}) + "\n")
    (root / "run_manifest.json").write_text(json.dumps({**identity, "execution_state": "terminal_audited"}) + "\n")
    (root / "logs").mkdir()
    (root / "logs" / "run.log").write_text("complete\n")
    return root


def test_failed_candidate_preserves_existing_valid_output(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result_dir = failed_result(tmp_path / "failed")
    output = tmp_path / "evidence.zip"
    sentinel = b"previous-valid-artifact"
    output.write_bytes(sentinel)
    built = package_failed(
        repo,
        result_dir,
        output,
        "--max-package-mib",
        "0.0001",
    )
    assert built.returncode == 2
    assert output.read_bytes() == sentinel
    assert not list(tmp_path.glob(".*evidence.zip.candidate-*"))


def test_command_start_failure_creates_failed_evidence_package(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    output_root = tmp_path / "run"
    artifact = tmp_path / "start_failed.zip"
    result = run(
        [
            sys.executable,
            "scripts/run_experiment_guard_hardened.py",
            "--experiment-id",
            "TEST-START-FAIL",
            "--repo-root",
            str(repo),
            "--output-root",
            str(output_root),
            "--artifact-output",
            str(artifact),
            "--",
            "/definitely/not/a/command",
        ],
        repo,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    marker = json.loads((output_root / "RUN_FAILED.json").read_text())
    assert marker["supervisor_error_type"] == "FileNotFoundError"
    assert (output_root / "logs" / "supervised_run.log").is_file()
    assert artifact.is_file()
    with zipfile.ZipFile(artifact) as zf:
        assert "results/TEST-START-FAIL/RUN_FAILED.json" in zf.namelist()


def test_head_change_failed_package_is_bound_to_launch_commit(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    launch = head(repo)
    output_root = tmp_path / "run"
    artifact = tmp_path / "head_changed.zip"
    code = (
        "from pathlib import Path; import subprocess; "
        "Path('README.md').write_text('changed and committed\\n'); "
        "subprocess.run(['git','add','README.md'], check=True); "
        "subprocess.run(['git','commit','-m','during run'], check=True)"
    )
    result = run(
        [
            sys.executable,
            "scripts/run_experiment_guard_hardened.py",
            "--experiment-id",
            "TEST-HEAD-CHANGE",
            "--repo-root",
            str(repo),
            "--output-root",
            str(output_root),
            "--artifact-output",
            str(artifact),
            "--source-file",
            "README.md",
            "--",
            sys.executable,
            "-c",
            code,
        ],
        repo,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    marker = json.loads((output_root / "RUN_FAILED.json").read_text())
    assert marker["base_commit"] == launch
    assert marker["end_commit"] != launch
    assert marker["provenance_compromised"] is True
    with zipfile.ZipFile(artifact) as zf:
        assert zf.read("BASE_COMMIT.txt").decode().strip() == launch
        manifest = json.loads(zf.read("ARTIFACT_MANIFEST.json"))
        assert manifest["base_commit"] == launch
        assert manifest["packaging_head"] == marker["end_commit"]
        assert zf.read("source_snapshot/README.md") == b"base\n"


def test_final_package_requires_manifest_logs_and_matching_identity(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = head(repo)
    result_dir = tmp_path / "final"
    result_dir.mkdir()
    (result_dir / "RUN_COMPLETE.json").write_text("{}\n")
    (result_dir / "TERMINAL_AUDIT.json").write_text("{}\n")
    output = tmp_path / "final.zip"
    checked = run(
        [
            sys.executable,
            "scripts/package_experiment_hardened.py",
            "--repo-root",
            str(repo),
            "--experiment-id",
            "TEST-FINAL",
            "--package-kind",
            "experiment-final",
            "--result-dir",
            str(result_dir),
            "--output",
            str(output),
            "--base-commit",
            base,
        ],
        repo,
    )
    assert checked.returncode == 2
    assert "run_manifest.json" in checked.stderr
    assert not output.exists()


def test_final_large_checkpoint_defaults_to_persistent_local_index_without_sidecar(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path)
    base = head(repo)
    (repo / "README.md").write_text("changed\n")
    result_dir = final_result(tmp_path / "final", "TEST-FINAL", base)
    checkpoint = result_dir / "method" / "adapter_model.safetensors"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(os.urandom(1024 * 1024))
    output = tmp_path / "final.zip"
    checked = run(
        [
            sys.executable,
            "scripts/package_experiment_hardened.py",
            "--repo-root",
            str(repo),
            "--experiment-id",
            "TEST-FINAL",
            "--package-kind",
            "experiment-final",
            "--result-dir",
            str(result_dir),
            "--output",
            str(output),
            "--base-commit",
            base,
            "--max-single-file-mib",
            "0.1",
        ],
        repo,
        timeout=60,
    )
    assert checked.returncode == 0, checked.stderr
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
        assert "results/TEST-FINAL/method/adapter_model.safetensors" not in names
        index = json.loads(zf.read("LARGE_FILE_INDEX.json"))
        assert index["sidecar_default"] is False
        assert index["sidecar"] is None
        row = next(x for x in index["entries"] if x["path"].endswith("adapter_model.safetensors"))
        assert row["persistence_status"] == "persistent_local"
        assert row["storage_path"] == str(checkpoint.resolve())
        assert row["sha256"]
        assert row["role"] == "checkpoint_or_model_state"


def test_sidecar_selection_and_size_limits_are_enforced(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result_dir = failed_result(tmp_path / "failed")
    checkpoint = result_dir / "method" / "adapter_model.safetensors"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(os.urandom(1024 * 1024))
    output = tmp_path / "evidence.zip"
    sidecar = tmp_path / "sidecar.zip"
    checked = package_failed(
        repo,
        result_dir,
        output,
        "--sidecar-output",
        str(sidecar),
        "--sidecar-purpose",
        "restart",
        "--sidecar-file",
        "method/adapter_model.safetensors",
        "--max-sidecar-mib",
        "0.01",
    )
    assert checked.returncode == 2
    assert "Sidecar is" in checked.stderr
    assert not sidecar.exists()
    assert not output.exists()



def test_supervisor_exception_after_launch_still_packages_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    output_root = tmp_path / "run"
    artifact = tmp_path / "supervisor_failed.zip"
    original_latest = hardened.latest_mtime
    calls = 0

    def failing_latest(root: Path) -> float | None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise RuntimeError("injected supervisor failure")
        return original_latest(root)

    monkeypatch.setattr(hardened, "latest_mtime", failing_latest)
    rc = hardened.guard_main(
        [
            "--experiment-id",
            "TEST-SUPERVISOR-FAIL",
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
            "import time; time.sleep(2)",
        ]
    )
    assert rc != 0
    marker_payload = json.loads((output_root / "RUN_FAILED.json").read_text())
    assert marker_payload["supervisor_error_type"] == "RuntimeError"
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
        ["git", "remote", "set-url", "origin", "https://example.invalid/drpo.git"],
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

def test_small_checkpoint_is_indexed_instead_of_embedded_in_final(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = head(repo)
    (repo / "README.md").write_text("changed\n")
    result_dir = final_result(tmp_path / "final", "TEST-SMALL-CKPT", base)
    checkpoint = result_dir / "method" / "adapter_model.safetensors"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"small-adapter")
    output = tmp_path / "final.zip"
    checked = run(
        [
            sys.executable,
            "scripts/package_experiment_hardened.py",
            "--repo-root",
            str(repo),
            "--experiment-id",
            "TEST-SMALL-CKPT",
            "--package-kind",
            "experiment-final",
            "--result-dir",
            str(result_dir),
            "--output",
            str(output),
            "--base-commit",
            base,
            "--large-file-persistence",
            "persistent_local",
        ],
        repo,
        timeout=60,
    )
    assert checked.returncode == 0, checked.stderr
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
        member = "results/TEST-SMALL-CKPT/method/adapter_model.safetensors"
        assert member not in names
        index = json.loads(zf.read("LARGE_FILE_INDEX.json"))
        row = next(x for x in index["entries"] if x["path"].endswith("adapter_model.safetensors"))
        assert row["reason"] == "large_or_checkpoint_index_only"
        assert row["persistence_status"] == "persistent_local"


def test_foundation_model_weight_cannot_be_sidecarred(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result_dir = failed_result(tmp_path / "failed")
    model = result_dir / "base_model" / "model.safetensors"
    model.parent.mkdir()
    model.write_bytes(b"foundation-weight")
    output = tmp_path / "evidence.zip"
    sidecar = tmp_path / "foundation-sidecar.zip"
    checked = package_failed(
        repo,
        result_dir,
        output,
        "--sidecar-output",
        str(sidecar),
        "--sidecar-purpose",
        "cross_machine_transfer",
        "--sidecar-file",
        "base_model/model.safetensors",
    )
    assert checked.returncode == 2
    assert "Foundation-model weights" in checked.stderr
    assert not output.exists()
    assert not sidecar.exists()


def test_sidecar_requires_purpose_and_refuses_existing_output(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result_dir = failed_result(tmp_path / "failed")
    checkpoint = result_dir / "adapter_model.safetensors"
    checkpoint.write_bytes(b"adapter")
    output = tmp_path / "evidence.zip"
    sidecar = tmp_path / "sidecar.zip"

    missing_purpose = package_failed(
        repo,
        result_dir,
        output,
        "--sidecar-output",
        str(sidecar),
        "--sidecar-file",
        "adapter_model.safetensors",
    )
    assert missing_purpose.returncode == 2
    assert "--sidecar-purpose" in missing_purpose.stderr
    assert not output.exists()
    assert not sidecar.exists()

    sentinel = b"previous-sidecar"
    sidecar.write_bytes(sentinel)
    overwrite = package_failed(
        repo,
        result_dir,
        output,
        "--sidecar-output",
        str(sidecar),
        "--sidecar-purpose",
        "restart",
        "--sidecar-file",
        "adapter_model.safetensors",
    )
    assert overwrite.returncode == 2
    assert "Refusing to overwrite an existing sidecar" in overwrite.stderr
    assert sidecar.read_bytes() == sentinel
    assert not output.exists()


def test_main_publish_failure_removes_new_sidecar_and_preserves_old_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    base = head(repo)
    result_dir = failed_result(tmp_path / "failed", base_commit=base)
    checkpoint = result_dir / "adapter_model.safetensors"
    checkpoint.write_bytes(b"adapter")
    output = tmp_path / "evidence.zip"
    output.write_bytes(b"previous-main")
    sidecar = tmp_path / "versioned-sidecar.zip"
    original_replace = hardened.os.replace

    def fail_main_publish(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        if Path(destination).resolve() == output.resolve():
            raise OSError("injected main publish failure")
        original_replace(source, destination)

    monkeypatch.setattr(hardened.os, "replace", fail_main_publish)
    rc = hardened.package_main(
        [
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
            base,
            "--sidecar-output",
            str(sidecar),
            "--sidecar-purpose",
            "restart",
            "--sidecar-file",
            "adapter_model.safetensors",
        ]
    )
    assert rc == 2
    assert output.read_bytes() == b"previous-main"
    assert not sidecar.exists()


def test_output_symlink_is_rejected_without_touching_target(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result_dir = failed_result(tmp_path / "failed")
    target = tmp_path / "target.zip"
    target.write_bytes(b"do-not-touch")
    output = tmp_path / "output.zip"
    try:
        output.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable in this environment")
    checked = package_failed(repo, result_dir, output)
    assert checked.returncode == 2
    assert "symbolic link" in checked.stderr
    assert target.read_bytes() == b"do-not-touch"
    assert output.is_symlink()


def test_sidecar_verifier_rejects_duplicate_member_names(tmp_path: Path) -> None:
    sidecar = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning):
        with zipfile.ZipFile(sidecar, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("duplicate.bin", b"one")
            zf.writestr("duplicate.bin", b"two")
    with pytest.raises(ValueError, match="duplicate member names"):
        hardened.verify_sidecar_candidate(sidecar, hard_limit_mib=1.0)


def test_sidecar_verifier_rejects_manifest_payload_mismatch(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result_dir = failed_result(tmp_path / "failed")
    checkpoint = result_dir / "method" / "adapter_model.safetensors"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"adapter")
    output = tmp_path / "main.zip"
    sidecar = tmp_path / "sidecar.zip"
    checked = package_failed(
        repo,
        result_dir,
        output,
        "--sidecar-output",
        str(sidecar),
        "--sidecar-purpose",
        "restart",
        "--sidecar-file",
        "method/adapter_model.safetensors",
    )
    assert checked.returncode == 0, checked.stderr
    with zipfile.ZipFile(sidecar) as zf:
        members = {
            name: zf.read(name)
            for name in zf.namelist()
            if not name.endswith("/")
        }
    manifest = json.loads(members["SIDECAR_MANIFEST.json"])
    manifest["files"][0]["path"] = "method/not-the-adapter.safetensors"
    members["SIDECAR_MANIFEST.json"] = (
        json.dumps(manifest, indent=2).encode("utf-8") + b"\n"
    )
    with zipfile.ZipFile(sidecar, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    with pytest.raises(ValueError, match="missing selected file|payload inventory mismatch"):
        hardened.verify_sidecar_candidate(sidecar, hard_limit_mib=1.0)


def test_formal_guard_requires_explicit_pin_when_origin_is_unavailable(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path)
    assert run(["git", "remote", "remove", "origin"], repo).returncode == 0
    output_root = tmp_path / "run"
    artifact = tmp_path / "artifact.zip"
    checked = run(
        [
            sys.executable,
            "scripts/run_experiment_guard_hardened.py",
            "--experiment-id",
            "TEST-OFFLINE-UNPINNED",
            "--repo-root",
            str(repo),
            "--output-root",
            str(output_root),
            "--artifact-output",
            str(artifact),
            "--",
            sys.executable,
            "-c",
            "print('must not run')",
        ],
        repo,
    )
    assert checked.returncode == 2
    assert "authoritatively verify origin/main" in checked.stderr
    assert not artifact.exists()


def test_formal_guard_accepts_explicit_commit_pin_in_offline_clone(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    expected = head(repo)
    assert run(["git", "remote", "remove", "origin"], repo).returncode == 0
    output_root = tmp_path / "run"
    artifact = tmp_path / "artifact.zip"
    checked = run(
        [
            sys.executable,
            "scripts/run_experiment_guard_hardened.py",
            "--experiment-id",
            "TEST-OFFLINE-PINNED",
            "--repo-root",
            str(repo),
            "--output-root",
            str(output_root),
            "--artifact-output",
            str(artifact),
            "--expected-commit",
            expected,
            "--",
            sys.executable,
            "-c",
            "print('pinned offline run')",
        ],
        repo,
        timeout=30,
    )
    assert checked.returncode == 0, checked.stderr
    assert artifact.is_file()


def test_missing_source_snapshot_file_is_rejected_before_launch(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    output_root = tmp_path / "run"
    artifact = tmp_path / "artifact.zip"
    side_effect = tmp_path / "started.txt"
    checked = run(
        [
            sys.executable,
            "scripts/run_experiment_guard_hardened.py",
            "--experiment-id",
            "TEST-MISSING-SOURCE",
            "--repo-root",
            str(repo),
            "--output-root",
            str(output_root),
            "--artifact-output",
            str(artifact),
            "--source-file",
            "missing_entrypoint.py",
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(side_effect)!r}).write_text('started')",
        ],
        repo,
    )
    assert checked.returncode == 2
    assert "unavailable at launch commit" in checked.stderr
    assert not side_effect.exists()
    assert not artifact.exists()


def test_nonempty_output_root_is_rejected_before_launch(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    output_root = tmp_path / "run"
    output_root.mkdir()
    stale = output_root / "required.csv"
    stale.write_text("stale\n")
    artifact = tmp_path / "artifact.zip"
    checked = run(
        [
            sys.executable,
            "scripts/run_experiment_guard_hardened.py",
            "--experiment-id",
            "TEST-STALE-ROOT",
            "--repo-root",
            str(repo),
            "--output-root",
            str(output_root),
            "--artifact-output",
            str(artifact),
            "--required-output",
            "required.csv",
            "--",
            sys.executable,
            "-c",
            "print('must not start')",
        ],
        repo,
    )
    assert checked.returncode == 2
    assert "new or empty" in checked.stderr
    assert stale.read_text() == "stale\n"
    assert not artifact.exists()


def test_internal_apply_check_uses_clean_base_not_callers_staged_index(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = head(repo)
    (repo / "README.md").write_text("staged-change\n")
    staged = run(["git", "add", "README.md"], repo)
    assert staged.returncode == 0
    result_dir = failed_result(tmp_path / "failed", base_commit=base)
    output = tmp_path / "staged.zip"
    checked = run(
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
            base,
        ],
        repo,
        timeout=60,
    )
    assert checked.returncode == 0, checked.stderr
    assert output.is_file()


def test_final_package_rejects_cross_file_identity_mismatch(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = head(repo)
    (repo / "README.md").write_text("changed\n")
    result_dir = final_result(tmp_path / "final", "TEST-FINAL-ID", base)
    audit = json.loads((result_dir / "TERMINAL_AUDIT.json").read_text())
    audit["base_commit"] = "0" * 40
    (result_dir / "TERMINAL_AUDIT.json").write_text(json.dumps(audit) + "\n")
    output = tmp_path / "final.zip"
    checked = run(
        [
            sys.executable,
            "scripts/package_experiment_hardened.py",
            "--repo-root",
            str(repo),
            "--experiment-id",
            "TEST-FINAL-ID",
            "--package-kind",
            "experiment-final",
            "--result-dir",
            str(result_dir),
            "--output",
            str(output),
            "--base-commit",
            base,
        ],
        repo,
    )
    assert checked.returncode == 2
    assert "TERMINAL_AUDIT.json base_commit" in checked.stderr
    assert not output.exists()



def test_small_numpy_evidence_is_embedded_in_final_package(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = head(repo)
    (repo / "README.md").write_text("changed\n")
    result_dir = final_result(tmp_path / "final", "TEST-NUMPY", base)
    array_file = result_dir / "raw" / "trajectory_metrics.npz"
    array_file.parent.mkdir()
    array_file.write_bytes(b"small-numpy-evidence")
    output = tmp_path / "final.zip"
    checked = run(
        [
            sys.executable,
            "scripts/package_experiment_hardened.py",
            "--repo-root",
            str(repo),
            "--experiment-id",
            "TEST-NUMPY",
            "--package-kind",
            "experiment-final",
            "--result-dir",
            str(result_dir),
            "--output",
            str(output),
            "--base-commit",
            base,
        ],
        repo,
        timeout=60,
    )
    assert checked.returncode == 0, checked.stderr
    with zipfile.ZipFile(output) as zf:
        member = "results/TEST-NUMPY/raw/trajectory_metrics.npz"
        assert member in zf.namelist()
        assert zf.read(member) == b"small-numpy-evidence"
        index = json.loads(zf.read("LARGE_FILE_INDEX.json"))
        assert not any(row["path"] == "raw/trajectory_metrics.npz" for row in index["entries"])


def test_failed_package_rejects_marker_identity_mismatch(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    base = head(repo)
    result_dir = failed_result(tmp_path / "failed", base_commit="0" * 40)
    output = tmp_path / "failed.zip"
    checked = run(
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
            base,
        ],
        repo,
    )
    assert checked.returncode == 2
    assert "run_manifest.json base_commit" in checked.stderr
    assert not output.exists()


def test_experiment_id_path_traversal_is_rejected(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result_dir = failed_result(tmp_path / "failed", base_commit=head(repo))
    output = tmp_path / "unsafe.zip"
    checked = run(
        [
            sys.executable,
            "scripts/package_experiment_hardened.py",
            "--repo-root",
            str(repo),
            "--experiment-id",
            "../../escape",
            "--package-kind",
            "experiment-failed",
            "--result-dir",
            str(result_dir),
            "--output",
            str(output),
            "--base-commit",
            head(repo),
        ],
        repo,
    )
    assert checked.returncode == 2
    assert "experiment_id must match" in checked.stderr
    assert not output.exists()
    assert not (tmp_path / "escape").exists()


def test_result_root_symlink_is_rejected(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    target = failed_result(tmp_path / "target", base_commit=head(repo))
    link = tmp_path / "result_link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable in this environment")
    output = tmp_path / "unsafe.zip"
    checked = run(
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
            str(link),
            "--output",
            str(output),
            "--base-commit",
            head(repo),
        ],
        repo,
    )
    assert checked.returncode == 2
    assert "symbolic link" in checked.stderr
    assert not output.exists()


def test_output_root_symlink_is_rejected_before_launch(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    target = tmp_path / "target_run"
    target.mkdir()
    output_root = tmp_path / "run_link"
    try:
        output_root.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable in this environment")
    artifact = tmp_path / "artifact.zip"
    checked = run(
        [
            sys.executable,
            "scripts/run_experiment_guard_hardened.py",
            "--experiment-id",
            "TEST-ROOT-LINK",
            "--repo-root",
            str(repo),
            "--output-root",
            str(output_root),
            "--artifact-output",
            str(artifact),
            "--",
            sys.executable,
            "-c",
            "print('must not run')",
        ],
        repo,
    )
    assert checked.returncode == 2
    assert "symbolic link" in checked.stderr
    assert not artifact.exists()


def test_supervisor_setup_failure_terminates_child_and_packages_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    output_root = tmp_path / "run"
    artifact = tmp_path / "setup_failed.zip"

    def fail_thread_start(_self: threading.Thread) -> None:
        raise RuntimeError("injected thread-start failure")

    monkeypatch.setattr(hardened.threading.Thread, "start", fail_thread_start)
    rc = hardened.guard_main(
        [
            "--experiment-id",
            "TEST-SETUP-FAIL",
            "--repo-root",
            str(repo),
            "--output-root",
            str(output_root),
            "--artifact-output",
            str(artifact),
            "--termination-grace-seconds",
            "0.2",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ]
    )
    assert rc != 0
    marker = json.loads((output_root / "RUN_FAILED.json").read_text())
    assert marker["supervisor_error_type"] == "RuntimeError"
    assert "thread-start" in marker["supervisor_error"]
    assert artifact.is_file()


def test_stale_child_ignoring_sigterm_is_escalated_to_sigkill(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    output_root = tmp_path / "run"
    artifact = tmp_path / "stale_killed.zip"
    code = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); "
        "time.sleep(30)"
    )
    checked = run(
        [
            sys.executable,
            "scripts/run_experiment_guard_hardened.py",
            "--experiment-id",
            "TEST-STALE-KILL",
            "--repo-root",
            str(repo),
            "--output-root",
            str(output_root),
            "--artifact-output",
            str(artifact),
            "--heartbeat-seconds",
            "0.05",
            "--stale-seconds",
            "3.0",
            "--termination-grace-seconds",
            "0.2",
            "--fail-on-stale",
            "--",
            sys.executable,
            "-c",
            code,
        ],
        repo,
        timeout=15,
    )
    assert checked.returncode != 0
    marker = json.loads((output_root / "RUN_FAILED.json").read_text())
    assert marker["stale_detected"] is True
    assert marker["stale_term_sent"] is True
    assert marker["stale_kill_sent"] is True
    assert artifact.is_file()


def test_verifier_rejects_non_object_manifest_and_unknown_kind(tmp_path: Path) -> None:
    def write_minimal(path: Path, manifest_payload: object) -> None:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("BASE_COMMIT.txt", "0" * 40 + "\n")
            zf.writestr("ARTIFACT_MANIFEST.json", json.dumps(manifest_payload))
            zf.writestr("update.patch", b"")
            zf.writestr("CHANGE_SUMMARY.md", "summary\n")
            zf.writestr("TEST_COMMANDS.sh", "#!/bin/bash\nset -euo pipefail\ntrue\n")
            zf.writestr("SHA256SUMS.txt", "")

    malformed = tmp_path / "malformed.zip"
    write_minimal(malformed, [])
    with pytest.raises(ValueError, match="must contain a JSON object"):
        hardened.verify_package(
            malformed,
            repo_root=None,
            skip_head_match=True,
            hard_limit_mib=1.0,
        )

    unknown = tmp_path / "unknown.zip"
    write_minimal(
        unknown,
        {
            "base_commit": "0" * 40,
            "package_kind": "unknown-kind",
            "experiment_id": "TEST",
            "modified_files": [],
        },
    )
    with pytest.raises(ValueError, match="Unknown or invalid package_kind"):
        hardened.verify_package(
            unknown,
            repo_root=None,
            skip_head_match=True,
            hard_limit_mib=1.0,
        )


def test_result_mutation_between_scan_and_copy_is_rejected(tmp_path: Path) -> None:
    result_dir = failed_result(
        tmp_path / "result",
        experiment_id="TEST-RACE",
        base_commit="1" * 40,
    )
    evidence = result_dir / "metrics.csv"
    evidence.write_text("before\n")
    entries = hardened.scan_result_tree(
        result_dir,
        package_kind="experiment-failed",
        max_single_bytes=1024 * 1024,
        selected_sidecar_paths=set(),
        large_file_persistence="persistent_local",
    )
    evidence.write_text("after mutation\n")
    with pytest.raises(ValueError, match="changed while packaging"):
        hardened.copy_result_entries(
            result_dir,
            tmp_path / "stage",
            "TEST-RACE",
            entries,
            sidecar=False,
        )


def test_guard_rejects_runtime_paths_overlapping_tracked_files(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    tracked_dir = repo / "tracked_results"
    tracked_dir.mkdir()
    (tracked_dir / "existing.txt").write_text("tracked\n")
    assert run(["git", "add", "tracked_results/existing.txt"], repo).returncode == 0
    assert run(["git", "commit", "-m", "tracked runtime path"], repo).returncode == 0
    artifact = tmp_path / "artifact.zip"
    checked = run(
        [
            sys.executable,
            "scripts/run_experiment_guard_hardened.py",
            "--experiment-id",
            "TEST-TRACKED-RUNTIME",
            "--repo-root",
            str(repo),
            "--output-root",
            str(tracked_dir),
            "--artifact-output",
            str(artifact),
            "--",
            sys.executable,
            "-c",
            "print('must not run')",
        ],
        repo,
    )
    assert checked.returncode == 2
    assert "overlaps tracked repository content" in checked.stderr
    assert not artifact.exists()


def test_small_foundation_model_bin_is_index_only(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result_dir = failed_result(tmp_path / "failed")
    model = result_dir / "base_model" / "weights.bin"
    model.parent.mkdir()
    model.write_bytes(b"foundation")
    output = tmp_path / "evidence.zip"
    checked = package_failed(repo, result_dir, output)
    assert checked.returncode == 0, checked.stderr
    with zipfile.ZipFile(output) as zf:
        assert "results/TEST-FAIL/base_model/weights.bin" not in zf.namelist()
        index = json.loads(zf.read("LARGE_FILE_INDEX.json"))
        row = next(x for x in index["entries"] if x["path"] == "base_model/weights.bin")
        assert row["reason"] == "foundation_model_forbidden_index_only"
        assert row["role"] == "foundation_model_weight"


def _freshness_cmd(
    repo: Path,
    ledger: Path,
    phase: str,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return run(
        [
            sys.executable,
            "scripts/resolve_main_commit.py",
            "--repo-root",
            str(repo),
            "--expected-sha",
            head(repo),
            "--phase",
            phase,
            "--ledger",
            str(ledger),
            *extra,
        ],
        repo,
    )


def test_three_phase_base_freshness_ledger_passes_on_stable_main(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    ledger = tmp_path / "freshness.json"
    start = _freshness_cmd(repo, ledger, "session_start", "--reset-ledger")
    assert start.returncode == 0, start.stderr
    before_run = _freshness_cmd(repo, ledger, "pre_execution")
    assert before_run.returncode == 0, before_run.stderr
    before_delivery = _freshness_cmd(repo, ledger, "pre_delivery")
    assert before_delivery.returncode == 0, before_delivery.stderr
    payload = json.loads(ledger.read_text())
    assert [row["phase"] for row in payload["checkpoints"]] == [
        "session_start",
        "pre_execution",
        "pre_delivery",
    ]
    assert all(row["status"] == "verified_current" for row in payload["checkpoints"])


def test_base_freshness_detects_remote_advance_before_delivery(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    remote = tmp_path / "remote.git"
    assert run(["git", "init", "--bare", str(remote)], tmp_path).returncode == 0
    assert run(["git", "remote", "set-url", "origin", str(remote)], repo).returncode == 0
    assert run(["git", "push", "-u", "origin", "main"], repo).returncode == 0
    ledger = tmp_path / "freshness.json"
    assert _freshness_cmd(repo, ledger, "session_start", "--reset-ledger").returncode == 0
    assert _freshness_cmd(repo, ledger, "pre_execution").returncode == 0

    old_head = head(repo)
    (repo / "REMOTE_ADVANCE.txt").write_text("advance\n")
    assert run(["git", "add", "REMOTE_ADVANCE.txt"], repo).returncode == 0
    assert run(["git", "commit", "-m", "remote advance"], repo).returncode == 0
    assert run(["git", "push", "origin", "main"], repo).returncode == 0
    assert run(["git", "reset", "--hard", old_head], repo).returncode == 0

    checked = _freshness_cmd(repo, ledger, "pre_delivery")
    assert checked.returncode == 3
    assert "origin/main advanced" in checked.stderr
    payload = json.loads(ledger.read_text())
    assert payload["latest_status"] == "base_advanced"


def test_official_external_sha_can_supply_authoritative_freshness(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    assert run(["git", "remote", "remove", "origin"], repo).returncode == 0
    ledger = tmp_path / "freshness.json"
    checked = _freshness_cmd(
        repo,
        ledger,
        "session_start",
        "--reset-ledger",
        "--authoritative-sha",
        head(repo),
        "--resolution-method",
        "github_commit_api",
    )
    assert checked.returncode == 0, checked.stderr
    payload = json.loads(ledger.read_text())
    row = payload["checkpoints"][0]
    assert row["remote_authoritative"] is True
    assert row["remote_resolution_method"] == "github_commit_api"


def test_user_expected_sha_alone_does_not_fake_remote_freshness(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    assert run(["git", "remote", "remove", "origin"], repo).returncode == 0
    ledger = tmp_path / "freshness.json"
    checked = _freshness_cmd(repo, ledger, "session_start", "--reset-ledger")
    assert checked.returncode == 2
    assert "could not be resolved authoritatively" in checked.stderr.lower()
    assert not ledger.exists()


def test_freshness_ledger_rejects_local_base_change_without_restart(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    ledger = tmp_path / "freshness.json"
    assert _freshness_cmd(repo, ledger, "session_start", "--reset-ledger").returncode == 0
    (repo / "LOCAL_CHANGE.txt").write_text("new base\n")
    assert run(["git", "add", "LOCAL_CHANGE.txt"], repo).returncode == 0
    assert run(["git", "commit", "-m", "local base change"], repo).returncode == 0
    checked = _freshness_cmd(repo, ledger, "pre_execution")
    assert checked.returncode == 2
    assert "changed within the freshness ledger" in checked.stderr
    payload = json.loads(ledger.read_text())
    assert payload["base_commit_used"] != head(repo)


def test_external_authoritative_sha_requires_resolution_method(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    ledger = tmp_path / "freshness.json"
    checked = _freshness_cmd(
        repo,
        ledger,
        "session_start",
        "--reset-ledger",
        "--authoritative-sha",
        head(repo),
    )
    assert checked.returncode == 2
    assert "requires --resolution-method" in checked.stderr
    assert not ledger.exists()
