from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCRIPT = ROOT / "scripts" / "package_experiment.py"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_experiment_package.py"
GUARD_SCRIPT = ROOT / "scripts" / "run_experiment_guard.py"


def run(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=check)


def init_repo(path: Path) -> str:
    path.mkdir(parents=True)
    run("git", "init", "-q", cwd=path)
    run("git", "config", "user.email", "test@example.com", cwd=path)
    run("git", "config", "user.name", "Test User", cwd=path)
    (path / "tracked.txt").write_text("before\n")
    run("git", "add", "tracked.txt", cwd=path)
    run("git", "commit", "-q", "-m", "base", cwd=path)
    return run("git", "rev-parse", "HEAD", cwd=path).stdout.strip()


def test_registry_declares_durable_artifact_governance() -> None:
    registry = yaml.safe_load((ROOT / "experiments" / "registry.yaml").read_text())
    assert registry["schema_version"] == 2
    assert registry["artifact_protocol"]["governance_id"] == "GOV-EXP-ARTIFACT-01"
    assert registry["rules"]["require_durable_artifact_before_completed_claim"] is True
    assert registry["rules"]["require_supervised_run_in_ephemeral_environment"] is True
    assert "delivered" in registry["allowed_execution_states"]


def test_governance_documents_define_completion_gate() -> None:
    agents = (ROOT / "AGENTS.md").read_text()
    handoff = (ROOT / "docs" / "handoff.md").read_text()
    protocol = (ROOT / "docs" / "formal_experiment_artifact_protocol.md").read_text()
    assert "raw_complete` is not a completed formal result" in agents
    assert "GOV-EXP-ARTIFACT-01" in handoff
    assert "packaged + delivered" in handoff
    assert "Starting a background process" in protocol


def test_build_and_verify_final_package(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    base_sha = init_repo(repo)
    (repo / "tracked.txt").write_text("after\n")

    result_dir = repo / "results" / "E-TEST"
    result_dir.mkdir(parents=True)
    (result_dir / "RUN_COMPLETE.json").write_text(json.dumps({"complete": True}))
    (result_dir / "TERMINAL_AUDIT.json").write_text(
        json.dumps(
            {
                "task_performance": "audited",
                "support_or_variance_boundary": "audited",
                "nan_inf_numerical_failure": "audited",
            }
        )
    )
    (result_dir / "run_manifest.json").write_text(json.dumps({"base_commit": base_sha}))
    output = tmp_path / "final.zip"

    build = run(
        sys.executable,
        str(PACKAGE_SCRIPT),
        "--repo-root",
        str(repo),
        "--experiment-id",
        "E-TEST",
        "--package-kind",
        "experiment-final",
        "--result-dir",
        str(result_dir),
        "--output",
        str(output),
        "--test-command",
        "python3 -m pytest -q",
        cwd=ROOT,
    )
    assert output.is_file(), build.stderr

    verified = run(
        sys.executable,
        str(VERIFY_SCRIPT),
        str(output),
        "--repo-root",
        str(repo),
        cwd=ROOT,
    )
    report = json.loads(verified.stdout)
    assert report["verified"] is True
    assert report["base_commit"] == base_sha

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "modified_files/tracked.txt" in names
        assert "results/E-TEST/RUN_COMPLETE.json" in names
        assert archive.read("BASE_COMMIT.txt").decode() == base_sha + "\n"


def test_guard_creates_raw_complete_recovery_artifact(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    init_repo(repo)
    scripts = repo / "scripts"
    scripts.mkdir()
    shutil.copy2(PACKAGE_SCRIPT, scripts / "package_experiment.py")
    shutil.copy2(GUARD_SCRIPT, scripts / "run_experiment_guard.py")
    run("git", "add", "scripts", cwd=repo)
    run("git", "commit", "-q", "-m", "add scripts", cwd=repo)

    output_root = repo / "run_output"
    artifact = tmp_path / "raw_complete.zip"
    command = (
        "from pathlib import Path; "
        f"p=Path({str(output_root)!r}); p.mkdir(parents=True, exist_ok=True); "
        "(p/'payload.json').write_text('{\"ok\": true}')"
    )
    guarded = run(
        sys.executable,
        str(scripts / "run_experiment_guard.py"),
        "--experiment-id",
        "E-GUARD",
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
        command,
        cwd=repo,
    )
    assert guarded.returncode == 0
    assert (output_root / "RUN_RAW_COMPLETE.json").is_file()
    assert artifact.is_file()

    verified = run(
        sys.executable,
        str(VERIFY_SCRIPT),
        str(artifact),
        "--skip-head-match",
        cwd=ROOT,
    )
    assert json.loads(verified.stdout)["verified"] is True
