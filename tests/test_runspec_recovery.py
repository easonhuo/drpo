from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN_LANE = ROOT / "scripts" / "agent" / "run_lane.py"


def run(cmd: list[str], *, cwd: Path, check: bool = True):
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


def make_repo(
    tmp_path: Path,
    *,
    stderr_text: str = "transient interruption",
    make_checkpoint: bool = True,
) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    run(["git", "init", "-q"], cwd=repo)
    run(["git", "config", "user.name", "Recovery Test"], cwd=repo)
    run(["git", "config", "user.email", "recovery@example.invalid"], cwd=repo)

    scripts = repo / "scripts" / "demo"
    scripts.mkdir(parents=True)
    checkpoint_line = (
        "printf 'checkpoint\\n' > outputs/e8/recovery/checkpoint.state"
        if make_checkpoint
        else ":"
    )
    initial = scripts / "initial.sh"
    initial.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p outputs/e8/recovery/logs\n"
        f"{checkpoint_line}\n"
        f"printf '%s\\n' {stderr_text!r} >&2\n"
        "exit 75\n",
        encoding="utf-8",
    )
    initial.chmod(0o755)
    resume = scripts / "resume.sh"
    resume.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "test -f outputs/e8/recovery/checkpoint.state\n"
        "printf '{\"ok\": true}\\n' > outputs/e8/recovery/summary.json\n"
        "printf '{\"audit\": true}\\n' > outputs/e8/recovery/audit.json\n"
        "printf 'resumed\\n' > outputs/e8/recovery/logs/resume.txt\n",
        encoding="utf-8",
    )
    resume.chmod(0o755)

    (repo / "experiments").mkdir()
    (repo / "experiments" / "registry.yaml").write_text(
        "schema_version: 2\nexperiments:\n  - experiment_id: EXT-C-E8-RECOVERY-01\n",
        encoding="utf-8",
    )
    (repo / ".agent_lane.yaml").write_text(
        "lane: e8\nforbid_cross_lane: true\nallowed_experiment_prefixes:\n  - EXT-C-E8-\n",
        encoding="utf-8",
    )
    (repo / "runspecs" / "ready").mkdir(parents=True)
    run(["git", "add", "."], cwd=repo)
    run(["git", "commit", "-q", "-m", "base"], cwd=repo)
    commit = run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()

    spec = {
        "version": 1,
        "run_id": "E8_RECOVERY_20260712",
        "lane": "e8",
        "priority": 10,
        "created_at": "2026-07-12T00:00:00Z",
        "experiment_id": "EXT-C-E8-RECOVERY-01",
        "repo_commit": commit,
        "entrypoint": {
            "cwd": "repo_root",
            "command": "bash scripts/demo/initial.sh",
        },
        "recovery": {
            "enabled": True,
            "max_attempts": 2,
            "resume_command": "bash scripts/demo/resume.sh",
            "retryable_exit_codes": [75],
            "checkpoint_globs": ["outputs/e8/recovery/checkpoint.state"],
            "backoff_seconds": 0,
        },
        "policy": {
            "existing_script_required": True,
            "forbid_new_launcher": True,
            "forbid_hparam_change": True,
            "forbid_cross_lane": True,
        },
        "outputs": {
            "run_dir": "outputs/e8/recovery",
            "summary_file": "outputs/e8/recovery/summary.json",
            "audit_file": "outputs/e8/recovery/audit.json",
        },
        "success_criteria": [
            "outputs/e8/recovery/summary.json exists",
            "outputs/e8/recovery/audit.json exists",
        ],
        "artifacts": {
            "package_policy": "manifest_only",
            "include": [
                "outputs/e8/recovery/summary.json",
                "outputs/e8/recovery/audit.json",
                "outputs/e8/recovery/logs/*.txt",
            ],
            "exclude": ["**/*.pt", "**/*.safetensors", "**/*.bin"],
            "max_package_size_mb": 10,
            "fail_if_excluded_matched": True,
            "fail_if_package_too_large": True,
        },
    }
    (repo / "runspecs" / "ready" / "E8_RECOVERY_20260712.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False),
        encoding="utf-8",
    )
    return repo


def test_retryable_failure_resumes_from_fresh_checkpoint(tmp_path: Path):
    repo = make_repo(tmp_path)
    proc = run(
        ["python", str(RUN_LANE), "--repo-root", str(repo), "--once", "--json"],
        cwd=repo,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASS"
    assert payload["attempts"] == 2
    assert payload["recovery_used"] is True
    report = json.loads((repo / payload["recovery_report"]).read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert [row["outcome"] for row in report["attempts"]] == ["failed", "passed"]
    assert report["attempts"][0]["retry_decision"] == "retry"
    assert (
        repo
        / ".runspec_state/logs/E8_RECOVERY_20260712/attempt-01/STDERR.log"
    ).is_file()
    assert (
        repo
        / ".runspec_state/logs/E8_RECOVERY_20260712/attempt-02/STDOUT.log"
    ).is_file()


def test_oom_log_blocks_retry_even_when_exit_code_is_allowlisted(tmp_path: Path):
    repo = make_repo(tmp_path, stderr_text="CUDA out of memory")
    proc = run(
        ["python", str(RUN_LANE), "--repo-root", str(repo), "--once", "--json"],
        cwd=repo,
        check=False,
    )
    assert proc.returncode != 0
    assert not (repo / "outputs/e8/recovery/logs/resume.txt").exists()
    report = json.loads(
        (
            repo
            / ".runspec_state/logs/E8_RECOVERY_20260712/RECOVERY_REPORT.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "failed"
    assert report["final_reason"] == "hard_stop:out_of_memory"
    assert (repo / ".runspec_state/failed/E8_RECOVERY_20260712.yaml").is_file()


def test_retry_requires_a_fresh_checkpoint(tmp_path: Path):
    repo = make_repo(tmp_path, make_checkpoint=False)
    proc = run(
        ["python", str(RUN_LANE), "--repo-root", str(repo), "--once", "--json"],
        cwd=repo,
        check=False,
    )
    assert proc.returncode != 0
    assert not (repo / "outputs/e8/recovery/logs/resume.txt").exists()
    report = json.loads(
        (
            repo
            / ".runspec_state/logs/E8_RECOVERY_20260712/RECOVERY_REPORT.json"
        ).read_text(encoding="utf-8")
    )
    assert report["final_reason"] == "no_fresh_checkpoint"


def test_invalid_recovery_attempt_limit_is_rejected_before_training(tmp_path: Path):
    repo = make_repo(tmp_path)
    spec_path = repo / "runspecs/ready/E8_RECOVERY_20260712.yaml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec["recovery"]["max_attempts"] = 4
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    proc = run(
        ["python", str(RUN_LANE), "--repo-root", str(repo), "--once", "--json"],
        cwd=repo,
        check=False,
    )
    assert proc.returncode != 0
    assert "between 2 and 3" in proc.stdout
    assert not (repo / "outputs/e8/recovery/checkpoint.state").exists()
