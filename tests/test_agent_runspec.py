from __future__ import annotations

import json
import subprocess
import textwrap
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VALIDATE = ROOT / "scripts" / "agent" / "validate_runspec.py"
CLAIM = ROOT / "scripts" / "agent" / "claim_next_runspec.py"
RUN_LANE = ROOT / "scripts" / "agent" / "run_lane.py"
PACKAGE = ROOT / "scripts" / "agent" / "package_runspec_artifacts.py"


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


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    run(["git", "init", "-q"], cwd=repo)
    run(["git", "config", "user.name", "RunSpec Test"], cwd=repo)
    run(["git", "config", "user.email", "runspec@example.invalid"], cwd=repo)
    (repo / "scripts" / "demo").mkdir(parents=True)
    runner = repo / "scripts" / "demo" / "run_demo.sh"
    runner.write_text(
        textwrap.dedent(
            """
            #!/usr/bin/env bash
            set -euo pipefail
            mkdir -p outputs/e8/demo/logs outputs/e8/demo/metrics
            printf '{"ok": true}\n' > outputs/e8/demo/summary.json
            printf '{"audit": true}\n' > outputs/e8/demo/audit.json
            printf '{"metric": 1}\n' > outputs/e8/demo/metrics/metric.json
            echo done > outputs/e8/demo/logs/stdout.txt
            echo should-not-package > outputs/e8/demo/model.pt
            """
        ).lstrip()
    )
    runner.chmod(0o755)
    (repo / "experiments").mkdir()
    (repo / "experiments" / "registry.yaml").write_text(
        "schema_version: 2\nexperiments:\n  - experiment_id: EXT-C-E8-DEMO-01\n"
    )
    (repo / "runspecs" / "ready").mkdir(parents=True)
    (repo / ".agent_lane.yaml").write_text(
        "lane: e8\nforbid_cross_lane: true\nallowed_experiment_prefixes:\n  - EXT-C-E8-\n"
    )
    run(["git", "add", "."], cwd=repo)
    run(["git", "commit", "-q", "-m", "base"], cwd=repo)
    commit = run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    spec = {
        "version": 1,
        "run_id": "E8_DEMO_20260711",
        "lane": "e8",
        "priority": 10,
        "created_at": "2026-07-11T00:00:00Z",
        "experiment_id": "EXT-C-E8-DEMO-01",
        "repo_commit": commit,
        "entrypoint": {
            "cwd": "repo_root",
            "command": "bash scripts/demo/run_demo.sh",
        },
        "policy": {
            "existing_script_required": True,
            "forbid_new_launcher": True,
            "forbid_hparam_change": True,
            "forbid_cross_lane": True,
        },
        "outputs": {
            "run_dir": "outputs/e8/demo",
            "summary_file": "outputs/e8/demo/summary.json",
            "audit_file": "outputs/e8/demo/audit.json",
        },
        "success_criteria": [
            "outputs/e8/demo/summary.json exists",
            "outputs/e8/demo/audit.json exists",
        ],
        "artifacts": {
            "package_policy": "manifest_only",
            "include": [
                "outputs/e8/demo/summary.json",
                "outputs/e8/demo/audit.json",
                "outputs/e8/demo/metrics/*.json",
                "outputs/e8/demo/logs/*.txt",
            ],
            "exclude": ["**/*.pt", "**/*.safetensors", "**/*.bin"],
            "max_package_size_mb": 10,
            "fail_if_excluded_matched": True,
            "fail_if_package_too_large": True,
        },
    }
    (repo / "runspecs" / "ready" / "E8_DEMO_20260711.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False)
    )
    return repo


def test_validate_claim_and_run_lane_packages_only_allowed_artifacts(tmp_path: Path):
    repo = make_repo(tmp_path)
    spec = repo / "runspecs" / "ready" / "E8_DEMO_20260711.yaml"

    validated = run(["python", str(VALIDATE), "--repo-root", str(repo), str(spec), "--json"], cwd=repo)
    assert json.loads(validated.stdout)["status"] == "PASS"

    claimed = run(["python", str(CLAIM), "--repo-root", str(repo), "--json"], cwd=repo)
    claimed_payload = json.loads(claimed.stdout)
    assert claimed_payload["status"] == "PASS"
    assert claimed_payload["claimed_path"] == ".runspec_state/claimed/E8_DEMO_20260711.yaml"
    assert (repo / "runspecs" / "ready" / "E8_DEMO_20260711.yaml").is_file()

    executed = run(
        ["python", str(RUN_LANE), "--repo-root", str(repo), "--run-id", "E8_DEMO_20260711", "--json"],
        cwd=repo,
        check=False,
    )
    # run_lane claims; this run_id is already claimed, so it should fail instead of double-running.
    assert executed.returncode != 0

    run_claimed = run(
        [
            "python",
            str(ROOT / "scripts" / "agent" / "run_claimed_runspec.py"),
            "--repo-root",
            str(repo),
            "--run-id",
            "E8_DEMO_20260711",
            "--json",
        ],
        cwd=repo,
    )
    payload = json.loads(run_claimed.stdout)
    assert payload["status"] == "PASS"
    zip_path = repo / payload["artifact_zip"]
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "outputs/e8/demo/summary.json" in names
    assert "outputs/e8/demo/audit.json" in names
    assert "outputs/e8/demo/model.pt" not in names
    manifest = json.loads((repo / "runspec_artifacts" / "E8_DEMO_20260711_manifest.json").read_text())
    assert all(not row["path"].endswith("model.pt") for row in manifest["included"])


def test_cross_lane_is_rejected(tmp_path: Path):
    repo = make_repo(tmp_path)
    (repo / ".agent_lane.yaml").write_text("lane: e1\nforbid_cross_lane: true\n")
    spec = repo / "runspecs" / "ready" / "E8_DEMO_20260711.yaml"
    proc = run(
        ["python", str(VALIDATE), "--repo-root", str(repo), str(spec), "--json"],
        cwd=repo,
        check=False,
    )
    assert proc.returncode != 0
    assert "does not match workspace lane" in proc.stdout


def test_artifact_packaging_rejects_model_like_include(tmp_path: Path):
    repo = make_repo(tmp_path)
    run(["bash", "scripts/demo/run_demo.sh"], cwd=repo)
    spec_path = repo / "runspecs" / "ready" / "E8_DEMO_20260711.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    spec["artifacts"]["include"].append("outputs/e8/demo/model.pt")
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    proc = run(
        ["python", str(PACKAGE), "--repo-root", str(repo), "--runspec", str(spec_path), "--json"],
        cwd=repo,
        check=False,
    )
    assert proc.returncode != 0
    assert "excluded/model-like" in proc.stdout
