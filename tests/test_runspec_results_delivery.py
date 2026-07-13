from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "scripts" / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import runspec_results_delivery as delivery  # noqa: E402
from runspec_lib import RunSpecError, sha256_file  # noqa: E402


def git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(proc.stderr or proc.stdout)
    return proc.stdout.strip()


def _artifact_row(repo: Path, rel: str) -> dict[str, object]:
    path = repo / rel
    return {
        "path": rel,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def make_completed_run(tmp_path: Path) -> tuple[Path, dict, dict]:
    repo = tmp_path / "source"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "RunSpec Test")
    git(repo, "config", "user.email", "runspec@example.invalid")
    script = repo / "scripts" / "run.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    git(repo, "add", "scripts/run.sh")
    git(repo, "commit", "-q", "-m", "base")

    summary = repo / "outputs" / "e7" / "run" / "RUN_SUMMARY.json"
    completed = (
        repo
        / "outputs"
        / "e7"
        / "run"
        / "branches"
        / "branch-1"
        / "COMPLETED.json"
    )
    figure = repo / "outputs" / "e7" / "run" / "curve.png"
    summary.parent.mkdir(parents=True)
    completed.parent.mkdir(parents=True)
    summary.write_text('{"completed": 1, "failed": 0}\n', encoding="utf-8")
    completed.write_text('{"return_code": 0}\n', encoding="utf-8")
    figure.write_bytes(b"not-a-real-image")

    run_id = "E7-DELIVERY-TEST-1"
    spec = {
        "version": 1,
        "run_id": run_id,
        "lane": "e7",
        "experiment_id": "EXT-H-E7-TEST-01",
        "entrypoint": {"cwd": "repo_root", "command": "bash scripts/run.sh"},
        "policy": {
            "existing_script_required": True,
            "forbid_new_launcher": True,
            "forbid_hparam_change": True,
            "forbid_cross_lane": True,
        },
        "outputs": {
            "run_dir": "outputs/e7/run",
            "summary_file": "outputs/e7/run/RUN_SUMMARY.json",
        },
        "artifacts": {
            "package_policy": "manifest_only",
            "include": ["outputs/e7/run/**"],
        },
        "delivery": {
            "enabled": True,
            "auto": True,
            "mode": "results_repo",
            "repository": "easonhuo/drpo-results",
            "branch": "ingest/e7",
            "export_profile": "manifest_text_v1",
            "max_total_size_mb": 30,
            "max_file_size_mb": 10,
        },
        "publish": {"enabled": False, "auto": False},
    }
    done = repo / ".runspec_state" / "done" / f"{run_id}.yaml"
    done.parent.mkdir(parents=True)
    done.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    artifact = {
        "schema_version": 1,
        "created_at": "2026-07-13T00:00:00+00:00",
        "run_id": run_id,
        "lane": "e7",
        "experiment_id": "EXT-H-E7-TEST-01",
        "repo_commit": git(repo, "rev-parse", "HEAD"),
        "zip_path": f"runspec_artifacts/{run_id}_results.zip",
        "zip_sha256": "a" * 64,
        "included": [
            _artifact_row(repo, "outputs/e7/run/RUN_SUMMARY.json"),
            _artifact_row(
                repo,
                "outputs/e7/run/branches/branch-1/COMPLETED.json",
            ),
            _artifact_row(repo, "outputs/e7/run/curve.png"),
        ],
    }
    artifact_path = repo / "runspec_artifacts" / f"{run_id}_manifest.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return repo, spec, artifact


def test_delivery_contract_is_lane_bound_and_excludes_legacy_publish() -> None:
    spec = {
        "delivery": {
            "enabled": True,
            "auto": True,
            "mode": "results_repo",
            "repository": "easonhuo/drpo-results",
            "branch": "ingest/e8",
        }
    }
    with pytest.raises(RunSpecError, match="ingest/e7"):
        delivery.validate_delivery_block(spec, "e7")

    spec["delivery"]["branch"] = "ingest/e7"
    spec["publish"] = {"enabled": True}
    with pytest.raises(RunSpecError, match="may not both be enabled"):
        delivery.validate_delivery_block(spec, "e7")


def test_review_export_compacts_branch_json_and_omits_non_text(tmp_path: Path) -> None:
    repo, spec, artifact = make_completed_run(tmp_path)
    policy = delivery.validate_delivery_block(spec, "e7")
    exported = delivery.export_review_package(repo, spec, artifact, policy)
    package = exported["package_dir"]

    assert (package / "files" / "outputs/e7/run/RUN_SUMMARY.json").is_file()
    branch_rows = [
        json.loads(line)
        for line in (package / "BRANCH_RESULTS.jsonl").read_text().splitlines()
    ]
    assert branch_rows[0]["source_path"].endswith("COMPLETED.json")
    source_manifest = json.loads(
        (package / "SOURCE_ARTIFACT_MANIFEST.json").read_text()
    )
    assert source_manifest["omitted_non_text"] == ["outputs/e7/run/curve.png"]
    ready = json.loads((package / "READY_FOR_REVIEW.json").read_text())
    result_manifest = json.loads((package / "RESULT_MANIFEST.json").read_text())
    assert ready["manifest_sha256"] == result_manifest["manifest_sha256"]


def test_results_repo_upload_is_append_only_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _spec, artifact = make_completed_run(tmp_path)
    remote = tmp_path / "results.git"
    remote.mkdir()
    git(remote, "init", "--bare", "-q")
    monkeypatch.setenv("DRPO_RESULTS_REMOTE_URL", str(remote))
    monkeypatch.setenv("DRPO_RESULTS_CACHE_DIR", str(tmp_path / "cache"))
    source_head = git(repo, "rev-parse", "HEAD")

    first = delivery.deliver_completed_run(repo, "E7-DELIVERY-TEST-1")
    assert first["status"] == "PASS"
    assert first["result_path"] == "runs/e7/E7-DELIVERY-TEST-1"
    assert git(repo, "rev-parse", "HEAD") == source_head

    inspect = tmp_path / "inspect"
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "--branch",
            "ingest/e7",
            str(remote),
            str(inspect),
        ],
        check=True,
    )
    target = inspect / "runs" / "e7" / "E7-DELIVERY-TEST-1"
    assert (target / "READY_FOR_REVIEW.json").is_file()
    assert not (target / "curve.png").exists()

    second = delivery.deliver_completed_run(repo, "E7-DELIVERY-TEST-1")
    assert second["status"] == "ALREADY_DELIVERED"
    assert second["idempotent"] is True

    summary = repo / "outputs" / "e7" / "run" / "RUN_SUMMARY.json"
    summary.write_text('{"completed": 2, "failed": 0}\n', encoding="utf-8")
    artifact["included"][0] = _artifact_row(
        repo,
        "outputs/e7/run/RUN_SUMMARY.json",
    )
    artifact_path = repo / "runspec_artifacts" / "E7-DELIVERY-TEST-1_manifest.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RunSpecError, match="RESULT_CONFLICT"):
        delivery.deliver_completed_run(repo, "E7-DELIVERY-TEST-1")
