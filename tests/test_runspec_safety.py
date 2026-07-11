from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "scripts" / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import runspec_safety as safety  # noqa: E402
from runspec_lib import RunSpecError  # noqa: E402


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr or proc.stdout)
    return proc.stdout.strip()


def init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "RunSpec Test")
    git(repo, "config", "user.email", "runspec@example.invalid")
    return repo


def test_completed_run_id_is_not_claimed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo = init_repo(tmp_path)
    ready = repo / "runspecs" / "ready" / "RUN-1.yaml"
    ready.parent.mkdir(parents=True)
    ready.write_text("run_id: RUN-1\n", encoding="utf-8")
    done = repo / ".runspec_state" / "done" / "RUN-1.yaml"
    done.parent.mkdir(parents=True)
    done.write_text("run_id: RUN-1\n", encoding="utf-8")
    spec = {
        "run_id": "RUN-1",
        "lane": "e8",
        "priority": 10,
        "created_at": "2026-07-11T00:00:00Z",
    }
    monkeypatch.setattr(safety, "iter_ready_specs", lambda _repo: [ready])
    monkeypatch.setattr(safety, "validate_runspec_safe", lambda *_a, **_k: spec)
    with pytest.raises(RunSpecError, match="ALREADY_COMPLETED"):
        safety.claim_next_runspec_safe(
            repo,
            lane_config={"lane": "e8"},
            run_id="RUN-1",
        )


def test_artifact_symlink_is_rejected(tmp_path: Path):
    repo = init_repo(tmp_path)
    outside = tmp_path / "checkpoint.json"
    outside.write_text("{}\n", encoding="utf-8")
    output = repo / "outputs" / "e8"
    output.mkdir(parents=True)
    (output / "summary.json").symlink_to(outside)
    spec = {"artifacts": {"include": ["outputs/e8/summary.json"]}}
    with pytest.raises(RunSpecError, match="symlink"):
        safety.reject_symlink_artifacts(repo, spec)


def test_protected_paths_allow_docs_only_commit_but_reject_script_drift(
    tmp_path: Path,
):
    repo = init_repo(tmp_path)
    script = repo / "scripts" / "run.sh"
    script.parent.mkdir()
    script.write_text("#!/bin/sh\necho one\n", encoding="utf-8")
    (repo / "README.md").write_text("one\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    pinned = git(repo, "rev-parse", "HEAD")
    spec = {
        "repo_commit": pinned,
        "entrypoint": {"command": "bash scripts/run.sh"},
        "provenance": {"commit_policy": "protected_paths_unchanged"},
    }
    (repo / "README.md").write_text("two\n", encoding="utf-8")
    git(repo, "add", "README.md")
    git(repo, "commit", "-q", "-m", "docs")
    safety.validate_provenance(repo, spec)
    script.write_text("#!/bin/sh\necho two\n", encoding="utf-8")
    git(repo, "add", "scripts/run.sh")
    git(repo, "commit", "-q", "-m", "change script")
    with pytest.raises(RunSpecError, match="protected experiment files changed"):
        safety.validate_provenance(repo, spec)


def test_workspace_configuration_uses_fixed_lane_branch(tmp_path: Path):
    repo = init_repo(tmp_path)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    branch = safety.ensure_workspace_branch(repo, "e8")
    assert branch == "dev/server-e8"
    assert git(repo, "branch", "--show-current") == branch


def test_publish_branch_must_match_configured_workspace_branch():
    spec = {"publish": {"dev_branch": "dev/e8-other"}}
    lane = {"lane": "e8", "publish_branch": "dev/server-e8"}
    with pytest.raises(RunSpecError, match="workspace publish_branch"):
        safety.validate_fixed_publish_branch(spec, lane)
