
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import handoff_authority as authority  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def test_side_branch_delta_maps_to_first_parent_merge_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "baseline.txt").write_text("base\n", encoding="utf-8")
    activation = _commit(repo, "activation")

    _git(repo, "switch", "-c", "feature")
    relative = "docs/handoff_deltas/EXAMPLE/HANDOFF_DELTA.yaml"
    delta = repo / relative
    delta.parent.mkdir(parents=True)
    delta.write_text("schema_version: 3\n", encoding="utf-8")
    first_add = _commit(repo, "add delta")

    _git(repo, "switch", "main")
    (repo / "main.txt").write_text("main\n", encoding="utf-8")
    _commit(repo, "main work")
    _git(repo, "merge", "--no-ff", "feature", "-m", "merge feature")
    head = _git(repo, "rev-parse", "HEAD")

    positions = authority._first_parent_positions(repo, activation, head)
    assert first_add not in positions
    assert authority._first_parent_integration_commit(
        repo,
        positions=positions,
        first_add=first_add,
        relative=relative,
    ) == head
