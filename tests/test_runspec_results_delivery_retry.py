from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "scripts" / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import runspec_results_delivery as delivery  # noqa: E402


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


def test_empty_remote_retry_discards_unpushed_local_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    remote = tmp_path / "results.git"
    remote.mkdir()
    git(remote, "init", "--bare", "-q")
    monkeypatch.setenv("DRPO_RESULTS_REMOTE_URL", str(remote))
    monkeypatch.setenv("DRPO_RESULTS_CACHE_DIR", str(tmp_path / "cache"))

    checkout = delivery._prepare_results_checkout(  # noqa: SLF001
        source,
        "easonhuo/drpo-results",
        "ingest/e7",
    )
    stale = checkout / "stale-local-only.txt"
    stale.write_text("not pushed\n", encoding="utf-8")
    git(checkout, "add", "stale-local-only.txt")
    git(checkout, "commit", "-m", "local push failed")
    assert stale.is_file()

    retried = delivery._prepare_results_checkout(  # noqa: SLF001
        source,
        "easonhuo/drpo-results",
        "ingest/e7",
    )
    assert retried == checkout
    assert not stale.exists()
    assert git(checkout, "branch", "--show-current") == "ingest/e7"
    assert git(checkout, "rev-parse", "--verify", "HEAD", check=False) == ""
