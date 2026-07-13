from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "scripts" / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import claude_runspec_guard as guard  # noqa: E402


def test_strict_guard_allows_only_canonical_results_upload() -> None:
    assert guard.bash_allowed(
        "python scripts/agent/upload_runspec_result.py --run-id E7-TEST-1"
    )
    assert not guard.bash_allowed(
        "git push git@github.com:easonhuo/drpo-results.git HEAD:ingest/e7"
    )
