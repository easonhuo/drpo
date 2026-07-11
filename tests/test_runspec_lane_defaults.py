from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "scripts" / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from configure_claude_workspace import (  # noqa: E402
    defaults_for_lane,
    validate_active_lane,
)


def test_active_server_lane_defaults_match_e7_e8_domains():
    assert validate_active_lane("e7") == "e7"
    assert validate_active_lane("E8") == "e8"
    assert defaults_for_lane("e7") == (["EXT-H-E7-"], ["EXT-C-E8-"])
    assert defaults_for_lane("e8") == (["EXT-C-E8-"], ["EXT-H-E7-"])


def test_inactive_e1_lane_is_rejected():
    with pytest.raises(Exception, match="unsupported active server lane"):
        validate_active_lane("e1")
