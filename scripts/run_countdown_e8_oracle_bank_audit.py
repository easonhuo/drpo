#!/usr/bin/env python3
"""Run the standardized audit for a generated Countdown E8 oracle bank v2 corpus."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drpo.countdown_e8_oracle_bank_audit import main  # noqa: E402


if __name__ == "__main__":
    main()
