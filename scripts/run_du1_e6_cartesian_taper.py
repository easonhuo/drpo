#!/usr/bin/env python3
"""One-click canonical launcher target for D-U1 E6 Cartesian + TAPER."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.du1_e6_cartesian_taper_v4 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
