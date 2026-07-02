#!/usr/bin/env python3
"""Repository wrapper for the EXT-H-E7-BENCH-01 parallel runner."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drpo.e7_bench import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
