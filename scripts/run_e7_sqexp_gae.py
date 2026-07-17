#!/usr/bin/env python3
"""Operator entrypoint for EXT-H-E7-SQEXP-GAE-01."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.e7_sqexp_gae import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
