#!/usr/bin/env python3
"""Repository entry point for the E7 canonical-agent two-dataset pilot."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from drpo.e7_canonical_two_dataset import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
