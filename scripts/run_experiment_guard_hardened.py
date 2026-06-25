#!/usr/bin/env python3
"""Run a DRPO experiment under the hardened foreground supervisor."""
from __future__ import annotations

import sys

try:
    from artifact_protocol_hardened import guard_main
except ModuleNotFoundError as exc:
    print(
        "ERROR: scripts/artifact_protocol_hardened.py is required; "
        "refusing to run without the hardened supervision protocol.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


if __name__ == "__main__":
    raise SystemExit(guard_main())
