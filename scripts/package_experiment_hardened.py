#!/usr/bin/env python3
"""Build and atomically verify a hardened DRPO artifact."""
from __future__ import annotations

import sys

try:
    from artifact_protocol_hardened import package_main
except ModuleNotFoundError as exc:
    print(
        "ERROR: scripts/artifact_protocol_hardened.py is required; "
        "refusing to fall back to the legacy packaging protocol.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


if __name__ == "__main__":
    raise SystemExit(package_main())
