#!/usr/bin/env python3
"""Verify a DRPO durable artifact with the hardened protocol."""
from __future__ import annotations

import sys

try:
    from artifact_protocol_hardened import verify_main
except ModuleNotFoundError as exc:
    print(
        "ERROR: scripts/artifact_protocol_hardened.py is required; "
        "refusing to use the legacy verifier.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


if __name__ == "__main__":
    raise SystemExit(verify_main())
