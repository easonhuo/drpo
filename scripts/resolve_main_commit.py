#!/usr/bin/env python3
"""Resolve DRPO main with Git, never a cached web commit listing."""
from artifact_protocol_hardened import resolve_main_cli


if __name__ == "__main__":
    raise SystemExit(resolve_main_cli())
