#!/usr/bin/env python3
"""Fail-closed execution contract for the E8 reproducibility audit."""
from __future__ import annotations

from typing import Any

from drpo import countdown_e8_repro_rng_audit_common as audit


def worker_cell_tuple(args: Any) -> tuple[float, float, int]:
    return float(args.alpha), float(args.c), int(args.seed_offset)


def validate_worker_cell(args: Any) -> None:
    cell = worker_cell_tuple(args)
    if cell not in audit.AUDIT_CELL_SPECS:
        raise ValueError(
            "Worker cell is outside the frozen six-cell reproducibility audit: "
            f"{cell}"
        )
