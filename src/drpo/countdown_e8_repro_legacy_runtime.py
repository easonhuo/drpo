#!/usr/bin/env python3
"""Historical evaluation-RNG behavior for the six-cell reproduction phase."""
from __future__ import annotations

import sys
from pathlib import Path

from drpo import countdown_e8_repro_rng_audit_common as audit

audit.activate("legacy_contaminated_v1")

from drpo import countdown_e8_alpha1_c_scan_runtime as _base_runtime  # noqa: E402
from drpo.countdown_e8_repro_contract import validate_worker_cell  # noqa: E402


def _worker_command(args, cell, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--model_path",
        args.model_path,
        "--bank",
        args.bank,
        "--val",
        args.val,
        "--base_config",
        args.base_config,
        "--grid_config",
        args.grid_config,
        "--output_dir",
        str(output_dir),
        "--alpha",
        str(cell.alpha),
        "--c",
        str(cell.c),
        "--seed_offset",
        str(cell.seed_offset),
    ]


_original_worker = _base_runtime.worker


def worker(args) -> int:
    validate_worker_cell(args)
    return _original_worker(args)


_base_runtime._worker_command = _worker_command
_base_runtime.worker = worker
parser = _base_runtime.parser
plan = _base_runtime.plan
smoke = _base_runtime.smoke
run = _base_runtime.run


def main(argv: list[str] | None = None) -> int:
    return _base_runtime.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
