#!/usr/bin/env python3
"""Runtime adapter for the E8 alpha=1 high-c extension pilot."""
from __future__ import annotations

import sys
from pathlib import Path

from drpo import countdown_e8_alpha1_highc_scan_common as highc

highc.activate()

from drpo import countdown_e8_alpha1_c_scan_runtime as _base_runtime  # noqa: E402


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


_base_runtime._worker_command = _worker_command

parser = _base_runtime.parser
plan = _base_runtime.plan
smoke = _base_runtime.smoke
run = _base_runtime.run
worker = _base_runtime.worker


def main(argv: list[str] | None = None) -> int:
    return _base_runtime.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
