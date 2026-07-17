#!/usr/bin/env python3
"""Runtime adapter for the E8 paper-aligned linear c scans."""
from __future__ import annotations

import sys
from pathlib import Path

from drpo import countdown_e8_alpha1_highc_scan_common as highc


def _pop_tau(argv: list[str]) -> float | None:
    if "--tau" not in argv:
        return None
    index = argv.index("--tau")
    if index + 1 >= len(argv): raise ValueError("--tau requires a value")
    value = float(argv[index + 1])
    del argv[index : index + 2]
    return value


def _grid_config_from_argv(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token == "--grid_config" and index + 1 < len(argv):
            return argv[index + 1]
    return None


_worker_tau = _pop_tau(sys.argv)
_grid_config = _grid_config_from_argv(sys.argv[1:])
if _grid_config is None:
    highc.activate()
else:
    highc.activate_for_grid_config(_grid_config)
if _worker_tau is not None:
    highc.set_active_tau(_worker_tau)

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
        "--tau",
        str(getattr(cell, "tau", 0.0)),
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
