#!/usr/bin/env python3
"""Thin auto-launcher adapter; resource selection remains the canonical E8 implementation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from drpo import countdown_e8_paper_aligned_lambda_minimal_common as paper

paper.activate()

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_LAUNCHER = _REPO_ROOT / "scripts" / "run_countdown_e8_oracle_offline_v2_alpha1_c_scan_auto.py"
_SPEC = importlib.util.spec_from_file_location("_e8_alpha1_c_scan_auto_base", _BASE_LAUNCHER)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load base launcher: {_BASE_LAUNCHER}")
_base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_base)

_base.ADAPTER_ID = "e8_paper_aligned_lambda_minimal_cuda_v1"


def _core_command(args, command: str, *, selected_ids: list[str]) -> list[str]:
    result = [
        sys.executable,
        str(_REPO_ROOT / "src" / "drpo" / "countdown_e8_paper_aligned_lambda_minimal_runtime.py"),
        command,
        "--model_path",
        str(Path(args.model_path).resolve()),
        "--work_dir",
        str(Path(args.work_dir).resolve()),
        "--bank",
        str(Path(args.bank).resolve()),
        "--val",
        str(Path(args.val).resolve()),
        "--base_config",
        str(Path(args.base_config).resolve()),
        "--grid_config",
        str(Path(args.grid_config).resolve()),
    ]
    if command == "run":
        result.extend(
            [
                "--gpus",
                ",".join(selected_ids),
                "--runtime-slots-per-gpu",
                str(paper.load_yaml(args.grid_config)["execution"]["parallel_cells_per_gpu"]),
            ]
        )
    return result


_base._core_command = _core_command
build_parser = _base.build_parser


def main(argv: list[str] | None = None) -> int:
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
