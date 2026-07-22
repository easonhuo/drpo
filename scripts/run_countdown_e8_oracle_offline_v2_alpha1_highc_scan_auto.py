#!/usr/bin/env python3
"""Autotuned launcher adapter for the E8 paper-aligned taper-family scans."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from drpo import countdown_e8_alpha1_highc_scan_common as highc

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_LAUNCHER = (
    _REPO_ROOT
    / "scripts"
    / "run_countdown_e8_oracle_offline_v2_alpha1_c_scan_auto.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_e8_alpha1_c_scan_auto_base", _BASE_LAUNCHER
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load base launcher: {_BASE_LAUNCHER}")
_base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_base)


def _grid_config_from_argv(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token == "--grid_config" and index + 1 < len(argv):
            return argv[index + 1]
    return None


def activate_config_driven_profile(path: str | Path) -> dict[str, Any]:
    """Install and activate the reviewed grid before resource planning."""

    from drpo import countdown_e8_alpha1_highc_scan_runtime as runtime_adapter

    profile = runtime_adapter.install_config_driven_profile(path)
    highc.activate_for_grid_config(path)
    return profile


def _adapter_id(profile: dict[str, Any]) -> str:
    kind = str(profile["kind"])
    if kind == "asymre_scan":
        return "e8_asymre_deltav_scan_cuda_dev_v2"
    if kind == "reciprocal_screen":
        return "e8_reciprocal_scan_cuda_dev_v2"
    if kind == "legacy_exp" and bool(profile.get("requires_positive_only")):
        return "e8_paper_aligned_linear_scan_cuda_dev_v2"
    if kind == "legacy_exp":
        return "e8_linear_c_extension_cuda_dev_v2"
    raise ValueError(f"Unsupported E8 config-driven profile kind: {kind}")


def _core_command(args, command: str, *, selected_ids: list[str]) -> list[str]:
    if len(selected_ids) != 8:
        raise RuntimeError("paper-aligned scan requires all eight configured GPUs")
    result = [
        sys.executable,
        str(
            _REPO_ROOT
            / "src"
            / "drpo"
            / "countdown_e8_alpha1_highc_scan_runtime.py"
        ),
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
                str(
                    highc.load_yaml(args.grid_config)["execution"][
                        "parallel_cells_per_gpu"
                    ]
                ),
            ]
        )
    return result


_base._core_command = _core_command
build_parser = _base.build_parser


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    grid_config = _grid_config_from_argv(tokens)
    if grid_config is None:
        highc.activate()
        _base.ADAPTER_ID = "e8_alpha1_highc_scan_cuda_dev_v1"
    else:
        profile = activate_config_driven_profile(grid_config)
        _base.ADAPTER_ID = _adapter_id(profile)
    return _base.main(tokens)


if __name__ == "__main__":
    raise SystemExit(main())
