#!/usr/bin/env python3
"""Frozen formal entrypoint for D-U1-E6-CONDITIONAL-GAP-01."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Sequence
import torch
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drpo.du1_e6_conditional_gap import execute, resolve_device, validate_config

DEFAULT_CONFIG = Path("configs/du1_e6_conditional_gap_longrun.yaml")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    validate_config(config, "formal")
    if args.check_only:
        print("D-U1-E6-CONDITIONAL-GAP-01 frozen config validation passed")
        return 0
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    execute(config, "formal", args.output_root, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
