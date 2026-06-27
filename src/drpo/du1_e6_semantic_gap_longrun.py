#!/usr/bin/env python3
"""Formal minimum-change D-U1 E6 semantic-gap long-run entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drpo.du1_e6_semantic import resolve_device
from drpo.du1_e6_semantic_gap import EXPERIMENT_ID, run_formal, validate_formal_config

DEFAULT_CONFIG = Path("configs/du1_e6_semantic_gap_longrun.yaml")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the frozen formal protocol without consuming held-out seeds.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    validate_formal_config(config)
    if args.check_only:
        print(f"{EXPERIMENT_ID} frozen config validation passed")
        return 0
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    run_formal(config, args.output_root, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
