#!/usr/bin/env python3
"""Run an existing DRPO command inside an explicit, auditable resource pool."""
from __future__ import annotations

import argparse
import json
import os
import sys

from drpo.runtime_resource_pool import (
    ResourcePoolError,
    activate_resource_pool,
    validate_delegated_gpu_pool,
    write_pool_identity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-pool",
        help="Linux CPU-list syntax, for example 0-31,64-95; omit to record inherited affinity",
    )
    parser.add_argument(
        "--gpu-pool",
        help="ordered comma-separated GPU ids owned by the delegated launcher",
    )
    parser.add_argument(
        "--gpu-enforcement",
        choices=("launcher_argument", "cuda_visible"),
        default="launcher_argument",
        help=(
            "launcher_argument requires an identical delegated --gpus value; "
            "cuda_visible exports CUDA_VISIBLE_DEVICES"
        ),
    )
    parser.add_argument(
        "--pool-identity",
        required=True,
        help="immutable RESOURCE_POOL.json path shared by plan and run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the exact delegation without executing it",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="delegated command after --",
    )
    return parser


def _delegated_command(values: list[str]) -> list[str]:
    command = list(values)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ResourcePoolError("a delegated command is required after --")
    return command


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = _delegated_command(args.command)
    enforcement = args.gpu_enforcement if args.gpu_pool else "none"
    pool = activate_resource_pool(
        cpu_pool=args.cpu_pool,
        gpu_pool=args.gpu_pool,
        gpu_enforcement=enforcement,
    )
    validate_delegated_gpu_pool(command, pool)
    identity_path = write_pool_identity(args.pool_identity, pool)
    payload = {
        "resource_pool": pool.as_dict(),
        "resource_pool_identity": str(identity_path),
        "delegated_command": command,
        "dry_run": bool(args.dry_run),
        "scientific_matrix_changed": False,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    if args.dry_run:
        return 0
    os.execvpe(command[0], command, dict(os.environ))
    raise AssertionError("os.execvpe returned unexpectedly")  # pragma: no cover


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResourcePoolError as exc:
        print(f"RESOURCE_POOL_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
