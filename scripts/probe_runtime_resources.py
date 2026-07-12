#!/usr/bin/env python3
"""Print a machine snapshot used by the runtime resource selectors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from drpo.runtime_resource_autotune import atomic_write_json, discover_machine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--meminfo", default="/proc/meminfo")
    parser.add_argument("--loadavg", default="/proc/loadavg")
    parser.add_argument("--cgroup-root", default="/sys/fs/cgroup")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snapshot = discover_machine(
        meminfo_path=args.meminfo,
        loadavg_path=args.loadavg,
        cgroup_root=args.cgroup_root,
        nvidia_smi=args.nvidia_smi,
    )
    payload = snapshot.as_dict()
    if args.output is not None:
        atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
