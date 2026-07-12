#!/usr/bin/env python3
"""Execute the registered real-data liveness smoke for E7 PPO stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drpo.e7_ppo_stability_smoke import run_smoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_smoke(
        repo_root=Path(args.repo_root),
        contract_path=args.contract,
        run_spec_path=args.run_spec,
        grid_path=args.grid,
        work_dir=args.work_dir,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
