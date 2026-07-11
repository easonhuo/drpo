#!/usr/bin/env python3
"""Claim the next READY RunSpec for this workspace lane."""
from __future__ import annotations

import argparse
from pathlib import Path

from runspec_lib import add_common_args, handle_cli_error, json_main, load_lane_config
from runspec_safety import claim_next_runspec_safe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--run-id", default=None, help="Optional explicit run_id")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    try:
        lane_config = load_lane_config(repo, args.lane, args.lane_file)
        claimed = claim_next_runspec_safe(repo, lane_config=lane_config, run_id=args.run_id)
        payload = {
            "status": "PASS",
            "claimed_path": claimed.relative_to(repo).as_posix(),
            "lane": lane_config["lane"],
        }
        if args.json:
            json_main(payload)
        else:
            print(f"Claimed RunSpec: {payload['claimed_path']}")
        return 0
    except Exception as exc:  # noqa: BLE001
        return handle_cli_error(exc, json_output=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
