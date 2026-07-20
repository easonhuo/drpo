#!/usr/bin/env python3
"""Claim and run one RunSpec for the current workspace lane."""
from __future__ import annotations

import argparse
from pathlib import Path

from run_claimed_runspec import execute_claimed_runspec
from runspec_delivery_policy import RESULT_TOO_LARGE
from runspec_lib import add_common_args, handle_cli_error, json_main, load_lane_config
from runspec_safety import claim_next_runspec_safe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--once", action="store_true", help="Run one task and exit")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    try:
        lane_config = load_lane_config(repo, args.lane, args.lane_file)
        claimed = claim_next_runspec_safe(repo, lane_config=lane_config, run_id=args.run_id)
        payload, code = execute_claimed_runspec(repo, claimed)
        payload["lane"] = lane_config["lane"]
    except Exception as exc:  # noqa: BLE001
        return handle_cli_error(exc, json_output=args.json)
    if args.json:
        json_main(payload)
    elif code == 0 and payload.get("delivery_status") == RESULT_TOO_LARGE:
        print(
            f"Lane run: PASS lane={payload['lane']} run_id={payload['run_id']} "
            f"delivery={RESULT_TOO_LARGE} artifact={payload['local_artifact_zip']}"
        )
    elif code == 0:
        print(f"Lane run: PASS lane={payload['lane']} run_id={payload['run_id']}")
    else:
        error = payload.get("delivery_error") or payload.get("publish_error")
        print(
            f"Lane run: PASS but result handoff: FAIL lane={payload['lane']} "
            f"run_id={payload['run_id']} error={error}"
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
