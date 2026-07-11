#!/usr/bin/env python3
"""Publish one completed RunSpec to its declared dev branch and Draft PR."""
from __future__ import annotations

import argparse
from pathlib import Path

from runspec_lib import (
    DEFAULT_LANE_FILE,
    DONE_DIR,
    handle_cli_error,
    json_main,
    load_lane_config,
    read_yaml,
    state_path,
)
from runspec_publish_contract import validate_publish_block
from runspec_publish_git import publish_completed_run as _publish_completed_run
from runspec_safety import validate_fixed_publish_branch


def publish_completed_run(repo: Path, run_id: str, *, lane: str | None = None):
    lane_config = load_lane_config(repo, lane, DEFAULT_LANE_FILE)
    spec = read_yaml(state_path(repo, DONE_DIR, run_id))
    validate_fixed_publish_branch(spec, lane_config)
    return _publish_completed_run(repo, run_id, lane=lane)


__all__ = ["publish_completed_run", "validate_publish_block"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lane", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    try:
        payload = publish_completed_run(repo, args.run_id, lane=args.lane)
    except Exception as exc:  # noqa: BLE001
        return handle_cli_error(exc, json_output=args.json)
    if args.json:
        json_main(payload)
    else:
        print(
            f"RunSpec publish: PASS run_id={payload['run_id']} "
            f"commit={payload['published_commit']} pr={payload['pr_url'] or '<disabled>'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
