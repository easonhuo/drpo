#!/usr/bin/env python3
"""Upload one completed RunSpec review package to its results-only repository."""
from __future__ import annotations

import argparse
from pathlib import Path

from runspec_lib import handle_cli_error, json_main
from runspec_results_delivery import deliver_completed_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    try:
        payload = deliver_completed_run(repo, args.run_id)
    except Exception as exc:  # noqa: BLE001
        return handle_cli_error(exc, json_output=args.json)
    if args.json:
        json_main(payload)
    else:
        print(
            "RunSpec delivery: "
            f"{payload['status']} run_id={payload['run_id']} "
            f"repository={payload['repository']} "
            f"commit={payload['results_commit']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
