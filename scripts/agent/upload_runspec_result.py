#!/usr/bin/env python3
"""Upload one completed RunSpec review package to its results-only repository."""
from __future__ import annotations

import argparse
from pathlib import Path

from runspec_delivery_policy import (
    RESULT_TOO_LARGE,
    is_result_too_large_error,
    record_completed_result_too_large,
    validate_simple_size_policy,
)
from runspec_lib import DONE_DIR, handle_cli_error, json_main, read_yaml, state_path
from runspec_registration import validate_registration_block
from runspec_results_delivery import deliver_completed_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    try:
        done = state_path(repo, DONE_DIR, args.run_id)
        spec = read_yaml(done)
        registration = validate_registration_block(spec)
        validate_simple_size_policy(spec)
        payload = deliver_completed_run(repo, args.run_id)
        payload.setdefault("registration_mode", registration["mode"])
        payload.setdefault(
            "registration_closure_required", registration["closure_required"]
        )
    except Exception as exc:  # noqa: BLE001
        if is_result_too_large_error(exc):
            try:
                payload = record_completed_result_too_large(repo, args.run_id, exc)
            except Exception as report_exc:  # noqa: BLE001
                return handle_cli_error(report_exc, json_output=args.json)
        else:
            return handle_cli_error(exc, json_output=args.json)
    if args.json:
        json_main(payload)
    elif payload["status"] == RESULT_TOO_LARGE:
        print(
            "RunSpec delivery: "
            f"{RESULT_TOO_LARGE} run_id={payload['run_id']} "
            f"artifact={payload['artifact_zip']}"
        )
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
