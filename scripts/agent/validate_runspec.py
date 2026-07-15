#!/usr/bin/env python3
"""Validate a DRPO RunSpec without executing it."""
from __future__ import annotations

import argparse
from pathlib import Path

from runspec_delivery_policy import validate_simple_size_policy
from runspec_lib import add_common_args, handle_cli_error, json_main, load_lane_config
from runspec_safety import validate_runspec_safe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("runspec", help="RunSpec YAML path")
    parser.add_argument(
        "--no-registry-check",
        action="store_true",
        help="Explicitly bypass registry lookup; deferred mode already does this automatically",
    )
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    try:
        lane_config = load_lane_config(repo, args.lane, args.lane_file)
        spec = validate_runspec_safe(
            repo,
            Path(args.runspec).resolve(),
            lane_config=lane_config,
            require_registry=False if args.no_registry_check else None,
        )
        validate_simple_size_policy(spec)
        registration = spec["registration"]
        payload = {
            "status": "PASS",
            "run_id": spec["run_id"],
            "lane": spec["lane"],
            "experiment_id": spec["experiment_id"],
            "entrypoint_command": spec["entrypoint"]["command"],
            "registration_mode": registration["mode"],
            "registration_closure_required": registration["closure_required"],
        }
        if args.json:
            json_main(payload)
        else:
            print(f"RunSpec validation: PASS run_id={spec['run_id']} lane={spec['lane']}")
        return 0
    except Exception as exc:  # noqa: BLE001
        return handle_cli_error(exc, json_output=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
