#!/usr/bin/env python3
"""Package RunSpec artifacts using fail-closed allow-list safety checks."""
from __future__ import annotations

import argparse
from pathlib import Path

from runspec_lib import handle_cli_error, json_main
from runspec_safety import package_artifacts_safe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--runspec", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    try:
        manifest = package_artifacts_safe(
            repo,
            Path(args.runspec).resolve(),
            output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        )
        payload = {
            "status": "PASS",
            "run_id": manifest["run_id"],
            "zip_path": manifest["zip_path"],
            "zip_sha256": manifest["zip_sha256"],
            "included_count": len(manifest["included"]),
            "total_included_size_bytes": manifest["total_included_size_bytes"],
        }
        if args.json:
            json_main(payload)
        else:
            print(f"Artifact package: PASS {manifest['zip_path']}")
        return 0
    except Exception as exc:  # noqa: BLE001
        return handle_cli_error(exc, json_output=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
