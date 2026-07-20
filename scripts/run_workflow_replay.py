#!/usr/bin/env python3
"""Run the DRPO A/B Replay Engine candidate composition path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.workflow_replay.orchestrate import (  # noqa: E402
    OrchestrationError,
    ProcessResult,
    run_candidate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    candidate = commands.add_parser("candidate")
    candidate.add_argument("--repo-root", default=".")
    candidate.add_argument("--spec", required=True)
    candidate.add_argument("--preparation-root", required=True)
    candidate.add_argument("--transaction-root", required=True)
    candidate.add_argument("--python", default=sys.executable)
    candidate.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = Path(args.repo_root).expanduser().resolve()

    def invoke(command) -> ProcessResult:
        try:
            process = subprocess.run(
                command.argv,
                cwd=repository,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except OSError as exc:
            return ProcessResult(127, "", str(exc))
        return ProcessResult(process.returncode, process.stdout, process.stderr)

    try:
        outcome = run_candidate(
            repo_root=repository,
            spec_path=args.spec,
            preparation_root=args.preparation_root,
            transaction_root=args.transaction_root,
            python_executable=args.python,
            invoke=invoke,
        )
        payload = asdict(outcome)
        payload["status"] = "PASS"
        payload["state"] = "READY"
        payload["command_count"] = len(outcome.commands)
        payload["placement_count"] = len(outcome.placements)
        print(
            json.dumps(payload, sort_keys=True)
            if args.json
            else f"PASS {outcome.preparation_id}: {outcome.ready_commit_sha}"
        )
        return 0
    except OrchestrationError as error:
        payload = {
            "status": "FAIL",
            "state": "BLOCKED",
            "step": error.step,
            "message": error.message,
        }
        print(
            json.dumps(payload, sort_keys=True) if args.json else f"FAIL {error}",
            file=sys.stdout if args.json else sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
