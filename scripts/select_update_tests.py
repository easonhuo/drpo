#!/usr/bin/env python3
"""Inspect or execute the canonical DRPO update fast/full test plan."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = REPO_ROOT / "tools" / "drpo-update"
sys.path.insert(0, str(TOOL_DIR))

from test_selection import (  # noqa: E402
    TestSelectionError,
    execute_test_plan,
    select_test_plan,
)


def git_paths(repo: Path, base: str, head: str) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), "diff", "--name-only", f"{base}..{head}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise TestSelectionError(proc.stderr.strip() or "git diff --name-only failed")
    return [line for line in proc.stdout.splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--mode", choices=("auto", "fast", "full"), default="auto")
    parser.add_argument("--map", dest="impact_map", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    impact_map = (args.impact_map or repo / "tools" / "drpo-update" / "test_impact_map.json").resolve()
    try:
        plan = select_test_plan(
            git_paths(repo, args.base, args.head),
            impact_map,
            requested_mode=args.mode,
        )
        executed = execute_test_plan(plan, worktree=repo) if args.execute else []
    except TestSelectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    payload = {**plan.to_dict(), "executed": executed}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "Update test selection: "
            f"mode={plan.selected_mode}, risk={plan.risk}, "
            f"groups={','.join(plan.matched_groups) or '-'}, "
            f"unknown={len(plan.unknown_paths)}"
        )
        print(f"Reason: {plan.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
