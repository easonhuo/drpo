#!/usr/bin/env python3
"""Claude Code PreToolUse guard for strict RunSpec executor workspaces.

When a checkout contains ``.agent_lane.yaml`` with ``executor_mode: strict``, the
local Claude Code agent is an executor rather than a planner or developer.  The
guard permits read-only inspection and the canonical RunSpec commands, while
blocking ad-hoc shell execution and all direct file-editing tools.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from runspec_lib import DEFAULT_LANE_FILE, RunSpecError, read_yaml  # noqa: E402

CANONICAL_SCRIPTS = {
    "claim_next_runspec.py",
    "configure_claude_workspace.py",
    "package_runspec_artifacts.py",
    "publish_runspec_result.py",
    "run_claimed_runspec.py",
    "run_lane.py",
    "validate_runspec.py",
}
READ_ONLY_GIT_SUBCOMMANDS = {
    "branch",
    "diff",
    "log",
    "rev-parse",
    "show",
    "status",
}
WRITE_TOOL_NAMES = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
SHELL_OPERATOR_RE = re.compile(r"(?:&&|\|\||[;|`]|\$\(|\n)")


def deny(reason: str) -> int:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    return 0


def project_root(payload: dict[str, Any]) -> Path:
    value = os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd()
    return Path(str(value)).expanduser().resolve()


def strict_executor_config(root: Path) -> dict[str, Any] | None:
    lane_path = root / DEFAULT_LANE_FILE
    if not lane_path.is_file():
        return None
    data = read_yaml(lane_path)
    mode = str(data.get("executor_mode", "strict")).strip().lower()
    if mode == "off":
        return None
    if mode not in {"strict", "advisory"}:
        raise RunSpecError(f"invalid executor_mode in {DEFAULT_LANE_FILE}: {mode}")
    if mode != "strict":
        return None
    if not data.get("lane"):
        raise RunSpecError(f"{DEFAULT_LANE_FILE} is missing lane")
    return data


def canonical_runspec_command(tokens: list[str]) -> bool:
    if not tokens:
        return False
    executable = Path(tokens[0]).name
    if executable not in {"python", "python3"}:
        return False
    index = 1
    while index < len(tokens) and tokens[index].startswith("-"):
        # ``-m`` is intentionally not accepted: executor commands must name the
        # checked-in script explicitly.
        if tokens[index] == "-m":
            return False
        index += 1
    if index >= len(tokens):
        return False
    script = Path(tokens[index].replace("\\", "/"))
    return (
        script.name in CANONICAL_SCRIPTS
        and script.parts[-3:-1] == ("scripts", "agent")
    )


def read_only_shell_command(tokens: list[str]) -> bool:
    if not tokens:
        return False
    if tokens[0] in {"pwd"} and len(tokens) == 1:
        return True
    if tokens[0] == "git" and len(tokens) >= 2:
        # ``git -C <path> status`` is allowed as long as the eventual
        # subcommand is read-only.
        index = 1
        while index < len(tokens) and tokens[index] == "-C":
            index += 2
        return index < len(tokens) and tokens[index] in READ_ONLY_GIT_SUBCOMMANDS
    return False


def bash_allowed(command: str) -> bool:
    if not command.strip() or SHELL_OPERATOR_RE.search(command):
        return False
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    return canonical_runspec_command(tokens) or read_only_shell_command(tokens)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001
        return deny(f"RunSpec guard could not parse Claude hook input: {exc}")
    if not isinstance(payload, dict):
        return deny("RunSpec guard received non-object hook input")

    try:
        config = strict_executor_config(project_root(payload))
    except Exception as exc:  # noqa: BLE001
        return deny(f"RunSpec executor configuration is invalid: {exc}")
    if config is None:
        return 0

    lane = str(config.get("lane"))
    tool_name = str(payload.get("tool_name") or "")
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        tool_input = {}

    if tool_name == "Bash":
        command = str(tool_input.get("command") or "")
        if bash_allowed(command):
            return 0
        return deny(
            f"This is a strict lane={lane} executor workspace. Direct shell commands are blocked. "
            "Use `python scripts/agent/run_lane.py --once` or another canonical "
            "scripts/agent RunSpec command; do not launch training or package results ad hoc."
        )

    if tool_name in WRITE_TOOL_NAMES:
        return deny(
            f"This is a strict lane={lane} executor workspace. Direct file edits are blocked so "
            "the server agent cannot invent launchers, alter configs, or bypass artifact policy."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
