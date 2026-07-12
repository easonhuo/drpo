#!/usr/bin/env python3
"""Configure one DRPO checkout as a lane-specific Claude Code executor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from runspec_lib import DEFAULT_LANE_FILE, LANE_RE, RunSpecError, json_main, write_yaml
from runspec_safety import ensure_workspace_branch, expected_workspace_branch

LOCAL_INSTRUCTION_FILE = "CLAUDE.local.md"
LOCAL_SETTINGS_FILE = Path(".claude") / "settings.local.json"
HOOK_COMMAND = 'python3 "${CLAUDE_PROJECT_DIR}/scripts/agent/claude_runspec_guard.py"'
ACTIVE_SERVER_LANES = frozenset({"e7", "e8"})


def validate_active_lane(lane: str) -> str:
    normalized = lane.strip().lower()
    if not LANE_RE.match(normalized):
        raise RunSpecError(f"invalid lane: {normalized}")
    if normalized not in ACTIVE_SERVER_LANES:
        raise RunSpecError(
            f"unsupported active server lane: {normalized}; "
            f"expected one of {sorted(ACTIVE_SERVER_LANES)}"
        )
    return normalized


def defaults_for_lane(lane: str) -> tuple[list[str], list[str]]:
    normalized = lane.lower()
    if normalized == "e7":
        return ["EXT-H-E7-"], ["EXT-C-E8-"]
    if normalized == "e8":
        return ["EXT-C-E8-"], ["EXT-H-E7-"]
    return [], []


def local_instruction_text(lane: str, publish_branch: str) -> str:
    return f"""# Local Claude Code executor role

This checkout is a strict server executor workspace for lane `{lane}`.
Its fixed publication branch is `{publish_branch}`.

- Read `.agent_lane.yaml` before acting.
- Do not infer a task from commits, registry order, or handoff prose.
- Do not choose another experiment or lane.
- Do not create or modify launcher scripts, configs, or hyperparameters.
- Do not launch training directly.
- Do not create ad-hoc archives or package an entire run directory.
- Execute the next READY task only through:

  `python scripts/agent/run_lane.py --once`

- To select an explicitly named task, use the same command with `--run-id`.
- When a completed RunSpec declares `publish.enabled: true`, publish only through:

  `python scripts/agent/publish_runspec_result.py --run-id <run_id>`

- Never switch branches or push manually in this executor workspace.
- After `FAILED`, `BLOCKED`, or `NO_READY_TASK`, stop and report the exact state.
- Never move, copy, delete, or recreate files under `runspecs/ready/` or
  `.runspec_state/`; the tracked READY definition and local terminal state are not
  duplicate tasks.
- Never rerun a failed `run_id`, manually launch remaining branches, or create
  `COMPLETED.json`, `RUN_SUMMARY.json`, or other success evidence by hand.
- On invalid RunSpec, duplicate run_id, provenance drift, missing script, missing output,
  or package-policy failure, report BLOCKED/FAILED. Do not repair experiment code here.
"""


def hook_entry() -> dict[str, Any]:
    return {
        "matcher": "Bash|Write|Edit|MultiEdit|NotebookEdit",
        "hooks": [{"type": "command", "command": HOOK_COMMAND}],
    }


def merge_local_settings(path: Path) -> None:
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RunSpecError(f"invalid JSON in {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise RunSpecError(f"{path} must contain a JSON object")
    else:
        data = {}
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise RunSpecError(f"hooks in {path} must be an object")
    pre = hooks.setdefault("PreToolUse", [])
    if not isinstance(pre, list):
        raise RunSpecError(f"hooks.PreToolUse in {path} must be a list")
    target = hook_entry()
    present = any(
        isinstance(row, dict)
        and any(
            isinstance(item, dict) and item.get("command") == HOOK_COMMAND
            for item in row.get("hooks", [])
        )
        for row in pre
    )
    if not present:
        pre.append(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_git_info_excludes(repo: Path, entries: list[str]) -> None:
    exclude = repo / ".git" / "info" / "exclude"
    if not exclude.parent.is_dir():
        import subprocess

        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--git-path", "info/exclude"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            raise RunSpecError((proc.stderr or proc.stdout).strip())
        exclude = Path(proc.stdout.strip())
        if not exclude.is_absolute():
            exclude = repo / exclude
    exclude.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
    lines = existing.splitlines()
    changed = False
    for entry in entries:
        if entry not in lines:
            lines.append(entry)
            changed = True
    if changed:
        exclude.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def configure(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve()
    if not (repo / "scripts" / "agent" / "run_lane.py").is_file():
        raise RunSpecError("run from a DRPO checkout containing scripts/agent/run_lane.py")
    lane = validate_active_lane(args.lane)
    publish_branch = ensure_workspace_branch(repo, lane)
    if publish_branch != expected_workspace_branch(lane):
        raise RunSpecError("workspace branch resolution returned an unexpected branch")
    default_allowed, default_forbidden = defaults_for_lane(lane)
    allowed = args.allowed_prefix if args.allowed_prefix is not None else default_allowed
    forbidden = args.forbidden_prefix if args.forbidden_prefix is not None else default_forbidden
    lane_payload = {
        "lane": lane,
        "executor_mode": args.mode,
        "forbid_cross_lane": True,
        "allowed_experiment_prefixes": allowed,
        "forbidden_experiment_prefixes": forbidden,
        "publish_branch": publish_branch,
    }
    lane_path = repo / DEFAULT_LANE_FILE
    write_yaml(lane_path, lane_payload)
    instruction_path = repo / LOCAL_INSTRUCTION_FILE
    instruction_path.write_text(local_instruction_text(lane, publish_branch), encoding="utf-8")
    settings_path = repo / LOCAL_SETTINGS_FILE
    merge_local_settings(settings_path)
    add_git_info_excludes(
        repo,
        [
            f"/{DEFAULT_LANE_FILE}",
            f"/{LOCAL_INSTRUCTION_FILE}",
            f"/{LOCAL_SETTINGS_FILE.as_posix()}",
            "/.runspec_state/",
            "/runspec_artifacts/",
        ],
    )
    return {
        "status": "PASS",
        "lane": lane,
        "publish_branch": publish_branch,
        "executor_mode": args.mode,
        "lane_file": lane_path.relative_to(repo).as_posix(),
        "claude_local_file": instruction_path.relative_to(repo).as_posix(),
        "claude_settings_file": settings_path.relative_to(repo).as_posix(),
        "next_command": "python scripts/agent/run_lane.py --once",
        "restart_claude_code_session": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--lane", required=True)
    parser.add_argument("--mode", choices=("strict", "advisory"), default="strict")
    parser.add_argument("--allowed-prefix", action="append", default=None)
    parser.add_argument("--forbidden-prefix", action="append", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        payload = configure(args)
    except Exception as exc:  # noqa: BLE001
        payload = {"status": "FAIL", "error": str(exc)}
        if args.json:
            json_main(payload)
        else:
            print(f"Claude workspace configuration: FAIL: {exc}")
        return 1
    if args.json:
        json_main(payload)
    else:
        print(
            f"Claude workspace configuration: PASS lane={payload['lane']} "
            f"branch={payload['publish_branch']} mode={payload['executor_mode']}"
        )
        print(f"Restart Claude Code, then run: {payload['next_command']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
