from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGURE = ROOT / "scripts" / "agent" / "configure_claude_workspace.py"
GUARD = ROOT / "scripts" / "agent" / "claude_runspec_guard.py"
HOOK_COMMAND = 'python3 "${CLAUDE_PROJECT_DIR}/scripts/agent/claude_runspec_guard.py"'


def run(cmd: list[str], *, cwd: Path, input_text: str | None = None):
    return subprocess.run(
        cmd,
        cwd=cwd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts" / "agent").mkdir(parents=True)
    (repo / "scripts" / "agent" / "run_lane.py").write_text("# fixture\n")
    run(["git", "init", "-q"], cwd=repo)
    return repo


def run_hook(repo: Path, payload: dict):
    return subprocess.run(
        ["python", str(GUARD)],
        cwd=repo,
        input=json.dumps(payload),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(repo)},
    )


def test_configure_is_idempotent_and_installs_strict_hook(tmp_path: Path):
    repo = make_repo(tmp_path)
    command = [
        "python",
        str(CONFIGURE),
        "--repo-root",
        str(repo),
        "--lane",
        "e8",
        "--json",
    ]
    first = run(command, cwd=repo)
    second = run(command, cwd=repo)
    assert first.returncode == second.returncode == 0
    assert json.loads(first.stdout)["status"] == "PASS"
    lane = yaml.safe_load((repo / ".agent_lane.yaml").read_text())
    assert lane["lane"] == "e8"
    assert lane["executor_mode"] == "strict"
    local_instructions = (repo / "CLAUDE.local.md").read_text()
    assert "run_lane.py --once" in local_instructions
    assert "Never move, copy, delete, or recreate files" in local_instructions
    assert "Never rerun a failed `run_id`" in local_instructions
    assert "create `COMPLETED.json`" in local_instructions
    settings = json.loads((repo / ".claude" / "settings.local.json").read_text())
    commands = [
        hook["command"]
        for group in settings["hooks"]["PreToolUse"]
        for hook in group["hooks"]
    ]
    assert commands.count(HOOK_COMMAND) == 1


def test_guard_allows_runspec_and_blocks_direct_training_and_edits(tmp_path: Path):
    repo = make_repo(tmp_path)
    (repo / ".agent_lane.yaml").write_text(
        "lane: e8\nexecutor_mode: strict\nforbid_cross_lane: true\n"
    )
    allowed = run_hook(
        repo,
        {
            "tool_name": "Bash",
            "tool_input": {"command": "python scripts/agent/run_lane.py --once"},
            "cwd": str(repo),
        },
    )
    assert allowed.returncode == 0
    assert allowed.stdout == ""

    for command in (
        "torchrun --nproc_per_node=8 train.py",
        "rm .runspec_state/failed/task.yaml",
        "mv .runspec_state/failed/task.yaml runspecs/ready/task.yaml",
    ):
        direct = run_hook(
            repo,
            {
                "tool_name": "Bash",
                "tool_input": {"command": command},
                "cwd": str(repo),
            },
        )
        decision = json.loads(direct.stdout)["hookSpecificOutput"]
        assert decision["permissionDecision"] == "deny"
        assert "run_lane.py" in decision["permissionDecisionReason"]

    edit = run_hook(
        repo,
        {
            "tool_name": "Write",
            "tool_input": {"file_path": str(repo / "launch.sh"), "content": "..."},
            "cwd": str(repo),
        },
    )
    assert json.loads(edit.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"
