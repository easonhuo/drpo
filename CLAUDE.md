@AGENTS.md

# Claude Code entrypoint for DRPO

This repository uses role-separated Claude Code workflows.

Before editing code, determine the active role:

- If the task is implementation or experiment execution on a dev branch, read
  `docs/agents/glm_dev_agent.md`.
- If the task is review, merge gating, or final scientific interpretation, read
  `docs/agents/reviewer_gatekeeper.md`.

If no role is explicitly specified, ask for clarification before changing files.
Implementation agents must not act as reviewers or merge controllers. Reviewer
agents must not silently change experiment code after results have been produced.

Machine-local role defaults, model routing, and private server notes belong in
`CLAUDE.local.md`; do not commit that file.

## RunSpec executor workspaces

If `.agent_lane.yaml` exists, this checkout is a lane-specific server executor,
not a planning or development workspace. Read `.agent_lane.yaml` and
`CLAUDE.local.md` before acting.

In executor mode:

- do not infer a task from commits, registry order, or handoff prose;
- do not choose another experiment or lane;
- do not create launchers or change configs/hyperparameters;
- do not launch training or package results with ad-hoc shell commands;
- execute only READY RunSpecs through `scripts/agent/run_lane.py` and the other
  canonical `scripts/agent/*runspec*` entrypoints;
- report BLOCKED/FAILED rather than repairing experiment code locally;
- after a successful run, publish only through
  `scripts/agent/publish_runspec_result.py` when the RunSpec declares
  `publish.enabled: true`; never push or open PRs ad hoc.

A strict executor workspace may install a Claude Code `PreToolUse` hook with
`scripts/agent/configure_claude_workspace.py`; do not bypass or disable it.
