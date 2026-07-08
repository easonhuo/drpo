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
