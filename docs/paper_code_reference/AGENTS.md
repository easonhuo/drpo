# PAPER-CODE-REFERENCE-01 task-local router

Root `AGENTS.md` keeps repository-wide rules. Current-main `docs/handoff.md` and
`experiments/registry.yaml` remain the scientific authority.

Before changing this task, read in order:

1. `TASK_EXECUTION_LOCK.yaml`;
2. `CURRENT_STATUS.md`;
3. `ACCEPTANCE_MATRIX.yaml`;
4. `SOURCE_MIGRATION_MAP.md`;
5. the actual `dev/paper-code-reference-01` branch, Draft PR #149, and exact-head CI.

Before any repository change, report the repository, branch, full SHA, task
purpose, lifecycle state, authorized next action, non-goals, exact proposed
paths, and unresolved uncertainties.

The active task is correctness acceptance of the existing migrated reviewer
code. Migration is closed. Backlog entries and remaining gaps do not authorize
implementation. Documents written in the current session do not expand that
session's authority.

Feature development, Countdown resume work, real-stack liveness, scientific
execution, method ranking, and integration of `main` are outside the active
scope. A concrete bug may be repaired only after the validation process
reproduces it.

All task changes remain on `dev/paper-code-reference-01`. Operations that alter
branch structure or repository history require a separate explicit user
instruction. Machine restrictions are implemented in
`.github/workflows/paper-code-single-branch-gate.yml`.
