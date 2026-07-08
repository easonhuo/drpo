# Dev-Branch Implementation and Reviewer-Gate Experiment Flow

This workflow fixes the default collaboration pattern for DRPO experiment code
changes. It keeps fast implementation and server-side execution separate from
independent review and merge decisions.

## Default branch model

```text
main
  -> dev/<experiment-or-claim>
       implementation agent writes code and runs liveness/experiment work
  -> reviewer gate
       independent review decides merge_ready / request_changes / reject
  -> main
```

Implementation agents may work quickly on dev branches. They may not merge to
`main` or decide that a scientific result is final.

## Phase 1: scope contract

Before editing, create or provide a scope contract that identifies:

- experiment ID or governance claim;
- base commit and dev branch;
- allowed files and forbidden files;
- allowed scientific-variable changes, if any;
- scientific variables explicitly unchanged;
- liveness gates;
- test commands;
- run commands;
- required result or failure artifacts;
- merge criteria.

No code change may start from an implicit or chat-memory-only scope.

## Phase 2: dev implementation

The implementation agent checks out the dev branch from the current `main`,
implements the approved scope, and records the base commit. It must keep the
worktree clean before starting formal or pilot runs unless the scope explicitly
allows a dirty pilot launch snapshot.

If the dev agent discovers a needed change outside scope, it stops and returns a
proposal instead of silently editing.

## Phase 3: liveness gates

Before any large sweep, run a small gate that proves the runner is observable and
recoverable. The gate must produce durable evidence of worker start, progress,
metrics flush, heartbeat/status update, and failure handling.

Smoke or liveness gates are engineering gates only. They are not scientific
results and must not be reported as method rankings.

## Phase 4: experiment execution

Runs must be bound to the dev branch `HEAD`. The result bundle must record the
launch command, commit, config, dataset identity, seeds, output root, terminal
state, logs, and failure inventory. Raw-complete, terminal-audited, packaged,
delivered, and repository-applied states remain separate.

If `main` advances before review, refresh the dev branch and rerun the relevant
tests and liveness gates. Rerun experiments when the refreshed code changes any
execution or scientific behavior.

## Phase 5: review and merge gate

The reviewer checks diff scope, forbidden scientific variables, tests, liveness,
provenance, and artifacts. The reviewer returns exactly one of:

- `merge_ready`;
- `request_changes`;
- `reject`.

A merge-ready decision does not imply that the reviewer has pushed or merged
anything unless the Git operation actually succeeded.
