# Dev Branch Handoff Template

## Repository state

- Repository:
- Base branch:
- Base commit:
- Dev branch:
- Dev branch HEAD:
- Worktree clean at launch:
- Worktree clean at handoff:

## Scope

- Experiment ID or claim:
- Scope contract path:
- Allowed files changed:
- Forbidden files unchanged:
- Scientific variables changed with approval:
- Scientific variables explicitly unchanged:

## Implementation summary

Describe what changed and why. Do not interpret final scientific results here
unless the reviewer requested a preliminary implementation note.

## Tests

| Command | Commit | Status | Log path |
|---|---|---|---|
|  |  |  |  |

## Liveness gates

| Gate | Command | Commit | Status | Evidence path |
|---|---|---|---|---|
|  |  |  |  |  |

## Experiment runs

| Run ID | Command | Commit | Status | Output root | Bundle path |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

## Failure inventory

Separate task-performance collapse, support/variance boundary events, and
NaN/Inf numerical failures. Preserve failed runs and logs.

## Known uncertainties

-

## Reviewer checklist

- [ ] diff matches scope contract
- [ ] no unauthorized scientific-variable changes
- [ ] tests/liveness ran on reviewed HEAD
- [ ] result provenance equals reviewed HEAD
- [ ] failure inventory is preserved
- [ ] raw-complete / terminal-audited / packaged / delivered states are not conflated
