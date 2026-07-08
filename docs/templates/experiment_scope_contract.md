# Experiment Scope Contract Template

## Identity

- Experiment ID or claim:
- User request:
- Base commit:
- Dev branch:
- Implementation agent:
- Reviewer/gatekeeper:

## Allowed changes

List exact repository paths and the reason each path may change.

- `path/to/file` - reason

## Forbidden changes

List paths, variables, formulas, or behaviors that must not change.

- activation/init/model profile:
- optimizer/lr/batch:
- datasets/data sizes:
- seeds:
- budgets:
- thresholds/convergence criteria:
- reward/advantage/loss formula:
- handoff/registry locked conclusions:

## Scientific variables explicitly unchanged

Record the current value and the source file/config for every variable that must
remain fixed.

| Variable | Current value | Source | Must remain unchanged? |
|---|---:|---|---|
|  |  |  | yes |

## Explicitly authorized scientific-variable changes

Only list changes approved by the user/reviewer before coding.

| Variable | Old value | New value | Rationale | Approval |
|---|---:|---:|---|---|
|  |  |  |  |  |

## Liveness gates

- Gate A command:
- Gate A required evidence:
- Gate B command:
- Gate B required evidence:
- Large sweep may start only after:

## Test commands

```bash
# fill in exact commands, no placeholder paths
```

## Run commands

```bash
# fill in exact commands, no placeholder paths
```

## Required artifacts

- `BASE_COMMIT.txt`
- `HEAD_COMMIT.txt`
- `CHANGE_SUMMARY.md`
- `TEST_COMMANDS.sh`
- test logs
- liveness logs
- run manifest
- result bundle or failure bundle
- terminal audit summary, when applicable

## Merge criteria

- scope diff passes;
- tests pass or failures are classified as unrelated baseline failures;
- liveness gates pass before large sweep;
- results are bound to dev branch `HEAD`;
- no unauthorized scientific-variable changes;
- reviewer returns `merge_ready`.
