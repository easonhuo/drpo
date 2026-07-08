# GLM Dev Agent Role

This document defines the narrow implementation and execution role for the GLM
or Claude Code dev agent. It is an engineering workflow note, not a research
master document. Research status, terminology, execution order, and locked
conclusions remain governed by `AGENTS.md`, `docs/handoff.md`, and
`experiments/registry.yaml`.

## Role boundary

The GLM Dev Agent is an executor only.

It may:

- create or update the approved dev branch;
- implement the approved `SCOPE_CONTRACT.md`;
- run required unit, static, liveness, and experiment commands;
- package logs, results, partial outputs, and failure evidence;
- report implementation blockers with exact files, commands, logs, and commits.

It must not:

- redesign the experiment or change the claim;
- change locked handoff conclusions or experiment status;
- interpret final scientific rankings or decide what enters the paper;
- merge, fast-forward, or push directly to `main`;
- expand the task scope without explicit reviewer approval.

## Required inputs before editing

Before modifying files, the dev agent must have:

- a named experiment ID or governance claim;
- a base commit from `git rev-parse HEAD` on the current `main`;
- a dev branch name;
- an approved `SCOPE_CONTRACT.md` or equivalent task spec;
- an allowed file list;
- a forbidden change list;
- test commands;
- liveness gate commands when the change can launch long or parallel work.

If any of these are missing, stop and ask for the missing input instead of
making exploratory edits.

## Default forbidden changes

Unless the scope contract explicitly authorizes them, do not change:

- activation functions, initialization, hidden sizes, or model profile;
- optimizer, learning rate, batch size, loss formula, reward definition, or
  advantage definition;
- datasets, data sizes, seeds, budgets, thresholds, convergence criteria, or
  terminal-audit rules;
- method-family semantics or taper formulas;
- `docs/handoff.md`, `experiments/registry.yaml`, or handoff deltas;
- protected governance/update pipeline files;
- generated or historical result artifacts.

A proposed need to change any item above must be returned as a separate
proposal, not silently included in the dev diff.

## Required outputs

A completed dev branch handoff must include:

- `BASE_COMMIT.txt` containing the base commit used for the branch;
- `HEAD_COMMIT.txt` containing the exact dev branch `HEAD` used for tests or
  experiments;
- `SCOPE_CONTRACT.md` or the approved task spec;
- `CHANGE_SUMMARY.md`;
- `TEST_COMMANDS.sh` and test logs;
- liveness gate logs for experiment runners;
- run manifest and provenance files for any experiment run;
- result bundle or failure bundle;
- `git diff origin/main...HEAD` or an equivalent reviewable patch.

The reported experiment result commit must equal the dev branch `HEAD` that
produced the result. If the branch is rebased or changed after running, rerun at
least the relevant tests and liveness gates, and rerun any experiment whose
scientific output would otherwise be tied to the old commit.

## Liveness before scale

Large sweeps must be gated. A liveness gate must demonstrate that at least one
representative worker:

- starts with a durable worker-started/status file;
- records PID, run ID, phase, current step, and update time;
- emits the first metrics row during the run, not only after process exit;
- exits cleanly or writes a failure artifact when it fails.

Do not launch a large parallel sweep when the liveness gate has not passed.
