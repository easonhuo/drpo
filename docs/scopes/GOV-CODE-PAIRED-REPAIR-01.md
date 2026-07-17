# GOV-CODE-PAIRED-REPAIR-01

## Status

- class: opt-in governance engineering workflow;
- repository: `easonhuo/drpo`;
- base branch: `main`;
- base commit: `d042a60e6e665fc7f8761e97d41fa0a621f78b87`;
- development branch: `dev/gov-code-paired-repair-01`;
- candidate feedback snapshot: `7826f5d60c83d8a58a11dc526b487cc09078d818`;
- scientific experiment status impact: none;
- default merge policy impact: none;
- execution state: implementation and review only; no paired observation recorded yet.

## Question

For a real large coding task, does structured reuse/code-size feedback turn the same
worker's frozen first complete implementation (`A0`) into a smaller, task-complete
repair (`B1`) without weakening tests, liveness, scientific scope, or reviewer
correctness?

This is not a two-worker randomized A/B claim. It is a low-cost before/after repair
observation intended to decide whether the feedback is useful enough for normal
engineering practice.

## Scope

Add only:

1. an opt-in paired-repair protocol;
2. one deterministic standard-library script that freezes A0 and compares B1;
3. focused tests for lineage, same-worker identity, code-size accounting, and
   correctness-first eligibility;
4. one validation template.

Do not add a worker service, queue, database, dashboard, container orchestrator,
automatic merge authority, or a second implementation arm.

## Frozen workflow

1. A real worker completes the task and commits A0 before seeing repair feedback.
2. A reviewer runs the frozen candidate code-change-budget rubric at
   `7826f5d60c83d8a58a11dc526b487cc09078d818` and records structured feedback.
3. The same worker receives that feedback and performs at most one bounded repair.
4. B1 is committed as a descendant of A0.
5. A0 and B1 receive explicit correctness/test/liveness evidence.
6. The report compares base-to-A0 and base-to-B1 production churn, new production
   files, test churn, and total changed files.
7. A human decides whether to merge A0, B1, or neither.

Correctness dominates size. A smaller B1 is ineligible when it loses an A0-passing
check, fails reviewer correctness, changes scientific scope, or fails required
liveness.

## Initial observation window

Use the workflow only on the next five suitable real large/structural coding tasks.
A task is suitable when it naturally produces a complete A0 and the reviewer has a
specific reuse or unnecessary-code finding. Do not manufacture tasks solely to fill
the window.

After five observations, review:

- whether feedback produced smaller eligible B1 implementations;
- whether duplicate modules/responsibilities were removed;
- repair time and reviewer burden;
- correctness, liveness, and scientific-scope regressions;
- cases where B1 had no size or reuse benefit.

No fixed success threshold or default activation is introduced by this claim.

## Governance boundaries

- no change to `docs/handoff.md` or `experiments/registry.yaml`;
- no change to scientific variables, seeds, thresholds, budgets, experiment roles,
  statuses, or execution order;
- no modification of Stage 1, 2, or 5 protected files;
- no automatic PR blocking or merge authorization;
- no claim that a single paired repair proves causal superiority;
- preserve A0, feedback, B1, validation, and failed repairs.

Because the workflow is opt-in and does not modify a closed stage's protected files,
this implementation does not reopen a closed governance stage.

## Rollback

Rollback is deletion of the files introduced by this claim. No existing production
workflow or protected governance authority is modified, so rollback does not require
state migration or historical evidence deletion. Existing paired observation records,
if any, remain immutable provenance even if the helper is later retired.
