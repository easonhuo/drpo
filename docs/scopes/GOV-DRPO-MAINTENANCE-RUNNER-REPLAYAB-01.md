# GOV-DRPO-MAINTENANCE-RUNNER-REPLAYAB-01 scope

## Identity

- claim: `GOV-DRPO-MAINTENANCE-RUNNER-REPLAYAB-01`;
- measurement authority: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`;
- current phase: Stage 0 documentation and review;
- controlling plan: `docs/development_workflow_optimization/M0_ATOMIC_DEV_TRANSACTION_PLAN.md`;
- scientific impact: none.

## Stage 0 authorized scope

Stage 0 may create or revise only:

- `docs/development_workflow_optimization/M0_ATOMIC_DEV_TRANSACTION_PLAN.md`;
- the preserved historical
  `docs/development_workflow_optimization/MAINTENANCE_RUNNER_REPLAYAB_PLAN.md`;
- `docs/development_workflow_optimization/maintenance_runner/**`;
- this scope document;
- Draft PR metadata and review comments for this claim.

Stage 0 may inspect existing repository code, PRs, commits, validators, workflow runs, and
Git objects. It may create unreferenced Git objects only for a non-publishing capability
check when no branch/ref/PR is created and the object identity is recorded in a review.

Stage 0 does not authorize behavior code, workflow changes, new Python paths, experiment
execution, routine use, default-policy activation, or merge.

## Conditionally proposed Stage 1 instrument scope

Stage 1 is not authorized by this document alone. After Stage 0 closes and the user
explicitly approves Stage 1, the only proposed executable change is:

- modify `scripts/run_workflow_replay.py` to add one bounded local
  `git-object-pair` measurement command;
- extend existing ReplayAB tests, primarily
  `tests/test_workflow_replay_execute.py`, with existing compare/model/orchestrate tests
  modified only when required;
- add non-Python fixtures under `tests/fixtures/workflow_replay/m0_atomic/**`.

No new Python file, dependency, GitHub workflow, service, backend, E7/E8 adapter, M0
publisher, or integration engine is proposed.

The measurement extension must remain local/bare-Git only and must reuse the existing
ReplayAB Core. It cannot contact GitHub, publish refs, change Candidate 01 behavior, or
alter ReplayAB thresholds.

## M0 qualification operations

After separate Stage 1 approval, two low-risk success transactions may use the existing
connected GitHub App operations to create reviewed blobs, one final tree, one commit,
one development branch, one Draft PR, and observed exact-head checks.

Stale-head and validation-failure qualification remain local and may not create
throwaway remote branches or PRs.

## Explicit exclusions

This claim does not authorize:

- the superseded M2 patch/apply/test/commit/push runner;
- M1 publisher implementation;
- changes to V1, Candidate 01 Core behavior, handoff authority, registry, governance
  pipeline implementation, formal execution, result delivery, or scientific code;
- arbitrary command or environment input;
- handoff, registry, schema-v3 delta, workflow, authority, governance-ledger, binary,
  symlink, gitlink, delete, rename, or mode-change M0 payloads;
- branch creation at an intermediate base state, force push, automatic retry/rebase,
  PR readiness, approval, merge, or experiment launch;
- post-result changes to cases, evaluators, thresholds, treatment orientation, or
  exclusions;
- default-route activation without a new reviewed policy change and explicit approval.

## Budgets and stop conditions

- M0 candidate production code: exactly zero;
- ReplayAB existing-file adapter preferred executable change: at most 100 lines;
- 101--140 lines: yellow review;
- above 140 lines: hard redesign;
- no new dependency or Python path;
- stop on ownership duplication, false acceptance, protected/scientific mutation,
  inability to evaluate independently, or need for broader responsibilities.

## Required next decision

Stage 1 remains blocked until:

- all Stage 0 review passes close;
- exact validation profiles, eight-case inventory, after-image hashes/modes, evaluator
  bindings, failure fixtures, and plan SHA-256 are frozen;
- the implementation branch is rebuilt from then-current `main`;
- the user explicitly approves the exact Stage 1 scope.

No merge or implementation authorization is inferred from this Stage 0 scope record.
