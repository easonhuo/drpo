# PAPER-CODE-REFERENCE-01 Shared D4RL Locomotion Delta

**Date:** 2026-07-19  
**Claim:** `PAPER-CODE-REFERENCE-01`  
**Scientific status impact:** none

Read this file after:

1. `docs/paper_code_reference/SESSION_HANDOFF.md`;
2. `docs/paper_code_reference/SESSION_HANDOFF_ROLLOUT_DELTA_20260718.md`;
3. `docs/paper_code_reference/SESSION_HANDOFF_HOPPER_PUBLIC_DELTA_20260718.md`.

> **Later correction:** read `SESSION_HANDOFF_D4RL_BACKEND_AUDIT_DELTA_20260719.md` immediately after this file. The later source audit preserves the one-task-catalog and no-per-task-copy conclusions but corrects the over-broad premise that Hopper E7-Q2 and D4RL-9 share an identical full actor/critic/training backend.

This append-only delta supersedes only the previous task-local next-slice statement that treated Hopper output binding as the immediate next engineering target. It does not replace `docs/handoff.md`, authorize a formal D4RL run, change scientific variables, or permit merging PR `#149`.

## 1. Repository snapshot

- repository: `easonhuo/drpo`;
- default branch used for this slice: `main@e99489e7435bc26e2a7e30cd8d1a3aa10f4fc67a`;
- only active development branch: `dev/paper-code-reference-01`;
- persistent Draft PR: `#149`;
- PR remains unmerged;
- scientific execution status remains unchanged.

## 2. Corrected manuscript scope

The manuscript does not contain only Hopper. Its task-performance section names the D4RL-9 locomotion matrix:

- HalfCheetah, Hopper, Walker2d;
- medium, medium-replay, medium-expert;
- nine task/dataset coordinates in total.

Hopper E7-Q2 and D4RL-9 have different scientific responsibilities:

- Hopper E7-Q2: external mechanism validation with matched near/far diagnostics and targeted interventions;
- D4RL-9: external task-performance validation with normalized return.

This delta established the no-per-task-copy rule and common task catalog. Its original full-engine sharing wording is superseded by the later backend-audit delta.

## 3. Human-approved new Python paths

The repository owner approved exactly:

- `paper_code/src/drpo_reference/external/d4rl_tasks.py`;
- `paper_code/src/drpo_reference/experiments/d4rl.py`;
- `paper_code/tests/test_d4rl_shared_core_differential.py`.

The durable approval record is PR conversation comment `5014487498`.

## 4. Implemented sharing boundary at this stage

### `external/d4rl_tasks.py`

- registers the exact D4RL-9 task order;
- records environment IDs, dataset IDs/basenames, and D4RL-v2 reference-score constants;
- binds the already registered Hopper medium-replay SHA to the matching task;
- leaves every other dataset SHA explicitly unresolved rather than inventing provenance;
- validates dataset identity and fails closed for formal use when SHA provenance is unresolved;
- contains no actor, critic, trainer, rollout loop, or aggregation implementation.

### `experiments/d4rl.py`

At this stage it introduced:

- the complete D4RL-9 matrix and seed coordinate;
- unresolved dataset provenance and unfrozen protocol blockers;
- `method_ranking_claim_allowed=false`;
- one dispatch boundary across all nine tasks;
- a prohibition on per-task trainer copies.

The later backend-audit delta adds an explicit backend specification and clarifies that the selected performance backend need not be the Hopper mechanism trainer.

### Shared-core differential test

The focused test covers:

- exact nine-task manuscript order;
- HalfCheetah/Hopper/Walker2d reference-score constants;
- identity with the frozen Hopper medium-replay protocol;
- unresolved-SHA fail-closed behavior;
- seed and dataset-matrix validation;
- one backend dispatch across all nine tasks;
- no method-ranking authorization.

## 5. Source map correction

`docs/paper_code_reference/SOURCE_MIGRATION_MAP.md` records one D4RL task catalog and no per-task trainer duplication. The current version of that document follows the later backend audit and distinguishes shared task/data/rollout contracts from backend-specific training contracts.

An intermediate connector mistake temporarily replaced this document with a placeholder in commit `ea56f4c7dfdd148c300f994045dd07e4d92b9b74`. The immediately following commit `1877d28f1799251a1749902c0b7e84b6d5abe98b` restored the complete document. No final content was lost. The history is retained transparently rather than rewritten.

## 6. Validation

At head `b78d00e9625c3399e68fcf50c463dc5ddd67fd17`, GitHub Actions run `29674497178` completed successfully, including:

- Python compile;
- shell syntax;
- handoff authority;
- formal execution channel;
- governance inventory;
- governance stage status;
- full repository pytest;
- Ruff.

Evidence Locator Gate run `29674497187` also passed.

These are engineering checks only. They did not load the nine registered datasets, run MuJoCo, train a model, or validate manuscript performance numbers.

## 7. Remaining work and blockers

The task catalog and dispatch boundary are established, but D4RL-9 is not formally runnable. The later backend-audit delta is authoritative for the next step: establish and freeze one scientifically valid performance backend across all nine tasks without copying a trainer per environment.
