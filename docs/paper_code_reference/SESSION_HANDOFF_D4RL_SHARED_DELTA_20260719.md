# PAPER-CODE-REFERENCE-01 Shared D4RL Locomotion Delta

**Date:** 2026-07-19  
**Claim:** `PAPER-CODE-REFERENCE-01`  
**Scientific status impact:** none

Read this file after:

1. `docs/paper_code_reference/SESSION_HANDOFF.md`;
2. `docs/paper_code_reference/SESSION_HANDOFF_ROLLOUT_DELTA_20260718.md`;
3. `docs/paper_code_reference/SESSION_HANDOFF_HOPPER_PUBLIC_DELTA_20260718.md`.

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

Hopper E7-Q2 and D4RL-9 have different scientific responsibilities but must share one locomotion implementation:

- Hopper E7-Q2: external mechanism validation with matched near/far diagnostics and targeted interventions;
- D4RL-9: external task-performance validation with normalized return.

The code must not implement separate Hopper and D4RL actors, critics, data loaders, training loops, or rollout stacks.

## 3. Human-approved new Python paths

The repository owner approved exactly:

- `paper_code/src/drpo_reference/external/d4rl_tasks.py`;
- `paper_code/src/drpo_reference/experiments/d4rl.py`;
- `paper_code/tests/test_d4rl_shared_core_differential.py`.

The durable approval record is PR conversation comment `5014487498`.

## 4. Implemented sharing boundary

### `external/d4rl_tasks.py`

- registers the exact D4RL-9 task order;
- records environment IDs, dataset IDs/basenames, and D4RL-v2 reference-score constants;
- binds the already registered Hopper medium-replay SHA to the matching task;
- leaves every other dataset SHA explicitly unresolved rather than inventing provenance;
- validates dataset identity and fails closed for formal use when SHA provenance is unresolved;
- contains no actor, critic, trainer, rollout loop, or aggregation implementation.

### `experiments/d4rl.py`

- resolves the complete D4RL-9 matrix and seed coordinate;
- exposes unresolved dataset provenance, unfrozen performance protocol, incomplete ten-run coordinates, and smoke status as explicit blockers;
- keeps `method_ranking_claim_allowed=false`;
- dispatches all nine tasks through one injected single-task runner;
- records that no second D4RL trainer is implemented;
- refuses formal dispatch while blockers remain.

### Shared-core differential test

The focused test covers:

- exact nine-task manuscript order;
- HalfCheetah/Hopper/Walker2d reference-score constants;
- identity with the frozen Hopper medium-replay protocol;
- unresolved-SHA fail-closed behavior;
- seed and dataset-matrix validation;
- one injected runner used for all nine tasks;
- no method-ranking authorization.

## 5. Source map correction

`docs/paper_code_reference/SOURCE_MIGRATION_MAP.md` now records one D4RL locomotion engine with two thin profiles. Existing `hopper_*` filenames are historical names for the first migrated locomotion core; they are not authorization to create a parallel D4RL implementation.

An intermediate connector mistake temporarily replaced this document with a placeholder in commit `ea56f4c7dfdd148c300f994045dd07e4d92b9b74`. The immediately following commit `1877d28f1799251a1749902c0b7e84b6d5abe98b` restored the complete document and added the intended D4RL sharing policy. No final content was lost. The history is retained transparently rather than rewritten.

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

The sharing boundary is now established, but D4RL-9 is not formally runnable yet. Before implementing the final single-task performance adapter, the next session must audit and freeze:

1. the authoritative D4RL-9 algorithm/backbone and method set actually supporting the manuscript table;
2. exact dataset provenance and SHA-256 values for the eight coordinates not already bound by Hopper medium-replay;
3. the registered ten-run seed set;
4. environment compatibility and normalization semantics for all three locomotion environments;
5. terminal-audit and task-performance protocol for the benchmark profile.

Do not copy the Hopper E7-Q2 six-branch mechanism suite into D4RL-9. Do not infer that the manuscript's provisional candidate table is a registered result. Do not implement HalfCheetah or Walker2d trainer copies. The next code slice must attach a verified performance profile to the existing shared locomotion engine.
