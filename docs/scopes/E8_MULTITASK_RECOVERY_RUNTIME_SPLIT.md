# E8 Multitask Recovery/Runtime Split

## Scope

This is a code-only architecture refactor originally branched from `main@9d41a784e05dbc8cff1a67379cc3227a926984d8` after the validated input/data, result/aggregation, and artifact/package extractions. Before final validation the branch was synchronized non-destructively with current `main@3c36b48637c7ea83314c235af172fa11b4c883f0`.

The goal is to move method-agnostic recovery/runtime lifecycle responsibilities out of `src/drpo/e8_multitask_exp_tuning.py` into the already-existing `src/drpo/e8_multitask_runtime.py` module.

No new Python file is created by this task.

## Scientific boundary

This refactor must not change:

- experiment IDs, tasks, banks, split sizes, seeds, parameter grids, or training horizon;
- EXP / AsymRE / TOPR / Reciprocal / DPO mathematics or DPO frozen-reference semantics;
- scheduler concurrency, GPU-slot geometry, seed barriers, queue ordering, or dynamic-refill semantics;
- recovery identity, reusable-cell eligibility, checkpoint cadence, import semantics, or incomplete-cell rerun policy;
- evaluation metrics, terminal-audit criteria, convergence semantics, or result status;
- artifact/package required members, archive ownership, or delivery semantics.

`EXT-C-E8-MULTITASK-BASELINE-MATRIX-01` remains `not_run`. No scientific run is part of this task.

## Design rule

Prefer mechanical relocation over redesign.

`e8_multitask_runtime.py` should own method-agnostic recovery/runtime helpers that inspect already-materialized manifests and filesystem state to determine resumability, construct recovery plans, import reusable completed cells, publish recovery checkpoints, and compact execution logs. `e8_multitask_exp_tuning.py` remains the composition root and supplies experiment-specific callbacks/values explicitly.

Do not move scheduler execution itself, scientific method kernels, model/optimizer code, terminal-audit adjudication, result aggregation, artifact/package construction, or engineering self-test orchestration in this task.

Do not add a new gate, manager/factory hierarchy, new runtime protocol, or new Python module.

## Candidate responsibilities to relocate

The implementation should inspect the current tree and move only the clean method-agnostic subset, expected to include some or all of:

- completed-attempt/current-identity checks used by recovery;
- effective recovery-config resolution;
- reusable completed-cell manifest inspection;
- recovery stage-plan construction;
- recovery-plan materialization;
- recovery import of identity-matched completed cells;
- recovery checkpoint snapshot construction and publication;
- transactional execution-log compaction.

Functions that directly own queue scheduling, subprocess launch, calibration/liveness execution, terminal scientific adjudication, or hardened final artifact packaging remain in the main runner.

## Dependency direction

Allowed:

`e8_multitask_exp_tuning.py -> e8_multitask_runtime.py`

Forbidden:

`e8_multitask_runtime.py -> e8_multitask_exp_tuning.py`

Experiment-specific concepts such as `experiment_id`, `stable_config_hash`, cell construction, method-specific recovery identity, atomic JSON writing, SHA-256 helpers, prepared/calibration/liveness checks, or hardened checkpoint packaging should be supplied as explicit callbacks or plain values where needed rather than importing the main runner.

## Implementation stages

1. Inventory current recovery/runtime functions and dependency edges.
2. Select the clean recovery-only subset and record any boundary intentionally left in the main runner.
3. Move functions mechanically into `e8_multitask_runtime.py`, preserving behavior and keeping compatibility wrappers only where existing callers/tests require them.
4. Run Python compile, focused E8 recovery/self-test tests, scheduler timing probe, method-integration contract, Ruff, governance checks, and broad pytest.
5. Review the final diff for scientific/runtime semantic drift and measure line-count reduction in `e8_multitask_exp_tuning.py`.

## Acceptance

The task is complete only if:

- moved recovery/runtime functions have a single implementation authority in `e8_multitask_runtime.py`;
- no circular import is introduced;
- reusable-cell, recovery-stage, checkpoint, import, and log-compaction semantics are unchanged;
- scheduler/recovery timing and terminal-audit scientific semantics are not changed;
- existing tests pass without weakening assertions;
- no new gate is added;
- the formal experiment remains `not_run`;
- the PR remains unmerged until explicit repository-owner approval.

## Pre-merge cleanup outcome

The post-refactor correctness/redundancy review found one latent schema-authority defect: `recovery_checkpoint_snapshot()` had hard-coded the recovery snapshot schema as `1` after relocation. The runtime helper now receives the caller's `RECOVERY_SNAPSHOT_SCHEMA_VERSION` explicitly, and a regression test verifies that a non-default caller-supplied schema value is preserved in `RECOVERY_SNAPSHOT.json`. The separate checkpoint `run_manifest.json` schema remains unchanged because it is a distinct manifest schema.

Redundancy cleanup removed callback/wrapper round trips that did not provide an abstraction boundary: runtime JSON readers no longer receive a main-runner wrapper that delegates back to runtime; recovery-stage planning receives already-materialized reusable/rejected mappings instead of calling a main wrapper that delegates back to runtime; and one-line hardlink/path-prefix wrappers were removed in favor of direct runtime calls. Compatibility/composition wrappers that existing callers/tests use were retained.

The targeted cleanup validation on the synchronized branch passed Python compile, `git diff --check`, Ruff, and the focused E8 regression suite with `113 passed`. Temporary transformation workflow files were removed by the validated cleanup commit. Full ordinary PR-gate validation is still required on the final human-authored head before merge.
