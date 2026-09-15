# E8 Multitask Results/Aggregation Split

## Scope

This is a code-only architecture refactor stacked on `dev/e8-multitask-inputs-split-01` after its validated input/data extraction.

The goal is to move method-agnostic result materialization and aggregation responsibilities out of `src/drpo/e8_multitask_exp_tuning.py` into the already-existing `src/drpo/e8_multitask_results.py` module.

No new Python file is created by this task.

## Scientific boundary

This refactor must not change:

- experiment IDs, tasks, banks, split sizes, seeds, parameter grids, or training horizon;
- EXP / AsymRE / TOPR / Reciprocal / DPO mathematics;
- DPO initialization or frozen-reference semantics;
- scheduler concurrency, execution geometry, seed barriers, or recovery identity;
- evaluation metrics, terminal-audit criteria, convergence semantics, or result status;
- output schema values, grouping semantics, parameter identities, or aggregate weighting.

`EXT-C-E8-MULTITASK-BASELINE-MATRIX-01` remains `not_run`. No scientific run is part of this task.

## Design rule

Prefer mechanical relocation over redesign.

`e8_multitask_results.py` should own pure or result-facing operations that consume cells/result rows and produce normalized rows, grouped summaries, aggregate records, or CSV/JSON-ready structures. The main runner remains the composition root and retains training kernels, method registration, scheduler orchestration, recovery/package lifecycle, self-test, and CLI dispatch.

Do not move model forward passes, optimizer code, DPO/paper training kernels, scheduler logic, recovery/package code, or engineering self-test into the results module.

Do not introduce a new manager/factory/base-class layer merely to enable the move.

## Candidate responsibilities to relocate

The implementation should inspect the current stacked tree and move only responsibilities with clean dependency direction, expected to include some or all of:

- result-row normalization/materialization beyond the existing `common_result_row` helper;
- task/method/parameter grouping for aggregation;
- single-method and matrix aggregate record construction;
- analysis-ready/result-summary derivation that depends only on result rows and method metadata supplied by the caller;
- CSV-ready aggregate ordering and stable parameter grouping/projection.

Method-specific scientific metadata should be passed in through existing `MethodSpec` callbacks or plain mappings rather than making `e8_multitask_results.py` import the main runner.

## Dependency direction

Allowed:

`e8_multitask_exp_tuning.py -> e8_multitask_results.py`

Forbidden:

`e8_multitask_results.py -> e8_multitask_exp_tuning.py`

The results module may depend only on standard-library types and generic protocols/helpers unless an already-existing dependency is demonstrably required.

## Implementation stages

1. Inventory current result/aggregation top-level functions and their dependencies.
2. Select the clean result-only subset; record any boundary left in the main runner and why.
3. Move functions mechanically, preserving names/signatures where practical and using compatibility aliases only where existing tests/callers require them.
4. Run compile, focused E8 tests, method-integration contract, Ruff, governance checks, and broad pytest.
5. Review the final diff for scientific-semantic drift and measure line-count reduction in `e8_multitask_exp_tuning.py`.

## Acceptance

The task is complete only if:

- the selected result/aggregation functions have a single implementation authority in `e8_multitask_results.py`;
- no circular import is introduced;
- current result schemas and grouping semantics are unchanged;
- existing tests pass without weakening scientific assertions;
- no new gate is added;
- the formal experiment remains `not_run`;
- the PR remains unmerged until explicit repository-owner approval.
