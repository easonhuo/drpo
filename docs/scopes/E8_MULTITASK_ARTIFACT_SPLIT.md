# E8 Multitask Artifact/Package Split

## Scope

This is a code-only architecture refactor stacked on `dev/e8-multitask-results-split-01` after the validated input/data and result/aggregation extractions.

The goal is to move result-artifact completion, portable package construction, and package verification responsibilities out of `src/drpo/e8_multitask_exp_tuning.py` into the already-existing `src/drpo/e8_multitask_results.py` module.

No new Python file is created by this task.

## Scientific boundary

This refactor must not change:

- experiment IDs, tasks, banks, split sizes, seeds, parameter grids, or training horizon;
- EXP / AsymRE / TOPR / Reciprocal / DPO mathematics or DPO frozen-reference semantics;
- scheduler concurrency, execution geometry, seed barriers, recovery identity, or recovery checkpoint timing;
- evaluation metrics, terminal-audit criteria, convergence semantics, or result status;
- package required members, package inventory rules, excluded model-weight paths, SHA-256 semantics, ZIP member names, or reopen/tamper rejection behavior;
- canonical archive ownership or hardened-guard responsibility.

`EXT-C-E8-MULTITASK-BASELINE-MATRIX-01` remains `not_run`. No scientific run is part of this task.

## Design rule

Prefer mechanical relocation over redesign.

`e8_multitask_results.py` should own result-facing artifact operations that consume already-audited result state and produce completion manifests, portable text-first ZIP packages, or package-verification records. `e8_multitask_exp_tuning.py` remains the composition root and supplies experiment-specific values/callbacks.

Do not move terminal-audit adjudication, method-specific audit logic, model/optimizer code, scheduler logic, recovery scheduling/checkpoint policy, or engineering self-test orchestration in this task.

Do not add a new gate, manager/factory hierarchy, or another artifact protocol.

## Responsibilities to relocate

The implementation should move the single implementation authority for:

- `PACKAGE_REQUIRED_MEMBERS`;
- completion-manifest materialization currently performed by `_write_completion_manifests`;
- result payload discovery/filtering currently performed by `_result_payload_paths`;
- independent ZIP reopen/inventory/hash verification currently performed by `verify_result_package`;
- package construction currently performed by `cmd_package`;
- final result-marker materialization currently performed by `cmd_finalize`.

Compatibility wrappers may remain in `e8_multitask_exp_tuning.py` where existing CLI/tests call those names. Experiment-specific values such as experiment ID, config hash, execution class, expected cell count, engineering-self-test status, JSON writer, and SHA-256 helper should be passed explicitly rather than importing the main runner from the results module.

## Dependency direction

Allowed:

`e8_multitask_exp_tuning.py -> e8_multitask_results.py`

Forbidden:

`e8_multitask_results.py -> e8_multitask_exp_tuning.py`

The results module must not become an owner of scheduler/recovery policy or scientific method semantics.

## Validation

Validation must cover at least:

1. Python compile for both touched Python files.
2. Focused E8 regression tests and the method-integration contract.
3. Existing engineering self-test path through aggregate -> audit -> finalize -> package.
4. Package reopen verification and tampered-ZIP rejection behavior.
5. Ruff on changed Python files.
6. Handoff/governance checks.
7. Broad pytest, with any pre-existing failures characterized against the stacked base rather than silently repaired in this PR.

## Acceptance

The task is complete only if:

- the moved artifact/package functions have one implementation authority in `e8_multitask_results.py`;
- package contents and validation semantics remain unchanged;
- no circular import is introduced;
- no scheduler/recovery/audit scientific semantics are changed;
- no new gate is added;
- the formal experiment remains `not_run`;
- the PR remains unmerged until explicit repository-owner approval.
