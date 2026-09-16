# E8 Multitask Training Splits

## Scope

This is a code-only architecture refactor stacked on the validated recovery/runtime split for `EXT-C-E8-MULTITASK-BASELINE-MATRIX-01`.

The formal experiment remains `not_run`. No scientific training run is part of this task.

The repository owner explicitly approved the following exact new Python paths in the active conversation:

- `src/drpo/e8_multitask_selftest.py`
- `src/drpo/e8_multitask_warmstart_training.py`
- `src/drpo/e8_multitask_canonical_bridge.py`

The work proceeds in independently validated stages. Stage P1 moves only the engineering self-test harness. P2 may then move the historical warm-start/rho/dense native trainer. P3 may then move the canonical cold-start compatibility bridge. No stage may silently redesign scientific logic.

## Scientific boundary

This refactor must not change:

- experiment IDs, tasks, banks, split sizes, seeds, parameter grids, training horizons, or evaluation budgets;
- EXP, AsymRE, TOPR, Reciprocal, Global, Positive-only, or DPO mathematics;
- DPO frozen-reference and shared-SFT identity semantics;
- canonical Countdown/paper implementation authority;
- scheduler concurrency, GPU-slot geometry, seed barriers, queue ordering, or dynamic-refill semantics;
- recovery identity, reusable-cell rules, checkpoint/import semantics, terminal-audit criteria, artifact/package semantics, or delivery semantics;
- task-performance, support/structure, and NaN/Inf reporting boundaries;
- result status or formal launch authorization.

Smoke tests, engineering self-tests, static checks, or limited probes remain engineering evidence only.

## Dependency rule

`e8_multitask_exp_tuning.py` remains the stable composition/CLI facade. Extracted modules may depend on lower-level stable modules, but must not create a circular import back through `e8_multitask_exp_tuning.py` for implementation authority.

The intended direction is:

`e8_multitask_exp_tuning.py -> {selftest, warmstart_training, canonical_bridge}`

and then into existing lower-level input/orchestration/runtime/results/canonical modules.

## P1 — engineering self-test extraction

Move the non-scientific engineering harness out of the production composition root. Candidate responsibilities include:

- engineering self-test config reduction;
- placeholder input/model fixture materialization;
- placeholder calibration/liveness gate materialization;
- dynamic-queue engineering audit;
- failure injection, recovery/repeat-run checks, package reopen/tamper checks, and final self-test report construction.

`e8_multitask_exp_tuning.py` should retain only the stable command-facing delegation required by existing callers/tests.

P1 must preserve the exact engineering semantics, including intentional failure, recovery, dynamic refill, repeat-run idempotence, aggregate/audit/finalize/package execution, tamper rejection, and `scientific_status=not_run`.

### P1 validated extraction outcome

The mechanical extraction was validated before publication and then pushed as source commit `a7398ec3da1f15a188b88fb9d3d771cabde331c4`. `e8_multitask_exp_tuning.py` decreased from about 8,073 lines to 7,535 lines; the new `e8_multitask_selftest.py` contains 674 lines. The main module keeps compatibility entry points while the extracted module owns the self-test implementation and does not import the main module back.

The exact transformed tree passed Python compilation, `tests/test_e8_multitask_p0.py` with `113 passed`, `bash tests/test_e8_method_integration_contract.sh`, Ruff on the changed Python/test surface, and `git diff --check`. The transformation workflow restored the repository's ordinary PR Gate workflow before committing the source tree. Ordinary PR Gate and Evidence Locator are still required on the final human-authored PR head before P1 is considered fully closed.

## P2 — historical warm-start/rho/dense training extraction

Only after P1 validation, move the native historical multitask trainer that is explicitly forbidden for formal cold-start execution. The extracted module may own the warm-start/rho/dense model loader, batch/sequence helpers, current-extreme selection, native training loop, native evaluator helpers, and related historical adapter/reload support.

Formal cold-start must continue to dispatch exclusively through the canonical old-code path.

## P3 — canonical cold-start compatibility bridge extraction

Only after P2 validation, move the compatibility layer that adapts the unified E8 interface to the frozen canonical Countdown/paper implementations. This may include legacy arena/paper runtime bridges, canonical grid/runtime adaptation, canonical cold cell execution/liveness, method-to-paper parameter conversion, DPO frozen-reference execution/identity, and shared-SFT contract enforcement.

This module is a bridge, not a reimplementation of the scientific algorithms.

## Validation

Each stage must be independently reviewable and must run at least:

- Python compilation;
- focused `tests/test_e8_multitask_p0.py` coverage relevant to the moved responsibility;
- engineering self-test regression where applicable;
- scheduler timing probe where applicable;
- `bash tests/test_e8_method_integration_contract.sh` where method/runtime boundaries are touched;
- Ruff;
- handoff/governance checks;
- broad pytest before the stage is considered complete.

The PR remains Draft and unmerged until explicit repository-owner approval. The formal experiment remains `not_run` throughout this refactor.
