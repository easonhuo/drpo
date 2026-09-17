# E8 Multitask Training Splits

## Scope

This is a code-only architecture refactor stacked on the validated recovery/runtime split for `EXT-C-E8-MULTITASK-BASELINE-MATRIX-01`.

The formal experiment remains `not_run`. No scientific training run is part of this task.

The repository owner explicitly approved the following exact new Python paths in the active conversation:

- `src/drpo/e8_multitask_selftest.py`
- `src/drpo/e8_multitask_warmstart_training.py`
- `src/drpo/e8_multitask_canonical_bridge.py`

The work proceeds in independently validated stages. Stage P1 moves only the engineering self-test harness. P2 isolates the historical warm-start/rho/dense native trainer as deprecated legacy reproduction code. P3 may then move the canonical cold-start compatibility bridge. No stage may silently redesign scientific logic.

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

The exact transformed tree passed Python compilation, `tests/test_e8_multitask_p0.py` with `113 passed`, `bash tests/test_e8_method_integration_contract.sh`, Ruff on the changed Python/test surface, and `git diff --check`. The transformation workflow restored the repository's ordinary PR Gate workflow before committing the source tree.

### P1 post-extraction review hardening

A follow-up correctness/redundancy review found one evidence-chain weakness in the engineering-only failure injection: the harness injected return code `73`, but the final report previously repeated that value as a constant without first proving that the scheduler-recorded failed cell and recorded return code were the intended injected failure. The P1 implementation now validates the scheduler's actual `failed_cells` and `results` against the intended first cell and return code `73` before recovery proceeds, and the report uses the validated evidence rather than treating the constant itself as proof. The existing end-to-end engineering self-test traverses this assertion, so a wrong failed cell or return code now fails the regression path closed.

The same review removed the remaining hard-coded `16` from the dynamic-refill diagnostic text and instead reports the configured `max_concurrent_cells`. The `bind_host()` compatibility bridge remains intentionally unchanged for P1: it is transitional composition glue, and redesigning that dependency surface before P2/P3 would expand this stage into a scheduler/composition refactor. No scientific or formal-execution semantics are changed by this hardening.

## P2 — deprecated historical warm-start/rho/dense trainer isolation

P2 isolates the native historical multitask trainer that is explicitly forbidden for formal cold-start execution. The extracted module is **deprecated legacy reproduction/compatibility code**, not a current first-class training path. Formal cold-start continues to dispatch exclusively through the canonical old-code path.

### P2 reviewed ownership boundary

The extracted `e8_multitask_warmstart_training.py` owns the historical-only implementation for:

- the warm-start reference-model loader used by historical calibration/training;
- historical prompt/completion encoding, batch movement, completion-log-probability, and current near/far extreme selection helpers;
- historical trainable-state hashing and gradient-budget calibration helpers;
- historical non-cold calibration execution;
- the historical native generation/evaluation helpers;
- the historical row dataset and split loader;
- the native rho/dense `_train_cell_impl` training loop, including the pre-existing Positive-only/EXP loss path, optimizer/scheduler behavior, logging, evaluation, and adapter saves.

The composition root intentionally retains shared functions that are also used by canonical cold-start, DPO, recovery, or common orchestration. In particular, P2 does **not** move `_seed_everything`, `_cell_identity`, `_prepare_cell_output`, `_summarize_evaluations`, `cmd_reload_adapter`, `_verify_fresh_process_adapter_reload`, `_adapter_weight_file`, the `train_cell` dispatcher, or any canonical/DPO trainer. Reference-manifest construction/validation and `_load_ready_inputs` also remain in the composition root because they participate in preparation/inheritance and are not the native trainer itself.

Compatibility symbols remain in `e8_multitask_exp_tuning.py` only where current callers/tests still require that surface. A post-extraction redundancy cleanup removed seven aliases with no remaining repository consumer: `RowDataset`, `_move_batch`, `_stack_encoded`, `_trainable_state_sha256`, `_raw_gradient_norm`, `_calibration_rows`, and `_generate_completions`. The compatibility facades that still have consumers remain deliberately in place.

The extracted module does not import `e8_multitask_exp_tuning.py` back. Shared composition-root behavior is supplied explicitly at delegation time through `WarmstartTrainingBindings`, avoiding another process-global host-binding layer.

P2 is a relocation/isolation only. It preserves the historical formulas and behavior, including the existing cold-start rejection in the historical model loader/native trainer. It does not make the deprecated historical native path reachable from formal cold-start.

### P2 validation outcome

P2 source and cleanup were validated without running a scientific experiment:

- Python compilation passed for `e8_multitask_exp_tuning.py`, `e8_multitask_selftest.py`, and `e8_multitask_warmstart_training.py`;
- `tests/test_e8_multitask_p0.py`: `113 passed`;
- `bash tests/test_e8_method_integration_contract.sh`: passed;
- broad pytest excluding two failures proven independently on the stacked PR base: `1365 passed, 27 skipped, 2 deselected`;
- the two deselected tests were rerun on base commit `79009c487be6d64ef7f1d54b3f5dfa33808cc4c2` and failed there identically, establishing that they predate P2;
- Ruff passed after correcting one P1 engineering-self-test exception type (`RuntimeError` to `TypeError`) with no scientific behavior change;
- the temporary validation workflow removed itself after the successful run.

The two pre-existing base failures are unrelated to P2: one is exact floating-point equality in the historical AsymRE boundary-dense test, and the other is a stale expected wording string in a Countdown result-delivery documentation test. P2 does not repair either unrelated baseline issue.

The formal experiment `EXT-C-E8-MULTITASK-BASELINE-MATRIX-01` remains `not_run`.

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
