# E8 Multitask Training Splits

## Scope

This is a code-only architecture refactor stacked on the validated recovery/runtime split for `EXT-C-E8-MULTITASK-BASELINE-MATRIX-01`.

The formal experiment remains `not_run`. No scientific training run is part of this task.

The repository owner explicitly approved the following exact new Python paths in the active conversation:

- `src/drpo/e8_multitask_selftest.py`
- `src/drpo/e8_multitask_warmstart_training.py`
- `src/drpo/e8_multitask_canonical_bridge.py`

The work proceeds in independently validated stages. Stage P1 moves only the engineering self-test harness. P2 isolates the historical warm-start/rho/dense native trainer as deprecated legacy reproduction code. P3 then isolates the canonical cold-start compatibility bridge. No stage may silently redesign scientific logic.

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

P3 isolates the compatibility layer that adapts the unified E8 interface to the frozen canonical Countdown/paper implementations. The bridge owns adaptation and dispatch only; scientific kernels remain the existing canonical modules.

### P3 reviewed ownership boundary

The pre-extraction call-site review assigns the following responsibilities to `e8_multitask_canonical_bridge.py`:

- method-to-paper parameter conversion for Positive-only, Global, EXP, Reciprocal-Linear, Reciprocal-Quadratic, AsymRE, and TOPR;
- canonical grid selection and runtime-grid validation/adaptation;
- the temporary legacy arena/paper runtime bridges, including exact patch/restore behavior for LoRA runtime parameters, generation sampling parameters, optimizer weight decay, scheduler warmup, and canonical validators;
- canonical baseline-grid identity metadata used by AsymRE/TOPR reporting;
- canonical cold cell execution and canonical method liveness, including transfer-task evaluator adaptation while keeping Countdown on the exact canonical task interface;
- DPO cold execution and liveness, including frozen-reference initialization/copy semantics, shared-SFT adapter contract/provenance verification, prompt-balanced DPO aggregation helpers, and DPO-specific numerical diagnostics;
- canonical no-calibration record verification needed by the canonical/DPO execution paths.

The composition root retains generic/shared orchestration that is not itself the canonical compatibility bridge: `Cell`/`MethodSpec` registration, cell construction/keying, `train_cell` failure capture/dispatch, `_cell_identity`, `_prepare_cell_output`, `_summarize_evaluations`, generic adapter reload subprocess support, split/input preparation, scheduler/recovery, aggregation/audit/finalization, and CLI handling.

The extracted bridge must not import `e8_multitask_exp_tuning.py` back. Generic composition-root behavior required by the bridge must be passed explicitly through a binding object, following the P2 direction rather than introducing another process-global `bind_host()` layer.

P3 must preserve every existing temporary monkey-patch restore path under exceptions. It must not reimplement or alter canonical loss math, DPO math, parameter grids, seeds, horizons, task-runtime budgets, or initialization semantics. The existing DPO implementation remains the audited PR #268 semantics port because no canonical DPO runtime is merged on the current stack.

### P3 validation plan

Before P3 is considered closed, the exact source head must demonstrate:

- Python compilation for the composition root and new bridge;
- focused multitask regression, including canonical runtime-bridge restoration and DPO initialization/provenance contracts;
- method-integration contract coverage;
- Ruff on the changed Python surface;
- broad pytest with any unrelated base-preexisting failures separately demonstrated rather than silently repaired in P3;
- ordinary PR/Evidence checks on the final human-authored P3 head;
- no formal scientific run and no change from `EXT-C-E8-MULTITASK-BASELINE-MATRIX-01 = not_run`.

### P3 validation outcome

P3 source extraction is committed at `7ec8146215ef208d8cdc3d42a75e7537345509dc` as a relocation-only canonical compatibility split. Relative to the P2 closing commit `697a9b62e3ca2e9a48e8f1996c39bdbea5bdd8e2`, `e8_multitask_exp_tuning.py` changed by `+63/-1652` (net `-1589` lines), while the new `e8_multitask_canonical_bridge.py` contains 1,823 lines. The composition root is about 5,174 lines after the extraction.

The bridge owns 28 canonical/DPO adaptation functions and is built per call through `CanonicalBridgeBindings`; it does not import `e8_multitask_exp_tuning.py` back and does not install a process-global host binding. Standard-library dependencies and `TaskInstance` are imported from their owning modules rather than being smuggled through the host surface. Existing compatibility entry points in `e8_multitask_exp_tuning.py` are thin delegates only; the full canonical/DPO implementations are not duplicated there.

The correctness/redundancy review confirmed that the legacy arena/paper bridges retain `finally` restoration for all temporary patched runtime symbols, that canonical/DPO dispatch still flows through the existing method-spec/cold-start facade, and that the 16 source-structure regression checks were retargeted to inspect the extracted implementation rather than weakened or removed. The remaining facade wrappers and four small composition dispatchers are intentionally retained as stable compatibility/monkey-patch surfaces rather than duplicate implementation authority.

The exact extracted source tree was validated without running a scientific experiment:

- Python compilation passed for the composition root, self-test module, deprecated warm-start module, canonical bridge, and focused test module;
- `tests/test_e8_multitask_p0.py`: `113 passed`;
- `bash tests/test_e8_method_integration_contract.sh`: passed across the existing method/runtime/recovery integration checks;
- Ruff passed on the changed Python/test surface;
- `git diff --check` passed;
- broad pytest excluding only the two failures already proven on stacked base `79009c487be6d64ef7f1d54b3f5dfa33808cc4c2`: `1365 passed, 27 skipped, 2 deselected`;
- both temporary P3 extraction workflows removed themselves before the final source commit.

The ordinary PR Gate and Evidence Locator Gate passed on the final human-authored P3 source head immediately preceding the workflow-authored extraction commit. The workflow-authored source commit itself reports `action_required` for those pull-request workflows because GitHub does not launch the ordinary PR jobs from that bot-authored commit; this is not a code-test failure. The repository-owner approval for all three exact new Python paths is also recorded durably in PR #370 so the existing code-change-budget approval check can validate the already-granted authorization rather than requiring a new decision.

No scientific training was launched. `EXT-C-E8-MULTITASK-BASELINE-MATRIX-01` remains `not_run`, and PR #370 remains Draft and unmerged pending the independent reviewer/owner merge decision.

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
