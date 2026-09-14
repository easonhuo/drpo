# E8 multitask input/data split

## Status

Code-only architecture refactor. No scientific experiment is launched by this task, no scientific result is produced, and no frozen E8 scientific variable is changed.

Base repository state: `easonhuo/drpo@1830e672395ed4b881b147ad3c3a42f1b36074bf` on `main`.

Development branch: `dev/e8-multitask-inputs-split-01`.

Relevant formal experiment remains `EXT-C-E8-MULTITASK-BASELINE-MATRIX-01`, status **not_run**. Its frozen 176-cell protocol, task set, seeds, method grids, 1,200-step horizon, seed-4000 -> seed-5000 barrier, evaluation semantics, and terminal-audit requirements are out of scope for scientific change.

## Problem statement

`src/drpo/e8_multitask_exp_tuning.py` remains a large composition/scientific module even after the method-integration refactor. A substantial contiguous responsibility block covers input preparation and frozen-bank semantics: task-row normalization, deterministic train/validation/test partitioning, negative-bank validation/reconstruction, task-adapter loading for prepared rows, reference-remoteness bank derivation, and conversion from prepared inputs into the canonical paper-runtime representation.

These responsibilities are scientifically important but are not method-specific training kernels. Keeping them physically co-located with cell planning, DPO training, runtime/recovery, aggregation, terminal audit, packaging, and CLI dispatch makes the main module harder to review and increases the blast radius of future maintenance.

This task performs a physical responsibility split only. It does not redesign the data protocol.

## Goal

Create one input/data module that owns the E8 multitask prepared-input pipeline while preserving every current observable scientific and artifact behavior.

The intended first-stage result is that `e8_multitask_exp_tuning.py` becomes smaller and delegates input preparation through a stable module boundary, without changing method kernels, scheduler behavior, result schemas, frozen config values, or experiment status.

## Owner-approved Python path

The repository owner explicitly approved the following new Python path and responsibility in the active conversation before implementation:

- `src/drpo/e8_multitask_inputs.py` — E8 multitask input/data layer: task-row normalization; deterministic split construction and split audits; prepared-bank loading/materialization helpers; verified negative-candidate reconstruction and selection helpers; fixed reference-remoteness bank preparation and its diagnostics; task-adapter/instance preparation needed by that input pipeline; and conversion/bridging of prepared task rows into canonical paper-runtime inputs where that conversion is data-facing rather than optimizer/method-specific.

Extending the existing `e8_multitask_exp_tuning.py` is insufficient because the purpose of this task is to remove an already coherent, large, non-method-specific input responsibility from that 9k+ line composition module. This is a physical decomposition, not a new abstraction layer.

This exact path is authorized under `GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02`. No additional new Python path is authorized by this scope.

## Design rules

1. **Move first, redesign later only if separately approved.** Prefer mechanical extraction of existing functions and constants over new classes, factories, managers, protocols, or generic helper frameworks.
2. **No scientific semantic changes.** P0/Countdown row normalization, split sizes, hash ordering, 16-negative requirements, verifier-wrong requirements, reference-remoteness selection, model-facing text, reference scoring, and canonical input conversion remain behaviorally identical.
3. **No method-kernel movement in this phase.** EXP/AsymRE/TOPR/DPO loss or optimizer code stays outside the new input module.
4. **No scheduler/runtime movement in this phase.** Cell scheduling, seed barriers, recovery, terminal audit, result aggregation, packaging, and CLI orchestration stay where they currently live or in their existing generic modules.
5. **No new gate.** Existing validation/rejection behavior may be preserved, but this refactor must not strengthen it or introduce a new mandatory acceptance condition.
6. **Keep compatibility at the existing entry module.** Existing tests/callers that import public or historically used helpers from `e8_multitask_exp_tuning.py` should continue to work through imports/aliases where needed; do not force broad caller rewrites merely to complete the split.
7. **Avoid circular dependencies.** The new input module may depend on lower-level E8 task/adaptor/config utilities, but it must not import the composition module that imports it. Where an extracted function currently reaches back into composition-only helpers, either leave that function in the old module for this phase or pass the already-existing dependency explicitly without changing scientific behavior.

## Proposed extraction boundary

### Phase 1A — pure prepared-row and split logic

Move the lowest-risk contiguous functions first, including the current responsibilities around:

- deterministic prompt-hash ordering;
- P0 row normalization;
- Countdown train/validation row normalization;
- training-row audits;
- partition prompt-ID overlap audits;
- P0 split construction;
- Countdown split construction;
- read/write/materialization helpers whose only responsibility is prepared input state.

This phase should contain no Torch/model dependency if the current dependency graph permits it.

### Phase 1B — negative candidate and reference-remoteness preparation

After Phase 1A regression parity, move the input-facing logic for:

- reference-surprisal summaries and deterministic coverage-first rank selection;
- error-class coverage diagnostics;
- deterministic verified-wrong candidate reconstruction;
- reference-policy candidate scoring;
- fixed reference-remoteness bank derivation/materialization and its input-side audit records.

If a helper is tightly coupled to canonical model/runtime loading, keep the model-loading bridge in the composition module and move only the bank/data transformation beneath it. Do not force a circular or over-abstracted design merely to maximize line count moved.

### Phase 1C — canonical prepared-input conversion

Move only the data-facing portion of canonical paper-runtime input conversion and task-adapter/instance preparation when it can be done without moving scientific optimizer/method semantics.

The final boundary is determined by dependency direction, not by a target line count.

## Explicit non-goals

This task must not:

- change `configs/e8_multitask_baseline_matrix_formal.yaml` or any frozen scientific value;
- change the 176-cell baseline-matrix geometry or reciprocal-method capability added on current `main`;
- alter EXP, AsymRE, TOPR, Reciprocal-Linear/Quadratic, DPO, Positive-only, or Global mathematics;
- change seeds, task sets, negative counts, split sizes, learning rates, horizons, evaluation cadence, stopping rules, convergence semantics, or terminal-audit thresholds;
- edit `docs/handoff.md` directly or change `experiments/registry.yaml`;
- add a new governance/scientific gate;
- launch GPU training or report a smoke/pilot as a scientific result;
- split artifacts/recovery/self-test/scientific backends in the same PR;
- create another new Python module.

## Characterization and acceptance

Before and after each extraction step, preserve and re-run focused behavior covering at minimum:

- P0 and Countdown normalization/split behavior;
- exact prepared row counts and prompt-ID disjointness;
- exact 16-negative and verifier-wrong invariants;
- reference-remoteness bank identity/selection semantics and existing audit fields;
- current canonical prepared-input conversion behavior;
- current formal 176-cell config expansion and seed barrier as a regression guard against accidental cross-cutting edits;
- existing method-integration contract;
- historical E8 focused tests that cover the moved helpers;
- Python compile and Ruff;
- repository governance/handoff authority checks required for code-only updates.

No new scientific acceptance threshold is introduced.

## Review checkpoints

### Checkpoint A — scope/plan only

This document exists on the dedicated branch before production code changes.

### Checkpoint B — first mechanical extraction

Create `src/drpo/e8_multitask_inputs.py`, move the lowest-risk pure row/split helpers, preserve compatibility imports in `e8_multitask_exp_tuning.py`, and run focused tests. Review the diff before moving model-coupled reference-remoteness logic.

### Checkpoint C — complete safe input boundary

Continue only with functions whose dependency direction remains clean. Any function that would require scientific-kernel redesign, circular imports, or a new abstraction should remain in the old module and be listed as deferred rather than forced across the boundary.

### Checkpoint D — final validation

Run focused E8 tests, the method-integration contract, full pytest when practical, Ruff, handoff authority, and governance-stage validation. Report unexecuted GPU/heavy tests explicitly. No scientific run is part of acceptance.

## Rollback / stop rule

Implementation stays on `dev/e8-multitask-inputs-split-01`. If the extraction cannot preserve existing scientific semantics or requires a new gate, frozen-variable change, or additional Python module, stop and report the conflict instead of expanding scope.

No merge to `main` is authorized until explicit repository-owner approval after diff/test review.
