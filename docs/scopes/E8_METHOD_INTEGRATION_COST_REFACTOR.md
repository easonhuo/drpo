# E8 method integration cost refactor

## Status

Code-only architecture refactor. No scientific experiment is launched by this task, no scientific result is produced, and no frozen E8 scientific variable is changed.

Base repository state: `easonhuo/drpo@51b54f1605f0417b0873269e4614d07b2507d3d7` on `main`.

Development branch: `dev/e8-method-integration-cost-refactor-01`.

Relevant formal experiment remains `EXT-C-E8-MULTITASK-BASELINE-MATRIX-01`, status **not_run**. Its frozen 176-cell protocol, task set, seeds, method grids, 1,200-step horizon, seed-4000 -> seed-5000 barrier, evaluation semantics, and terminal-audit requirements are out of scope for scientific change.

## Problem statement

The current `src/drpo/e8_multitask_exp_tuning.py` is both the E8 scientific experiment implementation and a large experiment engine. A new method can currently require edits across cell representation, planning, dispatch, liveness, scheduling, result materialization, aggregation, recovery identity, terminal audit, packaging, and tests.

The recent AsymRE / Joint Fitted-Reference beta-TOPR / canonical DPO baseline-matrix capability exposed this coupling. TOPR itself already had a canonical scientific implementation, but method integration propagated through the surrounding E8 execution pipeline. This task addresses that integration cost rather than changing any method mathematics.

## Goal

Make the cost of adding a future E8 method proportional to the method-specific work.

After this refactor, adding a method with an existing canonical trainer should normally require only:

1. method implementation or adapter registration;
2. method-specific configuration / parameter grid;
3. method correctness tests.

The generic scheduling, recovery, result materialization, packaging, and common terminal-audit infrastructure should not require edits merely because a new method name or method-specific hyperparameter exists.

This is an architecture and maintenance goal, not a promise about scientific-training time.

## Owner-approved Python paths

The repository owner explicitly approved the following three new Python paths and responsibilities in the active design discussion before implementation:

- `src/drpo/e8_multitask_orchestration.py` — method-agnostic experiment planning/execution orchestration: cell scheduling, waves/dynamic queue, GPU slots, seed-batch barrier, generic run-all/run-wave control, and execution bookkeeping.
- `src/drpo/e8_multitask_results.py` — method-agnostic result materialization and aggregation: task-local results, CSV/curve materialization, grouped metrics, and aggregate summaries.
- `src/drpo/e8_multitask_runtime.py` — method-agnostic reliability/runtime concerns: recovery/resume, provenance/manifests, terminal audit plumbing, finalize/package, log compaction, and engineering self-test support where those concerns are not scientific-method-specific.

`src/drpo/e8_multitask_exp_tuning.py` remains the existing E8 scientific entry module and retains method/scientific responsibilities. The existing shell/bootstrap entrypoints remain external launch wrappers; this refactor does not move Python orchestration into shell.

These exact new Python paths are authorized under `GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02`. Adding another new Python path or materially changing one of these responsibilities requires a new explicit approval.

## Architecture contract

### 1. Scientific/method layer

`src/drpo/e8_multitask_exp_tuning.py` owns scientific method semantics and method-specific lifecycle hooks, including existing canonical EXP, AsymRE, TOPR, and DPO behavior.

The method layer may know method names and method-specific parameters. Generic infrastructure must not need to understand the mathematics of those parameters.

### 2. Standard cell contract

Generic infrastructure must operate on a stable common cell identity plus an opaque method-parameter payload rather than growing a new top-level `Cell` field for every future method parameter.

Backward compatibility is mandatory. Existing historical cell keys, plan identities, recovery identities, and result artifacts must remain interpretable and, where currently frozen, unchanged.

A compatibility layer may preserve existing named fields such as `rho`, `lambda`, `delta_v`, `beta`, and `dpo_initialization` at public/artifact boundaries while internally reducing generic infrastructure dependence on them.

### 3. Method registry / method specification

Introduce one authority that maps a method identity to its method-specific behavior. The exact class/function shape may be adjusted during implementation, but it must cover the responsibilities that genuinely vary by method, such as:

- parameter-grid expansion / method parameters;
- stable method-specific cell identity formatting where required for backward compatibility;
- training dispatch;
- method-specific liveness behavior;
- method-specific audit evidence/invariants;
- method-specific result-parameter projection for backward-compatible artifacts.

The registry must not become a second experiment registry and must not duplicate scientific protocol state already owned by config/handoff.

### 4. Orchestration layer

`e8_multitask_orchestration.py` owns execution mechanics that should not vary with method mathematics:

- plan/wave execution;
- dynamic queue;
- GPU slot accounting;
- generic cell start/finish/failure handling;
- seed-batch barrier mechanics;
- generic task-completion publication trigger;
- generic recovery-checkpoint trigger.

It must dispatch through the method contract rather than accumulate `if method == ...` branches for future methods.

### 5. Results layer

`e8_multitask_results.py` owns generic materialization and aggregation. It must not hard-code future algorithms merely to discover their parameter names.

Method-specific parameter projection must come from the method contract. Existing output columns required by current experiments remain supported.

### 6. Runtime layer

`e8_multitask_runtime.py` owns generic execution reliability and artifacts. A method with a special invariant, such as DPO frozen-reference invariance, supplies method-specific audit evidence/hook logic through the method layer; runtime invokes it generically.

Runtime must not grow a new hard-coded branch for every future method.

## Compatibility and scientific invariants

This refactor must preserve all of the following:

- all historical E8 config identities and validation behavior unless a compatibility bug is discovered and separately approved;
- the exact current `EXT-C-E8-MULTITASK-BASELINE-MATRIX-01` 176-cell geometry;
- exact current transfer seeds `[4000, 5000]` and the hard seed-batch barrier;
- AsymRE grid `delta_v=[-1.0,-0.9,-0.75,-0.5,0.0]`;
- TOPR grid `beta=[0.25,0.5,1.0]` and the existing canonical Joint Fitted-Reference TOPR implementation without reimplementing its objective;
- DPO grid `beta=[0.05,0.1,0.2]`, fresh-LoRA formal initialization, and audited canonical PR-268 scientific semantics;
- fixed 1,200-update horizon and existing stopping/evaluation semantics;
- existing separation of task-performance failure, structural/support diagnostics, and NaN/Inf numerical failure;
- existing result status `not_run` before scientific execution;
- existing external shell/bootstrap invocation compatibility unless a purely mechanical import/entrypoint adjustment is required and covered by regression tests.

No scientific run is authorized by this scope.

## Non-goals

This task must not:

- modify TOPR, AsymRE, DPO, EXP, or DRPO mathematics;
- add DAPOPR or any other new scientific method;
- change method rankings, metrics, seeds, thresholds, task/data scale, training horizon, evaluation protocol, or convergence rules;
- launch smoke/pilot/formal model training as scientific evidence;
- edit `docs/handoff.md` directly or change `experiments/registry.yaml`;
- create new governance gates beyond already approved behavior;
- introduce a general repository-wide harness abstraction; this task is intentionally scoped to E8;
- create additional Python modules beyond the three owner-approved paths above without a new explicit approval.

## Implementation plan

### Phase 0 — characterization lock

Before moving responsibilities, lock current observable behavior with focused regression coverage. At minimum cover:

- complete 176-cell set and ordering/identity where ordering is contractual;
- current cell keys for AsymRE, TOPR, DPO, EXP/baselines;
- plan geometry and 12 seed-local nominal batches;
- seed-4000 -> seed-5000 barrier semantics;
- method-aware liveness dispatch;
- task-local result materialization;
- baseline-matrix aggregation schema/geometry;
- recovery/reuse identity;
- terminal audit of common invariants plus DPO frozen-reference evidence;
- CLI command availability and existing shell-facing invocation.

No new scientific acceptance threshold is introduced.

### Phase 1 — method contract inside the existing module

First establish the method registry/specification while code is still physically co-located. Convert method-specific branching at generic boundaries to the registry/hook contract without changing output behavior.

This phase must prove that method knowledge can be localized before files are split.

### Phase 2 — physical responsibility split

Move already-separated responsibilities into the three approved modules in this order:

1. results/materialization;
2. orchestration/scheduling;
3. runtime/recovery/audit/package.

Keep `e8_multitask_exp_tuning.py` as the stable scientific/CLI-facing module or a compatibility facade as required. Avoid broad renaming or unrelated cleanup.

### Phase 3 — extension-cost acceptance test

Add a test-only dummy method/specification with a synthetic parameter payload. It must pass the non-scientific path from cell expansion through plan, generic orchestration fixtures, result materialization/aggregation, recovery identity, and terminal-audit plumbing without production edits to `e8_multitask_orchestration.py`, `e8_multitask_results.py`, or `e8_multitask_runtime.py` for the dummy method name.

The dummy method is test-only and is not an experiment or scientific baseline.

### Phase 4 — cleanup and final compatibility audit

Remove transitional duplication only after regression parity is demonstrated. Keep compatibility exports/aliases where they materially reduce downstream breakage.

Review the final diff specifically for accidental scientific changes and method-name leakage into generic infrastructure.

## Acceptance criteria

The task is complete only when all of the following hold:

1. Current formal baseline-matrix config still expands to exactly 176 cells: 80 AsymRE, 48 TOPR, 48 DPO.
2. Existing frozen cell keys and recovery identities used by current/historical E8 configurations remain stable.
3. Current baseline-matrix aggregation and terminal-audit semantics remain behaviorally equivalent.
4. Historical EXP/cold-start tests remain green.
5. TOPR remains canonical/import-based; no TOPR objective is copied into the refactor.
6. DPO scientific semantics and frozen-reference invariant are unchanged.
7. `e8_multitask_orchestration.py`, `e8_multitask_results.py`, and `e8_multitask_runtime.py` do not require a production code edit when the test-only dummy method is added through the method contract.
8. A repository diff review finds no changes to frozen scientific config values or experiment status.
9. No scientific training is run or reported as a result.
10. Applicable static/unit/regression tests actually executed are reported; unexecuted heavy/GPU tests are explicitly listed as unexecuted.

## Method-integration cost target

For a future method whose canonical trainer already exists and whose lifecycle fits the method contract, target integration work is:

- method adapter/specification;
- config parameterization;
- method correctness/regression tests;

with no method-name-specific modifications to generic orchestration, result, or runtime modules.

The intended engineering target is to reduce a similar baseline integration from multi-day pipeline surgery to roughly half-day to one-day work, excluding genuinely new scientific algorithm implementation and GPU training time. This is a maintenance target, not a formal SLA.

## Review / rollback strategy

Implementation stays on the single branch `dev/e8-method-integration-cost-refactor-01`. Failed iterations, test fixes, simplifications, and review repairs stay on this branch rather than creating replacement branches.

Each phase should leave the branch in a reviewable state. If compatibility cannot be maintained without changing scientific semantics, stop the refactor and report the conflict instead of weakening the frozen experiment contract.

No merge to `main` is authorized until explicit owner approval after diff/test review.