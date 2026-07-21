# PAPER-CODE-VALIDATION-01 Countdown Correctness Re-audit

**Date:** 2026-07-21
**Parent claim:** `PAPER-CODE-REFERENCE-01`
**Validation claim:** `PAPER-CODE-VALIDATION-01`
**Experiment:** `EXT-C-E8-TAPER-0.5B-01`
**Base branch:** `dev/paper-code-reference-01`
**Base commit:** `a3dc2896652364a9942942ab54dc32aaf42e9f7c`
**Scientific-status impact:** none

## Scope

This audit covers the existing Countdown dependency-light algorithm core and the
reviewer-facing Transformer/PEFT runtime. It validates migration correctness and
engineering lifecycle behavior only. It does not implement interrupted-run
resume, run real Qwen/PEFT/CUDA liveness, launch a scientific experiment, change
the registered protocol, or authorize method ranking.

Countdown remains an external-validity environment and does not replace D-U1
controlled mechanism identification. `EXT-C-E8-TAPER-0.5B-01` remains pilot /
not-run.

## Existing correctness evidence

The reviewer package already covers:

- exact expression cleaning and arithmetic verification;
- prompt masking and completion-only likelihood/statistics;
- first-occurrence unique-negative banks and per-prompt denominators;
- historical linear-surprisal compatibility;
- active-tail remoteness and all six registered reviewer weights;
- detached weight computation followed by a distinct gradient-bearing negative
  forward;
- prompt-balanced sampling, joint objective, calibration, raw-gradient budget,
  and first clipped AdamW update;
- explicit schema validation, model/adapter/input identities, delayed test access,
  checkpoint/evaluation lifecycle, and non-formal evidence boundaries;
- controlled fake-HF multi-step execution and public CLI dispatch.

## Reproduced defect

The runtime used `_release_model(model)`, whose body executed `del model` and then
`torch.cuda.empty_cache()`. The deletion affected only the helper's local argument.
The caller still retained `calibration_model` or `model`, so the previous model
could remain strongly referenced while the next model was loaded. On a real GPU
this could leave two model instances resident simultaneously and cause avoidable
out-of-memory failure.

A weak-reference regression using cyclic fake models reproduced the defect: the
old implementation reached the next model load while the previous model remained
alive.

## Minimal repair

The repair:

1. replaces the misleading helper with `_clear_device_cache()`, which runs
   `gc.collect()` and then `torch.cuda.empty_cache()` when CUDA is available;
2. explicitly deletes the caller-owned model variable in each `finally` block
   before clearing the cache;
3. adds a regression proving that the calibration model and each method model are
   collected before the next model load.

No objective, remoteness definition, method, coefficient, seed, data coordinate,
optimizer, scheduler, budget, checkpoint-selection rule, evaluation metric, or
scientific status changes.

## Tests executed

On the exact pre-fix Countdown file blobs present at the base commit, after
applying this patch:

- Python compilation of both modified files: passed;
- `paper_code/tests/test_common.py`: passed;
- Countdown public CLI dispatch test: passed;
- shared self-contained reviewer tests (`aggregate`, `common`, `controls`,
  `events`): passed.

The regression was also run against the old implementation and failed at the
expected boundary before the repair.

Ruff was unavailable in the isolated local environment and is not claimed as
executed. Real Qwen/PEFT/CUDA liveness was not run.

## Acceptance decision

**Countdown engineering correctness is accepted for the currently migrated
reviewer scope, conditional on application of the separately delivered minimal
repair package based on commit
`a3dc2896652364a9942942ab54dc32aaf42e9f7c`.**

The branch at the time of this record does not yet contain that two-file repair.
Therefore this document records a completed correctness verdict and a pending
repository after-image, rather than claiming that the repaired runtime is already
present at branch HEAD.

This decision does not establish real GPU memory behavior, real Qwen/PEFT/CUDA
compatibility, convergence, a formal experiment result, a terminal scientific
audit, or method ranking. Real-stack liveness remains a separate, explicitly
authorized future gate.
