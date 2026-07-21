# PAPER-CODE-VALIDATION-01 Countdown Repair Application Record

**Date:** 2026-07-21  
**Claim:** `PAPER-CODE-VALIDATION-01`  
**Experiment:** `EXT-C-E8-TAPER-0.5B-01`  
**Base commit:** `db5fae4527b6a17dfb7ec0da32902d8ad4ca067c`  
**Scientific-status impact:** none

## Applied engineering repair

The stable public module `drpo_reference.experiments.countdown` is now a small
facade over `countdown_runtime`. Before exposing the runtime it installs a
completed-model release guard. The guard first attempts to move the completed
model to CPU; for wrappers that cannot move devices, including quantized-style
wrappers, it clears the completed top-level parameter, buffer, and child-module
registries. It then runs garbage collection and clears the optional CUDA cache.

This prevents the completed calibration, method-training, or saved-checkpoint
model tensor graph from remaining resident when the next model is loaded, even
while the caller's local Python model object has not yet left scope.

The repair changes no objective, remoteness coordinate, method, coefficient,
seed, dataset size, optimizer, scheduler, training budget, checkpoint rule,
evaluation metric, scientific status, or manuscript value.

## Validation

- Python compilation passed;
- Countdown release, public CLI, and common runtime tests passed: 34 tests;
- Ruff check and Ruff format check passed for the complete standalone package;
- the direct-import module remains the same runtime module object, so existing
  monkeypatch-based reviewer tests and public imports preserve their behavior.

## Remaining separate gates

Real Qwen/PEFT/CUDA liveness, real GPU allocator behavior, interrupted-run
optimizer/scheduler resume, registered scientific execution, terminal scientific
review, and method ranking remain outside this engineering acceptance.
