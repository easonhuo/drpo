# Scope contract: EXT-C-E8-MULTITASK-EXP-LAMBDA-DENSE-01

## Approved implementation scope

- Extend `src/drpo/e8_multitask_exp_tuning.py` without creating a new Python path.
- Add one task-local dense-lambda config and one shell entrypoint.
- Add protocol documentation and tests to the existing multitask P0 test file.
- Add a deferred E8 RunSpec only after the implementation SHA is frozen.

## Frozen scientific scope

- Seven tasks; Countdown and Spiral Matrix are excluded.
- Seven waves, exactly 16 Exp cells per wave, one task per wave.
- One predecessor tuning seed; no Positive-only rerun.
- One exact predecessor-selected lambda bridge per task.
- Frozen parent split/reference reuse; new-lambda calibration is rerun.
- Training, evaluation, no-test-access, and event-separation contracts are unchanged.

## Forbidden scope

- no new Python path;
- no change to P0 bank generation or task verifier semantics;
- no Countdown or Spiral Matrix rerun;
- no early stopping, test access, post-run private grid, or hidden cell deletion;
- no convergence, significance, universal superiority, categorical causal, or formal
  ranking claim;
- no merge or real GPU launch without separate explicit approval.

## Delivery

The RunSpec must use deferred registration and automatic text-first delivery to
`easonhuo/drpo-results@ingest/e8`. Model, adapter, checkpoint, optimizer, dataset, and
other binary artifacts are excluded.
