# Runtime-resource acceptance server correction 03

## Identity and evidence boundary

- Claim: `GOV-RUNTIME-RESOURCE-ACCEPTANCE-HARNESS-01`.
- Harness commit used by the third target-server run: `6db243cd0127d709cdbed8c24198ccb63b4a2081`.
- Pinned GPU selection-only commit: `133794a0c27e16cc546a3fdd052a2be3487b30aa`.
- Evidence package: `drpo_runtime_acceptance_20260715T114956Z.tar.gz`.
- This is engineering acceptance evidence only. It is not an E7/E8 scientific result and does not change any experiment state.

## Observed result

Stage 0 topology and Stage 1 resource-pool checks passed. Stage 2 E7 failed during the initial plan because measured CPU capacity could not support one worker. Stage 3 completed the H20 single-worker model-load, training-peak, evaluation-peak, and cleanup phases, then correctly blocked placement because measured system CPU occupancy left no safe capacity for one GPU worker. Stages 4 and 5 were consequently blocked.

No complete scientific sweep, OOM, NaN/Inf numerical failure, orphan process group, or repository mutation occurred. The evidence package manifest verifies successfully.

## Persistent external saturation

The third run still observed the unrelated high-CPU workload that blocked the previous attempt. Stage 0 recorded 605 processes, including 411 commands containing `joblib.externals.loky`, an active ResearchBench process, and an AIDE collector. The execution prompt required stopping before the acceptance run when this conflict remained visible, but the local executor proceeded. That protocol deviation did not alter repository code or scientific variables, but the run could not establish available production capacity.

## Newly exposed classification edge case

The previous correction normalized E7 capacity exhaustion only when an attempt-local `RUNTIME_REVALIDATION.json` existed. In this run the CPU pool was already saturated before a worker count could be selected, so the E7 selector stopped in the initial plan with the exact fail-closed message:

`measured CPU capacity cannot support one worker`

Because no runtime-revalidation document existed, the outer harness left Stage 2 as `FAIL`. Under the registered acceptance state machine this is safe-capacity unavailability and must be `BLOCKED`, provided no OOM or numerical-failure signature is present.

## Correction

The capacity normalizer now also recognizes the E7 plan-stage exact no-slot signatures:

1. `measured CPU capacity cannot support one worker`;
2. `measured CPU/RAM capacity produced no worker slot`.

The normalizer records `stage2_e7_cpu_v2/plan.log` as the evidence path and refuses normalization when an OOM or NaN/Inf fatal signature is present. Generic plan failures remain `FAIL`. Existing runtime-revalidation and GPU-placement classification behavior is unchanged.

Selector arithmetic, candidate grids, affinity/cgroup logic, resource thresholds, safety factors, worker counts, scientific variables, seeds, steps, data, models, evaluation, and immutable-selection rules are unchanged.

## Required rerun

A passing hardware acceptance still requires a fresh exact-head one-click run after the unrelated ResearchBench/AIDE/joblib workload has completed, or inside CPU pools/cgroups that are genuinely exclusive to this acceptance. The pre-run conflict gate must be obeyed: if the large external workload is still present, do not start the harness.
