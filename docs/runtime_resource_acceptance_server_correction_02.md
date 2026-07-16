# Runtime-resource acceptance server correction 02

## Identity and evidence boundary

- Claim: `GOV-RUNTIME-RESOURCE-ACCEPTANCE-HARNESS-01`.
- Harness commit used by the second target-server run: `2c420122ef4f68188ab908f446464825cce31692`.
- Pinned GPU selection-only commit: `133794a0c27e16cc546a3fdd052a2be3487b30aa`.
- Evidence package: `drpo_runtime_acceptance_20260715T094114Z.tar.gz`.
- This is engineering acceptance evidence only. It is not an E7/E8 scientific result and does not change any experiment state.

## Observed result

Stage 0 topology and Stage 1 resource-pool checks passed. Stage 2 E7 measured-CPU revalidation and Stage 3 GPU placement then failed closed; Stage 4 and Stage 5 were consequently blocked. No complete scientific sweep, OOM, NaN/Inf numerical failure, orphan process group, or repository mutation occurred.

The E7 plan selected 60 workers, but the immediate three-sample revalidation observed approximately 189--190 busy cores inside the 192-CPU E7 affinity while the frozen policy budget was 163.2 cores. Adding the selected workers' reserved 74.63 cores projected approximately 265 cores, so `revalidate_runtime` correctly emitted `cpu_capacity_changed` and prohibited a silent worker downshift.

The H20 single-worker probe completed all required model-load, training-peak, evaluation-peak, and cleanup phases without OOM. The placement selector then correctly refused to admit even one GPU worker because the 144-CPU E8 affinity was already saturated by external CPU work.

Stage 0 process inventory explains the abrupt pressure change: the server was concurrently running a large unrelated ResearchBench/AIDE workload with hundreds of `joblib.externals.loky` workers whose affinity covered the host CPUs. The harness is forbidden to kill, renice, migrate, or rebind unrelated processes, so the only safe result was to stop.

## Classification defect

The selector behavior was correct, but the outer stage wrappers converted every delegated exception into `FAIL`. That contradicts the registered acceptance state machine:

- unavailable safe CPU/RAM/GPU capacity is `BLOCKED`;
- contract, identity, process, cleanup, OOM, numerical, or code failures are `FAIL`.

The second server run is therefore a capacity-blocked acceptance attempt, not evidence that the E7 or GPU selector implementation is broken.

## Correction

The harness now normalizes only two narrowly evidenced cases from `FAIL` to `BLOCKED`:

1. E7 attempt-local `RUNTIME_REVALIDATION.json` has `decision=BLOCK` and every failure belongs to `cpu_capacity_changed` or `memory_capacity_changed`.
2. GPU placement log contains a registered safe-capacity exhaustion signature and no OOM/numerical-failure signature.

All other failures remain `FAIL`. Selector arithmetic, candidate grids, affinity/cgroup logic, resource thresholds, safety factors, worker counts, scientific variables, seeds, steps, data, models, evaluation, and immutable-selection rules are unchanged.

## Required rerun

A passing hardware acceptance still requires a fresh exact-head one-click run after the unrelated high-CPU workload has completed, or inside CPU pools/cgroups that are genuinely exclusive to this acceptance. Do not lower safety headroom, bypass revalidation, kill unrelated processes through the harness, or manually reduce the immutable selected worker count.
