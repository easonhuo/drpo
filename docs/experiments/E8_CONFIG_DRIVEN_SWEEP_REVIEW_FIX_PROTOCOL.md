# E8 config-driven sweep repair invariants

Status: engineering review / not a scientific result.

Scope: harden the `eight_task_coldstart_lambda_v1` orchestration refactor introduced by PR #340 without changing the frozen scientific kernel, paper taper, current-policy surprisal, all-unique-negative consumer, 1200-update horizon, optimizer, data, task runtime, scheduler topology, or result interpretation.

## Locked repair invariants

1. `experiment_id` is metadata selected by the YAML config. The shell runner and bootstrap must not maintain an independent default identity. An explicit `E8_COLDSTART_EXPERIMENT_ID` override is permitted only as a fail-closed equality assertion against the config value.
2. Historical experiment IDs remain immutable scientific/provenance identities. The existing 208-cell cold-start, 199-cell lambda-completion, and 140-cell lambda-curve-completion IDs must reject changes to their frozen parameterization, seeds, task grids, grid hashes, cell counts, or endpoint policy even when a caller recomputes self-consistent hashes.
3. A generic new cold-start-family ID may change transfer-task positive-only seeds, transfer Exp seed, active task grids, and cell count only within the existing locked trainer/runtime domain. This does not authorize changes to the canonical scientific kernel.
4. Countdown remains bound to the locked paper runtime. Its configured positive coefficients must be a non-empty unique subset of the locked round-1/extension paper coefficients, and its scheduled seed offsets must be a subset of the paper seed offsets accepted by the canonical worker. Countdown may schedule zero scientific cells by using an empty `countdown_seed_offsets` list while retaining a non-empty locked coefficient grid for engineering liveness.
5. Canonical engineering liveness remains a two-update Countdown smoke test on the locked round-1 anchor `c=0.916290732`; it is an engineering gate and is independent of whether Countdown scientific cells are scheduled in the sweep.
6. A transfer-task uncontrolled endpoint is represented explicitly as `METHOD_GLOBAL` / `lambda=0`, not as an exponential cell with a non-positive lambda. Positive Exp lambdas remain strictly positive. `include_global_endpoint: true` adds one Global cell per active transfer task using the transfer Exp seed; it does not alter Countdown's historical Global behavior.
7. A zero-cell cold-start sweep is invalid and must fail during config validation rather than later in wave construction.
8. Empty transfer-task lambda grids mean zero cells for that task. Empty Countdown seed offsets mean zero Countdown scientific cells, but the locked Countdown coefficient grid remains present for canonical liveness and provenance.
9. Aggregate, task-result, audit, recovery, and nominal-wave geometry must be derived from `build_cells(config)`; task-performance events, support/structure diagnostics, and NaN/Inf events remain reported separately.
10. Temporary CI trigger files, temporary repair scripts, and temporary workflow overrides must not remain in the final PR tree.

## Required regression checks

The final branch must preserve the existing 208/199/140 cell matrices and wave geometry; accept a synthetic unseen experiment ID with changed transfer grids/seeds/cell count; reject drift under each frozen historical ID; reject malformed/non-integer seeds and zero-cell matrices; preserve Countdown liveness with zero scheduled Countdown cells; verify runner/bootstrap config identity resolution and mismatch rejection; verify optional transfer Global endpoints without treating `lambda=0` as Exp; and run Python compilation, shell syntax checks, focused E8 tests, `git diff --check`, plus the repository PR gate.

No smoke test, static check, or focused unit test produced under this protocol is a scientific result.