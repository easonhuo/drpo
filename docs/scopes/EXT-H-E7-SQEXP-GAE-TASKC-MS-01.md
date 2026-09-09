# E7 task-specific top-4 c multi-seed pilot

## Identity

- Experiment: `EXT-H-E7-SQEXP-GAE-TASKC-MS-01`
- Parent benchmark: `EXT-H-E7-BENCH-01`
- Predecessor response-curve experiment: `EXT-H-E7-SQEXP-GAE-01`
- Profile: `d4rl9_task_specific_c_top4_multiseed`
- Execution claim: rerun the four strongest observed finite-c points for each D4RL task with five development seeds per task–c cell.
- Scientific role: D4RL/Hopper/Walker2d/HalfCheetah external-validity performance and stability screening only.

The task-specific multi-seed round has its own experiment identity. It must not deposit branches, manifests, aggregates, or terminal audits under the predecessor P1/P2/P3 experiment ID.

## Candidate-selection provenance

The four candidates per task are frozen before this multi-seed execution. They are the strict top four finite c values by `late_window_mean_800k_1m` across the completed P1, P2, and P3 development response curves on seeds `200,201`:

- P1 result commit: `33fa687352392df985679abddedb834535b10a3d`;
- P2 result commit: `7c1af9fc47ddd347c1bb28d43ad8a024ca95b8a8`;
- P3 result source: `easonhuo/drpo-results@ingest/e7`, run `E7_BENCH_JOINT_GAE_P3_LEFT_SATURATION_FULL_20260722_03`.

The exact per-task values are authoritative in `configs/e7_bench_joint_gae_taskc_top4_multiseed.json`. Positive-only is not a candidate c and is not part of this 180-branch confirmation matrix.

## Frozen matrix

The matrix contains the nine D4RL-v2 locomotion tasks. Every task–c cell uses seeds `200,201,202,203,208`.

`9 tasks × 4 c values × 5 seeds = 180 branches`.

Held-out seeds `204--207` remain untouched. Every branch uses canonical A2C, a jointly updated critic, trajectory-snapshot GAE with `lambda=0.95`, one million optimizer updates, evaluation every 50,000 updates, and ten evaluation episodes. No Positive-only or uncontrolled branch is added to this 180-branch candidate-confirmation matrix.

## Reporting protocol

The primary branch statistic is normalized return averaged over evaluations from 800k through 1M updates. For each task–c cell:

1. preserve all five seed results and complete evaluation trajectories;
2. sort the five seeds by the primary late-window statistic;
3. report the predeclared mean of the top three seeds as `top3_of_5_late_mean`;
4. also report all-five mean, standard deviation, median, final metrics, best-checkpoint diagnostics, and the exact three selected seed IDs.

For each task, the candidate c is selected by maximum `top3_of_5_late_mean`; ties are resolved by larger all-five median and then smaller c. This deterministic protocol is frozen before execution. External baselines remain quoted from their published sources; this pilot does not create a reproduction-versus-paper confrontation.

Task-performance degradation, support/variance boundary events, rollout failure, and NaN/Inf numerical failure remain separate. Failed runs and low-performing seeds are retained in raw outputs and all-five summaries even when they do not enter the top-three publication summary. Rollout and NaN/Inf counts in the terminal audit are derived from branch-level outputs rather than assumed.

## Execution boundary

The one-click entrypoint is:

```bash
bash scripts/run_e7_taskc_top4_multiseed_entrypoint.sh run
```

The outer entrypoint installs a run-local Python identity/profile shim. The shim binds the existing reviewed joint-GAE runner, bootstrap subprocess, and aggregate reader to `EXT-H-E7-SQEXP-GAE-TASKC-MS-01` only for this entrypoint, without changing the historical P1/P2/P3 experiment identity. It delegates to the fixed 180-branch launcher and performs the final branch-derived numerical/rollout audit update before RunSpec delivery.

Environment overrides are available for the canonical contract, run spec, work directory, and worker count. The launcher rejects a dirty checkout, validates the exact 180-branch matrix, supports identity-checked resume, and writes a terminal audit plus:

- `training_curves_long.csv`;
- `branch_results.csv`;
- `task_c_all5_summary.csv`;
- `task_c_top3_of5_summary.csv`;
- `task_selected_c_summary.csv`;
- `top3_selection_manifest.json`.

This is a development pilot. A fixed one-million-update horizon is not convergence or steady-state evidence, and the run does not authorize universal method-ranking claims.
