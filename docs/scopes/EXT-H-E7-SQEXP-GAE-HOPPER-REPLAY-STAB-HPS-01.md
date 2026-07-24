# Hopper-medium-replay stability hyperparameter pilot

## Identity and role

- Experiment ID: `EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-STAB-HPS-01`
- Parent benchmark: `EXT-H-E7-BENCH-01`
- Direct predecessor: `EXT-H-E7-SQEXP-GAE-TASKC-MS-01`
- Environment: `hopper-medium-replay-v2`
- Scientific role: D4RL external-validity stability screening only.
- Status before execution: `not_run`.
- Formal evidence allowed: `false`.

The completed predecessor showed that Hopper-medium-replay can reach normalized returns near 100, but success is highly seed-dependent and many trajectories return to the low-performance region. This successor tests whether conservative actor updates and larger batches improve multi-seed late-window performance and retention. It does not perform a mechanism diagnosis and does not alter the controlled C-U1, D-U1, product-manifold, or nonlinear Gaussian experiments.

## Frozen scientific implementation

Every branch keeps the predecessor scientific path unchanged except for the two explicitly scanned optimization controls:

- canonical A2C;
- jointly updated critic;
- trajectory-snapshot GAE with `lambda=0.95`;
- thresholded exponential remoteness control with `taper_lambda=1`, threshold `0`, and reference distance `2`;
- one million optimizer updates;
- evaluation every 50,000 updates with ten episodes;
- canonical base learning rate `3e-4`;
- critic learning rate remains `3e-4` in every branch;
- actor optimizer learning rate is `3e-4 × actor_lr_multiplier`;
- batch size changes both the canonical minibatch and the replay-equivalent trajectory-snapshot refresh interval.

No critic expectile, GAE lambda, reward normalization, dataset, network, optimizer family, horizon, evaluation cadence, or remoteness formula is changed.

## Phase A: difficult-seed screening

Frozen grid:

- `c`: `[0.08, 0.125]`;
- actor learning-rate multiplier: `[1.0, 0.5, 0.25]`;
- batch size: `[256, 512]`;
- development seeds: `[200, 202, 203, 208]`.

This gives `2 × 3 × 2 × 4 = 48` independently trained branches. Seed 201 is deliberately excluded from screening because it was the clearly successful predecessor seed across several c values and would weaken the difficult-seed stability test.

For each of the twelve hyperparameter configurations, preserve every seed result and complete evaluation trajectory. Rank configurations lexicographically by:

1. larger median `late_window_mean_800k_1m` over the four seeds;
2. larger mean of the two lowest seed-level late-window means;
3. larger median final score;
4. smaller mean best-to-final drop;
5. smaller actor learning-rate multiplier;
6. larger batch size;
7. smaller c.

The first four criteria are scientific selection criteria. The final three are deterministic exact-tie breakers only.

## Phase B: fresh five-seed confirmation

The top two Phase-A configurations are rerun from fresh initialization with seeds `[200, 201, 202, 203, 208]`, producing ten new branches. Phase-B results are not assembled by reusing Phase-A trajectories.

The final confirmed configuration is selected with the same lexicographic rule, now over five seeds. Confirmation output must retain:

- all-five late mean, standard deviation, and median;
- mean of the two lowest late-window means;
- all-five final mean, standard deviation, and median;
- top-three-of-five late mean and exact seed IDs;
- mean best-to-final drop;
- complete evaluation curves;
- exact Phase-A selection manifest and Phase-B confirmation manifest.

Total new training budget is `48 + 10 = 58` branches.

## Seed and reporting boundary

Development seeds are `200,201,202,203,208`. Held-out seeds `204--207` remain untouched. Low-performing runs and failed runs may not be deleted. The top-three-of-five confirmation summary is secondary to the all-five stability statistics and may not be used to conceal the full seed distribution.

Task-performance degradation, support or variance-boundary events, rollout failure, and NaN/Inf numerical failure remain separate event classes. No task-collapse count may be asserted because this pilot registers no task-performance-collapse threshold. Support/variance boundary remains uninstrumented unless separately registered.

## Claim boundary

This fixed one-million-update pilot may identify configurations with better finite-horizon multi-seed retention on Hopper-medium-replay. It does not establish convergence, steady state, statistical significance, universal superiority, or a formal method ranking. A good result on Hopper-medium-replay does not replace controlled causal identification.

## Planned execution and outputs

One-click entrypoint:

```bash
bash scripts/run_e7_hopper_replay_stability_hps.sh run
```

Planned work root:

`outputs/e7/hopper_replay_stability_hps_001`

Required aggregate outputs:

- `phase_a/aggregate/branch_results.csv`;
- `phase_a/aggregate/training_curves_long.csv`;
- `phase_a/aggregate/config_summary.csv`;
- `phase_a/aggregate/phase_a_selection_manifest.json`;
- `phase_b/aggregate/branch_results.csv`;
- `phase_b/aggregate/training_curves_long.csv`;
- `phase_b/aggregate/config_summary.csv`;
- `phase_b/aggregate/phase_b_confirmation_manifest.json`;
- `aggregate/final_selected_config.json`;
- `aggregate/terminal_audit.json`.

## Pre-run validation

The training-free validation command

```bash
bash scripts/test_e7_hopper_replay_stability_hps.sh
```

passed against runner blob `a046c18c082786b4f5c0e23b6b27770d04f3fef6`. This validates Python extraction/compilation, shell syntax, the frozen 48+10 matrix, optimizer-control self-test, held-out-seed exclusion, and exact branch-level agreement among batch size, trajectory-snapshot refresh interval, and recorded snapshot-batch provenance. It is not D4RL liveness or scientific evidence.

The RunSpec must bind the exact frozen implementation commit, use deferred registration with closure required, preserve protected implementation paths, and automatically deliver text-first evidence to `easonhuo/drpo-results@ingest/e7` only after terminal audit completion.
