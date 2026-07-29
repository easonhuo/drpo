# E7 D4RL-9 Exponential taper-onset tau tuning v1

## Status and scope

Experiment ID: `EXT-H-E7-D4RL9-EXP-TAU-TUNE-01`.

This is a development-only pilot stacked on the completed joint Exponential
negative-scale/coefficient tuning run. It does not populate a formal D4RL-9
ranking table, does not use held-out seeds, and does not change the frozen
one-step-TD trainer.

The experiment asks one narrow question:

> After freezing three useful `(negative_scale, exponential_coefficient)`
> operating points per task, how does a small DRPO near-field onset threshold
> affect task performance and training variability?

## The tau being tuned

The tuned parameter is named `taper_onset_tau`. It is the DRPO threshold in the
E7 implementation's normalized RMS standardized-distance coordinate

```text
u = RMS((action - policy_mean) / policy_std) / reference_distance
```

and the Exponential shape is

```text
shape(u) = exp(-c * max(u - taper_onset_tau, 0))
```

Consequently:

- `taper_onset_tau = 0` exactly recovers the predecessor E7 formula
  `exp(-c*u)`;
- `u <= taper_onset_tau` receives shape factor exactly one;
- only the remote tail beyond the threshold is attenuated.

The canonical trainer also has a pre-existing CLI flag `--tau 0.5`. That is the
critic expectile parameter and remains frozen. It is not the DRPO taper
threshold. The implementation, output schema, and branch names never use a bare
`tau` for the new control.

## Frozen matrix

The local threshold grid is

```text
[0.0, 0.125, 0.25, 0.375, 0.5]
```

Because `reference_distance = 2`, these values correspond to raw RMS
standardized-distance thresholds `[0, 0.25, 0.5, 0.75, 1.0]`.

Each task has three frozen operating points. They are role-based, not claimed to
be the three highest-scoring predecessor cells:

| Dataset | Role | negative_scale | c |
|---|---|---:|---:|
| hopper-medium-v2 | performance | 1 | 0.15 |
|  | stronger_taper_contrast | 1 | 0.30 |
|  | scale_contrast | 0.065 | 0.15 |
| hopper-medium-replay-v2 | performance | 1 | 0.122666666667 |
|  | local_c_contrast | 1 | 0.184 |
|  | scale_contrast | 0.6 | 0.122666666667 |
| hopper-medium-expert-v2 | performance | 0.01 | 0.425 |
|  | lowest_seed_std | 0.01 | 0.2125 |
|  | robust_worst_seed | 0.003 | 0.85 |
| walker2d-medium-v2 | performance | 0.003 | 0.15 |
|  | local_c_contrast | 0.003 | 0.20 |
|  | scale_contrast | 0.03 | 0.20 |
| walker2d-medium-replay-v2 | performance | 1 | 0.0096 |
|  | local_c_contrast | 1 | 0.0048 |
|  | scale_contrast | 0.03 | 0.0096 |
| walker2d-medium-expert-v2 | performance | 1 | 0.64 |
|  | stronger_taper_contrast | 1 | 1.28 |
|  | scale_contrast | 0.1 | 0.64 |
| halfcheetah-medium-v2 | performance | 1 | 0.0416666666666 |
|  | stronger_taper_contrast | 1 | 0.0833333333332 |
|  | scale_contrast | 0.02 | 0.0416666666666 |
| halfcheetah-medium-replay-v2 | performance | 1 | 0.0046875 |
|  | stronger_taper_contrast | 1 | 0.009375 |
|  | scale_contrast | 0.01 | 0.0046875 |
| halfcheetah-medium-expert-v2 | performance | 1 | 2.3 |
|  | stronger_taper_contrast | 1 | 4.6 |
|  | scale_contrast | 0.03 | 2.3 |

The total is

```text
9 tasks × 3 operating points × 5 taper_onset_tau values × 4 seeds
= 540 branches
```

Development seeds remain `[200, 201, 202, 203]`. Held-out seeds
`[204, 205, 206, 207]` are excluded.

## Frozen trainer

The following remain unchanged:

- canonical alpha `0.11`;
- reference distance `2`;
- variant `iqlv_exp_rank`;
- trainer expectile `--tau 0.5`;
- temperature `5`;
- one million updates;
- batch size `256`;
- learning rate `3e-4`;
- evaluation every `50k` updates;
- ten evaluation episodes;
- late window `750k, 800k, 850k, 900k, 950k, 1M`.

## Runtime implementation

No new tracked Python path is introduced. The tracked shell launcher generates a
SHA-bound runtime Python adapter under the output root. The adapter:

1. verifies the canonical source contract and dataset SHA;
2. imports the existing canonical injection layer;
3. changes only the detached Exponential shape to the thresholded formula;
4. leaves the actor loss, critic target/loss, optimizer order, network, data loop,
   trainer arguments, and evaluation code unchanged;
5. writes the exact `taper_onset_tau` into branch identity, branch config, and
   terminal audit.

The generated adapter runs a fail-closed self-test before preparation:

- onset zero is bitwise identical to the predecessor taper;
- the near-field plateau is exactly one;
- the tail matches the registered formula;
- a negative onset threshold is rejected.

## Outputs and selection

The launcher writes:

- `EXECUTION_PLAN.json`;
- `CANDIDATE_MATRIX.json`;
- `TERMINAL_AUDIT.json`;
- `TASKWISE_SELECTION.json`;
- `TASKWISE_TAU_CURVES.json`;
- generated per-unit configs, branch identities, logs, and trainer summaries.

Primary taskwise selection remains the predecessor late-window rule, extended
with a smaller-`taper_onset_tau` tie-break. This does not retroactively change
the predecessor selection rule.

In addition to late mean and seed-level standard deviation, the audit reports:

- mean single-seed late temporal standard deviation;
- mean instantaneous cross-seed standard deviation over late checkpoints;
- final-score seed standard deviation;
- best-to-late and best-to-final drops;
- terminal slope.

These diagnostics distinguish similar time-averaged seed means from genuinely
less variable training trajectories.

## Execution

Validation only:

```bash
bash scripts/run_e7_canonical_d4rl9_exp_tau_tuning_one_click.sh --validate-only
```

Preparation and source-shape validation only:

```bash
CONTRACT=/root/d4rl2/configs/e7_canonical_contract_9task.json \
SOURCE_RUN_SPEC=/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json \
WORK_ROOT=outputs/e7/d4rl9_exp_tau_tuning_run_001 \
bash scripts/run_e7_canonical_d4rl9_exp_tau_tuning_one_click.sh --prepare-only
```

Development run:

```bash
CONTRACT=/root/d4rl2/configs/e7_canonical_contract_9task.json \
SOURCE_RUN_SPEC=/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json \
WORK_ROOT=outputs/e7/d4rl9_exp_tau_tuning_run_001 \
bash scripts/run_e7_canonical_d4rl9_exp_tau_tuning_one_click.sh
```

## Interpretation boundary

- This is development tuning, not held-out confirmation.
- A fixed one-million-update horizon is not convergence or steady state.
- No cross-method ranking or formal D4RL-9 table may be populated from this run.
- Task-performance collapse, support/variance boundary availability, and NaN/Inf
  numerical failure remain separately reported.
- No post-hoc seed exclusion or operating-point replacement is permitted.
