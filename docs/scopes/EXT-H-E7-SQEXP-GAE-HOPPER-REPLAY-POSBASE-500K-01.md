# EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-POSBASE-500K-01

## Status

- result status: `not_run`
- run kind: `pilot`
- formal evidence allowed: `false`
- environment role: Hopper / D4RL external-validity screening only
- predecessor: `EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-ADVSHIFT-200K-01`

## Claim under test

The predecessor 200k pilot showed that a fixed raw-GAE shift changes the actor--critic
trajectory, but DRPO negative-branch effects and the short horizon prevented a clean test of the
sample-quality hypothesis. This successor isolates Positive-only and asks whether raising its
advantage admission threshold improves Hopper-medium-replay performance or retention.

For every replay transition,

```text
shifted_advantage = raw_trajectory_snapshot_gae - fixed_baseline
actor coefficient = max(shifted_advantage, 0)
```

No transition is deleted from the replay data and no replay row is masked before GAE lookup.
Transitions with shifted advantage at or below zero remain present but contribute zero to the
Positive-only actor objective. This experiment does not use the DRPO negative branch.

## Frozen matrix

Dataset:

- `hopper-medium-replay-v2`

Fixed baselines in raw trajectory-snapshot GAE units:

- `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10`

Development seeds:

- `200, 201, 202, 203, 208, 209`

Held-out seeds excluded:

- `204, 205, 206, 207`

Exact matrix:

```text
11 baselines x 6 development seeds = 66 fresh branches
```

No baseline or seed may be pruned based on intermediate results.

## Frozen training protocol

- Positive-only actor objective for every branch;
- canonical A2C actor update and jointly updated critic;
- trajectory-snapshot GAE with lambda `0.95`;
- batch size `512`;
- actor learning rate `3e-4`;
- critic learning rate `3e-4`;
- `500,000` optimizer updates;
- evaluation every `20,000` updates;
- `10` evaluation episodes;
- no early stopping and no result-conditioned pruning.

The canonical remoteness fields required by the shared E7 bootstrap remain inert because every
branch uses Positive-only and therefore assigns zero actor weight to shifted-negative samples.

## Reporting

Every branch retains its complete 25-point evaluation trajectory and every trajectory-snapshot
baseline diagnostic. Aggregate reporting includes:

- early mean over `20k--200k`;
- middle mean over `220k--340k`;
- late mean over `360k--500k`;
- trend slope over `200k--500k`;
- score at `500k`;
- best checkpoint and best-to-final drop;
- all-six mean, median, standard deviation, and worst-two-seed late mean;
- raw advantage mean and standard deviation;
- effective Positive-only fraction `P(A>b)`;
- `b/std(A)`;
- mean raw and shifted advantage among actor-active samples;
- paired differences against baseline zero.

The full snapshot series is required because one fixed baseline has different relative strength as
the critic and raw-GAE distribution evolve. A fixed 500k horizon is not convergence or steady-state
evidence. Method ranking, task-collapse rates, and universal claims remain forbidden.

## Event separation

The terminal audit reports separately:

- task-performance degradation: `not_adjudicated_no_registered_threshold`;
- support or variance boundary: `not_instrumented_in_this_pilot`;
- rollout failure count;
- NaN/Inf numerical failure count.

## Code and output paths

- config: `configs/e7_hopper_replay_positive_only_baseline_500k.json`
- entrypoint: `scripts/run_e7_hopper_replay_positive_only_baseline_500k.sh`
- validation: `scripts/test_e7_hopper_replay_positive_only_baseline_500k.sh`
- default output: `outputs/e7/hopper_replay_positive_only_baseline_500k_001`
- RunSpec: `runspecs/ready/E7_HOPPER_REPLAY_POSBASE_500K_20260725_01.yaml`
