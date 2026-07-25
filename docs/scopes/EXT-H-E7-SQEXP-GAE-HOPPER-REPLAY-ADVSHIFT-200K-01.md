# EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-ADVSHIFT-200K-01

## Status

- result status: `not_run`
- run kind: `pilot`
- formal evidence allowed: `false`
- environment role: Hopper / D4RL external-validity screening only
- predecessor: `EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-STAB-HPS-01`

## Claim under test

Hopper-medium-replay may contain many transitions whose raw trajectory-snapshot GAE is
positive only relative to a weak state baseline, without being sufficiently strong imitation
targets. This pilot tests whether applying one fixed positive baseline to every raw GAE value
before the sign split improves the first 200,000 updates of the existing DRPO training path.

For every sampled transition,

```text
shifted_advantage = raw_trajectory_snapshot_gae - fixed_baseline
```

The shifted value, not the raw value, determines whether the transition enters the positive or
negative DRPO branch. No transition is deleted or masked. A transition that crosses from positive
to negative remains in the actor loss and receives the existing distance-dependent negative
weight.

## Frozen matrix

Dataset:

- `hopper-medium-replay-v2`

Development seeds:

- `200, 202, 203, 208`

Held-out seeds excluded:

- `204, 205, 206, 207`

Controls and candidates:

1. `positive_only`: raw GAE, negative terms zeroed, baseline `0`;
2. `drpo_b0`: current DRPO, baseline `0`;
3. `drpo_b0p25`: DRPO with fixed raw-GAE baseline `0.25`;
4. `drpo_b0p5`: DRPO with fixed raw-GAE baseline `0.5`;
5. `drpo_b1`: DRPO with fixed raw-GAE baseline `1.0`.

Exact matrix:

```text
5 controls x 4 seeds = 20 fresh branches
```

## Frozen training protocol

- canonical A2C actor update;
- jointly updated critic;
- trajectory-snapshot GAE with lambda `0.95`;
- batch size `512`;
- actor learning rate `3e-4`;
- critic learning rate `3e-4`;
- DRPO remoteness scale `c=0.08`;
- remoteness threshold `0`;
- taper lambda `1`;
- reference distance `2`;
- `200,000` optimizer updates;
- evaluation every `20,000` updates;
- `10` evaluation episodes;
- no early stopping and no result-conditioned pruning.

## Reporting

Every branch retains the complete ten-point evaluation trajectory. The compact comparison reports:

- early mean over `20k--100k`;
- late mean over `120k--200k`;
- final score at `200k`;
- linear trend slope over `100k--200k`;
- best-to-final drop;
- raw and shifted positive fractions from the trajectory-snapshot provider;
- paired differences against both `positive_only` and `drpo_b0`.

A rising or approximately retained 200k trajectory is only a signal that the baseline-shift idea
is worth longer confirmation. It is not convergence, steady state, a task-collapse adjudication,
or a formal method ranking.

## Event separation

The terminal audit reports separately:

- task-performance degradation: `not_adjudicated_no_registered_threshold`;
- support or variance boundary: `not_instrumented_in_this_pilot`;
- rollout failure count;
- NaN/Inf numerical failure count.

## Code and output paths

- config: `configs/e7_hopper_replay_advshift_200k.json`
- entrypoint: `scripts/run_e7_hopper_replay_advshift_200k.sh`
- validation: `scripts/test_e7_hopper_replay_advshift_200k.sh`
- default output: `outputs/e7/hopper_replay_advshift_200k_001`
- RunSpec: `runspecs/ready/E7_HOPPER_REPLAY_ADVSHIFT_200K_20260725_01.yaml`
