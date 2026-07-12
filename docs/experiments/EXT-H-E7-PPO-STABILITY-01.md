# EXT-H-E7-PPO-STABILITY-01

## Status

`pilot / not_run`

## Claim

Test whether replacing only the canonical E7 actor surrogate with a PPO clipped surrogate reduces training instability, seed sensitivity, and branch-wise BEST-to-FINAL degradation while preserving the existing network, critic, advantage, EXP remoteness taper, normalization, dataset, optimizer, learning rate, batch size, evaluation protocol, and one-million-step horizon.

## Scientific boundary

This is an external-validity Hopper/D4RL stability pilot. It does not replace C-U1 or D-U1 controlled mechanism identification. It does not establish convergence, steady-state ranking, or universal PPO superiority.

## Fixed implementation delta

The historical A2C-style actor objective is

```text
L_actor = -mean(log pi_theta(a|s) * A_tilde)
```

The PPO branch replaces only that actor objective with

```text
ratio = exp(log pi_theta(a|s) - log pi_old(a|s))
L_actor = -mean(min(ratio * A_tilde,
                    clip(ratio, 1-epsilon, 1+epsilon) * A_tilde))
```

where `pi_old` is a frozen snapshot of the same actor at the beginning of the current PPO update block. It is not the unknown behavior policy that generated the offline dataset.

`A_tilde` is produced by the existing canonical E7 implementation without modification:

- critic target and expectile value update unchanged;
- positive advantages unchanged;
- Positive-only still zeros only negative terms;
- EXP still uses the existing detached policy-relative standardized distance and existing exponential taper;
- full-batch normalization semantics unchanged.

## Fixed PPO settings

- `clip_epsilon: 0.2`
- `updates_per_old_policy: 4`
- one independent offline minibatch per optimizer step;
- `pi_old` frozen for four actor optimizer steps, then refreshed from the current actor;
- no KL penalty;
- no target-KL early stop;
- no entropy bonus;
- no actor gradient clipping;
- no value clipping;
- no learning-rate or optimizer change.

## Diagnostics

The PPO branch must record, without changing the objective:

- pre-update likelihood ratio mean/min/max and selected quantiles;
- absolute log-ratio magnitude;
- ratio-outside-clip fraction;
- sign-aware objective clip fraction;
- positive-advantage objective clip fraction;
- negative-advantage objective clip fraction;
- within-block position `1..4`;
- sparse post-update ratio and single-step ratio diagnostics;
- sparse actor gradient norm and actor parameter-update norm;
- finite-value checks and old-policy refresh count.

A high clip fraction only shows that the proximal constraint is active. Scientific effectiveness requires lower BEST-to-FINAL drop and lower seed variability without materially reducing attainable BEST.

## Task selection

The task subset is selected for the registered stability claim, not for general benchmark coverage:

1. `hopper-medium-expert-v2`: largest overall trajectory oscillation and BEST-to-FINAL degradation across the relevant EXP settings;
2. `walker2d-medium-replay-v2`: second-largest degradation and strong seed-dependent final performance;
3. `walker2d-medium-v2`: largest seed-cohort reversal for `EXP c=1.5` relative to Positive-only.

`hopper-medium-replay-v2` is highly oscillatory in absolute trajectory metrics, but is not in the top three for the specific `c=1.5` paired-effect seed-sensitivity criterion. It is reserved as a follow-up task rather than expanding this first pilot.

## Frozen matrix

Datasets:

- `hopper-medium-expert-v2`
- `walker2d-medium-v2`
- `walker2d-medium-replay-v2`

Development seeds:

- `200`
- `201`
- `202`
- `203`

Held-out seeds `204--207` remain untouched.

Negative-control settings:

- Positive-only;
- EXP scale `1.0`, coefficient `0.5`;
- EXP scale `1.0`, coefficient `1.0`;
- EXP scale `1.0`, coefficient `1.5`.

Actor update modes:

- historical A2C-style surrogate;
- PPO clipped surrogate.

Total matrix:

```text
3 datasets x 4 seeds x 4 negative-control settings x 2 actor updates = 96 branches
```

All branches use exactly `1,000,000` optimizer steps. No two-million-step continuation is registered.

## Primary metrics

- branch-wise BEST normalized score;
- FINAL normalized score at one million steps;
- branch-wise BEST-to-FINAL drop;
- across-seed standard deviation of FINAL;
- paired PPO-minus-A2C differences at matched dataset, seed, and negative-control setting;
- paired EXP-minus-Positive-only differences within each actor-update mode.

## Secondary diagnostics

- common-step aggregate curve;
- best checkpoint step distribution;
- late-window mean and variance;
- PPO ratio and clipping diagnostics listed above;
- separate reporting of task-performance degradation, support/variance boundary events, and NaN/Inf numerical failures.

## Interpretation gates

The pilot supports the stability hypothesis only if PPO materially reduces BEST-to-FINAL drop and seed variability while preserving comparable BEST performance. A large ratio or clip fraction without improved task trajectories is implementation/mechanism evidence, not a successful method result. A fixed one-million-step endpoint is not convergence.

## Execution order

1. implementation and deterministic unit tests;
2. synthetic liveness check proving that ratio leaves one, clipping activates for both advantage signs, and old policy stays frozen within a four-step block;
3. one short real-data smoke branch outside scientific aggregation;
4. the frozen 96-branch development pilot;
5. terminal audit and durable packaging before any held-out confirmation.
