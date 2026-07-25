# Scope: EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-POSBASE-DRPO-CGRID-500K-01

## Identity

- Repository: `easonhuo/drpo`
- Development route: stacked Draft PR above `dev/ext-h-e7-hopper-replay-posbase-500k-01`
- Stacked base commit: `5ce2084035a94b86a04b4f70e5337cb6b06eac03`
- Parent experiment: `EXT-H-E7-BENCH-01`
- Immediate predecessor: `EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-POSBASE-500K-01`
- Environment role: Hopper/D4RL external-validity pilot
- Result status before execution: `not_run`
- Allowed scientific status after this code change: `not_run + implemented`; smoke, validation, CI, or plan output is not a scientific result.

## Claim under test

On `hopper-medium-replay-v2`, test the finite-horizon interaction between:

1. a stricter fixed raw trajectory-snapshot GAE baseline used before the actor sign split; and
2. the remoteness scale of the existing thresholded squared-exponential DRPO negative branch.

The experiment asks whether DRPO remains beneficial relative to Positive-only at the same baseline after more replay transitions cross into the negative branch, and whether a shifted baseline improves DRPO relative to baseline-zero DRPO at the same taper scale.

This experiment does not test the controlled C-U1 causal claim and does not replace C-U1. It does not establish D4RL-wide method ranking, convergence, steady state, task-collapse rates, or held-out-seed confirmation.

## Frozen matrix

### Positive baseline

```text
b in {0, 4, 5, 6}
```

The shift is applied to every trajectory-snapshot GAE value:

```text
shifted_advantage = raw_trajectory_snapshot_gae - b
```

No replay transition is deleted or masked before trajectory-snapshot lookup.

### Methods

Positive-only anchors:

```text
PO(b), b in {0,4,5,6}
```

DRPO cells:

```text
DRPO(b,c),
b in {0,4,5,6},
c in {0.06,0.08,0.10,0.125}
```

Here `c` is the existing `remoteness_scale` in:

```text
w(D) = exp(-lambda * relu((D - threshold) / c))
D = normalized squared standardized distance
```

Frozen shared values:

- `lambda = 1`
- `threshold = 0`
- `reference distance = 2`
- positive shifted advantages remain unattenuated
- Positive-only zeros negative shifted advantages
- DRPO applies the detached remoteness taper only to negative shifted advantages

The exact method count is:

```text
4 Positive-only + 16 DRPO = 20 controls
```

### Seeds

Development seeds:

```text
200, 201, 202, 203, 208, 209, 210, 211, 212
```

Held-out confirmation seeds, prohibited in this pilot:

```text
204, 205, 206, 207
```

Exact branch count:

```text
20 controls x 9 development seeds = 180 branches
```

The full factorial must be preserved. No result-conditioned cell removal, seed pruning, or early stopping is allowed.

## Frozen training protocol

- dataset: `hopper-medium-replay-v2`
- canonical A2C actor update
- jointly updated critic
- trajectory-snapshot GAE, `lambda=0.95`
- batch size: `512`
- actor learning rate: `3e-4`
- critic learning rate: `3e-4`
- optimizer updates: exactly `500000`
- evaluation interval: `20000`
- evaluation episodes: `10`
- only NaN/Inf or an execution failure may prevent a branch from reaching the scheduled horizon
- fixed 500k is a finite horizon, not a convergence criterion

## Required comparisons

Every DRPO cell must be paired by seed against:

1. `DRPO(b,c) - PO(b)`, to isolate the contribution of controlled negative gradients at the same positive admission baseline;
2. `DRPO(b,c) - DRPO(0,c)`, for `b in {4,5,6}`, to isolate the baseline shift at the same DRPO taper scale.

The finite-horizon screening order is frozen as:

1. late-window median over `360k--500k`;
2. worst-three-seed late mean;
3. paired late win count against Positive-only at the same baseline;
4. nonnegative median trend over `200k--500k`;
5. final median at `500k`;
6. smaller mean best-to-final drop.

This order may screen candidates for later held-out confirmation. It is not a steady-state method ranking.

## Required diagnostics

The implementation must preserve, per branch:

- complete 25-point evaluation curve;
- complete trajectory-snapshot advantage series;
- raw advantage mean and standard deviation;
- raw and shifted positive/negative fractions;
- positive-to-nonpositive crossing fraction;
- `b / std(A)`;
- raw and shifted negative advantage mass;
- taper-adjusted effective negative mass;
- near/far raw and effective negative mass using standardized-distance boundary `2`;
- sampled raw actor-gradient norm;
- sampled Adam actor parameter-update norm;
- sampled actor `log_std` extrema and support/variance-boundary fractions;
- rollout failure;
- NaN/Inf numerical failure.

Task-performance collapse is not adjudicated because this pilot does not register a task-collapse threshold. Support/variance-boundary events and NaN/Inf numerical failure must remain separate.

## Code scope

New files:

- `configs/e7_hopper_replay_posbase_drpo_cgrid_500k.json`
- `docs/scopes/EXT-H-E7-SQEXP-GAE-HOPPER-REPLAY-POSBASE-DRPO-CGRID-500K-01.md`
- `scripts/run_e7_hopper_replay_posbase_drpo_cgrid_500k.sh`
- `scripts/e7_hopper_replay_posbase_drpo_cgrid_500k_runtime/*.inc` (reviewable runtime Python source fragments; reconstructed and compiled before use)
- `scripts/test_e7_hopper_replay_posbase_drpo_cgrid_500k.sh`
- `runspecs/ready/E7_HOPPER_REPLAY_POSBASE_DRPO_CGRID_500K_20260725_01.yaml` after the implementation commit is frozen

No new Python repository path is created. The runner reconstructs runtime Python from reviewable non-`.py` source fragments, compiles it before every invocation, and minimally extends the already-approved E7 bootstrap interfaces without editing the frozen predecessor implementation.

## Excluded changes

This task must not modify:

- C-U1, D-U1, Countdown, or Figure 1 experiments;
- canonical actor/critic architecture;
- canonical critic target or expectile loss;
- GAE recursion, timeout handling, or trajectory-snapshot refresh semantics;
- optimizer family or learning rates;
- batch size, seeds, horizon, evaluation cadence, taper family, threshold, lambda, or reference distance;
- held-out seeds `204--207`;
- the frozen files of predecessor PR #280;
- `docs/handoff.md` directly;
- `experiments/registry.yaml` before the code-first implementation SHA is frozen and the governed registration transaction is prepared.

## Execution and interpretation gates

- Code-first implementation is permitted; experiment launch is not.
- A RunSpec may be added only after the exact implementation commit is frozen.
- The real 180-branch pilot must use the governed RunSpec and exact protected implementation files.
- All 180 branches, aggregate outputs, and the terminal audit must be complete before result delivery.
- Failed and low-performing branches must be preserved.
- A completed 500k pilot may support finite-horizon candidate screening only.
- Held-out-seed confirmation and longer-horizon terminal assessment require a separately registered successor.
