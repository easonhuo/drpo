# E7 canonical EXP coefficient and horizon joint pilot v1

**Experiment:** `EXT-H-E7-BENCH-01`  
**Status:** pilot only; not a formal ranking or convergence result.

## Claim

Measure the stability of the scale-one exponential coefficient response across four tuning seeds while testing whether the prior 1M horizon is sufficient for the three core coefficient candidates.

## Frozen shared protocol

The existing nine-task canonical run spec remains the source of dataset paths, dataset hashes, canonical trainer arguments, evaluation interval, evaluation episodes, checkpoint policy, and thread environment. That previously validated file contains seeds `200,201`; the adapter requires exactly those source seeds and expands only the seed list to the registered tuning set `200,201,202,203`. Seeds `204-207` remain untouched for later held-out confirmation.

The adapter also requires the source run spec to retain `--steps 1000000`, then replaces only that value with a branch-specific `{steps}` placeholder. The canonical actor/critic implementation, `alpha=0.11`, reference distance `2.0`, batch size, learning rate, evaluation protocol, single-thread worker environment, and validated queue size `max_workers=60` remain unchanged.

## Matrix

Three scale-one EXP coefficients run to 2M:

- `0.374162511054291`
- `1.0`
- `1.5`

Six additional scale-one EXP coefficients run to 1M:

- `0.25`
- `0.5`
- `0.75`
- `1.25`
- `2.0`
- `3.0`

Three fixed 1M controls are retained:

- Positive-only;
- legacy EXP with `negative_scale=0.1` and coefficient `0.374162511054291`;
- unchanged original ExpRank-MR passthrough baseline.

This gives `12 branches × 9 datasets × 4 seeds = 432 runs`. The 108 long branches are ordered first; the existing fixed-size queue then fills freed worker slots with 1M branches.

## Interpretation and terminal audit

`FINAL` remains the primary fixed-horizon comparison. It does not by itself establish convergence. For the three 2M coefficients, report the registered 750k-1M, 1.25M-1.5M, and 1.75M-2M windows, including mean, slope, variability, best-to-final drop, and whether the terminal trajectory is plateaued, drifting, oscillatory, or inconclusive.

Task-performance collapse, support or boundary events, and NaN/Inf numerical failure must remain separate. Seeds `204-207` must not be used to select the coefficient in this pilot.

## Server execution

A configured E7 RunSpec executor launches the READY specification with:

```bash
python scripts/agent/run_lane.py --once
```

The underlying one-click entrypoint is:

```bash
bash scripts/run_e7_canonical_exp_horizon_joint_one_click.sh
```

It validates a 432-branch plan first, then starts the fixed 60-worker run in `outputs/e7/exp_horizon_joint_run_001`.
