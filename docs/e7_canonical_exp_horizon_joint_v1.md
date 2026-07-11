# E7 canonical EXP coefficient and horizon joint pilot v1

**Experiment:** `EXT-H-E7-BENCH-01`  
**Status:** pilot only; not a formal ranking or convergence result.

## Claim

Measure the stability of the scale-one exponential coefficient response across four tuning seeds while testing whether the prior 1M horizon is sufficient for the three core coefficient candidates.

## Frozen shared protocol

The existing nine-task canonical run spec remains the source of dataset paths, dataset hashes, canonical trainer arguments, evaluation interval, evaluation episodes, checkpoint policy, and thread environment. The adapter requires the exact dataset order and tuning seeds `200,201,202,203`, requires the source run spec to retain `--steps 1000000`, then replaces only that value with a branch-specific `{steps}` placeholder.

The canonical actor/critic implementation, `alpha=0.11`, reference distance `2.0`, batch size, learning rate, evaluation protocol, and validated queue size `max_workers=60` remain unchanged.

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

Task-performance collapse, support or boundary events, and NaN/Inf numerical failure must remain separate. Seeds `204-207` are reserved for later held-out confirmation and must not be used to select the coefficient in this pilot.

## Server command

The existing server-local contract and nine-task run spec are reused:

```bash
python scripts/run_e7_canonical_exp_horizon_joint.py run \
  --contract /root/d4rl2/configs/e7_canonical_contract_9task.json \
  --run-spec /root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json \
  --grid configs/e7_canonical_exp_horizon_joint_grid_v1.json \
  --work-dir /root/e7_canonical_exp_horizon_joint/run_001 \
  --max-workers 60
```

Run `plan` first with the same arguments and verify `432` branches before replacing `plan` with `run`.
