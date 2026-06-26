# C-U1-E4-CONV-01 result summary

## Status

- Formal 20-seed computation completed on run commit `c869df8b203f13eb8389d1d300b33f1928502871`.
- Repository closure package base: `ba1e3710df4140ffaf54db3ecf12cd6f40ac531a`.
- Registered grid: fixed variance `alpha in {0.75, 1.00, 1.25}`, seeds 50--69, 4000 E4 updates per branch.
- Integrity: 60/60 seed-alpha rows, all fixed-variance Adam, no additional positive-only run, preflight and reference-regression checks passed.
- The original pre-registered 18/20 per-alpha consensus **did not pass** and remains permanently recorded.
- After explicit user review on 2026-06-26, the scoped E4 long-horizon phase claim is **closed as long-run validated**. This is a transparent post-run acceptance decision, not a retroactive claim that the 18/20 gate passed.

## Terminal-state counts

| alpha | expected terminal state | expected | inconclusive | explicit opposite |
|---:|---|---:|---:|---:|
| 0.75 | stable_beneficial_extrapolation | 15/20 | 5/20 | 0/20 |
| 1.00 | stable_beneficial_extrapolation | 16/20 | 4/20 | 0/20 |
| 1.25 | stable_over_extrapolation | 15/20 | 5/20 | 0/20 |

All 60 runs preserved their registered scientific role from step 2000 to step 4000. There were no explicit opposite terminal states, task-performance collapses, support/variance-boundary events, or NaN/Inf numerical failures.

## Aggregate long-horizon behavior

| alpha | held-out-context reward | normalized displacement | W2 displacement change | W2 reward change | raw-gradient W2/W1 | Adam-update W2/W1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 0.920641 | 0.566691 | 0.000362 | 0.000218 | 1.033273 | 1.076483 |
| 1.00 | 0.998282 | 1.028898 | -0.000437 | 0.000241 | 1.026425 | 1.137236 |
| 1.25 | 0.638814 | 2.012683 | 0.000959 | -0.000263 | 1.004254 | 1.156999 |

The aggregate trajectories are nearly stationary and no scientific role reverses. The 14 inconclusive rows were caused by individual raw-gradient or Adam-update W2/W1 ratios above the frozen `1.25` classifier threshold, not by displacement runaway, reward reversal, support failure, or non-finite values. No threshold, seed label, optimizer, horizon, or result value was changed.

## Closed scientific claim

The accepted long-horizon E4 claim is:

- `alpha=0.75/1.00` retain bounded beneficial extrapolation over 4000 E4 updates rather than exhibiting a late scientific-role reversal;
- `alpha=1.25` behaves as stable over-extrapolation rather than slow runaway;
- together with the already delivered `alpha=1.50` task-performance collapse, `alpha=1.75` continuing raw-gradient/parameter runaway, and learnable-variance support contraction, the E4 non-monotonic phase and separated failure-type chain is closed for paper use.

## Reporting boundary

Required disclosure:

- report the exact 15/20, 16/20, and 15/20 counts;
- report that the original 18/20 gate did not pass;
- state that closure followed explicit user evidence review;
- keep held-out-context terminology and separate task collapse, support/variance boundary, and NaN/Inf.

Not allowed:

- saying the original 18/20 gate passed;
- saying 20/20 seeds received fixed-point certification or every seed was strictly stationary;
- describing C-U1 as OOD generalization;
- changing any result or claiming a universal method ranking.
