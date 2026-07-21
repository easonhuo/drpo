# E7 D4RL-9 Joint-Critic GAE P1 Result Review

- Experiment: `EXT-H-E7-SQEXP-GAE-01`
- Profile: `d4rl9_common_c_p1`
- Run ID: `E7_BENCH_JOINT_GAE_P1_FULL_20260719_02`
- Result repository: `easonhuo/drpo-results`, branch `ingest/e7`
- READY manifest SHA-256: `9f1ea69f0759bcd3bd79a91c7ccdb5e5d1a22f49ea67e114aaa1e7d4f5f6dbc1`
- Matrix: 9 tasks x 2 development seeds x 11 controls = 198 branches
- Completion: 198 completed, 0 failed
- Scientific class: development screening pilot; formal evidence and method ranking are not allowed

## Equal-task-weighted summary

| Control | Best mean | Late mean | Final mean | Paired late wins vs Positive-only | Task wins vs Positive-only |
|---|---:|---:|---:|---:|---:|
| Positive-only | 77.3923 | 65.3054 | 68.3166 | — | — |
| c=0.25 | 66.0387 | 47.6785 | 45.7866 | 4/18 | 2/9 |
| c=0.5 | 60.7757 | 37.5388 | 37.1540 | 2/18 | 1/9 |
| c=1 | 53.2169 | 28.3853 | 27.3390 | 2/18 | 1/9 |
| c=2 | 43.1042 | 24.4181 | 23.8511 | 2/18 | 1/9 |
| c=4 | 35.1630 | 17.0786 | 15.4961 | 1/18 | 0/9 |
| c=8 | 34.4421 | 16.7762 | 16.7193 | 0/18 | 0/9 |
| c=16 | 30.0255 | 14.9371 | 11.7980 | 0/18 | 0/9 |
| c=32 | 29.2320 | 12.7514 | 10.1549 | 0/18 | 0/9 |
| c=64 | 25.8822 | 12.0044 | 10.6663 | 0/18 | 0/9 |
| Uncontrolled | 4.5271 | 0.4611 | 0.4161 | 0/18 | 0/9 |

## Interpretation

Within the registered development matrix, weaker tapering produced progressively worse equal-task-weighted late performance. Positive-only remained the best overall anchor. The best controlled point was the smallest tested scale, `c=0.25`, so P1 did not bracket an interior optimum and motivates a separately registered left-boundary extension.

This is not evidence that all negative signal is useless. HalfCheetah-medium showed local improvement for moderate controlled negative pressure, while Hopper—especially Hopper-medium-expert—was highly sensitive. The task heterogeneity and two-seed instability prohibit a universal coefficient claim.

## Terminal and failure audit

- terminal audit: PASS;
- fixed horizon is not convergence: true;
- selected control: null;
- selection status: `response_curve_only_pending_protocol_freeze`;
- held-out seeds `204--207`: untouched;
- NaN/Inf numerical failures: 0;
- rollout failures: 0;
- task-performance collapse: not adjudicated because no threshold was registered;
- support/variance boundary: not instrumented in this pilot.

Low normalized return must not be rewritten as numerical collapse or a support-boundary event.

## Provenance limitation

The result package records source commit `d0ba443154d847065965b18a43ffe897f19530fa`, but that commit is not currently resolvable from the remote repository. The result remains reviewable development-pilot evidence and cannot be upgraded to authoritative formal evidence until provenance is closed.
