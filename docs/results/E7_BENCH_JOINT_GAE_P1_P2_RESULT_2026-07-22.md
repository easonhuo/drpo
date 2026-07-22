# E7 joint-critic GAE P1/P2 result closure

P1: `198/198`, result commit `33fa687352392df985679abddedb834535b10a3d`, READY manifest `9f1ea69f0759bcd3bd79a91c7ccdb5e5d1a22f49ea67e114aaa1e7d4f5f6dbc1`.
P2: `180/180`, result commit `7c1af9fc47ddd347c1bb28d43ad8a024ca95b8a8`, READY manifest `37bbc4ce2fbbd98cd2c2ad742329a9124c578955512ebe4819b75e1aa5bea9e4`.
Both: failed=0, terminal audit PASS, held-out seeds untouched, NaN/Inf=0, rollout failures=0.

| control | best | late | final | drop | task wins | seed-task wins |
|---|---:|---:|---:|---:|---:|---:|
| P1 positive_only | 77.3923 | 65.3054 | 68.3166 | 9.0757 | — | — |
| P2 c0.015625 | 78.2069 | 64.8863 | 62.1744 | 16.0325 | 4 | 9 |
| P2 c0.025 | 78.9350 | 63.7141 | 59.7329 | 19.2022 | 6 | 10 |
| P2 c0.04 | 78.1482 | 62.1268 | 58.9491 | 19.1991 | 2 | 4 |
| P2 c0.0625 | 80.3389 | 60.2982 | 54.7541 | 25.5849 | 5 | 9 |
| P2 c0.08 | 77.4900 | 59.1869 | 54.2560 | 23.2340 | 6 | 10 |
| P2 c0.10 | 79.7162 | 56.7544 | 54.9766 | 24.7396 | 2 | 6 |
| P2 c0.125 | 78.4317 | 57.3451 | 55.5435 | 22.8883 | 4 | 8 |
| P2 c0.16 | 78.9693 | 55.4924 | 49.1398 | 29.8295 | 4 | 7 |
| P2 c0.20 | 79.8624 | 50.6457 | 49.4462 | 30.4161 | 3 | 6 |
| P1 c0.25 | 66.0387 | 47.6785 | 45.7866 | 20.2521 | 2 | 4 |

The response broadly recovers as `c` decreases. The smallest finite point `c=0.015625` has late mean `64.8863`, only `0.4191` below Positive-only `65.3054`, but final mean remains `6.1422` lower. No tested finite common `c` exceeds Positive-only on the equal-task-weighted late metric, and no interior common optimum is identified.

Negative signal is not uniformly useless: at `c=0.015625`, medium and medium-replay strata exceed Positive-only by about `1.35` and `0.93`, while medium-expert is lower by about `3.53`. Several cells improve, but larger losses on sensitive tasks offset them. Transient best values above Positive-only coexist with worse final values and larger best-to-final decline; this is not a synchronized checkpoint or steady-state result.

Task-performance collapse is `not_adjudicated` because no threshold was registered. Support/variance boundary is `not_instrumented`. Low return is not relabeled as numerical collapse or a support event.

P1 records unresolved source commit `d0ba443154d847065965b18a43ffe897f19530fa`, blocking formal-evidence promotion. P2 binds resolvable implementation `909249875c190a75301ceb2dc2c2062ca0efcb16`; its stale pre-run `GAE_STAGE_STATUS.json` is retained but cannot override the completed 180/180 terminal audit.

Full response values are in `experiments/results/e7_sqexp_gae_p1_p2_closure_20260722/P1_P2_RESPONSE_CURVE.csv`. This closure does not merge stacked scientific-code PRs or authorize P3.
