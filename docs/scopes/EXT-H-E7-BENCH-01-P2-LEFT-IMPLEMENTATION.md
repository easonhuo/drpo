# EXT-H-E7-BENCH-01 P2 left-boundary implementation

This stacked branch extends the completed P1 development response curve to smaller common remoteness scales. It is based on the P1 implementation branch at `68d4ba27823abc68c3671858c5437093aae2b3e3`; it does not change the canonical actor--critic algorithm, GAE estimator, datasets, seeds, training horizon, evaluation cadence, taper formula, or aggregation responsibilities.

## Identity and responsibility

- experiment ID: `EXT-H-E7-SQEXP-GAE-01`;
- parent claim: `EXT-H-E7-BENCH-01`;
- profile ID: `d4rl9_common_c_p2_left`;
- responsibility: development-only left-boundary response-curve extension after P1 placed the best controlled point at the smallest tested scale;
- formal evidence allowed: false;
- method-ranking claim allowed: false.

## Frozen matrix

- datasets: HalfCheetah, Hopper, and Walker2d, each at medium, medium-replay, and medium-expert;
- development seeds: `200,201`;
- held-out seeds `204--207`: forbidden;
- canonical A2C with joint critic updates;
- trajectory-snapshot GAE lambda: `0.95`;
- steps: `1,000,000`;
- evaluation interval: `50,000`;
- evaluation episodes: `10`;
- `tau=0`, taper lambda `1`, reference distance `2`;
- nine common scales: `0.20, 0.16, 0.125, 0.10, 0.08, 0.0625, 0.04, 0.025, 0.015625`;
- one internally rerun Positive-only anchor;
- no `c=0.25` rerun and no Uncontrolled branch;
- exact branch count: `(9 + 1) x 9 x 2 = 180`.

The historical P1 `c=0.25` point may be displayed only as a cross-run boundary reference. It is not a P2 paired observation and must not be used in P2 paired-seed counts.

## Decision boundary

P2 asks whether stronger tapering reveals a common interior scale that exceeds the internally rerun Positive-only anchor, or whether the response curve only approaches Positive-only as `c` decreases. No coefficient is selected automatically. A fixed one-million-step endpoint is not convergence or steady state.

Task-performance collapse, support or variance boundary, rollout failure, and NaN/Inf numerical failure remain separate. Per-task degradation must be reported alongside the equal-task-weighted mean; a gain on one environment cannot silently offset severe degradation on another.

## Implementation boundary

No new Python file is authorized or required. Reuse the existing squared-night runner, bootstrap, canonical injection, aggregator, runtime autotuner, one-click script, and standard E7 RunSpec lane. Only the existing profile constants and validation branch, a new JSON grid, a new RunSpec template, and existing tests may change. Both P1 and its result artifacts remain immutable at their pinned commits.

The checked-in RunSpec stays under `runspecs/templates/`. Promotion to `runspecs/ready/`, launch, merge, authoritative registration closure, and scientific-status upgrade each require separate review and approval.