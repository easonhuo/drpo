# EXT-H-E7-SQEXP-GAE P3 left-saturation implementation scope

## Identity

- Parent experiment: `EXT-H-E7-SQEXP-GAE-01`
- Profile: `d4rl9_common_c_p3_left_saturation`
- Development claim: implement and validate, but do not register or launch, the D4RL-9 joint-critic GAE left-tail response-curve profile.
- Base: current `main` at branch creation and refresh.
- Scientific role: Hopper/D4RL external-validity screening only. It does not replace C-U1 or D-U1 controlled mechanism identification.

## Frozen scientific matrix

- Canonical A2C actor with jointly updated critic.
- Trajectory-snapshot GAE with `lambda=0.95`.
- Nine D4RL-v2 tasks: Hopper, Walker2d, and HalfCheetah × medium, medium-replay, and medium-expert.
- Development seeds: `200,201`; held-out seeds `204--207` remain forbidden.
- One million updates, evaluation every 50,000 updates, ten evaluation episodes.
- Positive-only plus nine exact quarter-decade scales from `0.01` through `0.0001`.
- Exact branch count: `(9 c + 1 Positive-only) × 9 tasks × 2 seeds = 180`.
- Historical P2 `c=0.015625` is not rerun. P2 and P3 may be connected only through within-run `Delta(c)=J(c)-J(Positive-only)` curves.

## Complete development route

This task uses the normal repository development route, not M0. The implementation must cover runner, runtime adapter identity, resource probe, bootstrap profile validation, aggregation, launcher authorization, and regression tests. A Draft PR and exact-head CI are required. Real server resource selection and a two-branch P3 liveness run must pass before implementation-SHA freeze and registration. Full launch remains blocked by the digest-bound zero-training preflight and later explicit approval.

## Claim limits

The profile may characterize overall, environment-stratified, and tier-stratified response curves, including recovery region, elbow, practical saturation onset, and candidate nonzero overshoot. It must not select a formal best `c`, claim convergence from the fixed horizon, form a steady-state or formal method ranking, or touch held-out seeds.

Task-performance degradation, support/variance boundary events, rollout failures, and NaN/Inf failures remain separate. Task collapse is not adjudicated without a registered threshold, and support/variance is not reported as instrumented.
