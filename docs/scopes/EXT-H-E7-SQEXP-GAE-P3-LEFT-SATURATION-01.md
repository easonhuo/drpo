# EXT-H-E7-SQEXP-GAE P3 left-saturation implementation scope

## Identity

- Parent experiment: `EXT-H-E7-SQEXP-GAE-01`
- Profile: `d4rl9_common_c_p3_left_saturation`
- Development claim: implement, but do not register or launch, the D4RL-9 joint-critic GAE left-tail response-curve profile agreed by the repository owner.
- Base for this implementation transaction: resolved current `main` at publication time.
- Scientific role: Hopper/D4RL external-validity screening only. It does not replace C-U1 or D-U1 controlled mechanism identification.

## Frozen scientific matrix

- Canonical A2C actor with jointly updated critic.
- Trajectory-snapshot GAE with `lambda=0.95`.
- Nine D4RL-v2 tasks: Hopper, Walker2d, and HalfCheetah × medium, medium-replay, and medium-expert.
- Development seeds: `200,201`, fully paired across all controls.
- Held-out seeds `204--207` remain forbidden.
- One million updates, evaluation every 50,000 updates, ten evaluation episodes.
- Positive-only plus nine exact quarter-decade scales `10^(-2-k/4)`, `k=0,...,8`, from `0.01` through `0.0001`.
- Exact branch count: `(9 c + 1 Positive-only) × 9 tasks × 2 seeds = 180`.
- Historical P2 `c=0.015625` is not rerun. P2 and P3 may be joined only through within-run `Delta(c)=J(c)-J(Positive-only)` curves, never as one raw paired run.

## Intended outputs and claim limits

The profile may characterize the overall, environment-stratified, and tier-stratified response curves, including recovery region, elbow, practical saturation onset, and a candidate nonzero overshoot. It must not select a formal best `c`, claim convergence from the fixed horizon, form a formal method ranking, or touch held-out seeds.

Task-performance degradation, support/variance boundary events, rollout failures, and NaN/Inf failures remain separate. This implementation preserves P2's honest boundaries: task collapse is not adjudicated without a registered threshold, and support/variance is not reported as instrumented.

## Zero-training-cost launch gate

Before any full 180-branch launch, a digest-bound report must evaluate the nine scales on compatible saved learner-relative distance evidence with zero training updates. Required fields are median/P90/P99 weight, effective negative mass, and fractions below `1e-3` and `1e-6`, overall and by environment/tier. The check is diagnostic and may not delete grid points. This implementation adds the contract only; it does not assert that an acceptable raw P2 distance artifact is already present in `main`.

## M0 boundary

This change is classified as `NARROW_M0`: seven complete UTF-8 `100644` after-images, no deletes, renames, mode changes, workflow edits, handoff/registry edits, governance edits, RunSpec promotion, or scientific execution. M0 publishes one atomic implementation commit to a new dedicated dev branch. A Draft PR, exact-head CI, independent review, implementation-SHA freeze, and the normal pilot-registration fastpath remain mandatory. Merge and experiment launch require later explicit approval.
