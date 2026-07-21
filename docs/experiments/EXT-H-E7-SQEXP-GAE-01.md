# EXT-H-E7-SQEXP-GAE-01 — Joint-Critic GAE Common-c Screening

## Responsibility

This experiment is an external-validity development screen on the public D4RL-v2 MuJoCo locomotion suite. It evaluates the finite-horizon task-performance response to a common squared-remoteness exponential negative-weight scale under the canonical joint actor--critic path with trajectory-snapshot GAE. It does not replace C-U1 or D-U1 controlled mechanism identification and cannot establish convergence, steady state, or universal method superiority.

## Shared frozen contract

- tasks: HalfCheetah, Hopper, and Walker2d, each at medium, medium-replay, and medium-expert;
- development seeds: `200,201`;
- held-out seeds: `204--207`, forbidden in development screening;
- actor update: canonical A2C;
- critic: updated on every actor-training step with the canonical objective;
- advantage estimator: trajectory-snapshot GAE with lambda `0.95`;
- optimizer horizon: `1,000,000` updates;
- evaluation: every `50,000` updates with ten episodes;
- distance coordinate: normalized squared standardized action distance;
- public taper: `w(D)=w(0) exp[-lambda_taper relu((D-tau)/c)]`;
- `tau=0`, `lambda_taper=1`, reference distance `2`;
- fixed one-million-step endpoints are not convergence or steady state.

Task-performance collapse, support or variance boundary events, rollout failures, and NaN/Inf numerical failures are reported separately.

## P1 broad response curve — completed pilot

Profile `d4rl9_common_c_p1` compared Positive-only, Uncontrolled, and common `c={0.25,0.5,1,2,4,8,16,32,64}` across nine tasks and two development seeds. Run `E7_BENCH_JOINT_GAE_P1_FULL_20260719_02` completed `198/198` branches with zero failed branches and passed the registered terminal audit. The bounded text result package is available in `easonhuo/drpo-results`, branch `ingest/e7`, with READY manifest SHA-256 `9f1ea69f0759bcd3bd79a91c7ccdb5e5d1a22f49ea67e114aaa1e7d4f5f6dbc1`.

The equal-task-weighted late-window normalized return was `65.3054` for Positive-only, `47.6785` for `c=0.25`, and then decreased to `37.5388, 28.3853, 24.4181, 17.0786, 16.7762, 14.9371, 12.7514, 12.0044` as `c` increased; Uncontrolled was `0.4611`. Positive-only was the best overall development anchor, while the strongest tested taper `c=0.25` was the best controlled point and lay on the left boundary of the tested interval. This supports further left-boundary localization, not a claim that smaller `c` must universally improve every task.

The audit reported NaN/Inf `0` and rollout failures `0`. Task-performance collapse remained `not_adjudicated_no_registered_threshold`, and support/variance boundary remained `not_instrumented_in_this_pilot`; neither may be retroactively inferred from low returns. No coefficient was selected, method-ranking claims remain disallowed, and held-out seeds were untouched.

The result package records source commit `d0ba443154d847065965b18a43ffe897f19530fa`, which is not currently resolvable from the remote repository. Therefore P1 remains development pilot evidence with an explicit source-provenance blocker; it is not authoritative formal evidence.

## P2 left-boundary extension — implemented, not launched

Profile `d4rl9_common_c_p2_left` freezes nine smaller scales:

`0.20, 0.16, 0.125, 0.10, 0.08, 0.0625, 0.04, 0.025, 0.015625`.

P2 reruns Positive-only as its internal paired anchor, excludes Uncontrolled, and does not rerun `c=0.25`. The historical P1 `c=0.25` point may appear only as a cross-run boundary reference and must never enter P2 paired-seed counts. The exact matrix is `(9 c values + Positive-only) x 9 tasks x 2 seeds = 180 branches`.

The scientific implementation is frozen at `909249875c190a75301ceb2dc2c2062ca0efcb16` on branch `dev/ext-h-e7-bench-joint-gae-p2-left-01`; Draft PR `#223` contains the review object. The RunSpec template is `runspecs/templates/E7_BENCH_JOINT_GAE_P2_LEFT_FULL_20260721_01.yaml`, run ID `E7_BENCH_JOINT_GAE_P2_LEFT_FULL_20260721_01`. The template remains outside `runspecs/ready/`; P2 has started zero branches and requires separate approval before promotion or launch.

P2 asks whether stronger tapering yields a common interior scale that exceeds its internally rerun Positive-only anchor, or whether the curve only approaches Positive-only. It does not authorize per-task retuning, automatic coefficient selection, formal ranking, convergence, steady-state, or universal DRPO-superiority claims.
