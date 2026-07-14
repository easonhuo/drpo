# EXT-H-E7-SQUARED-EXP-NIGHT-01

## Status

- route: code-first development branch;
- scientific role: Hopper/Walker external-validity development screening;
- execution status: runnable development pilot; schema-v3 registration may follow launch;
- predecessor: `EXT-H-E7-W0-HIGHC-ACTOR-01`;
- base commit: `9c49824558b1eb7f697f299b246a135ff35a2017`;
- held-out seeds `204--207`: forbidden and untouched.

## Claim

The predecessor used the linear-distance envelope

\[
w(d)=\exp[-c(d/2)].
\]

The manuscript defines Gaussian learner-relative remoteness by negative log
probability, which is proportional (up to an additive and dimensional constant)
to squared standardized Mahalanobis distance.  This successor therefore first
tests the manuscript-consistent envelope

\[
w(d)=\exp[-c(d/2)^2].
\]

It then asks whether a longer old-policy reuse window with analytic Gaussian-KL
early refresh changes the behavior of the existing offline PPO-style actor.  A
third GAE stage is represented in the suite contract but must remain fail-closed
until the canonical trainer exposes verified ordered-trajectory,
terminal-versus-truncation, and return-construction semantics.

## Frozen code-first development matrix

### Stage A — squared-remoteness kernel

- datasets: `hopper-medium-expert-v2`, `walker2d-medium-v2`,
  `walker2d-medium-replay-v2`;
- development seeds: `200, 201`;
- controls: Positive-only plus `w(0)=1` at
  `c in {0.25, 0.5, 1, 2, 4, 8}`;
- actor updates: canonical A2C and the existing PPO-clip path;
- horizon: `1,000,000` optimizer updates;
- evaluation: every `50,000` updates, ten episodes;
- PPO reference: `epsilon=0.2`, old-policy cadence `K=4`.

This stage has `7 x 2 x 3 x 2 = 84` branches.

### Stage B — KL-early-refresh PPO

Stage B reuses the exact Stage-A datasets, seeds, controls, network, critic,
optimizer, batch, learning rate, evaluation protocol, squared kernel, and
one-step TD advantage.  Only the PPO reference lifecycle changes:

- clipping epsilon remains `0.2`;
- maximum old-policy reuse window is `K_max=16` optimizer updates;
- after every actor update, compute analytic
  `KL(pi_old(.|s) || pi_new(.|s))` on the current offline states;
- refresh the old actor immediately when mean KL exceeds `0.01`;
- scheduled refresh at `K_max` remains a hard upper bound;
- no KL penalty is added to the loss.

This adds `7 x 3 x 2 = 42` branches.  Stages A and B together contain 126
branches.

### Stage C — GAE gate

The intended first GAE value is `lambda=0.95`, following the common continuous
control PPO default.  Stage C is not allowed to fabricate GAE from shuffled
independent transitions.  Before any Stage-C branch can be generated, a
trajectory contract must verify:

1. stable ordered trajectory or episode identifiers;
2. distinction between environment terminal and time-limit truncation;
3. exact reward, discount, and bootstrap semantics;
4. no use of held-out evaluation trajectories in actor or value fitting.

Until that contract exists, the one-click suite records Stage C as blocked and
continues with Stages A and B.  Stage-C blocking does not block the rest of the
night suite.

## Reporting

Report BEST, FINAL, 800k--1M late-window mean/std/slope, the 500k intermediate
snapshot, paired seed differences, effective negative mass, PPO clip fractions,
analytic KL, KL-triggered refreshes, and numerical failures.  Keep task
performance degradation, support/variance-boundary events, and NaN/Inf failure
separate.

`LAUNCH_REGISTRATION_STATUS.json` must preserve whether the initial server
launch occurred before schema-v3 materialization.  Pre-registration launch is
allowed only as a development pilot and does not permit formal-evidence claims.

## Non-claims

This two-seed development sweep does not establish convergence, a steady-state
method ranking, universal A2C/PPO superiority, a causal actor-update claim, OOD
generalization, or a formal paper result.  Stage C being blocked is not a GAE
result.

## Execution gate

A clean checkout of the code-first dev branch, the frozen config, deterministic
tests, protocol document, and required canonical data are sufficient to run
liveness and start Stages A and B.  Schema-v3 registration proceeds immediately
after launch and may not interrupt an already running development pilot.  The
suite must still refuse a dirty checkout, preserve the launch commit, record the
registration state, and never treat an unregistered or partially registered run
as formal evidence.
