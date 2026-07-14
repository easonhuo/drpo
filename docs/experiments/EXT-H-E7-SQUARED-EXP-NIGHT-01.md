# EXT-H-E7-SQUARED-EXP-NIGHT-01

## Status

- scientific class: external-validity development screening pilot;
- implementation route: dedicated code-first branch from `main`;
- registration: schema-v3 authority update `EXT-H-E7-SQUARED-EXP-NIGHT-RESULT-2026-07-14` materialized;
- launch policy: code-complete clean dev-branch checkout may run before registration;
- result status: `pilot`; 126/126 branches completed and terminal audit passed;
- held-out seeds `204--207`: untouched and forbidden;
- predecessor: `EXT-H-E7-W0-HIGHC-ACTOR-01`.
- compact result evidence: `experiments/results/e7_squared_exp_night_1m_pilot/`;
- full package SHA-256: `8a63ffd3aceef73e3e1998c66813b74d5dbef0879aca80f1db5625229de68a88`;
- GAE result: none; Stage C remained blocked and started zero branches.

## Why this successor is required

The predecessor implemented

\[
u=d/2,\qquad w(d)=w(0)\exp(-cu),
\]

where `d` is detached RMS standardized action distance.  The manuscript defines
Gaussian learner-relative remoteness through negative log probability, which is
proportional, up to additive and dimensional constants, to squared standardized
Mahalanobis distance.  The manuscript-consistent E7 envelope is therefore

\[
u=d/2,\qquad w(d)=w(0)\exp(-cu^2).
\]

The predecessor remains useful directional evidence that stronger attenuation
can recover severe finite-horizon degradation, but it is not the squared-
remoteness EXP experiment required by the manuscript.

## Shared protocol

All runnable branches preserve the historical canonical E7 source fingerprint,
actor and critic architecture, critic target and expectile loss, one-step TD
advantage, actor-before-critic order, optimizer, batch size `256`, learning rate
`3e-4`, datasets, and evaluation protocol.

- datasets: `hopper-medium-expert-v2`, `walker2d-medium-v2`,
  `walker2d-medium-replay-v2`;
- development seeds: `200, 201`;
- horizon: `1,000,000` optimizer updates;
- evaluation: every `50,000` updates with ten episodes;
- controls per actor mode: Positive-only plus `w(0)=1` at
  `c in {0.25,0.5,1,2,4,8}`;
- 500k is retained only as an intermediate comparison point;
- terminal reporting uses the 800k--1M late window in addition to BEST and
  FINAL.

## Stage A: squared-remoteness actor comparison

Stage A runs the seven controls under:

1. historical canonical A2C signed actor update;
2. the existing PPO-clipped actor with clip epsilon `0.2` and old-policy
   cadence `K=4`.

This stage contains

\[
7\times2\times3\times2=84
\]

branches.  It asks whether the paper-consistent squared envelope improves the
near/far trade-off and whether the finite-horizon behavior differs between the
two actor updates.  Hopper and Walker remain external-validity tasks; the paired
comparison is not controlled causal identification.

## Stage B: longer PPO reference window with analytic-KL early refresh

Stage B runs the same seven controls, datasets, seeds, horizon, critic,
advantage, optimizer, and evaluation protocol under one additional PPO reference
lifecycle:

- clip epsilon remains `0.2`;
- maximum old-policy reuse window is `K_max=16` optimizer updates;
- after every actor update, compute analytic diagonal-Gaussian
  `KL(pi_old(.|s) || pi_new(.|s))` on the current offline states;
- if mean KL exceeds `0.01`, end the current reference block by immediately
  refreshing the old actor;
- the scheduled `K_max=16` refresh remains a hard upper bound;
- no KL penalty, entropy bonus, actor gradient clipping, or value clipping is
  introduced.

This stage adds

\[
7\times3\times2=42
\]

branches.  The current/old ratio remains a proximal ablation rather than a claim
of behavior-policy importance correction.

## Stage C: GAE gate

The first intended GAE value is `lambda=0.95`, following a common continuous-
control PPO default.  The current canonical trainer callback exposes shuffled
transition batches and does not supply a reviewed ordered-trajectory contract.
GAE may not be reconstructed until the following are verified:

1. stable ordered trajectory or episode identity;
2. environment terminal versus time-limit truncation semantics;
3. exact reward, discount, and bootstrap semantics;
4. separation of training trajectories from evaluation evidence.

Accordingly, Stage C is fail-closed.  The one-click suite records it as
`BLOCKED`, starts zero GAE branches, and does not treat the block record as a GAE
result.

## Diagnostics

Every actor mode records the exact negative-sample tensors used by the objective:

- standardized-distance quantiles;
- squared-EXP effective-weight quantiles and exact mean;
- exact weight-threshold fractions;
- advantage-weighted effective negative mass.

PPO branches additionally record ratio-outside and true objective-clip
fractions.  Stage-B branches record analytic KL, interval maxima, triggered
refresh counts, and total old-policy refresh counts.

`LAUNCH_REGISTRATION_STATUS.json` records the initial and latest registration
state.  A pre-registration launch is permitted, is explicitly labeled
`code_first_pre_registration`, and never upgrades the run to formal evidence.

## Terminal reporting

The terminal audit requires all 126 runnable branches and reports:

- branch-wise 500k, BEST, best step, FINAL, BEST-to-FINAL drop;
- 800k--1M mean, standard deviation, and slope;
- two-seed group summaries and actor-update differences;
- geometry, clipping, KL, and numerical diagnostics;
- a separate Stage-C blocked record.

Task-performance degradation or collapse, support or variance-boundary events,
and NaN/Inf numerical failure remain separate categories.  This pilot does not
instrument a post hoc task-collapse threshold or a complete support-boundary
criterion; unavailable categories must be reported as not adjudicated rather
than silently inferred.

## Interpretation limits

This is a two-development-seed, fixed-1M screening suite.  It does not establish
convergence, steady state, a universal actor-update ranking, a causal PPO/A2C
claim, a GAE result, OOD generalization, or a formal paper result.  Dataset-
specific winning coefficients may not be cherry-picked and presented as one
common robust method.  The Hopper/Walker evidence does not replace C-U1 or D-U1
controlled mechanism identification.

## Execution sequence

1. code, deterministic tests, and protocol document on the dedicated dev branch;
2. update the server checkout to the reviewed code-first commit;
3. run short real-data liveness and, after it passes, start the resumable
   126-branch suite immediately;
4. materialize schema-v3 handoff/registry registration in parallel without
   blocking the already started development pilot;
5. record server-start evidence and change result status only through the
   subsequent non-destructive registration delta;
6. perform terminal audit and result-status registration after durable evidence
   is delivered.
