# EXT-H-E7-SQUARED-EXP-KL-TUNE-01

## Status

- scientific class: Hopper/Walker external-validity development screening pilot;
- stage: `stage_a_kl_threshold_and_reference_lifecycle_screen`;
- implementation commit: `2d4d295022c75b0c2cde283d2d9c3402779c5764`;
- registration status: registered before or in parallel with code-first execution;
- result status: `not_run` at registration time;
- held-out seeds `204--207`: untouched and forbidden;
- predecessor: `EXT-H-E7-SQUARED-EXP-NIGHT-01`;
- GAE: excluded from Stage A;
- Stage B: not authorized by this registration.

## Question

Stage A tests whether the positive finite-horizon signal from analytic-KL-triggered
old-policy refresh is robust to the KL threshold and whether its benefit comes from
adaptive refresh timing rather than merely allowing a longer fixed reference window.
It also probes the squared-EXP region above `c=8` without selecting a different
coefficient per dataset.

## Frozen matrix

All 150 branches preserve the predecessor's canonical actor and critic architecture,
critic target and expectile loss, one-step TD advantage, actor-before-critic order,
optimizer, batch size `256`, learning rate `3e-4`, datasets, and evaluation protocol.

- datasets: `hopper-medium-expert-v2`, `walker2d-medium-v2`,
  `walker2d-medium-replay-v2`;
- development seeds: `200, 201`;
- horizon: `1,000,000` optimizer updates;
- evaluation: every `50,000` updates with ten episodes;
- kernel: `w(d)=w(0)exp[-c(d/2)^2]`;
- controls per lifecycle: Positive-only plus `w(0)=1` at
  `c in {4,8,16,32}`;
- PPO clip epsilon: `0.2`;
- 500k: intermediate checkpoint only;
- terminal reporting window: 800k--1M.

Reference lifecycles:

1. fixed K4 PPO;
2. fixed K16 PPO with no KL-triggered refresh;
3. `K_max=16`, analytic `KL(old||new)` threshold `0.003`;
4. `K_max=16`, analytic `KL(old||new)` threshold `0.01`;
5. `K_max=16`, analytic `KL(old||new)` threshold `0.03`.

The branch count is:

```text
5 × 5 × 3 × 2 = 150
```

The old/current ratio remains a proximal ablation and is not described as
behavior-policy importance correction.

## Code-first launch policy

The clean implementation checkout at `2d4d295022c75b0c2cde283d2d9c3402779c5764` may run before the
materialized registration commit reaches the server. The runner records the initial
registration state but registration does not block liveness or the development pilot.
The scientific matrix must remain byte-equivalent to the registered config.

## Qualification output

Terminal aggregation writes `stage_a_qualification.json`. An adaptive threshold
qualifies only when all branches are terminal-audited, pooled paired late mean and
median differences versus fixed K4 are positive, and it wins more than half of the
15 `(dataset, control)` cells. Positive-only and squared-EXP effects are reported
separately. The tie-break order is mean difference, median difference, then cell wins.

The qualification record does not launch Stage B. Stage B requires a separate frozen
registration and explicit launch decision.

## Diagnostics and terminal audit

Every branch records squared-EXP geometry, effective negative mass, PPO ratio-outside
and true objective-clip fractions. Adaptive branches additionally record analytic KL,
interval maxima, triggered refresh counts, and total reference refresh counts.

Task-performance collapse or degradation, support or variance-boundary events, and
NaN/Inf numerical failure remain separate categories. Missing registered thresholds
must be reported as not adjudicated rather than inferred after the run.

## Interpretation limits

This is two-development-seed external-validity screening. It does not establish
convergence, steady state, a universal PPO ranking, lower seed variance from KL
refresh, controlled causal actor-update identification, a GAE result, OOD
generalization, or a formal D4RL method ranking. Hopper/Walker evidence does not
replace C-U1 or D-U1 controlled mechanism evidence.
