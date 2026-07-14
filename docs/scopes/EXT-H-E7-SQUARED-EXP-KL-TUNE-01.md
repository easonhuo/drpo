# Scope contract: EXT-H-E7-SQUARED-EXP-KL-TUNE-01 Stage A

## Allowed changes

- add the dedicated Stage A config, runner, bootstrap, aggregation, runtime autotune,
  launch scripts, and tests already frozen at `2d4d295022c75b0c2cde283d2d9c3402779c5764`;
- run exactly the registered 150-branch matrix;
- autotune only active subprocess count;
- record pre-registration code-first launch state without blocking execution;
- preserve and report failures and partial outputs.

## Frozen scientific variables

- datasets: Hopper medium-expert, Walker2d medium, Walker2d medium-replay;
- development seeds: `200,201`;
- held-out seeds: `204--207` forbidden;
- horizon: 1M;
- evaluation: 50k cadence, ten episodes;
- squared-EXP coefficients: `4,8,16,32` plus Positive-only;
- lifecycle controls: fixed K4, fixed K16, and adaptive K16 at
  `target_kl=0.003,0.01,0.03`;
- PPO epsilon: `0.2`;
- one-step TD advantage and all inherited optimizer/critic controls.

## Forbidden changes

- GAE or any other advantage-estimator change;
- KL penalty, entropy bonus, actor-gradient clipping, or value clipping;
- different coefficients per dataset presented as one common method;
- held-out execution;
- automatic Stage B launch;
- convergence, steady-state, universal ranking, stability, OOD, or causal claims;
- destructive changes to predecessor code, results, registry history, or handoff history.

## Completion gate

All 150 branches must be accounted for and the terminal audit must separately report
task performance, support/variance boundary status, and NaN/Inf status. A fixed 1M
endpoint is not convergence.
