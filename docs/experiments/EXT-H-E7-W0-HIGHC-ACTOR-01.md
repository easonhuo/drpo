# EXT-H-E7-W0-HIGHC-ACTOR-01

## Status

- scientific class: external-validity development screening pilot;
- implementation route: stacked dev branch on `dev/e7-ppo-w0-grid-pilot`;
- registration: pending schema-v3 authority update after code review;
- result status: `not_run`;
- held-out seeds `204--207`: untouched and forbidden.

## Claim

This follow-up asks two separate questions under the same canonical E7 D4RL setup:

1. does direct `w(0)=1` approach the Positive-only endpoint when the exponential coefficient is extended beyond the previous `c<=1.5` range;
2. is the observed Hopper-medium-expert failure specific to PPO clipping, or does it also appear under the historical canonical A2C actor update?

The public parameterization remains

\[
u=d/2,\qquad w(d)=w(0)\exp(-cu).
\]

## Frozen development matrix

- datasets: `hopper-medium-expert-v2`, `walker2d-medium-v2`, `walker2d-medium-replay-v2`;
- development seeds: `200, 201`;
- actor updates: `a2c`, `ppo_clip`;
- controls per actor update:
  - Positive-only anchor `w(0)=0`;
  - `w(0)=1` with `c in {2,3,4,6,8,12}`;
- horizon: `500,000` optimizer updates;
- branches: `2 actor modes x 7 controls x 3 datasets x 2 seeds = 84`.

The PPO branch preserves clip epsilon `0.2`, four actor updates per old-policy snapshot, and no KL penalty, target-KL stop, entropy bonus, actor gradient clipping, or value clipping. A2C uses the existing canonical signed actor update. Network, critic, advantage definition, full-batch normalization, optimizer, dataset, batch size, learning rate, and evaluation protocol remain shared.

## Geometry diagnostics

Both actor-update modes record the exact negative-sample tensors already used by the actor objective. Every 1,000 updates the branch writes:

- negative standardized-distance sampled quantiles;
- negative effective-weight sampled quantiles and exact mean;
- exact fractions with `w(d)>0.5,0.1,0.05,0.01`;
- advantage-weighted effective negative mass

\[
\frac{\sum_{A<0}|A|w(d)}{\sum_{A<0}|A|}.
\]

The observer adds no actor or critic forward pass and does not change the controlled advantage returned to either update implementation.

## Reporting limits

Report branch-wise BEST, FINAL, BEST-to-FINAL drop, 400k--500k mean and slope, paired PPO-minus-A2C differences, geometry diagnostics, and numerical failures. Keep task-performance degradation, support/variance-boundary events, and NaN/Inf failures separate.

This is a two-seed 500k screening pilot. It does not establish convergence, a steady-state ranking, universal PPO/A2C superiority, or a causal claim about actor-update choice. Dataset-specific cells may not be cherry-picked as a common robust method.

## Execution sequence

1. code and tests on the dedicated dev branch;
2. Draft PR and CI review;
3. short real-data liveness and server launch from the reviewed implementation commit;
4. schema-v3 handoff/registry registration in parallel;
5. terminal audit and result registration after the durable server artifact is delivered.
