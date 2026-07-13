# Scope contract: EXT-H-E7-W0-HIGHC-ACTOR-01

## Approved claim

Determine whether the previous `w(0)=1,c<=1.5` range ended before the Positive-only limit became visible, and whether Hopper-medium-expert sensitivity differs between canonical A2C and PPO clipping.

## Allowed changes

- add one non-destructive follow-up experiment ID;
- freeze `w(0)=1`, `c in {2,3,4,6,8,12}`, Positive-only anchors, A2C/PPO actor modes, development seeds `200,201`, and 500k horizon;
- reuse the canonical E7 source contract, datasets, network, critic, advantage, optimizer, batch, learning rate, and evaluation protocol;
- add side-effect-only geometry/weight diagnostics shared by A2C and PPO;
- add one-click liveness, auto-resource, run, resume, aggregation, and tests;
- register the experiment through one schema-v3 authority delta after implementation review.

## Forbidden changes

- no held-out seeds `204--207`;
- no change to PPO clip epsilon `0.2` or old-policy refresh cadence `4`;
- no KL penalty, target-KL stop, entropy bonus, actor gradient clipping, or value clipping;
- no change to dataset hashes, network, critic, advantage normalization, optimizer, batch size, learning rate, evaluation interval, or evaluation episodes;
- no modification or deletion of predecessor experiment history;
- no convergence, steady-state ranking, universal actor-update superiority, OOD, or controlled-causal claim;
- no dataset-specific parameter cherry-picking as a common method.

## Branch and merge gate

This is a stacked development branch based on the unmerged predecessor implementation branch because direct-`w(0)` support is not yet on `main`. The Draft PR must target `dev/e7-ppo-w0-grid-pilot` until the predecessor is merged. It may be retargeted to `main` only after ancestry is clean. Explicit user approval remains required before any merge.
