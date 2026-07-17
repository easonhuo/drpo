# EXT-H-E7-SQEXP-GAE-01 — Canonical Joint-Critic GAE Repair

## Status

- development class: code-first external-validity screening pilot;
- result status: `not_run`;
- formal evidence allowed: `false`;
- repository base: `64001f2e7d8636642cf30e57bf6ffc57882bf6ac`;
- implementation branch: `dev/e7-canonical-gae-repair-01`;
- predecessor evidence: `EXT-H-E7-SQUARED-EXP-NIGHT-01` Stage C remained blocked and started zero GAE branches;
- superseded unmerged design: the earlier prepared-advantage / frozen-critic draft must not be launched or treated as this experiment.

## Claim

On the verified canonical D4RL joint actor--critic path, compare one-step TD and
trajectory-aware GAE (`lambda=0.95`) while preserving the same current critic
training objective, actor-before-critic update order, random transition sampler,
squared-remoteness control, datasets, development seeds, optimizer, batch size,
training horizon, and evaluation cadence.

This is a Hopper/Walker external-validity development pilot. It does not provide
controlled causal identification, convergence, steady-state ranking, OOD
generalization, or a universal GAE claim.

## Correct lineage

The authoritative algorithmic parent is the canonical `iqlv_exp_rank` source
contract used by the E7 joint actor--critic pilots:

- `src/drpo/e7_canonical_vendor/d4rl/agents.py`;
- `src/drpo/e7_canonical_vendor/d4rl/train_sna2c_variant.py`;
- `src/drpo/e7_canonical_injection.py`;
- `src/drpo/e7_squared_exp_kernel.py`.

The following behavior is frozen:

1. actor and critic are both initialized by the canonical agent;
2. actor updates before critic;
3. critic uses the existing one-step Bellman target and expectile loss;
4. `c_opt.step()` remains active during all actor training;
5. the canonical trainer continues uniform random transition sampling;
6. no separate trainer, critic pretraining stage, prepared checkpoint, or frozen
   advantage artifact is introduced.

`EXT-H-E7-Q2` remains a separate completed frozen-critic mechanism experiment and
is not a parent implementation for this pilot.

## Ordered trajectory contract

The canonical HDF5 loader already exposes aligned arrays:

- observations;
- actions;
- normalized rewards;
- next observations;
- environment terminals;
- time-limit timeouts.

The file order is the trajectory order. GAE recursion must stop at either an
environment terminal or timeout and must never cross the dataset tail.

Bootstrap and recursion have different masks:

- environment terminal: no `V(next_state)` bootstrap and no continuation;
- timeout: retain `V(next_state)` in the TD residual but stop GAE recursion;
- dataset tail without an explicit boundary: retain the final one-step bootstrap
  but stop recursion because no following dataset transition exists.

Overlapping terminal and timeout flags fail closed.

## Joint-critic snapshot estimator

The canonical trainer samples independent transitions, so exact per-update
full-trajectory GAE would require a full replay pass before every optimizer
update. This pilot instead uses a matched periodic snapshot protocol:

1. before the first optimizer update, evaluate the current critic over the full
   ordered replay dataset;
2. compute both the one-step TD table and `GAE(lambda=0.95)` table from the same
   critic snapshot;
3. refresh both tables once per transition-equivalent replay epoch, where the
   interval is `ceil(dataset_transition_count / canonical_batch_size)` updates;
4. use the sampled transition IDs only to look up the selected TD or GAE actor
   advantage;
5. update the critic on every optimizer step using the original current one-step
   expectile objective.

TD and GAE therefore share exactly the same snapshot age and refresh schedule.
The comparison does not mix current-step TD with stale GAE. Snapshot refresh
cadence is derived from dataset size and the frozen canonical batch size rather
than introduced as a tunable hyperparameter.

Every full branch must record snapshot critic hashes and fail if critic evolution
is not observed. A short liveness run may contain only one snapshot and therefore
cannot establish critic evolution or scientific evidence.

## Frozen matrix

- datasets:
  - `hopper-medium-expert-v2`;
  - `walker2d-medium-v2`;
  - `walker2d-medium-replay-v2`;
- development seeds: `200,201,202,203`;
- held-out seeds: `204,205,206,207` remain forbidden;
- actor update: canonical A2C only;
- estimators: one-step TD and GAE with `lambda=0.95`;
- controls:
  - Positive-only;
  - squared EXP `c=64`;
  - squared EXP `c=128`;
  - squared EXP `c=256`;
- public control: `w(d)=w(0) exp[-c(d/2)^2]`;
- horizon: `1,000,000` optimizer updates;
- evaluation: every `50,000` updates with ten episodes;
- total: `3 x 4 x 2 x 4 = 96` branches.

PPO is excluded because the completed actor-decision gate did not retain PPO for
the E7 mainline under this backbone. Reopening PPO requires a separately scoped
ablation; it must not be silently restored through this repair.

The coefficient set is a previously used development bracket, not a selected
common optimum. No dataset-specific winner may be promoted as a universal
coefficient.

## Required diagnostics and gates

Before a full sweep:

1. deterministic GAE boundary tests;
2. `lambda=0` exact reduction to the TD snapshot;
3. ordered replay identity and transition-index lookup tests;
4. critic-update test proving `c_opt.step()` changes critic parameters;
5. exact 96-branch matrix test;
6. current-main CI;
7. real-data short liveness for one TD and one GAE branch on the same dataset and
   seed;
8. runtime overhead and snapshot cadence audit.

The liveness run is engineering evidence only. It cannot be reported as a pilot
result.

## Terminal reporting

Report per branch and paired groups:

- BEST, FINAL, and 800k--1M late-window mean;
- late-window standard deviation and slope;
- BEST-to-FINAL drop;
- GAE-minus-TD paired differences;
- snapshot count, refresh interval, first/final critic hashes, and whether critic
  evolution was observed;
- task-performance collapse, support/variance boundary, and NaN/Inf numerical
  failure as separate fields.

A fixed 1M endpoint is not convergence or steady state. No method ranking is
authorized without the required terminal audit.