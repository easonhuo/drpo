# PAPER-CODE-VALIDATION-01 Hopper E7-Q2 Correctness Re-audit

**Date:** 2026-07-21  
**Parent claim:** `PAPER-CODE-REFERENCE-01`  
**Validation claim:** `PAPER-CODE-VALIDATION-01`  
**Canonical branch:** `dev/paper-code-reference-01`  
**Audited branch head:** `7e6d16fda35c59e3eaab0faf4c6856de3dfb29b4`  
**Scientific-status impact:** none

## 1. Purpose and boundary

This re-audit checks whether the migrated Hopper E7-Q2 reviewer code preserves
the authoritative external-mechanism-validation pipeline.

Hopper E7-Q2 is kept separate from the D4RL-9 task-performance backend. This
slice validates dataset handling, critic and frozen-advantage construction,
near/far mechanism diagnostics, six-branch actor behavior, optimizer updates,
rollout semantics, output provenance, and event separation.

This slice does not run the registered Hopper HDF5, Gymnasium/MuJoCo, ten formal
seeds, or a scientific terminal review. No method ranking is produced.

Reviewer paths:

- `paper_code/src/drpo_reference/external/hopper_data.py`;
- `paper_code/src/drpo_reference/external/hopper_models.py`;
- `paper_code/src/drpo_reference/external/hopper_critic.py`;
- `paper_code/src/drpo_reference/external/hopper_advantages.py`;
- `paper_code/src/drpo_reference/external/hopper_metrics.py`;
- `paper_code/src/drpo_reference/external/hopper_actor.py`;
- `paper_code/src/drpo_reference/external/hopper_suite.py`;
- `paper_code/src/drpo_reference/external/hopper_rollout.py`;
- `paper_code/src/drpo_reference/external/hopper_protocol.py`;
- `paper_code/src/drpo_reference/experiments/hopper.py`.

Authoritative sources:

- `src/drpo/e7_hopper_q2.py`;
- `configs/e7_hopper_q2_medium_replay_v2.yaml`;
- `scripts/run_e7_hopper_q2.py`.

## 2. Source identity and executable evidence

The verified reviewer artifact `8464318725` was produced from commit
`479b1dadef168c9e42a0fd67cc60c66842e8f799`.

A repository comparison from that source commit to the audited head contains no
change under `paper_code/`. The validated package therefore carries the same
Hopper reviewer implementation as the audited branch head.

GitHub Actions at the artifact source commit passed:

- full repository pytest, including all Hopper differential suites;
- Python compilation and Ruff;
- package build, manifest verification, isolated installation, and all public
  command help checks.

In this session, the extracted package command

```text
python3 -m drpo_reference hopper --help
```

parsed successfully and exposed the explicit dataset, output, seed subset,
device, canonical-critic reuse, and smoke arguments. No dataset-backed Hopper
execution was launched.

## 3. Frozen protocol identity

The reviewer protocol preserves:

- experiment ID `EXT-H-E7-Q2`;
- role `external_mechanism_validation`;
- dataset basename `hopper_medium_replay-v2.hdf5`;
- registered dataset SHA-256
  `e121c5f7c9857a307baa9edc6a2c3b48e85fedb9ac316ecddd0f48ca7ef4e39b`;
- rollout dataset identity `hopper-medium-replay-v2`;
- evaluation environment `Hopper-v4` through Gymnasium/MuJoCo;
- process-isolated rollout preflight with no legacy D4RL fallback;
- critic budget 100,000 steps with 50,000 minimum steps;
- one-time frozen advantage standardization;
- Positive-only preparation followed by six identical-start branches;
- ten formal seeds 100--109;
- 100,000 Positive-only preparation steps and 200,000 branch steps;
- registered near/far matching, gradient-probe, audit, and rollout coordinates.

The active six methods remain:

1. `positive_only`;
2. `signed`;
3. `near_zero`;
4. `far_zero`;
5. `far_cap`;
6. `dynamic_budget_matched_global`.

## 4. Differential evidence reviewed

### 4.1 Dataset, episode, return, split, and normalization

`test_hopper_data_differential.py` binds the reviewer implementation to the
authoritative runner for:

- HDF5 arrays with and without optional `timeouts` and `next_observations`;
- episode-ID construction;
- timeout/terminal-aware discounted returns;
- episode-level train/validation/test splitting;
- observation normalizer fitting and transformation.

### 4.2 Model initialization and policy geometry

`test_hopper_models_differential.py` checks exact state initialization and forward
identity for the critic and squashed Gaussian actor, including:

- log probability;
- standardized action distance;
- mean and log-scale output-score components;
- joint output-score norm.

### 4.3 Critic selection and frozen advantages

`test_hopper_critic_differential.py` checks:

- deterministic sampling, rank, correlation, gradient, and update utilities;
- value, next-value, raw-advantage, and standardized frozen-advantage arrays;
- the frozen critic contract of 100,000 total and 50,000 minimum steps;
- a fixed-budget short critic trajectory against the authoritative runner;
- best/final checkpoint metrics, stationarity, operational gates, quality gates,
  selected-checkpoint role, and frozen-advantage acceptance;
- exact resulting critic parameters and advantage arrays.

### 4.4 Near/far matching and mechanism diagnostics

`test_hopper_metrics_differential.py` checks:

- scalar statistics and terminal classifications;
- advantage-matched near/far index selection;
- per-sample and aggregate negative-gradient norms;
- analytic output-score versus autograd agreement;
- mean-score and corrected quadratic log-scale distance slopes;
- dynamic Global budget matching;
- complete matched-pair and distance-bin diagnostic artifacts.

This preserves the intended mechanism claim: near/far comparisons are matched on
negative-advantage magnitude, while distance and score geometry vary.

### 4.5 Actor objectives, gradients, first updates, and short trajectories

`test_hopper_actor_differential.py` checks every registered actor method for:

- actor loss and diagnostics;
- full raw gradient;
- first clipped AdamW update;
- evaluation and boundary metrics.

It also checks fixed-seed short trajectories for Positive-only and dynamic
budget-matched Global, and verifies that a non-finite loss applies no optimizer
update and is reported as a separate numerical event.

### 4.6 Prepared actor and six-branch suite

`test_hopper_suite_differential.py` checks:

- Positive-only preparation seed and branch seed;
- identical branch initial states without shared storage;
- prepared-checkpoint reload identity;
- audit and fixed-negative indices;
- near/far matching, gradient probe, far threshold, Far-cap score, and Global
  budget;
- all six branch terminal states and checkpoints;
- isolated failure of one branch without corrupting other branches;
- rejection of mixed/stale output roots.

### 4.7 Rollout and public-runner semantics

`test_hopper_rollout_differential.py` checks:

- frozen Gymnasium/MuJoCo protocol;
- normalized-return calculation;
- four-tuple and five-tuple environment APIs;
- deterministic reset/action behavior;
- persisted required and optional preflight failures;
- process-isolated native-signal and timeout handling;
- explicit prohibition of legacy D4RL fallback;
- policy rollout/evaluation behavior and failure reporting.

`test_hopper_public_differential.py` checks:

- formal, formal-subset-non-evidence, and smoke execution identities;
- registered seed order and duplicate/subset rejection;
- dataset basename and SHA-256 enforcement;
- hash-strict canonical critic reuse with exactly one critic training identity;
- aggregation and separation of task performance, support/variance boundary, and
  numerical events;
- root completion without authorizing method ranking;
- wiring of dataset, canonical critic, six branches, rollouts, and completion
  records.

## 5. Correctness findings

The migration preserves:

1. the registered medium-replay dataset identity and episode-level split;
2. critic training, checkpoint selection, and one-time frozen advantages;
3. a shared canonical critic across actor seeds;
4. Positive-only preparation and identical branch starts;
5. matched near/far mechanism probes with frozen negative advantages;
6. all six actor methods, losses, raw gradients, clipping, AdamW update, and short
   trajectories;
7. process-isolated Gymnasium/MuJoCo rollout semantics without silent fallback;
8. dataset, critic, preparation, branch, checkpoint, and output provenance;
9. separate task-performance, support/variance-boundary, numerical, rollout, and
   incomplete-terminal states;
10. fail-closed behavior for non-finite loss, bad dataset identity, corrupted
    critic artifacts, native rollout failures, and stale output roots.

No Hopper E7-Q2 migration defect was reproduced.

## 6. Acceptance decision

**Hopper E7-Q2 reviewer mechanism runner: engineering correctness accepted for
the currently migrated scope.**

This acceptance covers code-level identity and synthetic/fake-environment
execution paths. It does not establish:

- real registered HDF5 identity in the current runtime;
- actual Gymnasium/MuJoCo interaction;
- ten-seed reproduction;
- terminal scientific acceptance;
- any fresh method ranking.

The real HDF5/MuJoCo gate remains blocked and must stay separately reported.

## 7. Next authorized validation slice

Proceed to the D4RL-9 reviewer task-performance backend correctness acceptance.
Keep its responsibility separate from Hopper E7-Q2 mechanism validation. Validate
backend initialization, first update, deterministic short trajectory, task
contracts, aggregation, and non-formal evidence boundaries without launching a
real D4RL sweep.
