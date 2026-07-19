# PAPER-CODE-REFERENCE-01 D4RL Backend Migration Delta

**Date:** 2026-07-19  
**Claim:** `PAPER-CODE-REFERENCE-01`  
**Registered benchmark ID:** `EXT-H-E7-BENCH-01`  
**Scientific status impact:** none

Read this file after:

1. `docs/paper_code_reference/SESSION_HANDOFF.md`;
2. `docs/paper_code_reference/SESSION_HANDOFF_ROLLOUT_DELTA_20260718.md`;
3. `docs/paper_code_reference/SESSION_HANDOFF_HOPPER_PUBLIC_DELTA_20260718.md`;
4. `docs/paper_code_reference/SESSION_HANDOFF_D4RL_SHARED_DELTA_20260719.md`.

This delta records selection and code migration of the D4RL performance backend. It does not replace `docs/handoff.md`, authorize a formal run, change frozen scientific variables, or permit merging PR `#149`.

## Repository snapshot

- repository: `easonhuo/drpo`;
- authoritative main inspected: `85b0a68d77ed085a7f6e67771fb0f7672c43da09`;
- development branch before the migration: `dev/paper-code-reference-01@18c062fe19fe1d3e2870addc0dd45be3cf32bb7b`;
- migrated implementation/test head: `3cfeb7000ab6cf3b7db2487a445edad27948b765`;
- persistent Draft PR: `#149`;
- no scientific experiment was launched.

## Selected backend

The repository owner confirmed that `SNA2C_IQLV_ExpRankAgent` is the correct D4RL performance backend and instructed that it be migrated.

The selected lineage uses:

- a two-layer ReLU actor with bounded mean and learned independent log standard deviation;
- a two-layer ReLU value critic;
- dynamic TD advantages recomputed at every update;
- expectile value regression;
- rank-based negative weighting `alpha * exp(-T * score)`;
- actor update followed by critic update;
- one implementation for HalfCheetah, Hopper, and Walker2d.

It remains scientifically distinct from the Hopper E7-Q2 frozen-advantage actor-only mechanism backend.

Authoritative differential oracles:

- `src/drpo/e7_canonical_vendor/d4rl/agents.py`;
- `src/drpo/e7_canonical_vendor/d4rl/train_sna2c_variant.py`;
- `src/drpo/e7_canonical_vendor/d4rl/d4rl_common/train_loop.py`;
- `src/drpo/e7_canonical_vendor/d4rl/d4rl_common/normalize.py`.

## Migrated implementation

`paper_code/src/drpo_reference/experiments/d4rl.py` now contains the paper-facing implementation of:

- `CanonicalActor` and `CanonicalCritic`;
- `SNA2CIQLVExpRankAgent`;
- canonical rank weights;
- dynamic TD/expectile loss computation;
- first-order actor and critic updates;
- locomotion reward normalization and episode-aware Monte Carlo returns;
- canonical action clipping and dataset preparation;
- deterministic uniform-minibatch training;
- legacy-compatible actor checkpoints and non-formal completion records;
- one D4RL-9 execution plan and one backend dispatch across all nine tasks.

No `halfcheetah_*`, `hopper_*`, or `walker2d_*` performance trainer was created. The Hopper mechanism trainer is not reused.

## Differential validation

`paper_code/tests/test_d4rl_shared_core_differential.py` compares the migrated code with the authoritative legacy source on:

- network parameter initialization;
- actor and critic forward values;
- rank-weight formula;
- first Adam update;
- a fixed three-step training trajectory;
- locomotion dataset preparation and action clipping;
- checkpoint and completion records;
- exact nine-task task/rollout identities;
- one trainer implementation across all nine tasks;
- remaining formal blockers.

GitHub Actions run `29681768089` passed Python compilation, shell syntax, handoff authority, formal execution channel, governance checks, full repository pytest, and Ruff. These are engineering and short differential checks, not scientific results.

## Remaining formal blockers

Backend selection and code migration are no longer blockers. Formal D4RL-9 execution still requires:

1. SHA-256 values for the eight unresolved dataset coordinates;
2. the final common method controls and coefficients;
3. the registered ten-run seed coordinate;
4. formal budget, checkpoint policy, and terminal-audit rules;
5. real nine-task HDF5 and Gymnasium/MuJoCo liveness;
6. separate reporting of task-performance collapse, support/variance-boundary events, rollout failure, incomplete terminal state, and NaN/Inf failure.

Until these are frozen and executed, `formal_evidence_eligible=false` and `method_ranking_claim_allowed=false` remain mandatory.

## Execution status

No real HDF5 dataset, MuJoCo environment, full-budget actor/critic training, manuscript-number validation, or method ranking was executed in this migration. PR `#149` remains Draft and unmerged.
