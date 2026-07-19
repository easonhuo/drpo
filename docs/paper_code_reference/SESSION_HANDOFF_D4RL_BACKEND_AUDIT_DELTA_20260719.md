# PAPER-CODE-REFERENCE-01 D4RL Backend Audit Delta

**Date:** 2026-07-19  
**Claim:** `PAPER-CODE-REFERENCE-01`  
**Registered benchmark ID:** `EXT-H-E7-BENCH-01`  
**Scientific status impact:** none

Read this file after:

1. `docs/paper_code_reference/SESSION_HANDOFF.md`;
2. `docs/paper_code_reference/SESSION_HANDOFF_ROLLOUT_DELTA_20260718.md`;
3. `docs/paper_code_reference/SESSION_HANDOFF_HOPPER_PUBLIC_DELTA_20260718.md`;
4. `docs/paper_code_reference/SESSION_HANDOFF_D4RL_SHARED_DELTA_20260719.md`.

This delta corrects the earlier full-engine-sharing premise and records the selected D4RL performance backend. It does not replace `docs/handoff.md`, authorize a formal run, change frozen scientific variables, or permit merging PR `#149`.

## Repository snapshot

- repository: `easonhuo/drpo`;
- authoritative main inspected: `85b0a68d77ed085a7f6e67771fb0f7672c43da09`;
- development branch before this slice: `dev/paper-code-reference-01@18c062fe19fe1d3e2870addc0dd45be3cf32bb7b`;
- persistent Draft PR: `#149`;
- no scientific experiment was launched.

## Selected backend

The repository owner confirmed that the correct D4RL performance backend is `SNA2C_IQLV_ExpRankAgent` and instructed that it be migrated.

The selected lineage uses dynamic TD advantages, joint actor and expectile-value updates, a bounded actor mean, and rank-based negative weighting. It is scientifically distinct from the Hopper E7-Q2 frozen-advantage actor-only mechanism backend.

Authoritative source paths:

- `src/drpo/e7_canonical_vendor/d4rl/agents.py`;
- `src/drpo/e7_canonical_vendor/d4rl/train_sna2c_variant.py`;
- `src/drpo/e7_canonical_vendor/d4rl/d4rl_common/train_loop.py`;
- `src/drpo/e7_canonical_vendor/d4rl/d4rl_common/normalize.py`.

## Migration decision

The paper-facing implementation does not copy a second actor, critic, or trainer. `paper_code/src/drpo_reference/experiments/d4rl.py` now acts as a stable adapter to the vendored canonical source.

It now provides:

- backend identity `canonical_sna2c_iqlv_exprank`;
- `implementation_selected=true` and `implementation_migrated=true`;
- SHA-256 provenance for every authoritative source file;
- dynamic verification that `SNA2C_IQLV_ExpRankAgent` is present;
- exact construction of the `train_sna2c_variant.py --variant iqlv_exp_rank` command;
- one task runner for all nine D4RL coordinates;
- plan-only mode by default and optional non-formal execution;
- explicit refusal of formal-evidence and method-ranking flags.

The vendored source remains the only algorithm implementation. No per-environment performance trainer was added, and the Hopper mechanism trainer is not reused.

## Differential coverage

The existing D4RL differential test now checks:

- the exact nine-task matrix and score references;
- all task rollout identities;
- backend selection and migration state;
- loading and one finite update of the authoritative ExpRank agent;
- source provenance inventory;
- canonical trainer command arguments;
- one trainer path across all nine tasks;
- plan-only records with formal claims disabled;
- remaining formal blockers.

These are engineering checks, not scientific results.

## Remaining formal blockers

Backend selection and migration are no longer blockers. Formal D4RL-9 execution still requires:

1. SHA-256 values for the eight unresolved datasets;
2. the final common method controls and coefficients;
3. the registered ten-run seed coordinate;
4. formal budget, checkpoint policy, and terminal-audit rules;
5. real nine-task HDF5 and Gymnasium/MuJoCo liveness;
6. separate reporting of task-performance collapse, support/variance-boundary events, rollout failure, and NaN/Inf failure.

Until these are frozen, `formal_evidence_eligible=false` and `method_ranking_claim_allowed=false` remain mandatory.

## Execution status

No real HDF5 dataset, MuJoCo environment, actor training, critic training, manuscript-number validation, or method ranking was executed in this slice. PR `#149` remains Draft and unmerged.
