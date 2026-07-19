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

This append-only delta corrects one engineering premise in the preceding D4RL sharing delta. It does not replace `docs/handoff.md`, authorize a formal run, change a scientific variable, or permit merging PR `#149`.

## 1. Repository snapshot

- repository: `easonhuo/drpo`;
- authoritative main inspected for this audit: `e99489e7435bc26e2a7e30cd8d1a3aa10f4fc67a`;
- development branch before the audit: `dev/paper-code-reference-01@b2c33539c75462f3cb5cccd68a25f19b072ace51`;
- persistent Draft PR: `#149`;
- no scientific experiment was launched.

## 2. Premise corrected by source audit

The preceding delta correctly prohibited a separate trainer for each HalfCheetah, Hopper, or Walker2d task. It was too broad when it described Hopper E7-Q2 and D4RL-9 as sharing one complete actor/critic/training implementation.

Inspection of the authoritative code showed a material scientific difference.

### Hopper E7-Q2 mechanism backend

The migrated Hopper mechanism path uses:

- one canonical critic and frozen advantages;
- actor-only method branches after preparation;
- a latent squashed-Gaussian policy;
- inverse-tanh action mapping and tanh-Jacobian correction in the logged action density;
- mechanism methods `positive_only`, `signed`, `near_zero`, `far_zero`, `far_cap`, and `dynamic_budget_matched_global`;
- matched near/far probes and targeted mechanism interventions.

Relevant reference paths include:

- `paper_code/src/drpo_reference/external/hopper_models.py`;
- `paper_code/src/drpo_reference/external/hopper_actor.py`;
- `paper_code/src/drpo_reference/external/hopper_suite.py`.

### Legacy D4RL performance candidate

The latest registered canonical performance candidate inspected in `main` uses:

- the canonical SNA2C/IQLV agent family;
- dynamic TD advantages recomputed during training;
- joint actor and expectile-value updates;
- a bounded actor mean with a Normal likelihood evaluated directly on the bounded dataset action;
- an ExpRank negative-weight schedule and injected shortlist controls;
- a pilot-only two-Hopper-dataset, four-seed, one-million-step protocol.

Relevant source paths are:

- `src/drpo/e7_canonical_vendor/d4rl/agents.py`;
- `src/drpo/e7_canonical_vendor/d4rl/train_sna2c_variant.py`;
- `src/drpo/e7_canonical_shortlist_protocol.py`.

The shortlist protocol is explicitly pilot-only, covers only Hopper medium-replay and medium-expert, and does not authorize the manuscript D4RL-9 table or a formal method ranking.

## 3. Correct sharing boundary

The implementation should share contracts only where they are actually equivalent:

- D4RL task and dataset identities;
- HDF5 schema validation;
- observation/action shape checks;
- Gymnasium/MuJoCo interaction boundary;
- D4RL reference-score normalization;
- common provenance, I/O, and event taxonomy;
- one nine-task catalog.

The following remain backend-specific unless later evidence proves exact equivalence:

- actor likelihood semantics;
- critic objective and lifecycle;
- advantage estimator and lifecycle;
- optimizer schedule;
- method matrix;
- terminal-audit protocol.

Therefore the correct anti-duplication rule is:

> Select one scientifically frozen D4RL performance backend for all nine tasks, but do not force the Hopper mechanism backend to serve as that trainer and do not create separate trainers per task.

## 4. Implementation change

`paper_code/src/drpo_reference/experiments/d4rl.py` now records an explicit `D4RLPerformanceBackendSpec` and the audited candidate:

- experiment ID: `EXT-H-E7-BENCH-01`;
- backend ID: `legacy_canonical_sna2c_iqlv_candidate`;
- algorithm family: `SNA2C_IQLV_ExpRank`;
- status: `pilot_only_unfrozen`;
- formal task-matrix eligibility: false;
- Hopper mechanism-runner reuse: false.

The execution plan now blocks formal eligibility on both:

- `d4rl9_performance_backend_not_frozen`;
- `d4rl9_backend_not_formal_matrix_eligible`.

Its manifest distinguishes:

- `shared_task_data_rollout_boundary=true`;
- `shared_full_training_engine=false`;
- `separate_per_task_trainers_allowed=false`.

The focused differential test verifies this boundary and preserves the complete nine-task dispatch through one backend-specific runner.

## 5. Source-map correction

`docs/paper_code_reference/SOURCE_MIGRATION_MAP.md` now replaces the over-broad full-engine sharing statement with the audited boundary above. Historical content was not deleted from Git history; the correction is recorded here and in the PR conversation.

The result-reporting scope was also narrowed to a minimal scientific summary and protocol audit. A general table/figure-generation framework and byte-identical stochastic-output verifier remain out of scope.

## 6. Scientific and execution status

Unchanged:

- Hopper E7-Q2 remains a mechanism external-validity path;
- D4RL-9 remains a task-performance external-validity path;
- the inspected canonical shortlist remains pilot evidence only;
- the D4RL-9 formal backend, methods, datasets, seeds, budget, and terminal rule remain unfrozen;
- no method ranking is authorized;
- no real HDF5, MuJoCo, critic, actor, or benchmark execution occurred in this audit.

## 7. Next allowed step

Before migrating a D4RL-9 performance trainer, the next session must establish from the research authority and registered evidence:

1. which algorithm/backbone actually supports the manuscript's D4RL-9 candidate table;
2. whether the canonical SNA2C/IQLV backend is selected, superseded, or only a pilot lineage;
3. the final method shortlist and coefficients;
4. exact dataset identities and SHA-256 values for all nine cells;
5. the formal ten-run seed coordinate, budget, and terminal-audit rule.

Until those are frozen, only backend-independent task, dataset, rollout, normalization, and audit-boundary work may proceed. Do not migrate the pilot trainer as though it were the final formal backend.
