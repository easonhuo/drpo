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

### Canonical D4RL performance lineage

The canonical performance lineage inspected in `main` uses:

- the vendored historical `SNA2C_IQLV_ExpRankAgent` backbone;
- dynamic TD advantages recomputed during training;
- joint actor and expectile-value updates;
- a bounded actor mean with a Normal likelihood evaluated directly on the bounded dataset action;
- a passthrough ExpRank branch plus injected negative-control variants that change only the negative-advantage multiplier;
- fixed-horizon pilot profiles, not a frozen formal D4RL-9 protocol.

Relevant source paths include:

- `src/drpo/e7_canonical_vendor/d4rl/agents.py`;
- `src/drpo/e7_canonical_vendor/d4rl/train_sna2c_variant.py`;
- `src/drpo/e7_canonical_injection.py`;
- `src/drpo/e7_canonical_sweep.py`;
- `src/drpo/e7_canonical_two_dataset.py`;
- `configs/e7_canonical_weight_grid_v1.json`;
- `src/drpo/e7_canonical_shortlist_protocol.py`.

The two-dataset wrapper fixes `SNA2C_IQLV_ExpRankAgent` as the target class and defines the original ExpRank passthrough plus injected controls. The later shortlist remains pilot-only and covers only Hopper medium-replay and medium-expert.

## 3. Nine-task archive evidence and limit

The repository closure audit records an external canonical D4RL archive with:

- nine datasets;
- seeds `200` and `201`;
- one-million-step fixed horizons;
- `576/576` compact rows;
- no reported numerical failure;
- pilot status and `formal_ranking=false`.

The row count is consistent with `9 datasets × 2 seeds × 32 branches`. The canonical weight grid defines 31 injected branches, and the canonical wrapper adds one original ExpRank passthrough, so the archived matrix is structurally consistent with this lineage.

This is an inference from the committed row counts and current canonical source family, not proof of an exact launch commit. The closure explicitly records that the archive's repository commit is absent and unresolved. The original report's post-hoc per-task best cells and broad-grid ranking remain descriptive only. Therefore the archive cannot be promoted into the manuscript's claimed ten-run formal benchmark or used to freeze a method winner.

## 4. Correct sharing boundary

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

## 5. Implementation changes

`paper_code/src/drpo_reference/experiments/d4rl.py` records an explicit `D4RLPerformanceBackendSpec` and the audited candidate:

- experiment ID: `EXT-H-E7-BENCH-01`;
- backend ID: `legacy_canonical_sna2c_iqlv_candidate`;
- algorithm family: `SNA2C_IQLV_ExpRank`;
- status: `pilot_only_unfrozen`;
- formal task-matrix eligibility: false;
- Hopper mechanism-runner reuse: false.

The execution plan blocks formal eligibility on both:

- `d4rl9_performance_backend_not_frozen`;
- `d4rl9_backend_not_formal_matrix_eligible`.

Its manifest distinguishes:

- `shared_task_data_rollout_boundary=true`;
- `shared_full_training_engine=false`;
- `separate_per_task_trainers_allowed=false`.

`paper_code/src/drpo_reference/external/d4rl_tasks.py` now owns the backend-independent rollout identity for every D4RL-9 cell:

- backend `gymnasium_mujoco`;
- exact dataset ID;
- exact Gymnasium environment ID;
- the corresponding D4RL reference-score constants.

This advances the shared task/environment boundary without selecting or migrating an unfrozen trainer. The focused differential test verifies all nine identities, the backend separation, formal blockers, and one backend dispatch across the matrix.

## 6. Source-map correction

`docs/paper_code_reference/SOURCE_MIGRATION_MAP.md` replaces the over-broad full-engine sharing statement with the audited boundary above. Historical content was not deleted from Git history; the correction is recorded here and in the PR conversation.

The result-reporting scope was also narrowed to a minimal scientific summary and protocol audit. A general table/figure-generation framework and byte-identical stochastic-output verifier remain out of scope.

## 7. Scientific and execution status

Unchanged:

- Hopper E7-Q2 remains a mechanism external-validity path;
- D4RL-9 remains a task-performance external-validity path;
- the canonical nine-task archive and shortlist remain pilot evidence only;
- the D4RL-9 formal backend, methods, datasets, seeds, budget, and terminal rule remain unfrozen;
- no method ranking is authorized;
- no real HDF5, MuJoCo, critic, actor, or benchmark execution occurred in this audit.

## 8. Next allowed step

Before migrating a D4RL-9 performance trainer, the next session must establish from the research authority and registered evidence:

1. whether the canonical ExpRank/injection lineage is the intended formal backend rather than only the source of provisional broad-grid numbers;
2. the single method and coefficients to be evaluated on untouched runs, rather than post-hoc per-task best cells;
3. exact dataset identities and SHA-256 values for all nine cells;
4. the formal ten-run seed coordinate, budget, checkpoint policy, and terminal-audit rule;
5. whether the manuscript's current ten-run wording is a future protocol target or already has unseen evidence not present in the repository.

Until those are frozen, only backend-independent task, dataset, rollout, normalization, and audit-boundary work may proceed. Do not migrate the broad-grid pilot as though it were the final formal backend.
