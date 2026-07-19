# PAPER-CODE-REFERENCE-01 Current Status

**Document role:** canonical task-local current snapshot and continuation index.  
**Not a research master:** `docs/handoff.md` remains the unique research source of truth.  
**Claim:** `PAPER-CODE-REFERENCE-01`  
**Scientific-status impact:** none.  
**Last audited branch head before this document:** `b2d485c19dc0c1d7e5e7bb2f51252770f3833d8a`.

This file consolidates the current engineering state without deleting historical records. It supersedes stale *current-status* and *next-slice* statements in the older task-local handoff and delta files. Those older files remain provenance for the order in which the migration was implemented and reviewed.

## 1. Mandatory continuation order

A new session must read and verify, in this order:

1. `AGENTS.md` from current `main`;
2. Section 0 of `docs/handoff.md` from current `main`;
3. `experiments/registry.yaml` from current `main`;
4. this file;
5. `docs/paper_code_reference/ACCEPTANCE_MATRIX.yaml`;
6. `docs/paper_code_reference/SOURCE_MIGRATION_MAP.md` for source ownership and historical inclusion/exclusion decisions;
7. `docs/paper_code_reference/IMPLEMENTATION_PLAN.md` as the original architecture and acceptance plan, not as a live file inventory;
8. the actual `dev/paper-code-reference-01` branch, Draft PR `#149`, changed files, legacy differential oracles, and exact-head CI.

The older `SESSION_HANDOFF*` files are append-only historical implementation records. They are no longer the preferred way to discover the current next step.

## 2. Repository and branch snapshot

At the audit immediately before this document:

- repository: `easonhuo/drpo`;
- default branch: `main`;
- current `main`: `85b0a68d77ed085a7f6e67771fb0f7672c43da09`;
- task base and current merge base: `4544005bd7df69c53bad70a9dcac846af01285e4`;
- only active development branch: `dev/paper-code-reference-01`;
- development head before this document: `b2d485c19dc0c1d7e5e7bb2f51252770f3833d8a`;
- persistent cumulative Draft PR: `#149`;
- PR state: open, Draft, unmerged;
- overall task state: `in_development`.

The SHA values above are audit facts, not reusable assumptions. Every continuation session must resolve both heads again. The branch remains separate from `main` until the user explicitly authorizes a merge decision.

The branch still lags newer unrelated `main` work. This does not prove a conflict, but it requires an integration-freshness audit before a final merge proposal. Do not silently rebase, merge `main`, or reinterpret newer scientific registrations as part of this task.

## 3. Reviewer-facing code boundary

This boundary governs all remaining migration work.

The public `paper_code` package is reviewer-facing reference code. Its primary obligations are:

- readable algorithm implementation;
- explicit dataset and environment identities;
- runnable training entry points;
- runnable rollout evaluation;
- checkpoints and basic run metadata;
- a lightweight completion/failure record;
- simple multi-seed mean/std aggregation.

It is **not** required to duplicate the repository's internal scientific-governance platform. The following remain internal responsibilities and are not hard requirements for reviewer-facing migration closure:

- registry or handoff mutation;
- formal-evidence eligibility decisions;
- full task × method × seed completeness governance;
- selected-versus-terminal checkpoint scientific adjudication;
- manuscript table-cell and artifact-hash binding;
- internal collapse-taxonomy adjudication and formal result promotion.

The public code still fails clearly on missing files, invalid shapes, non-finite training, unavailable rollout environments, and incomplete commands. Lightweight fields such as `training_completed`, `final_step`, `final_checkpoint`, `finite`, and `evaluation_completed` are normal software robustness, not an internal formal audit.

Training and rollout scores may vary across hardware, dependency versions, and random seeds. Reviewer-facing reproducibility means the algorithm and stated protocol are runnable and the reported trend is reproducible under the specified coordinate; it does not promise byte-identical single-run scores on every machine.

## 4. Two-axis status model

Do not conflate the scientific status registered in the main repository with the reproduction status of `paper_code`.

| Component | Existing scientific status | `paper_code` migration status | Remaining reviewer-code gate |
|---|---|---|---|
| Shared utilities and controls | no independent scientific claim | implementation complete; engineering validated | final integration review |
| C-U1 | experiment-specific statuses remain authoritative in `docs/handoff.md` and the registry | implementation complete | registered reproduction remains an internal scientific task |
| D-U1 revision 4 | `not_run` for the active formal matrix | implementation complete | formal run remains an internal scientific task |
| Hopper E7-Q2 | `long_run_validated` for the existing learned-critic external mechanism result | implementation complete | real registered-data reproduction through the new runner |
| D4RL-9 / `EXT-H-E7-BENCH-01` | historical archive is pilot provenance only; no formal ranking | algorithm, public multi-method training, rollout evaluation, and simple aggregation implemented | real HDF5/MuJoCo liveness |
| Countdown | final manuscript-facing protocol/result not frozen | blocked at the experiment-entry layer | freeze final protocol and result before migration |

A smoke, static check, first update, or fixed short trajectory is engineering evidence only. It does not change scientific status.

## 5. Current implementation ownership

### 5.1 C-U1

The live implementation is split by responsibility across `continuous/cu1*.py`, with public dispatch in `cli.py`. It covers source, causal, phase/control, taper, artifacts, aggregation, and audit for the controlled continuous environment.

### 5.2 D-U1 revision 4

The live implementation is split across `categorical/du1_*.py`, with public dispatch in `cli.py`. It covers the revision-4 environment, categorical policy, six frozen controls, shared-start training, metrics, public execution, and reports. Revisions 1/2/3 and the historical reciprocal-quartic method remain excluded.

### 5.3 Hopper E7-Q2 mechanism profile

The Hopper mechanism implementation includes data, models, critic, frozen advantages, actor training, diagnostics, rollout, suite execution, public runner, aggregation, and terminal records. It remains scientifically distinct from the D4RL-9 task-performance backend.

### 5.4 D4RL-9 performance profile

Implemented reviewer-facing code:

- exact nine-task catalog, dataset basenames, environment IDs, reference-score constants, and fail-closed SHA state in `external/d4rl_tasks.py`;
- `CanonicalActor`, `CanonicalCritic`, and the shared `SNA2CIQLVExpRankAgent` lifecycle;
- dynamic TD advantages, expectile critic regression, and actor-then-critic Adam updates;
- reviewer-selectable `exprank`, `positive_only`, `signed`, `global`, `reciprocal_linear`, `reciprocal_quadratic`, and `exponential` actor-side controls;
- ExpRank as the only implicit default;
- explicit `legacy-pilot-v1` acknowledgement before any historical non-ExpRank control may run;
- historical pilot coefficients carried as provenance, with `profile_is_final=false` and `final_method_matrix_frozen=false`;
- one shared trainer across every task, method, and seed, with no per-task trainer copies and no post-hoc per-task method selection;
- locomotion reward normalization, episode-aware returns, action clipping, deterministic minibatch training, and legacy-compatible ExpRank checkpoints;
- public `drpo-reference d4rl` CLI with optional `--methods` and `--method-profile`;
- selected-task or all-nine-task training, method × seed output roots, final checkpoints, and lightweight completion/failure records;
- optional direct Gymnasium/MuJoCo evaluation in `HalfCheetah-v4`, `Hopper-v4`, and `Walker2d-v4`;
- raw return and D4RL normalized score per episode;
- episode mean/std and task × method mean/std across seed means;
- differential and controlled tests for formulas, detached distance, legacy ExpRank compatibility, method-profile gates, execution paths, rollout semantics, and aggregation.

The canonical legacy D4RL files remain differential oracles, not runtime dependencies of the paper package. The D4RL evaluator is intentionally direct and readable rather than duplicating Hopper's heavier process-isolated mechanism-validation preflight. Environment failure is reported clearly and does not become a task-performance score.

## 6. D4RL remaining work

### 6.1 Reviewer-facing runtime

No additional core actor, critic, multi-method training-loop, rollout, or simple aggregation migration is currently required. The remaining reviewer-facing gate is a real liveness run using compatible HDF5 files and Gymnasium/MuJoCo. That liveness is execution evidence only and must not be reported as a formal method result.

A standalone checkpoint-only evaluation command could be added later for convenience, but it is not required for the current train-and-evaluate reviewer workflow because `drpo-reference d4rl --eval-episodes N` evaluates each trained actor before command completion.

### 6.2 Reviewer-selectable methods versus final paper matrix

The code exposes seven reviewer-selectable arms. This is an implementation migration, **not** a freeze of the final D4RL-9 comparison matrix. Historical non-ExpRank arms are available only under the explicitly named `legacy-pilot-v1` profile. Their coefficients, negative scales, and provenance are recorded in manifests, but they must not be interpreted as final nine-task paper choices.

The final comparison arms, common coefficients, formal ten-run seeds, budgets, and checkpoint policy still require separate protocol approval. No command may perform post-hoc per-task method selection, and no reviewer run may claim a formal ranking.

### 6.3 Internal formal experiment lifecycle

Formal seeds, budgets, checkpoint roles, full matrix completeness, terminal scientific adjudication, result promotion, and manuscript artifact binding remain in the internal repository workflow. They are not reviewer-code migration defects.

## 7. D4RL items that are not missing code

The following are provenance, protocol, resource, execution, or internal review gates:

- SHA-256 values for eight unresolved dataset coordinates;
- authoritative resolution of the historical archive launch commit;
- final common comparison arms and coefficients;
- registered ten-run seed coordinate;
- formal budgets and checkpoint policy;
- availability of all real HDF5 datasets and compatible Gymnasium/MuJoCo dependencies;
- real nine-task liveness and full-budget execution;
- internal terminal scientific review and final manuscript values.

## 8. Exact next sequence

1. Perform the integration-freshness audit against current `main` before any final merge proposal.
2. Run a low-cost real HDF5/Gymnasium/MuJoCo liveness on one representative coordinate.
3. Expand real liveness to the three environment families before any nine-task execution.
4. Keep formal experiment registration, final method-matrix freeze, full execution, terminal scientific review, and manuscript artifact binding in the internal repository workflow.

The multi-method code slice passed exact-head Python compile, full pytest, Ruff, handoff authority, formal execution channel, governance inventory, and governance stage gates at `b2d485c19dc0c1d7e5e7bb2f51252770f3833d8a`. No formal experiment was launched. No method ranking, scientific status, seed, threshold, formal budget, or final coefficient was changed.
