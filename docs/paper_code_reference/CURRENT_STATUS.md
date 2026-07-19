# PAPER-CODE-REFERENCE-01 Current Status

**Document role:** canonical task-local current snapshot and continuation index.  
**Not a research master:** `docs/handoff.md` remains the unique research source of truth.  
**Claim:** `PAPER-CODE-REFERENCE-01`  
**Scientific-status impact:** none.  
**Last audited branch head before this document:** `1394a29ef79296ab3ae1e1f3793417fb9d430444`.

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
- development head before this document: `1394a29ef79296ab3ae1e1f3793417fb9d430444`;
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

The public code should still fail clearly on missing files, invalid shapes, non-finite training, unavailable rollout environments, and incomplete commands. A lightweight record such as `training_completed`, `final_step`, `final_checkpoint`, `finite`, and `evaluation_completed` is normal software robustness, not an internal formal audit.

Training and rollout scores may vary across hardware, dependency versions, and random seeds. Reviewer-facing reproducibility means the algorithm and stated protocol are runnable and the reported trend is reproducible under the specified coordinate; it does not promise byte-identical single-run scores on every machine.

## 4. Two-axis status model

Do not conflate the scientific status registered in the main repository with the reproduction status of `paper_code`.

| Component | Existing scientific status | `paper_code` migration status | Remaining reviewer-code gate |
|---|---|---|---|
| Shared utilities and controls | no independent scientific claim | implementation complete; engineering validated | final integration review |
| C-U1 | experiment-specific statuses remain authoritative in `docs/handoff.md` and the registry | implementation complete | registered reproduction remains an internal scientific task |
| D-U1 revision 4 | `not_run` for the active formal matrix | implementation complete | formal run remains an internal scientific task |
| Hopper E7-Q2 | `long_run_validated` for the existing learned-critic external mechanism result | implementation complete | real registered-data reproduction through the new runner |
| D4RL-9 / `EXT-H-E7-BENCH-01` | historical archive is pilot provenance only; no formal ranking | algorithm core and public training runner implemented | rollout evaluator, simple aggregation, real liveness |
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

Already migrated:

- exact nine-task catalog, dataset basenames, environment IDs, reference-score constants, and fail-closed SHA state in `external/d4rl_tasks.py`;
- `CanonicalActor`, `CanonicalCritic`, and `SNA2CIQLVExpRankAgent`;
- dynamic TD advantages, expectile critic regression, and ExpRank negative weighting;
- actor-then-critic Adam updates;
- locomotion reward normalization, episode-aware returns, action clipping, and deterministic minibatch training;
- legacy-compatible actor checkpoint payloads;
- differential tests for initialization, forward values, rank weights, first Adam update, fixed short trajectory, dataset preparation, checkpoint records, task identities, and one-backend dispatch;
- reviewer-facing `run_d4rl` entry point in `experiments/__init__.py`;
- public `drpo-reference d4rl` CLI;
- selected-task or all-nine-task training, per-seed output roots, final checkpoints, `RUN_MANIFEST.json`, `SUMMARY.json`, `COMPLETED.json`, and `FAILED.json`;
- explicit non-formal status and explicit `evaluation_completed: false` until rollout evaluation exists.

The canonical legacy D4RL files remain differential oracles, not runtime dependencies of the paper package.

## 6. D4RL code still required

### 6.1 Three-environment rollout evaluator

The next core reviewer-facing slice is deterministic evaluation of trained ExpRank actors in:

- `HalfCheetah-v4`;
- `Hopper-v4`;
- `Walker2d-v4`.

Required behavior:

- process-isolated Gymnasium/MuJoCo preflight;
- observation and action dimension checks against the checkpoint and dataset task;
- deterministic actor action with clipping;
- correct reset/step, termination/truncation, episode seeding, timeout, and environment-unavailable behavior;
- raw return and D4RL normalized-score calculation;
- mean/std across evaluation episodes;
- no silent `d4rl` or `mujoco_py` fallback.

The characterized low-level contracts in the Hopper rollout code may be reused, but the Hopper mechanism trainer must not become the D4RL performance trainer.

### 6.2 Simple aggregation

After rollout evaluation, add a small deterministic aggregator that reports per-task and per-seed raw return and normalized-score mean/std. It should reject missing or malformed reviewer-run outputs but does not need manuscript table-cell binding or internal formal-evidence governance.

### 6.3 Method-matrix execution remains protocol blocked

Only the selected `SNA2C_IQLV_ExpRank` backend is currently exposed. The final comparison arms and common coefficients are not frozen, so no multi-method matrix should be invented. After explicit protocol approval, exact frozen arms may be added through the same shared backend without per-task trainer copies or post-hoc per-task method selection.

## 7. D4RL items that are not missing code

The following are provenance, protocol, resource, execution, or internal review gates rather than current reviewer-code defects:

- SHA-256 values for eight unresolved dataset coordinates;
- authoritative resolution of the historical archive launch commit;
- final common comparison arms and coefficients;
- registered ten-run seed coordinate;
- formal budgets and checkpoint policy;
- availability of all real HDF5 datasets and compatible Gymnasium/MuJoCo dependencies;
- real liveness and full-budget execution;
- internal terminal scientific review and final manuscript values.

## 8. Exact next sequence

1. Run exact-head CI for the new D4RL public training runner.
2. Perform the integration-freshness audit against current `main` before the next substantial slice.
3. Implement the three-environment rollout evaluator by extending existing Python files where responsibilities fit; any new `.py` path still requires explicit human approval.
4. Add controlled fake-environment tests and focused checkpoint/evaluation tests.
5. Run real environment liveness before any large execution.
6. Add simple reviewer-facing multi-seed aggregation.
7. Keep formal experiment registration, full execution, terminal scientific review, and manuscript artifact binding in the internal repository workflow.

No formal experiment was launched by this code slice. No method ranking, scientific status, seed, threshold, formal budget, or coefficient was changed.
