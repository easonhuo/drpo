# PAPER-CODE-REFERENCE-01 Current Status

**Document role:** canonical task-local current snapshot and continuation index.  
**Not a research master:** `docs/handoff.md` remains the unique research source of truth.  
**Claim:** `PAPER-CODE-REFERENCE-01`  
**Scientific-status impact:** none.  
**Last audited branch head before this document:** `cf559e3becd4ee292edb1b6f3062b855ffb3a8d1`.

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

The older files below are append-only historical implementation records. They are no longer the preferred way to discover the current next step:

- `SESSION_HANDOFF.md`;
- `SESSION_HANDOFF_ROLLOUT_DELTA_20260718.md`;
- `SESSION_HANDOFF_HOPPER_PUBLIC_DELTA_20260718.md`;
- `SESSION_HANDOFF_D4RL_SHARED_DELTA_20260719.md`;
- `SESSION_HANDOFF_D4RL_BACKEND_AUDIT_DELTA_20260719.md`.

## 2. Repository and branch snapshot

At the audit immediately before this document:

- repository: `easonhuo/drpo`;
- default branch: `main`;
- current `main`: `85b0a68d77ed085a7f6e67771fb0f7672c43da09`;
- task base and current merge base: `4544005bd7df69c53bad70a9dcac846af01285e4`;
- only active development branch: `dev/paper-code-reference-01`;
- development head: `cf559e3becd4ee292edb1b6f3062b855ffb3a8d1`;
- relation to current `main`: 95 commits ahead and 60 commits behind;
- persistent cumulative Draft PR: `#149`;
- PR state: open, Draft, unmerged;
- overall task state: `in_development`.

The SHA values above are audit facts, not reusable assumptions. Every continuation session must resolve both heads again. The branch must remain separate from `main` until the user explicitly authorizes a merge decision.

The 60-commit lag does not by itself prove a code conflict. It does require an integration-freshness audit before the next substantial code slice and before any final merge proposal. Do not silently rebase, merge `main`, or reinterpret newer scientific registrations as part of this task.

## 3. Two-axis status model

Do not conflate the scientific status already registered in the main repository with the reproduction status of the new `paper_code` implementation.

| Component | Existing scientific status | `paper_code` migration status | Remaining gate |
|---|---|---|---|
| Shared utilities and controls | no independent scientific claim | implementation complete; engineering validated | none beyond integration and final review |
| C-U1 | existing experiment-specific statuses remain authoritative in `docs/handoff.md` and the registry | implementation complete | registered full CPU reproduction, terminal review, selected conclusion report |
| D-U1 revision 4 | `not_run` for the active revision-4 formal matrix | implementation complete | registered 20-seed × six-method × 8000-step run and terminal review |
| Hopper E7-Q2 | `long_run_validated` for the existing learned-critic external mechanism result | implementation complete | real registered-data reproduction through the new paper runner and terminal review |
| D4RL-9 / `EXT-H-E7-BENCH-01` | historical archive is pilot provenance only; no formal ranking | selected ExpRank algorithm core migrated; formal runtime incomplete | protocol freeze, remaining runtime code, real liveness, registered execution, terminal review |
| Countdown | final manuscript-facing protocol/result not frozen | blocked at the experiment-entry layer | freeze final protocol and result before migration |

A smoke, static check, first update, or fixed three-step trajectory is engineering evidence only. It does not change any scientific status.

## 4. Current implementation ownership

### 4.1 C-U1

The live paper-facing implementation is split by responsibility rather than represented by the early conceptual `experiments/cu1.py` sketch:

- `paper_code/src/drpo_reference/continuous/cu1.py`: environment, policy objectives, evaluation primitives;
- `continuous/cu1_training.py`: Positive-only and shared training lifecycle;
- `continuous/cu1_mechanism.py` and `continuous/cu1_source_causal.py`: source and causal paths;
- `continuous/cu1_phase.py`, `continuous/cu1_control.py`, and `continuous/cu1_phase_taper.py`: strength scans and controls;
- `continuous/cu1_taper.py`: taper-family execution;
- `continuous/cu1_suite.py`, `continuous/cu1_public_protocol.py`, `continuous/cu1_public_audit.py`, and `continuous/cu1_artifacts.py`: public execution, artifacts, aggregation, and audit;
- `paper_code/src/drpo_reference/cli.py`: public command dispatch.

### 4.2 D-U1 revision 4

The live implementation is:

- `categorical/du1_environment.py`: utility × rarity environment and roles;
- `categorical/du1_policy.py`: categorical policy;
- `categorical/du1_controls.py`: six frozen controls and calibration;
- `categorical/du1_training.py`: shared-start training and updates;
- `categorical/du1_metrics.py`: task and mechanism metrics;
- `categorical/du1_protocol.py`: frozen revision-4 coordinate;
- `categorical/du1_suite.py`, `categorical/du1_public.py`, and `categorical/du1_reports.py`: execution, artifacts, aggregation, and terminal reports;
- `paper_code/src/drpo_reference/cli.py`: public command dispatch.

Revision 1/2/3 and the historical reciprocal-quartic method remain excluded.

### 4.3 Hopper E7-Q2 mechanism profile

The Hopper mechanism implementation is complete at the engineering layer:

- `external/hopper_data.py`, `hopper_models.py`, `hopper_critic.py`, and `hopper_advantages.py`;
- `external/hopper_actor.py`, `hopper_metrics.py`, and `hopper_optim.py`;
- `external/hopper_rollout.py` and `hopper_protocol.py`;
- `external/hopper_suite.py`;
- `experiments/hopper.py` for the public runner and aggregation;
- `cli.py` for `drpo-reference hopper`.

It remains scientifically distinct from the D4RL-9 task-performance backend.

### 4.4 D4RL-9 performance profile

Already migrated:

- exact nine-task catalog, dataset basenames, environment IDs, reference-score constants, and fail-closed SHA state in `external/d4rl_tasks.py`;
- `CanonicalActor` and `CanonicalCritic`;
- `SNA2CIQLVExpRankAgent`;
- dynamic TD advantages and expectile critic regression;
- rank-based negative weighting;
- actor-then-critic Adam updates;
- locomotion reward normalization, episode-aware returns, and action clipping;
- deterministic uniform-minibatch training;
- legacy-compatible actor checkpoint payloads and non-formal completion records;
- one execution-plan and dispatch boundary across all nine tasks;
- differential tests for initialization, forward values, rank weights, first Adam update, fixed three-step trajectory, dataset preparation, checkpoint records, task identities, and one-backend dispatch.

The canonical legacy D4RL files remain differential oracles, not the runtime implementation of the paper package.

## 5. D4RL code that is still missing

The selected algorithm core is migrated. The remaining code is primarily the formal runtime and evidence lifecycle, not another actor/critic rewrite.

### 5.1 Concrete D4RL-9 public runner

`dispatch_d4rl9` currently accepts an injected `task_runner`; the branch does not yet provide the complete concrete runner that:

- loads and validates each of the nine HDF5 datasets;
- prepares the canonical locomotion tensors;
- creates each registered method/seed run;
- invokes training and evaluation;
- enforces new-or-empty output roots;
- writes per-task and root completion or failure state;
- exposes a stable `drpo-reference d4rl` command.

### 5.2 Backend-compatible three-environment rollout evaluation

The task catalog owns rollout identities, but D4RL-9 does not yet have a complete evaluator for the migrated ExpRank actor across `HalfCheetah-v4`, `Hopper-v4`, and `Walker2d-v4`.

Remaining behavior includes:

- process-isolated Gymnasium/MuJoCo preflight;
- observation/action dimension checks against each dataset;
- deterministic actor evaluation and action clipping;
- reset/step, termination/truncation, episode seeding, timeout, and environment-unavailable diagnostics;
- raw and normalized return calculation for all three environments;
- fail-closed real-liveness behavior without `d4rl` or `mujoco_py` fallback.

The existing Hopper E7-Q2 rollout module is a mechanism-profile implementation. It may supply characterized low-level contracts, but it must not silently turn the Hopper mechanism trainer into the D4RL performance trainer.

### 5.3 Frozen method-matrix execution

Only the selected `SNA2C_IQLV_ExpRank` backend core is migrated. The final D4RL-9 comparison matrix and common coefficients are not frozen, so the corresponding method-selection and multi-arm execution code must not be invented yet.

After protocol approval, the code must implement exactly the frozen arms through one shared backend and prohibit post-hoc per-task method selection. Until then, this is a protocol-blocked code item.

### 5.4 Formal budget, checkpoint, and terminal-audit lifecycle

The current non-formal trainer writes checkpoints and completion records, but D4RL-9 still lacks the final registered implementation of:

- fixed training budget and evaluation cadence;
- checkpoint roles and any selection rule;
- terminal versus selected checkpoint reporting;
- per-run terminal-state classification;
- root completeness verification;
- separate task-performance collapse, support/variance-boundary, rollout failure, incomplete terminal state, and NaN/Inf counts;
- formal-evidence eligibility only after all expected tasks, methods, and seeds pass audit.

These details must follow a frozen protocol rather than be inferred from the historical pilot.

### 5.5 D4RL-9 aggregation and minimal paper binding

The branch still needs deterministic aggregation that consumes one explicitly selected completed result root and produces the minimum result summary required by the manuscript. It must bind each value to task, method, seed, checkpoint role, and source artifact path. It must reject smoke, subsets, incomplete roots, mixed identities, and tampered inputs.

This does not require a universal table/figure framework or training-time plotting code.

## 6. D4RL blockers that are not missing code

The following items must not be reported as implementation defects:

- SHA-256 values for eight unresolved dataset coordinates;
- authoritative resolution of the historical archive launch commit;
- final common method controls and coefficients;
- the registered ten-run seed coordinate;
- formal budgets, checkpoint policy, and terminal-audit rules before they are coded;
- availability of nine real HDF5 datasets and compatible Gymnasium/MuJoCo dependencies;
- real liveness, full-budget execution, and post-run scientific audit;
- final manuscript candidate values.

These are provenance, protocol, resource, execution, or review gates.

## 7. Exact next sequence

1. **Document and freeze the D4RL-9 formal protocol first:** dataset identities, method arms, coefficients, seeds, budgets, checkpoint roles, terminal rules, artifact paths, and liveness gate.
2. **Perform an integration-freshness audit** against current `main`; do not silently absorb unrelated scientific work.
3. **Implement the concrete D4RL runtime** using existing files where responsibilities fit. Any exact new `.py` path requires prior human approval under `GOV-NEW-PYTHON-FILE-HUMAN-APPROVAL-01`.
4. **Run focused differential and controlled fake-environment tests.**
5. **Run real liveness** on the frozen backend and dataset identities before a nine-task sweep.
6. **Register and launch the formal run only through the repository's hardened execution channel.**
7. **Perform terminal review and minimal paper-result binding.**

No formal experiment was launched by this documentation consolidation. No method ranking, scientific status, seed, threshold, budget, or coefficient was changed.
