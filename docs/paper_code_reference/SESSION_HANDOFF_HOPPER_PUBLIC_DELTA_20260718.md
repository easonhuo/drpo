# PAPER-CODE-REFERENCE-01 Hopper Public Runner Handoff Delta

**Date:** 2026-07-18  
**Claim:** `PAPER-CODE-REFERENCE-01`  
**Document role:** append-only task-local continuation delta  
**Research status impact:** none

This file must be read after, in order:

```text
docs/paper_code_reference/SESSION_HANDOFF.md
docs/paper_code_reference/SESSION_HANDOFF_ROLLOUT_DELTA_20260718.md
```

It supersedes only the stale implementation-status and next-slice statements in the earlier task-local documents. It does not replace `docs/handoff.md`, change any registered experiment status, authorize a formal rerun, or weaken the locked single-branch rule.

## 1. Repository snapshot

- repository: `easonhuo/drpo`;
- default branch observed for this slice: `main@e99489e7435bc26e2a7e30cd8d1a3aa10f4fc67a`;
- only active development branch: `dev/paper-code-reference-01`;
- public-runner exact head before this delta: `bc9749a9277968755be1174459befa6de94abfe4`;
- persistent cumulative Draft PR: `#149`;
- overall acceptance state: `in_development`;
- branch relation at the public-runner head: 66 commits ahead of and 59 commits behind current `main`, with merge base `4544005bd7df69c53bad70a9dcac846af01285e4`.

The development branch remains separate from `main`. PR `#149` remains Draft and must not be merged without a separate explicit user instruction.

## 2. Human-approved Python paths

After the exact paths, responsibilities, and nearest-file insufficiency were stated, the repository owner explicitly approved creation of:

```text
paper_code/src/drpo_reference/experiments/__init__.py
paper_code/src/drpo_reference/experiments/hopper.py
paper_code/tests/test_hopper_public_differential.py
```

and modification of the existing:

```text
paper_code/src/drpo_reference/cli.py
```

The durable approval record is PR `#149` conversation comment `5011310629`.

That approval does not authorize compact regeneration, paper table or figure-data generation, Countdown migration, a registered formal rerun, another branch, or merge to `main`.

## 3. Hopper public-runner slice now durably implemented

The development branch now contains a public Hopper execution layer that composes the already migrated modules rather than copying their implementations.

Implemented behavior:

1. strict dataset basename and SHA-256 verification before HDF5 loading;
2. explicit execution-plan separation among:
   - the complete registered ten-seed formal coordinate;
   - a registered-order formal seed subset, always marked non-evidence;
   - the smoke protocol, always marked non-evidence;
3. rejection of duplicate, unknown, empty, or out-of-order formal seed sets;
4. one canonical critic and frozen-advantage context shared by every actor seed;
5. canonical critic training exactly once, or strict artifact reuse only after identity and per-file hash verification;
6. persistence and verification of canonical split indices, observation normalizer, selected and terminal critic checkpoints, critic audit, and frozen advantages;
7. required process-isolated rollout preflight before critic or actor execution;
8. deterministic rollout evaluator injection into Positive-only preparation and every one of the six actor branches;
9. exact rollout seed derivation `seed * 100000 + step`;
10. registered intermediate versus terminal rollout episode counts;
11. isolated per-seed output roots and isolated branch failures;
12. continuously refreshed per-seed CSV and aggregate JSON records;
13. paired normalized-return records and paired deltas against `signed` without authorizing a ranking;
14. separate counts for:
    - task-performance availability and collapse;
    - support or variance boundary events;
    - NaN/Inf numerical events;
    - branch failures;
15. root terminal audit covering dataset identity, critic artifact identity, fixed budgets, branch clone identity, rollout availability, terminal-audit completeness, and seed completeness;
16. `RUN_COMPLETE.json` only when the engineering pipeline is complete;
17. `RUN_INCOMPLETE.json` plus `SCIENTIFIC_RUN_FAILED.json` for incomplete or failed execution;
18. a public command:

```text
drpo-reference hopper --dataset ... --output ...
```

with optional registered-order `--seeds`, `--device`, `--critic-artifact`, and `--smoke` controls.

Every aggregate, terminal audit, and completion record keeps:

```text
method_ranking_claim_allowed = false
formal_scientific_gate_passed = false
```

A full registered-coordinate engineering run may reach `raw_complete_pending_review`, but only a separate post-run scientific review can evaluate any scientific claim.

## 4. Differential and repository validation

Exact head `bc9749a9277968755be1174459befa6de94abfe4` passed:

- Evidence Locator Gate;
- Python compilation;
- shell syntax;
- handoff authority;
- formal execution channel;
- governance inventory;
- governance stage validation;
- full repository pytest;
- Ruff.

Focused public-runner tests cover:

- formal, formal-subset, and smoke execution semantics;
- duplicate, unknown, and out-of-order seed rejection;
- dataset basename and SHA-256 validation;
- strict canonical critic identity and hash-verified reuse;
- tampered-artifact rejection;
- aggregate terminal-state and event-separation compatibility with the authoritative legacy runner;
- explicit prohibition on method-ranking claims;
- root terminal-audit completion logic;
- Positive-only and six-branch rollout wiring;
- public CLI parsing and dispatch;
- incomplete-run and failure-record persistence.

Local pre-upload checks also executed Python compilation and controlled stub composition checks. Local Ruff was unavailable, so Ruff success is established only by exact-head GitHub CI.

These validations are engineering migration evidence only. They did not load the registered Hopper HDF5 artifact, did not exercise a real Gymnasium/MuJoCo runtime, did not run the registered fixed budgets, and did not change Hopper scientific status.

## 5. Hopper implementation status after this delta

Durably migrated:

1. protocol and HDF5 data contract;
2. episode handling, normalization, and splits;
3. canonical value critic, selected checkpoint, and frozen advantages;
4. squashed-Gaussian actor and six weighting modes;
5. fixed-budget actor training and terminal classification;
6. advantage-matched near/far mechanism diagnostics;
7. Positive-only preparation and exact six-branch per-seed suite;
8. Gymnasium/MuJoCo rollout adapter and process-isolated preflight;
9. public Hopper CLI and root runner;
10. canonical critic train-or-strict-reuse lifecycle;
11. multi-seed aggregation, paired records, root terminal audit, and completion/failure markers.

Still missing:

1. registered-input compact regeneration;
2. paper-facing Hopper table and figure-data binding;
3. clean-checkout real-environment liveness using compatible Gymnasium/MuJoCo dependencies;
4. any newly authorized registered-data fixed-budget reproduction;
5. post-run terminal scientific audit for any future registered execution;
6. implementation-status refresh of `SOURCE_MIGRATION_MAP.md` and `ACCEPTANCE_MATRIX.yaml` without falsely upgrading formal-result gates;
7. Countdown migration, still blocked by its unfrozen final manuscript-facing protocol and result.

## 6. Exact next implementation slice

The next slice is **Hopper compact regeneration and paper-facing output binding**.

It must consume a specifically selected completed result root and must not train a model or search historical directories. At minimum it must:

1. require and verify the root completion marker, terminal audit, dataset identity, protocol identity, seed identity, method identity, critic identity, prepared-checkpoint identities, and terminal checkpoint roles;
2. reject incomplete, mixed, tampered, smoke, or seed-subset artifacts for formal paper-output regeneration;
3. preserve task-performance, support/variance-boundary, NaN/Inf, rollout-unavailable, and branch-failure records as distinct outputs;
4. generate deterministic compact JSON and CSV artifacts from terminal records only;
5. generate the exact Hopper table inputs and figure-data files used by the manuscript;
6. bind every generated value to its source seed and artifact path;
7. never select a best run, best checkpoint, or favorable method after inspecting outcomes;
8. keep method-ranking authorization false unless a separately registered claim and post-run audit explicitly permit it.

This next slice must not launch a registered run, modify paper prose, migrate Countdown, or merge PR `#149`.

No exact new Python path for compact regeneration or paper-output binding is authorized by this delta. Before creating any new `.py` path, a continuation session must inspect current files, state each exact proposed path and responsibility, explain why the nearest existing file is insufficient, and obtain explicit human approval.

## 7. Remaining uncertainties

- Real Gymnasium/MuJoCo liveness has not been exercised in this CI environment.
- The registered HDF5 artifact has not been consumed by the paper-facing runner.
- No new full-budget Hopper execution is authorized; the historical Hopper scientific status remains unchanged.
- The development branch remains diverged from newer unrelated `main` work; integration freshness must be reviewed before further writes and before any eventual merge decision.
- The source migration map and acceptance matrix lag the implementation state.
- Countdown final manuscript-facing protocol and formal result remain unresolved and blocked from migration.
