# PAPER-CODE-REFERENCE-01 Session Handoff

**Document role:** task-local engineering continuation record.  
**Not a research master:** `docs/handoff.md` remains the unique research source of truth.  
**Claim:** `PAPER-CODE-REFERENCE-01`  
**Task class:** paper-facing reference implementation; no new scientific experiment.  
**Scientific-status impact:** none.

## 0. Mandatory continuation protocol

A new session must not continue from chat memory alone. Read and verify, in this order:

1. `AGENTS.md` from current `main`.
2. Section 0 of `docs/handoff.md` from current `main`.
3. `experiments/registry.yaml` from current `main`.
4. `docs/paper_code_reference/IMPLEMENTATION_PLAN.md`.
5. `docs/paper_code_reference/SIZE_AND_COMPLETENESS_POLICY.md`.
6. `docs/paper_code_reference/SOURCE_MIGRATION_MAP.md`.
7. `docs/paper_code_reference/ACCEPTANCE_MATRIX.yaml`.
8. This file.
9. The actual `dev/paper-code-reference-01` branch, relevant legacy source files, changed files, tests, persistent Draft PR `#149`, and exact-head CI.

Before writing code, report:

- repository and default branch;
- current `main` commit SHA;
- active development branch and exact head SHA;
- active claim;
- scientific statuses affected or explicitly unaffected;
- unresolved uncertainties;
- exact next implementation slice.

Do not edit `docs/handoff.md` directly. This engineering-only task must remain a handoff/registry no-op unless a separately authorized scientific-status change is required.

### 0.1 Locked single-branch development rule

The user explicitly replaced the former stacked-branch workflow on 2026-07-18. Every later session working on `PAPER-CODE-REFERENCE-01` must follow this rule:

1. The only active long-lived development branch is:

   ```text
   dev/paper-code-reference-01
   ```

2. Continue implementation and task-document updates directly on that branch. Do **not** create one new staging branch or stacked Draft PR per implementation slice.
3. Keep one persistent Draft PR only:

   ```text
   #149: dev/paper-code-reference-01 -> main
   ```

   It is the cumulative diff, discussion, and clean-checkout CI window. It must remain Draft and must not be merged into `main` without a separate explicit user instruction.
4. Before every write, resolve the current exact head of `dev/paper-code-reference-01`; never continue from a stale session SHA.
5. Every commit pushed to the development branch must update PR `#149` and pass the applicable exact-head CI before the slice is called complete.
6. The former stacked PRs `#125`, `#126`, `#127`, `#128`, `#129`, `#131`, `#132`, `#133`, `#134`, `#135`, `#136`, `#140`, and `#144` are closed historical review records. They are not active continuation branches and must not be reopened or used as the base of new work.
7. Old staging branch refs may remain visible because the available GitHub connector did not expose branch-ref deletion. Their existence does not authorize their reuse. The consolidated branch already contains their commits.
8. A temporary branch is allowed only when a concrete repository/tool limitation makes direct work on the long-lived branch impossible, and only after explicit user approval. It must be reconciled immediately after exact-head CI rather than becoming another permanent stack layer.
9. This rule does not weaken the new-Python-file human-approval gate, scientific scope gates, terminal audits, differential tests, or the prohibition on direct `main` changes.

## 1. Stable objective and scope

Build a readable, complete, paper-facing reference implementation for the manuscript evidence paths only:

- shared deterministic utilities and scientific controls;
- C-U1 controlled continuous path;
- D-U1 protocol revision 4 controlled categorical path;
- Hopper E7-Q2 external mechanism-validation path;
- Countdown only after its final manuscript-facing protocol and result are frozen;
- aggregation, terminal audit, and paper-facing result/figure-data regeneration.

Do not refactor the entire repository, create a universal framework, import superseded pilots, import unmerged Hopper GAE/benchmark development, anonymize/export another repository, delete historical code, or change scientific variables.

Correctness and completeness determine implementation size. No line-count target or maximum acceptance threshold exists.

## 2. Current repository snapshot

- Repository: `easonhuo/drpo`
- Default branch: `main`
- Current `main` observed before the Hopper suite slice: `e99489e7435bc26e2a7e30cd8d1a3aa10f4fc67a`
- Overall task base: `4544005bd7df69c53bad70a9dcac846af01285e4`
- Only active development branch: `dev/paper-code-reference-01`
- Consolidated branch head before this document update: `563138b186baef56212430e4911c0c4aa33177c7`
- Persistent cumulative Draft PR: `#149`
- Overall acceptance state: `in_development`

At consolidation, `dev/paper-code-reference-01` was fast-forwarded from `b009fbb91f214aa5052c9b8bb9d74704bd2ba304` to `32c6d6ef979e28809ffade9e260fd333f0057fb8`. GitHub compare reported the long-lived branch and the latest Hopper-mechanism head as identical. No paper-code commit was merged into `main`.

The development branch and current `main` have subsequently diverged because unrelated repository work continued on `main`. A continuation session must resolve both current heads and review integration freshness before writing; it must not reuse either SHA above as an assumed current value.

## 3. Scientific boundaries that must not drift

- Product-manifold experiments answer only where far-field large gradients originate.
- Nonlinear Gaussian causal experiments answer whether abnormal far-field negative gradients transmit into drift and collapse.
- C-U1 and D-U1 are controlled mechanism environments.
- Hopper and Countdown are external-validity paths and do not replace controlled identification.
- C-U1 uses same-distribution held-out-context / unseen-state generalization terminology, not OOD generalization.
- Task-performance collapse, support/variance or probability-boundary events, and NaN/Inf numerical failure must remain separate.
- No implementation or smoke test changes a registered scientific result status.
- No method winner may be assumed.

## 4. Durable implementation status

### 4.1 Planning and shared kernel

Durably present on the consolidated branch:

- implementation objective, scope, phase plan, and correctness gates;
- size/completeness policy;
- source migration map;
- acceptance matrix;
- isolated `paper_code/` package;
- common deterministic I/O, seeding, events, and terminal-audit support;
- shared negative-control formulas, masks, taper coefficients, and budget matching;
- fixed-input characterization and differential tests.

The acceptance matrix may lag the implementation commits and must be refreshed before reviewer-ready closure. Do not infer acceptance from compilation or smoke tests alone.

### 4.2 C-U1

Durably implemented on the consolidated branch:

- Gaussian/environment core;
- Positive-only preparation;
- source and causal paths;
- phase scan and controls;
- taper family;
- public CLI, per-seed artifacts, aggregation, and terminal audit.

Verified during development:

- Python compilation;
- strict legacy differential tests;
- integrated C-U1 smoke;
- full repository pytest;
- Ruff;
- repository governance checks.

Not completed or not authorized by these engineering checks:

- a new full formal-budget C-U1 rerun;
- a new paper-table/figure numerical freeze;
- any scientific-status upgrade.

### 4.3 D-U1 revision 4

Durably implemented on the consolidated branch:

- revision-4 environment and policy core;
- utility × rarity Cartesian geometry;
- normalized excess surprisal and dynamic common/rare roles;
- six frozen methods only;
- shared start, deterministic minibatches, training, two-window terminal audit;
- paired Positive-only task-collapse assignment;
- public CLI, per-seed artifacts, aggregation, mechanism report, and terminal audit.

Exact-head differential CI and the complete six-method smoke passed during development.

Scientific status remains `not_run`. No 20-seed × 6-method × 8000-step formal matrix was executed by this task, and no method ranking is authorized.

### 4.4 Hopper E7-Q2

Durably implemented on the consolidated branch:

- frozen protocol and registered identity;
- HDF5 data contract, episode handling, returns, splits, and observation normalization;
- value and squashed-Gaussian actor architectures;
- Gaussian output-score components and separate terminal-event classification;
- fixed-budget canonical critic training and best-validation checkpoint selection;
- critic audit, selected-vs-final advantage stability, and frozen advantages;
- one shared actor with six weighting modes: `positive_only`, `signed`, `near_zero`, `far_zero`, `far_cap`, and `dynamic_budget_matched_global`;
- fixed-budget AdamW actor training, terminal checkpoint/audit, and non-finite no-step behavior;
- advantage-matched near/far pairing;
- per-sample and aggregate full-parameter gradient diagnostics;
- score decomposition, analytic/autograd agreement, far-field slopes, and budget-matching artifacts;
- one Positive-only preparation per actor seed;
- exact prepared-checkpoint persistence and reload identity;
- registered far threshold, Far-cap reference score, and initial global-budget diagnostics derived from the prepared actor;
- six branches cloned from one identical reloaded prepared state and run in the frozen method order;
- branch-local trajectories, checkpoints, terminal audits, failure records, and new-or-empty output enforcement;
- strict legacy differential tests for preparation, matching, probes, calibration, all six branch endpoints, seed derivation, clone independence, and failure isolation.

The Hopper suite was added in:

- `paper_code/src/drpo_reference/external/hopper_suite.py`;
- `paper_code/tests/test_hopper_suite_differential.py`.

The user explicitly approved both exact Python paths and responsibilities. The approval is preserved in PR `#149` discussion. Exact-head `563138b186baef56212430e4911c0c4aa33177c7` passed Evidence Locator, Python compile, full repository pytest, Ruff, handoff authority, formal execution channel, and governance checks.

Still missing:

- real Gymnasium/MuJoCo rollout adapter and process-isolated preflight;
- public Hopper runner and aggregation;
- registered-input compact regeneration and paper-facing output binding;
- full registered-data and fixed-budget rerun.

The suite smoke/differential execution is engineering evidence only. It did not use the registered D4RL dataset, did not perform real rollouts, and did not change Hopper scientific status.

### 4.5 Countdown

Blocked by the implementation plan until the final manuscript-facing Countdown protocol and result are frozen. Only task-independent shared interfaces may be prepared before that point.

Do not migrate a historical giant one-file runner and do not treat a pilot or coefficient scan as the final paper protocol.

## 5. Branch and PR governance after consolidation

The old incremental stack is retained only as historical review provenance:

- `#125` C-U1 differential core;
- `#126` C-U1 Positive-only path;
- `#127` C-U1 source and causal mechanisms;
- `#128` C-U1 phase and controls;
- `#129` C-U1 taper family;
- `#131` C-U1 public runner;
- `#132` D-U1 revision-4 core;
- `#133` D-U1 training and terminal audit;
- `#134` D-U1 public runner;
- `#135` Hopper E7-Q2 core;
- `#136` Hopper critic and frozen advantages;
- `#140` Hopper actor training;
- `#144` Hopper mechanism diagnostics.

These PRs are closed. Their commits were inherited by the consolidated branch; closing them did not delete code. GitHub may mark the first PR as merged because its original base was the long-lived branch when that branch fast-forwarded over its commits. This does not mean any paper-code change entered `main`.

PR `#149` is the only active paper-code Draft PR. Future work updates this PR by committing directly to `dev/paper-code-reference-01`. Do not create another stacked PR chain.

## 6. Exact next implementation slice

The next slice is **Hopper real rollout evaluation and process-isolated environment preflight**, based only on:

- `src/drpo/e7_hopper_q2.py` at the overall task base;
- `configs/e7_hopper_q2_medium_replay_v2.yaml`;
- the migrated Hopper protocol, data, normalization, model, actor, and suite layers in `paper_code/`.

Required behavior:

1. use the frozen Gymnasium MuJoCo backend and `Hopper-v4` identity;
2. verify the registered D4RL dataset/environment identity before evaluation;
3. perform process-isolated preflight with the registered timeout and maximum-step contract;
4. evaluate the deterministic actor using the training observation normalizer;
5. preserve exact action shape, clipping, reset/step semantics, episode seeding, and termination/truncation handling;
6. compute raw return and the frozen normalized-score percentage using the registered reference minimum and maximum;
7. write explicit environment and rollout diagnostics;
8. fail closed when a required rollout environment is unavailable;
9. never substitute a synthetic environment or the forbidden legacy D4RL fallback;
10. keep rollout unavailability, task-performance collapse, support/variance boundary, and NaN/Inf numerical failure distinct.

This next slice must not yet add the public Hopper CLI, multi-seed aggregation, compact regeneration, or launch a formal run.

No exact new Python path for the rollout slice has been approved by this handoff. Before creating any new `.py` path, the next session must inspect the current branch, name the exact proposed path and responsibility, explain why existing Hopper files are insufficient, and obtain explicit human approval under the repository hard gate.

Minimum gates before calling the rollout slice complete:

- exact legacy/reference preflight behavior;
- reset, step, termination, truncation, action, and seed identity on controlled fake environments;
- observation-normalization identity;
- raw and normalized return identity;
- required-versus-optional failure behavior;
- process timeout/failure diagnostics;
- no synthetic or legacy fallback;
- Python compilation, focused tests, full repository pytest, Ruff, handoff/formal/governance gates, and exact-head PR `#149` CI.

## 7. New-file governance

Before creating any new `.py` path, follow the current `AGENTS.md` human-approval gate:

- name the exact proposed path;
- explain why extending the nearest existing Python file is insufficient;
- obtain explicit human approval;
- preserve the approval in PR `#149` discussion or another durable repository record.

A continuation session must inspect which Python paths already exist at the current branch head before proposing new paths. This handoff does not itself authorize a new Python filename.

## 8. Validation and claim discipline

For every new commit:

1. resolve and inspect the exact current head of `dev/paper-code-reference-01`;
2. inspect the changed files and legacy authority;
3. run focused old/new differential tests;
4. run clean-checkout CI through persistent Draft PR `#149` on the exact new head;
5. do not count a cancelled or superseded workflow run as passing;
6. record local tests separately from GitHub tests;
7. do not call a smoke or short trajectory a scientific reproduction;
8. do not claim formal Hopper readiness before registered data, budget, rollouts, terminal audit, and compact/full regeneration gates pass.

## 9. Remaining uncertainties

- Countdown final manuscript-facing protocol and formal result are not frozen.
- Full Hopper reproduction needs the registered D4RL data and Gymnasium/MuJoCo runtime resources.
- Hopper rollout, public runner, aggregation, compact regeneration, and paper-output binding remain incomplete.
- The acceptance matrix needs a later implementation-status refresh.
- The development branch has diverged from newer unrelated `main` work; integration freshness must be reviewed before future writes and before any eventual merge decision.
- Old staging branch refs may still exist remotely, but they are historical only and must not be used for new work.
- The consolidated branch must remain separate from `main` until the user explicitly authorizes a final merge decision.

## 10. Completion definition

This task is not complete when modules compile or smoke tests pass. Completion requires:

- all included paper experiments have one readable public entry point;
- every non-TBD paper number has an exact command and result path;
- legacy/reference function, gradient, update, trajectory, and terminal-audit gates pass;
- registered scientific runs are regenerated at the required budgets where resources permit;
- clean-checkout reproduction commands work without hidden local paths;
- code and docs are reviewed as one coherent paper-facing implementation;
- reviewer-facing acceptance is audited as `reviewer_ready` rather than inferred;
- merge to `main`, if ever desired, occurs only after a separate explicit user approval.
