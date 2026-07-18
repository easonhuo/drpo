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
6. This file.
7. The actual stacked branch, relevant legacy source files, changed files, tests, Draft PRs, and exact-head CI.

Before writing code, report:

- repository and default branch;
- current `main` commit SHA;
- continuation branch and exact head SHA;
- active claim;
- scientific statuses affected or explicitly unaffected;
- unresolved uncertainties;
- exact next implementation slice.

Do not edit `docs/handoff.md` directly. This engineering-only task should remain a handoff/registry no-op unless a separately authorized scientific-status change is required.

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

## 2. Repository snapshot at this handoff

- Repository: `easonhuo/drpo`
- Default branch: `main`
- Confirmed `main` SHA at handoff creation: `4544005bd7df69c53bad70a9dcac846af01285e4`
- Overall task base: `4544005bd7df69c53bad70a9dcac846af01285e4`
- Root development branch: `dev/paper-code-reference-01`
- Continuation branch before this document commit: `dev/paper-code-reference-01-hopper-critic-staging`
- Continuation head before this document commit: `a617cf99aa05c425d18ed7043e8605aff7ed7c4f`
- Latest stacked Draft PR before this document commit: PR `#136`

The session that continues this work must resolve the new exact branch head after this document commit and must not reuse the pre-document SHA as the current head.

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

Completed and present in the stack:

- implementation objective, scope, phase plan, and correctness gates;
- size/completeness correction;
- isolated `paper_code/` package;
- common deterministic I/O, seeding, events, and terminal-audit support;
- shared negative-control formulas, masks, taper coefficients, and budget matching;
- fixed-input characterization/differential tests.

Still missing as consolidated task documents:

- a complete source-to-target migration inventory;
- a single claim-to-entrypoint-to-artifact-to-command acceptance matrix.

Do not infer that these Phase-0 deliverables exist merely because parts of the information appear in PR bodies or tests.

### 4.2 C-U1

Durably implemented through stacked Draft PR `#131`:

- Gaussian/environment core;
- Positive-only preparation;
- source and causal paths;
- phase scan and controls;
- taper family;
- public CLI, per-seed artifacts, aggregation, and terminal audit.

Verified at exact heads during development:

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

Durably implemented through stacked Draft PRs `#132`, `#133`, and `#134`:

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

Durably implemented:

- PR `#135`: frozen protocol, HDF5 data contract, episode handling, returns/splits/normalization, value and squashed-Gaussian networks, score components, scalar metrics, and separate terminal event classification;
- PR `#136`: fixed-budget critic training, best-validation checkpoint selection, critic terminal audit, selected-vs-final advantage stability, one-time frozen-advantage standardization, checkpoint/artifact output, and strict legacy differential tests.

The corrected Hopper core and critic exact heads passed GitHub clean-checkout CI, including full repository pytest and Ruff.

Not yet durably implemented:

- Positive-only actor preparation;
- the six actor intervention objectives and fixed-budget actor training layer;
- advantage-matched near/far mechanism diagnostics;
- rollout adapter;
- public Hopper runner and aggregation;
- registered-input compact regeneration;
- full formal rerun.

Important interruption boundary:

- The previous session prepared actor-layer code in a transient workspace and created at least one unattached Git object, but no actor-layer commit, branch, or PR was completed.
- Unattached blobs and transient files are not authoritative continuation material.
- A new session must reconstruct or recover the actor slice from repository-visible sources and verify it before committing. It must not claim the actor layer already exists in GitHub.

### 4.5 Countdown

Blocked by the implementation plan until the final manuscript-facing Countdown protocol and result are frozen. Only task-independent shared interfaces may be prepared before that point.

Do not migrate a historical giant one-file runner and do not treat a pilot or coefficient scan as the final paper protocol.

## 5. Stacked Draft PR chain

The current paper-code stack is intentionally incremental. At handoff creation, the durable chain is:

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
- `#136` Hopper critic and frozen advantages.

All are Draft and unmerged. `main` is unchanged. Do not merge, retarget, squash, or close the stack without explicit user approval and a separate stack-integration review.

PR bodies contain useful slice-local scope and test records, but this file is the task-level continuation index.

## 6. Exact next implementation slice

The next durable slice is **Hopper actor objectives and fixed-budget actor training**, based only on the authoritative legacy path:

- `src/drpo/e7_hopper_q2.py`
- `configs/e7_hopper_q2_medium_replay_v2.yaml`
- the already migrated Hopper protocol, data, models, metrics, optimizer utilities, critic, and frozen advantages in `paper_code/`.

Required behavior:

1. exact `actor_eval_metrics` behavior and compatibility fields;
2. exact objectives and diagnostics for:
   - `positive_only`;
   - `signed`;
   - `near_zero`;
   - `far_zero`;
   - `far_cap`;
   - `dynamic_budget_matched_global`;
3. detached distance/output-score control semantics;
4. fixed optimizer-step budget; terminal audit must not shorten the registered horizon;
5. selected rollout-evaluation schedule without inventing an environment fallback;
6. separate task-performance, support/variance-boundary, and numerical-failure reporting;
7. terminal actor checkpoint and curves/audit artifacts.

Minimum differential gates before publication:

- loss and every diagnostic field for all six methods on fixed tensors;
- raw gradients for all six methods;
- first AdamW update identity;
- fixed-seed short trajectories and terminal audit identity for representative methods, including Positive-only and dynamic budget matching;
- non-finite loss/gradient path does not apply an optimizer update;
- smoke profile remains explicitly non-formal;
- Python compilation, focused tests, full repository pytest, Ruff, handoff/formal/governance gates, and exact-head CI.

Do not combine this slice with near/far pair construction, rollout integration, aggregation, or the public runner unless the actor slice first becomes a coherent, reviewable, exact-head-green commit.

## 7. New-file governance

Before creating any new `.py` path, follow the current `AGENTS.md` human-approval gate:

- name the exact path;
- explain why the nearest existing Python file cannot own the responsibility;
- obtain explicit human approval;
- preserve the approval in durable PR discussion or repository documentation.

A continuation session must inspect which Python paths already exist at the current stack head before proposing new paths. This handoff does not itself authorize a new Python filename.

## 8. Validation and claim discipline

For every new commit:

1. inspect the exact base and changed files;
2. run focused old/new differential tests;
3. run clean-checkout CI on the exact new head;
4. do not count a cancelled or superseded workflow run as passing;
5. record local tests separately from GitHub tests;
6. do not call a smoke or short trajectory a scientific reproduction;
7. do not claim formal Hopper readiness before registered data, budget, rollouts, terminal audit, and compact/full regeneration gates pass.

## 9. Remaining uncertainties

- Countdown final manuscript-facing protocol and formal result are not frozen.
- Full Hopper reproduction needs the registered D4RL data and environment/runtime resources.
- The paper-code source-to-target inventory and unified acceptance matrix still need a durable consolidated form.
- The large stacked PR chain requires an eventual integration/review strategy before anything can merge to `main`.
- The interrupted Hopper actor work is not committed and must be rebuilt or recovered from authoritative repository sources.

## 10. Completion definition

This task is not complete when modules compile or smoke tests pass. Completion requires:

- all included paper experiments have one readable public entry point;
- every non-TBD paper number has an exact command and result path;
- legacy/reference function, gradient, update, trajectory, and terminal-audit gates pass;
- registered scientific runs are regenerated at the required budgets where resources permit;
- clean-checkout reproduction commands work without hidden local paths;
- code and docs are reviewed as one coherent paper-facing implementation;
- merge occurs only after explicit user approval.
