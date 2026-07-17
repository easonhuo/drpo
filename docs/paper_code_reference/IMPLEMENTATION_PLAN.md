# PAPER-CODE-REFERENCE-01 Implementation Plan

**Base commit:** `4544005bd7df69c53bad70a9dcac846af01285e4`  
**Development branch:** `dev/paper-code-reference-01`  
**Task class:** paper-facing reference implementation; no new scientific experiment  
**Scientific status impact:** none

## 1. Objective

Build a compact, readable, paper-facing reference implementation for the experiments that actually support the manuscript. The work does not refactor the full internal repository and does not export or anonymize a second repository.

The existing implementations and delivered result artifacts remain the scientific and provenance reference. The new code is accepted only after it reproduces the registered behavior and paper-facing outputs under the gates below.

## 2. Scope

### Included

1. Shared deterministic utilities used by paper experiments:
   - seeding;
   - JSON/CSV output;
   - event taxonomy;
   - numerical-finiteness checks;
   - terminal-audit records.
2. One authoritative implementation of paper-facing negative controls:
   - Positive-only;
   - uncontrolled signed negative;
   - global scaling;
   - hard near/far selection;
   - far cap;
   - reciprocal-linear taper;
   - reciprocal-quadratic taper;
   - exponential taper;
   - raw-negative-gradient budget matching.
3. Controlled continuous code required by C-U1.
4. Controlled categorical code required by D-U1 protocol revision 4.
5. Hopper E7-Q2 external mechanism validation code.
6. Countdown code only after the manuscript-facing protocol and result are frozen. Before that point, only stable shared interfaces may be added.
7. Paper-result aggregation and figure-data regeneration for the experiments above.

### Excluded

- repository governance and handoff machinery;
- historical experiment revisions and superseded pilots;
- E7-BENCH method benchmark code unless the final manuscript explicitly uses it;
- unmerged GAE or other development branches;
- generic experiment orchestration frameworks;
- artifact anonymization/export;
- model weights, datasets, checkpoints, and large internal result archives;
- destructive deletion or rewriting of existing scientific code.

## 3. Current manuscript boundary

The current manuscript names two controlled environments and two external families: C-U1, D-U1, Hopper/D4RL, and Countdown. The controlled and external result tables are still intentionally unfilled until the corresponding terminal-audited artifacts are frozen. Therefore this task must not invent or pre-fill scientific values.

Current authoritative scientific boundaries remain:

- C-U1: same-distribution held-out-context / unseen-state generalization, never OOD under the current protocol;
- D-U1: controlled categorical mechanism, protocol revision 4 for the current formal path;
- Hopper: external learned-critic mechanism validation, not a replacement for C-U1;
- Countdown: external Transformer validation, not a replacement for D-U1;
- task-performance collapse, support/variance or probability-boundary events, and NaN/Inf numerical failure are reported separately.

## 4. Target directory

The reference implementation is isolated from the historical implementation:

```text
paper_code/
├── README.md
├── pyproject.toml
├── src/
│   └── drpo_reference/
│       ├── common/
│       │   ├── io.py
│       │   ├── seeding.py
│       │   ├── events.py
│       │   └── audit.py
│       ├── controls/
│       │   ├── weights.py
│       │   ├── selection.py
│       │   └── budget.py
│       ├── continuous/
│       │   ├── gaussian.py
│       │   ├── cu1.py
│       │   └── hopper.py
│       ├── categorical/
│       │   ├── surprisal.py
│       │   ├── du1.py
│       │   └── countdown.py
│       └── experiments/
│           ├── cu1.py
│           ├── du1.py
│           ├── hopper.py
│           └── countdown.py
└── tests/
```

This is a shallow-composition design. It deliberately avoids a universal trainer, deep inheritance, plugin registries, manager/factory layers, or a repository-wide rewrite.

## 5. Sharing boundary

### Must be shared

- taper formulas and coefficient calibration;
- remoteness-coordinate validation;
- hard near/far masks;
- global and capped negative controls;
- raw-gradient norm and budget matching;
- deterministic seeding;
- task/boundary/numerical event taxonomy;
- result serialization and terminal-audit schema.

### Must not be forced into one implementation

- C-U1 small-policy optimization and Hopper actor-critic training;
- D-U1 categorical-policy training and Countdown Transformer/LoRA training;
- continuous Gaussian support diagnostics and categorical probability-support diagnostics;
- controlled-environment exact evaluation and external-environment rollout or verifier evaluation.

The shared layer supplies pure functions and small data contracts. Task-specific trainers compose those functions.

## 6. Phases

### Phase 0 — Scope and architecture freeze

Deliverables:

- this implementation plan;
- a source-to-target migration inventory;
- an acceptance matrix mapping each manuscript claim to source implementation, reference entry point, expected result artifact, and validation command.

Exit gate:

- every included paper experiment has one authoritative legacy source path;
- superseded and development-only paths are explicitly excluded;
- no scientific variable is changed.

### Phase 1 — Shared kernel and characterization tests

Deliverables:

- `paper_code` package skeleton;
- common event/audit records;
- canonical taper, selection, and budget functions;
- tests that compare the reference functions with the authoritative legacy formulas on fixed tensors.

Exit gate:

- formula outputs, masks, detached-gradient behavior, coefficients, and budget scales match the authoritative source within declared numerical tolerance;
- no trainer is migrated yet.

### Phase 2 — C-U1 migration

Deliverables:

- C-U1 environment, Gaussian policy, objectives, evaluation, and experiment entry points required by the paper;
- no copied historical packaging/governance code;
- differential tests against `src/drpo/cu1_core.py` and `src/drpo/drpo_cu1_e1_e4_oneclick.py`.

Correctness gates:

1. environment tensors match for fixed seeds;
2. initial model parameters match;
3. losses and raw gradients match on fixed batches;
4. first Adam update matches;
5. fixed short trajectories match;
6. registered full CPU result is regenerated before C-U1 is marked ready.

### Phase 3 — D-U1 revision-4 migration

Deliverables:

- protocol-revision-4 environment and policy;
- normalized excess surprisal and common/rare dynamic assignment;
- six-method formal path only;
- no historical protocol revisions in the paper package.

Correctness gates mirror C-U1 and additionally cover:

- exact action/reward geometry;
- hidden rare-action task cost;
- initial common/rare gap;
- calibration threshold and scale;
- shared-rarity gradient ratio;
- support/probability-boundary classification.

The full registered CPU matrix must be regenerated before D-U1 is marked ready.

### Phase 4 — Hopper E7-Q2 migration

Deliverables:

- dataset contract;
- critic and frozen-advantage preparation;
- Gaussian actor and negative controls;
- rollout evaluation and terminal audit;
- only the delivered E7-Q2 external mechanism path.

Correctness gates:

1. deterministic dataset split identity;
2. critic checkpoint selection identity;
3. advantage sign/rank and near/far-pair identity;
4. actor loss, raw gradient, and first update differential tests;
5. compact result regeneration from the registered input artifact;
6. full formal rerun on the registered data and budget before Hopper is marked ready for public release.

A smoke or short pilot cannot satisfy gate 6.

### Phase 5 — Countdown migration after protocol freeze

Deliverables:

- verifier/data/model/training/evaluation modules for the final manuscript-facing protocol;
- stable completion-surprisal and taper functions reused from the shared layer;
- no copy of the historical giant one-file execution stack in the paper package.

Correctness gates:

- token masks, EOS handling, prompt/padding exclusion, sequence log-probability, taper weights, verifier labels, optimizer update, selected and terminal evaluation all match the frozen source;
- real Qwen/CUDA execution at the registered budget is required before ready status.

Until the final Countdown result/protocol is frozen, this phase remains blocked except for task-independent shared components.

### Phase 6 — Paper-output closure

Deliverables:

- one command per paper experiment;
- one command to regenerate controlled-result inputs;
- one command to regenerate external-result inputs;
- figure-data generation from reference outputs;
- clean-checkout installation and reproduction guide.

Exit gate:

- every non-TBD manuscript number has an exact code command and result path;
- generated summaries match the frozen manuscript inputs;
- tests and reproduction commands pass from a clean checkout;
- no hidden internal path or cached file is required.

## 7. Correctness policy

The reference code is not accepted because it imports, compiles, passes a smoke test, or produces plausible curves. It is accepted only after three layers pass:

1. **Function equivalence:** fixed inputs produce matching formulas, masks, losses, gradients, updates, and event labels.
2. **Trajectory equivalence:** fixed seeds produce matching short trajectories and checkpoints.
3. **Scientific reproduction:** the registered seeds, budgets, terminal audits, summaries, and paper-facing outputs are regenerated.

For deterministic CPU controlled environments, numerical comparisons are strict and tolerances are frozen before observing mismatches. For GPU/Transformer paths, nondeterministic tolerances must be declared before the full run and may not be relaxed after seeing the result.

Any failed gate leaves the corresponding phase incomplete. No phase is called reproduced based only on static inspection, unit tests, or limited-step pilot execution.

## 8. Code-quality policy

- one authoritative implementation for each mathematical control;
- experiment entry points are thin composition layers;
- no new `onefile` implementation;
- no scientific formula duplicated across task modules;
- no inheritance deeper than one project-defined layer;
- no generic framework added without a paper-code need;
- existing historical code remains intact as provenance and differential oracle;
- file-size and duplication audits are added after the Phase-0 inventory establishes justified thresholds.

## 9. Rollback

All work is isolated under `paper_code/`, its tests, and this task's documentation. Rollback is a branch/PR revert. Existing experiments, outputs, handoff, registry, manuscript values, and execution routes remain unchanged.

## 10. Current uncertainties

1. Countdown's final manuscript-facing protocol and formal result are not yet frozen.
2. The unmerged Hopper GAE/benchmark work is not part of this task unless it later becomes part of the final manuscript.
3. Full Hopper and Countdown reproduction requires the registered data/model/compute environment; absence of those resources blocks ready status rather than being replaced by smoke evidence.
