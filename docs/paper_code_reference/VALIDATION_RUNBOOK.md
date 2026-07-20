# PAPER-CODE-VALIDATION-01 Correctness Validation Runbook

**Role:** executable engineering-validation plan for the reviewer-facing `paper_code` package.  
**Research authority:** `docs/handoff.md` remains the unique research master.  
**Migration authority:** `docs/paper_code_reference/CURRENT_STATUS.md` and `ACCEPTANCE_MATRIX.yaml`.  
**Scientific-status impact:** none.  
**Planning base head:** `8b81b4b72e38538a0b2ea4b50595059a67838d63`.  
**Main observed during planning:** `4b718e7439cf78a04f4affa1987ac15582d702d1`.

This runbook converts the existing three-layer correctness policy into an ordered,
checkable campaign. It does not register a new scientific experiment, alter a
frozen variable, or authorize a method-ranking claim.

## 1. Validation object and non-goals

The object under validation is the exact `paper_code/` tree at the resolved
validation head. Every validation record must bind:

- repository and full commit SHA;
- package-file inventory and SHA-256 values;
- command and working directory;
- Python, PyTorch, dependency, OS, CPU, GPU, CUDA, Gymnasium, MuJoCo,
  Transformers, and PEFT versions when applicable;
- input file identities and hashes;
- seed, device, and execution class;
- stdout, stderr, exit code, produced files, and pass/fail decision.

The campaign does **not**:

- refactor or shorten the implementation;
- silently repair mismatches while a gate is running;
- change data geometry, methods, coefficients, seeds, budgets, thresholds,
  convergence rules, checkpoint selection, or test access;
- promote a smoke, fake-backend, short trajectory, or liveness result to a
  scientific result;
- merge Draft PR #149 without a separate explicit user instruction.

## 2. Status vocabulary

Every gate uses exactly one status:

- `pending`: prerequisites are known but the gate has not run;
- `running`: a supervised execution is active;
- `passed`: the declared command and pass criteria completed successfully;
- `failed`: the gate ran and violated at least one frozen criterion;
- `blocked`: a required input, environment, identity, or protocol is unresolved;
- `deferred`: intentionally outside the deadline-critical path;
- `not_applicable`: the gate does not apply to that component.

`passed` may only be written after actual execution. Static inspection and prior
chat statements are not execution evidence.

## 3. Correctness layers

The implementation plan defines three acceptance layers, preserved here:

1. **Function equivalence:** fixed inputs reproduce formulas, masks, losses, raw
   gradients, clipped gradients, optimizer updates, and event labels.
2. **Trajectory equivalence:** fixed seeds reproduce the declared short
   trajectories, checkpoint decisions, and lightweight output schemas.
3. **Scientific reproduction:** registered data, seeds, budgets, terminal audits,
   summaries, and manuscript-facing inputs are regenerated.

A later layer cannot repair or waive a failed earlier layer.

## 4. Evidence directory contract

Validation artifacts are written outside the source tree under a new or empty
root:

```text
validation_outputs/PAPER-CODE-VALIDATION-01/
├── V0_baseline/
├── V1_clean_package/
├── V2_function_equivalence/
├── V3_short_trajectory/
├── V4_real_liveness/
└── V5_scientific_reproduction/
```

Each gate directory must contain, when applicable:

```text
SOURCE_COMMIT.txt
COMMAND.txt
ENVIRONMENT.json
INPUT_MANIFEST.json
STDOUT.log
STDERR.log
EXIT_CODE.txt
RESULT.json
OUTPUT_MANIFEST.json
```

A failure preserves the same records plus traceback and partial outputs. Output
roots must be new or empty.

## 5. Tolerance policy

- Existing differential tests keep their already-declared tolerances.
- No tolerance may be relaxed after observing a mismatch.
- New deterministic CPU comparisons must freeze exact fields and numerical
  tolerances in `VALIDATION_MATRIX.yaml` before execution.
- GPU/Transformer liveness does not claim score equality. Its pass criteria are
  explicit lifecycle completion and invariant checks.
- Any later GPU numerical-reproduction tolerance must be documented before the
  corresponding full run.

## 6. Ordered stages

### V0 — Baseline and documentation freeze

Purpose: make the validation target and decision rules immutable before running
comparisons.

Required gates:

1. resolve current `main`, validation branch, and exact head;
2. inventory the extracted `paper_code/` files and hashes;
3. commit this runbook and `VALIDATION_MATRIX.yaml`;
4. synchronize `CURRENT_STATUS.md` and `ACCEPTANCE_MATRIX.yaml`;
5. pass exact-head CI after the documentation commit.

No experiment or liveness command runs in V0.

### V1 — Clean package, installation, and entry-point validation

Run from a clean checkout or exact extracted package:

```bash
cd paper_code
python -m pip install -e '.[test]'
python -m compileall -q src tests
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m drpo_reference --help
python -m drpo_reference cu1 --help
python -m drpo_reference du1 --help
python -m drpo_reference hopper --help
python -m drpo_reference d4rl --help
python -m drpo_reference countdown --help
```

Pass criteria:

- installation succeeds without the historical repository on `PYTHONPATH`;
- compile, full pytest, Ruff check, and Ruff format check pass;
- all five public subcommands parse and display help;
- no generated output is written inside tracked source paths;
- no internal registry, handoff, governance, result archive, or cached model is
  required by core tests.

### V2 — Function and first-update equivalence

Run focused suites before any real-environment training.

#### Shared controls

```bash
cd paper_code
python -m pytest -q \
  tests/test_common.py \
  tests/test_controls.py \
  tests/test_events.py \
  tests/test_aggregate.py
```

Required invariants include taper formulas, detached weights, hard masks,
budget matching, event separation, verifier/masking statistics, and aggregation.

#### C-U1

```bash
cd paper_code
python -m pytest -q \
  tests/test_cu1_differential.py \
  tests/test_cu1_mechanism_differential.py \
  tests/test_cu1_phase_differential.py \
  tests/test_cu1_positive_training.py \
  tests/test_cu1_taper_differential.py
```

Required comparison fields: environment tensors, initialization, losses, raw
gradients, first Adam update, remoteness controls, and event labels.

#### D-U1 revision 4

```bash
cd paper_code
python -m pytest -q \
  tests/test_du1_environment_differential.py \
  tests/test_du1_controls_differential.py \
  tests/test_du1_training_differential.py \
  tests/test_du1_update_differential.py
```

Additional required fields: action/reward geometry, hidden rare-action cost,
initial common/rare gap, calibration threshold/scale, shared-rarity gradient
ratio, and support/probability-boundary classification.

#### Hopper E7-Q2

```bash
cd paper_code
python -m pytest -q \
  tests/test_hopper_data_differential.py \
  tests/test_hopper_models_differential.py \
  tests/test_hopper_critic_differential.py \
  tests/test_hopper_metrics_differential.py \
  tests/test_hopper_actor_differential.py \
  tests/test_hopper_suite_differential.py \
  tests/test_hopper_rollout_differential.py \
  tests/test_hopper_public_differential.py
```

Required fields: split identity, critic selection, frozen-advantage identity,
near/far matching, actor loss, raw gradient, first update, rollout semantics,
and aggregation schema.

#### D4RL-9

```bash
cd paper_code
python -m pytest -q tests/test_d4rl_shared_core_differential.py
```

This gate validates the selected backend code only. It does not close unresolved
dataset hashes, final methods, formal seeds, budgets, or checkpoint policy.

#### Countdown

```bash
cd paper_code
python -m pytest -q tests/test_common.py tests/test_cli.py
```

Required fields: expression verification, token masks, EOS and padding exclusion,
sequence log probability, unique-negative denominator, active-tail coordinate,
two-forward detachment, model-backed calibration, first clipped AdamW update,
canonical-config mutation rejection, structure metrics, and delayed test access.

### V3 — Fixed short-trajectory equivalence

Purpose: detect differences in RNG order, train/eval mode, scheduler timing,
checkpoint selection, and exception boundaries that first-update tests miss.

- C-U1: run the existing fixed-seed short-trajectory and suite tests.
- D-U1: run training/update/public-suite tests with fixed seeds.
- Hopper: run critic, actor, six-branch, rollout, and public-runner short paths.
- D4RL-9: run the deterministic minibatch trajectory in the shared-core
  differential suite.
- Countdown: run the controlled fake-HF end-to-end path covering calibration,
  at least two methods, multiple optimizer steps, checkpoints, generation,
  delayed test access, and aggregation.

V3 passes only if every component's declared short path passes on the exact
validation head. It remains engineering evidence, not a paper result.

### V4 — Real dependency and hardware liveness

Purpose: prove the reviewer package operates against the actual external stack.
These are supervised, non-scientific runs unless separately registered.

#### C-U1 CPU liveness

```bash
cd paper_code
python -m drpo_reference cu1 \
  --stage source \
  --output "$VALIDATION_ROOT/V4_real_liveness/cu1_source" \
  --device cpu \
  --smoke
```

Pass: command completes, outputs are finite, expected records exist, and event
categories remain separate.

#### D-U1 CPU liveness

```bash
cd paper_code
python -m drpo_reference du1 \
  --output "$VALIDATION_ROOT/V4_real_liveness/du1" \
  --device cpu \
  --workers 1 \
  --smoke
```

Pass: all six methods execute in the smoke matrix, expected records exist, and no
smoke result is marked formal.

#### Hopper HDF5/MuJoCo liveness

Prerequisites: `HOPPER_HDF5` points to the registered
`hopper_medium_replay-v2.hdf5`; its basename and SHA-256 must match the frozen
protocol; Gymnasium/MuJoCo must be installed.

```bash
cd paper_code
python -m pip install -e '.[test,rollout]'
python -m drpo_reference hopper \
  --dataset "$HOPPER_HDF5" \
  --output "$VALIDATION_ROOT/V4_real_liveness/hopper" \
  --device auto \
  --smoke
```

Pass: dataset identity, isolated environment preflight, one real environment
interaction, training lifecycle, checkpoint reload, and rollout evaluation pass.
Environment failure is rollout unavailability, not a task score.

#### Countdown Qwen/PEFT/CUDA liveness

Prerequisites: real Qwen2.5-0.5B-Instruct model, prepared reference adapter,
replay/calibration/validation inputs, and CUDA. A separate explicit schema-1
liveness config must be frozen before execution; it must be visibly
non-scientific and may not impersonate canonical v79.

```bash
cd paper_code
python -m pip install -e '.[test,countdown]'
python -m drpo_reference countdown \
  --config "$COUNTDOWN_LIVENESS_CONFIG" \
  --output "$VALIDATION_ROOT/V4_real_liveness/countdown"
```

Pass: tokenizer/model/adapter identity checks, one real forward/backward and
optimizer update, checkpoint save/reload, Greedy and sampled generation,
finite-state checks, delayed test behavior, and completion/failure records work.
No score or ranking criterion is attached to liveness.

#### D4RL-9 liveness

Status: `blocked` until the selected dataset identities required for the chosen
task are available. A one-task reviewer liveness may run after identity closure,
but it cannot freeze or substitute for the unresolved formal nine-task protocol.

### V5 — Registered scientific reproduction and terminal review

This stage runs only after V1--V4 pass for the corresponding component and the
formal coordinate is authoritative.

- C-U1: registered full CPU reproduction, terminal audit, selected conclusion
  report.
- D-U1 revision 4: seeds 200--219, six methods, 8000 updates, registered terminal
  windows, terminal audit, selected conclusion report.
- Hopper E7-Q2: registered HDF5, canonical critic/frozen advantages, registered
  seeds and budgets, real rollouts, root terminal audit.
- Countdown: canonical v79 only after real liveness; best and terminal/last-finite
  remain separate; test is accessed only after all method training; a scientific
  terminal review is required before any ranking.
- D4RL-9: remains blocked until final methods, coefficients, seeds, budget,
  checkpoint policy, all dataset identities, and manuscript role are frozen.

For dynamics, fixed-horizon completion is not convergence. Task-performance
collapse, support/variance/probability boundary events, NaN/Inf numerical failure,
environment invalidity, and unresolved terminal state are reported separately.

## 7. Stop and repair rules

- A failed V1 gate blocks all later stages.
- A failed component gate in V2 blocks V3--V5 for that component.
- A failed V3 gate blocks real liveness for that component.
- A real-environment dependency failure is preserved as evidence and repaired
  without altering scientific coordinates.
- Any code repair creates a new exact head; all affected earlier gates rerun.
- Unaffected component evidence may be retained only when the diff proves the
  component and its shared dependencies did not change.
- No failed gate may be reclassified as passed by explanation alone.

## 8. Immediate execution order

1. Finish V0 documentation synchronization and exact-head CI.
2. Freeze and archive the exact `paper_code` v0.1 baseline.
3. Execute V1 clean-package checks.
4. Execute V2 in this order: shared, C-U1, D-U1, Hopper, D4RL-9 code, Countdown.
5. Execute V3 fixed short trajectories.
6. Execute V4 C-U1 and D-U1 CPU liveness.
7. Execute V4 Hopper real HDF5/MuJoCo liveness when the registered dataset is
   available.
8. Freeze the non-scientific Countdown liveness config, then execute real
   Qwen/PEFT/CUDA liveness.
9. Enter V5 only for experiments actually required by the final manuscript.

The live gate statuses are maintained in
`docs/paper_code_reference/VALIDATION_MATRIX.yaml`.