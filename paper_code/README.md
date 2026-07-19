# DRPO paper reference code

This directory contains the paper-facing implementation developed under
`PAPER-CODE-REFERENCE-01`. It is intentionally separate from the repository's
historical experiment drivers, registries, governance tooling, and packaging
code.

The current task-level implementation snapshot, live file ownership, and exact
D4RL remaining-work inventory are maintained in:

```text
../docs/paper_code_reference/CURRENT_STATUS.md
```

`docs/handoff.md` remains the unique research source of truth. The current-status
document is an engineering continuation index only.

C-U1 and D-U1 both use independent train and held-out contexts drawn from the
same distribution. Their result is **same-distribution held-out-context
generalization**, not OOD generalization.

## Reviewer-facing scope

The public package gives reviewers readable, runnable algorithms with explicit
data identities, training commands, checkpoints, rollout evaluation, and simple
result summaries. It is not a second copy of the repository's internal
scientific-governance system.

Public runners retain normal software safeguards: new-or-empty output roots,
clear failures, non-finite checks, final checkpoints, and lightweight completion
records. Registry updates, formal-evidence eligibility, full matrix governance,
scientific terminal adjudication, and manuscript table-cell/artifact binding
remain internal responsibilities.

Scores may vary across seeds, hardware, MuJoCo/Gymnasium versions, and numerical
libraries. The goal is reproducibility of the algorithm and stated protocol,
not byte-identical single-run scores on every machine.

## Install and test

Core training and tests:

```bash
cd paper_code
python -m pip install -e '.[test]'
python -m pytest
```

Install the optional Gymnasium/MuJoCo rollout dependencies when real environment
evaluation is required:

```bash
python -m pip install -e '.[test,rollout]'
```

## C-U1

The public C-U1 stage names follow the paper's evidence roles rather than the
internal E-number history:

- `source`: equal-advantage near/far gradient amplification;
- `causal`: near/far interventions and drift/collapse transmission;
- `phase`: negative-strength scans and far-pressure controls;
- `taper`: remoteness-aware taper comparison.

```bash
python -m drpo_reference cu1 \
  --stage source \
  --output outputs/cu1_source

python -m drpo_reference cu1 \
  --stage causal \
  --output outputs/cu1_causal

python -m drpo_reference cu1 \
  --stage phase \
  --output outputs/cu1_phase

python -m drpo_reference cu1 \
  --stage taper \
  --output outputs/cu1_taper
```

These commands expose the migrated implementation. A new registered full-budget
reproduction and terminal review are still required before the paper-code
migration is called scientifically reproduced.

## D-U1 revision 4

D-U1 is the controlled categorical utility×rarity environment. The active
matrix contains exactly six methods: Positive-only, All-negative,
matched-global, reciprocal-linear distance, reciprocal-quadratic distance, and
exponential-quadratic distance. The historical quartic method is not active.
Hidden high-reward rare actions are evaluation-only and make rarity-support
contraction task-visible.

The complete registered matrix uses CPU, seeds 200--219, 8000 updates, and the
frozen two-window terminal audit:

```bash
python -m drpo_reference du1 \
  --output outputs/du1_rev4 \
  --device cpu \
  --workers 8
```

A small integration run exercises all six methods but is never scientific
evidence:

```bash
python -m drpo_reference du1 \
  --output outputs/du1_smoke \
  --smoke
```

The active revision-4 scientific status remains `not_run` until the registered
full matrix and terminal review are complete.

## Hopper E7-Q2 mechanism validation

Hopper E7-Q2 is the external learned-critic mechanism profile. It uses the
frozen Hopper protocol, canonical critic and frozen advantages, one
Positive-only preparation, six actor branches, matched near/far diagnostics,
process-isolated Gymnasium/MuJoCo preflight, rollout evaluation, aggregation,
and root terminal audit.

```bash
drpo-reference hopper \
  --dataset /ABS/PATH/TO/hopper_medium_replay-v2.hdf5 \
  --output outputs/hopper_e7_q2
```

Equivalent module form:

```bash
python -m drpo_reference hopper \
  --dataset /ABS/PATH/TO/hopper_medium_replay-v2.hdf5 \
  --output outputs/hopper_e7_q2
```

Optional registered-order seed subsets and `--smoke` are always marked
non-evidence. The existing main-repository Hopper E7-Q2 scientific result is
`long_run_validated`; the new paper-facing runner still requires real
registered-data reproduction before migration closure.

## D4RL-9 locomotion performance

The D4RL-9 implementation uses one migrated `SNA2C_IQLV_ExpRank` trainer for
HalfCheetah, Hopper, and Walker2d across medium, medium-replay, and
medium-expert. It remains scientifically and operationally separate from the
Hopper E7-Q2 frozen-advantage mechanism runner.

The migrated code contains the actor, critic, dynamic TD/expectile update,
rank-based negative weighting, locomotion preparation, deterministic minibatch
training, checkpoint payload, nine-task catalog, fail-closed dataset identity,
real Gymnasium/MuJoCo rollout evaluation, and seed-level mean/std aggregation.

A selected-task reviewer run with real evaluation:

```bash
drpo-reference d4rl \
  --dataset-root /ABS/PATH/TO/D4RL_V2_HDF5 \
  --tasks hopper-medium-replay-v2 \
  --seeds 200,201 \
  --steps 100000 \
  --eval-episodes 10 \
  --output outputs/d4rl_hopper_medium_replay
```

Omit `--tasks` to run all nine tasks. The dataset root must contain each selected
task's canonical filename. `--eval-episodes 0`, the default, runs training only.
A tiny integration path is available:

```bash
drpo-reference d4rl \
  --dataset-root /ABS/PATH/TO/D4RL_V2_HDF5 \
  --tasks halfcheetah-medium-v2 \
  --seeds 7 \
  --output outputs/d4rl_smoke \
  --smoke
```

The runner writes per-seed final checkpoints plus `RUN_MANIFEST.json`,
`SUMMARY.json`, `COMPLETED.json`, or `FAILED.json`. When rollout evaluation is
enabled it also writes per-seed `EVALUATION.json`, raw returns, normalized
scores, episode mean/std, and task-level mean/std across seed means.

The final comparison matrix, formal seeds, formal budgets, and eight unresolved
dataset hashes remain protocol/provenance issues. They are not missing actor or
critic code and are not silently frozen by the public runner. Real nine-task
HDF5 and MuJoCo liveness has not yet been executed by this migration task.

## Artifact and evidence boundary

A public runner's completion record only answers whether that command finished,
wrote its expected checkpoint, stayed finite, and completed any configured
evaluation. It does not promote the output to formal scientific evidence.

Task-performance collapse, support/variance or probability-boundary events,
NaN/Inf numerical failures, environment invalidity or rollout unavailability,
and incomplete terminal state remain distinct in internal scientific review.
The public code never assumes that Distance, exponential, global scaling, SBRC,
Hybrid, or any other method must win.

Current migration status:

- C-U1 implementation candidate complete; registered reproduction pending;
- D-U1 revision-4 implementation candidate complete; formal run pending;
- Hopper E7-Q2 implementation candidate complete; registered real reproduction pending;
- D4RL-9 reviewer-facing algorithm, training, rollout, and simple aggregation implemented; real liveness and formal protocol remain pending;
- Countdown blocked pending final manuscript-facing protocol and result freeze.

No smoke or short differential result is a paper result. The machine-readable
acceptance contract is in:

```text
../docs/paper_code_reference/ACCEPTANCE_MATRIX.yaml
```
