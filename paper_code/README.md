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

## Install and test

```bash
cd paper_code
python -m pip install -e '.[test]'
python -m pytest
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

Public entry point:

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
registered-data reproduction and terminal review before migration closure.

## D4RL-9 locomotion performance

The paper-facing D4RL-9 code uses one migrated
`SNA2C_IQLV_ExpRank` performance implementation for HalfCheetah, Hopper, and
Walker2d across medium, medium-replay, and medium-expert. It is scientifically
and operationally separate from the Hopper E7-Q2 frozen-advantage mechanism
runner.

Already migrated:

```text
src/drpo_reference/experiments/d4rl.py
src/drpo_reference/external/d4rl_tasks.py
```

The migrated code contains the actor, critic, dynamic TD/expectile update,
rank-based negative weighting, locomotion preparation, deterministic minibatch
training, checkpoint payload, nine-task catalog, fail-closed dataset identity,
and one-backend dispatch boundary.

There is **not yet a public `drpo-reference d4rl` command**. Formal D4RL-9
execution remains disabled because the concrete nine-task runtime, generic
three-environment rollout evaluator, frozen method-matrix execution, formal
budget/checkpoint/terminal-audit lifecycle, aggregation, and minimal paper
binding are still incomplete. Eight dataset hashes, final controls and
coefficients, ten-run seeds, budgets, runtime resources, and real execution are
separate protocol/provenance/resource blockers rather than missing actor or
critic code.

See `../docs/paper_code_reference/CURRENT_STATUS.md` for the exact code-versus-
protocol split.

## Artifact and evidence boundary

Every complete public runner writes protocol manifests, per-seed artifacts,
aggregate results, and a terminal audit. Task-performance collapse,
support/variance or probability-boundary events, NaN/Inf numerical failures,
environment invalidity or rollout unavailability, and incomplete terminal state
remain separate fields.

Supplying a seed subset or `--smoke` always writes
`formal_evidence_allowed: false`. A full matrix is not accepted unless every
registered run is present and its terminal audit is resolved. The code never
assumes that Distance, exponential, global scaling, SBRC, Hybrid, or any other
method must win.

Current migration status:

- C-U1 implementation candidate complete; registered reproduction pending;
- D-U1 revision-4 implementation candidate complete; formal run pending;
- Hopper E7-Q2 implementation candidate complete; registered real reproduction pending;
- D4RL-9 selected algorithm core migrated; formal runtime and protocol closure pending;
- Countdown blocked pending final manuscript-facing protocol and result freeze.

No smoke or short differential result is a paper result. The machine-readable
acceptance contract is in:

```text
../docs/paper_code_reference/ACCEPTANCE_MATRIX.yaml
```
