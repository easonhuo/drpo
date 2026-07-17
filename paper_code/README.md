# DRPO paper reference code

This directory contains the paper-facing implementation developed under
`PAPER-CODE-REFERENCE-01`. It is intentionally separate from the repository's
historical experiment drivers, registries, governance tooling, and packaging
code.

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

## Artifact and evidence boundary

Every public runner writes protocol manifests, per-seed artifacts, aggregate
results, and a terminal audit. Task-performance collapse, support/variance or
probability-boundary events, NaN/Inf numerical failures, and environment
invalidity remain separate fields.

Supplying a seed subset or `--smoke` always writes
`formal_evidence_allowed: false`. A full matrix is not accepted unless every
registered run is present and its terminal audit is resolved. The code never
assumes that Distance, exponential, global scaling, SBRC, Hybrid, or any other
method must win.

Current migration status: C-U1 and D-U1 have paper-facing implementation
candidates and clean integration paths. Formal full-budget reruns,
paper-output numerical identity, Hopper, and Countdown remain pending. No smoke
result is a paper result.

The acceptance contract is in
`../docs/paper_code_reference/ACCEPTANCE_MATRIX.yaml`.
