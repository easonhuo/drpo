# DRPO paper reference code

This directory contains the paper-facing implementation developed under
`PAPER-CODE-REFERENCE-01`. It is intentionally separate from the repository's
historical experiment drivers, registries, governance tooling, and packaging
code.

The public experiment names follow the paper's evidence roles rather than the
internal E-number history:

- `source`: equal-advantage near/far gradient amplification;
- `causal`: near/far interventions and drift/collapse transmission;
- `phase`: negative-strength scans and far-pressure controls;
- `taper`: remoteness-aware taper comparison.

C-U1 uses independent train and held-out states from the same state
distribution. Its result is **same-distribution held-out-context
generalization**, not OOD generalization.

## Install and test

```bash
cd paper_code
python -m pip install -e '.[test]'
python -m pytest
```

## Run C-U1

Formal defaults are frozen in the protocol dataclasses. The commands below use
the registered seed sets and budgets unless `--seeds` or `--smoke` is supplied.
A seed subset and every smoke run are written with
`formal_evidence_allowed: false`.

```bash
# Source diagnostic
python -m drpo_reference cu1 \
  --stage source \
  --output outputs/cu1_source

# Causal near/far interventions
python -m drpo_reference cu1 \
  --stage causal \
  --output outputs/cu1_causal

# Strength phase scans and controls
python -m drpo_reference cu1 \
  --stage phase \
  --output outputs/cu1_phase

# Taper comparison and 2x terminal audit
python -m drpo_reference cu1 \
  --stage taper \
  --output outputs/cu1_taper

# Developer-only integration path; never scientific evidence
python -m drpo_reference cu1 \
  --stage source \
  --output outputs/cu1_source_smoke \
  --smoke
```

Each stage writes:

- exact protocol and environment manifests;
- per-seed summaries and trajectories;
- initialization and terminal checkpoints where training occurs;
- deterministic aggregate CSV files;
- an explicit terminal-audit JSON file;
- separate task-performance, support/variance-boundary, NaN/Inf, and
  environment-invalid event fields.

Current migration status: the C-U1 scientific modules and public stage runner
are implementation candidates. Formal full-budget reruns, paper-output identity,
D-U1, Hopper, and Countdown remain pending. No smoke result is a paper result.

The acceptance contract is in
`../docs/paper_code_reference/ACCEPTANCE_MATRIX.yaml`.
