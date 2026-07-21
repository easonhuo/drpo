# PAPER-CODE-VALIDATION-01 D-U1 Revision-4 Correctness Re-audit

**Date:** 2026-07-21  
**Parent claim:** `PAPER-CODE-REFERENCE-01`  
**Validation claim:** `PAPER-CODE-VALIDATION-01`  
**Canonical branch:** `dev/paper-code-reference-01`  
**Audited branch head:** `5de50275f37859331479878bdeb69484acab4b42`  
**Scientific-status impact:** none

## 1. Purpose and boundary

This re-audit checks whether the already migrated D-U1 protocol-revision-4
reviewer implementation preserves the authoritative categorical environment,
policy, controls, optimizer update, short trajectory, event taxonomy, and public
execution boundary.

Migration remains closed. This is engineering correctness evidence only. It is
not a registered formal run, convergence evidence, steady-state evidence, or a
method-ranking result.

Reviewer paths:

- `paper_code/src/drpo_reference/categorical/du1_environment.py`;
- `paper_code/src/drpo_reference/categorical/du1_policy.py`;
- `paper_code/src/drpo_reference/categorical/du1_controls.py`;
- `paper_code/src/drpo_reference/categorical/du1_training.py`;
- `paper_code/src/drpo_reference/categorical/du1_metrics.py`;
- `paper_code/src/drpo_reference/categorical/du1_protocol.py`;
- `paper_code/src/drpo_reference/categorical/du1_suite.py`;
- `paper_code/src/drpo_reference/categorical/du1_public.py`;
- `paper_code/src/drpo_reference/categorical/du1_reports.py`.

Authoritative oracle:

- `src/drpo/du1_e6_cartesian_taper_v4.py`;
- `configs/du1_e6_cartesian_taper_v4.yaml`.

## 2. Executable source identity

The verified reviewer artifact `8464318725` was produced from commit
`479b1dadef168c9e42a0fd67cc60c66842e8f799`.

A repository comparison from that artifact source commit to the audited head
contains no change under `paper_code/`. The artifact therefore carries the same
D-U1 reviewer implementation as the audited branch head.

The artifact and inner package checksums, 68-file manifest, extracted inventory,
and source commit were independently verified in the preceding shared/C-U1
re-audit.

## 3. Independent execution in this session

Using the verified extracted package:

```text
python3 -m pytest -q tests/test_du1_cli.py tests/test_du1_public.py
```

Result: **2 passed**.

The public command was executed on CPU:

```text
python3 -m drpo_reference du1 \
  --output /tmp/du1-recheck \
  --device cpu \
  --workers 1 \
  --smoke
```

Observed:

- process exit zero;
- all six frozen revision-4 methods executed in the registered order:
  `positive_only`, `all_negative`, `global_matched`,
  `reciprocal_linear_distance`, `reciprocal_quadratic_distance`, and
  `exponential_quadratic_distance`;
- expected and actual runs were both 6;
- no run identity was missing or unexpected;
- the environment audit passed;
- no environment-invalid, task-collapse, support-boundary, or NaN/Inf event was
  reported;
- all terminal classifications were accepted;
- `formal_scientific_acceptance`, `method_ranking_allowed`, and
  `formal_evidence_allowed` remained `false` because this was a smoke run;
- terminology remained same-distribution held-out-context generalization;
- expected manifests, summaries, reports, trajectories, calibration records, and
  per-seed checkpoint-completion records were written.

## 4. Differential evidence reviewed

The repository-internal exact-head full-pytest record passed the D-U1
revision-4 differential suites. The current reviewer code is unchanged from the
validated package source commit.

The reviewed tests bind the following invariants to the authoritative revision-4
runner.

### 4.1 Environment, policy, calibration, and metrics

`test_du1_environment_differential.py` checks:

- exact prototype embeddings, action-to-prototype and action-to-rarity mappings;
- rarity signs, action embeddings, utility directions, train split, and held-out
  split;
- complete environment audit identity;
- exact positive, cell, and residual log probabilities;
- coordinate-calibration identity;
- evaluation metrics and policy-geometry audit identity;
- useful/unhelpful rare-to-common shared-rarity gradient ratios;
- utility-oracle sign validity and rarity-shift task costs.

### 4.2 Taper coordinates, detached weights, losses, and gradients

`test_du1_controls_differential.py` checks:

- normalized excess current surprisal;
- reciprocal-linear, reciprocal-quadratic, and exponential-quadratic control
  formulas;
- exact coefficient calibration;
- detached taper weights;
- Positive-only, All-negative, reciprocal-linear, reciprocal-quadratic, and
  exponential-quadratic active-cell losses;
- raw gradients against the authoritative implementation.

### 4.3 Stepwise global budget matching and first update

`test_du1_update_differential.py` checks:

- raw All-negative gradient norm;
- exponential target-gradient norm;
- stepwise Global scale and budget-match error;
- rarity-logit anchor term;
- joint objective;
- the first Adam parameter update.

### 4.4 Shared start and fixed short trajectories

`test_du1_training_differential.py` checks the shared model/optimizer start and
fixed-seed four-step trajectories for Positive-only and Global-matched, including
every trajectory and summary field.

The existing public tests additionally check six-method execution, frozen output
order, aggregate/report files, checkpoint-completion records, event separation,
and non-formal evidence semantics.

## 5. Correctness findings

The migration preserves:

1. the 80-action revision-4 environment, including 64 observed actions and 16
   evaluation-only hidden rare actions;
2. fixed utility/rarity geometry and same-distribution train/held-out contexts;
3. the shared contextual rarity residual parameterization without trainable
   per-action bias;
4. common/rare calibration threshold and scale;
5. dynamic policy-relative surprisal coordinates with detached weights;
6. the active six-method matrix and exclusion of historical reciprocal quartic;
7. stepwise raw-negative-gradient matching for Global;
8. shared initialization, optimizer state, minibatch stream, objective, raw
   gradients, and first Adam update;
9. short trajectory and output ordering;
10. separate task-performance, support-boundary, numerical-failure, and
    environment-invalid events.

No D-U1 migration defect was reproduced.

## 6. Non-blocking reporting observation

`RUN_COMPLETE.json` uses the field name
`terminal_audit_all_checks_passed`, but assigns it the value of
`formal_scientific_acceptance`. Consequently, a successful smoke has complete
runs, accepted terminal classifications, and zero failure events while this
field remains `false`.

The authoritative revision-4 runner uses the same assignment. This is therefore
legacy-compatible and not a migration mismatch. The reviewer package also emits
the explicit fields `smoke`, `formal_scientific_acceptance`, and
`formal_evidence_allowed`, so the evidence boundary remains machine-readable.
No schema change is made in this re-audit.

## 7. Acceptance decision

**D-U1 revision 4: engineering correctness accepted for the currently migrated
scope.**

This acceptance covers environment geometry, policy initialization, calibration,
losses, detached controls, raw gradients, first update, selected short
trajectories, public six-method CPU smoke, output order, and event separation.

It does not mean:

- the registered 20-seed formal matrix was run;
- terminal plateau or steady state was established;
- a method ranking was established;
- scientific status changed from `not_run`;
- D-U1 is an OOD-generalization experiment.

## 8. Next authorized validation slice

Proceed to Hopper E7-Q2 mechanism-runner correctness acceptance. Preserve its
frozen-advantage role and keep it separate from the D4RL-9 task-performance
backend. Do not launch real HDF5/MuJoCo execution under this correctness slice.
