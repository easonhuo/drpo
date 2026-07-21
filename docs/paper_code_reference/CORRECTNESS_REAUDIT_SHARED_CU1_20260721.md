# PAPER-CODE-VALIDATION-01 Shared Kernel and C-U1 Correctness Re-audit

**Date:** 2026-07-21  
**Parent claim:** `PAPER-CODE-REFERENCE-01`  
**Validation claim:** `PAPER-CODE-VALIDATION-01`  
**Canonical branch:** `dev/paper-code-reference-01`  
**Audited branch head:** `08a605679302ee442b468238d2e37a6a3ce6538b`  
**Scientific-status impact:** none

## 1. Purpose and boundary

This re-audit checks whether the already migrated shared controls and C-U1
reviewer-facing implementation preserve the authoritative algorithm behavior.
Migration remains closed. This is not feature development, scientific
reproduction, convergence evidence, method ranking, real-stack liveness, or an
authorization to implement remaining gaps.

Audited reviewer paths:

- `paper_code/src/drpo_reference/controls/weights.py`;
- `paper_code/src/drpo_reference/controls/selection.py`;
- `paper_code/src/drpo_reference/controls/budget.py`;
- `paper_code/src/drpo_reference/common/events.py`;
- `paper_code/src/drpo_reference/continuous/gaussian.py`;
- `paper_code/src/drpo_reference/continuous/cu1*.py`;
- the corresponding self-contained and repository-internal differential tests.

Authoritative C-U1 oracles remain:

- `src/drpo/cu1_core.py`;
- `src/drpo/drpo_cu1_e1_e4_oneclick.py`;
- `src/drpo/cu1_distance_taper_formal.py`;
- `src/drpo/cu1_taper_near_retention_formal.py`;
- `src/drpo/cu1_taper_budget_match_formal.py`.

## 2. Executable source identity

The available validated reviewer artifact is GitHub Actions artifact
`8464318725`, produced from source commit
`479b1dadef168c9e42a0fd67cc60c66842e8f799`.

Verified locally:

- outer artifact SHA-256:
  `430048d3d0fbe280423625478703f11a9241c90dd67654052e5aa40f1693d28c`;
- inner package SHA-256:
  `de0f49cbfdc87329e737f6f619425bde2fda2801563351a3e9c8a502d2ec39ab`;
- inner checksum file matches the package;
- package manifest declares 68 files excluding the manifest and the extracted
  inventory contains exactly those 68 declared payload files plus the manifest.

A repository compare from artifact source commit `479b1dade...` to audited head
`08a60567...` contains no change under `paper_code/`. The artifact is therefore
an exact executable carrier for the current shared/C-U1 reviewer code. It is not
used as a substitute for current GitHub source facts.

The container could not clone GitHub because DNS resolution failed. No package
fallback or repository write was triggered by that limitation.

## 3. Independent execution in this session

Using the verified extracted package with `PYTHONPATH=src`:

```text
python3 -m pytest -q \
  tests/test_controls.py \
  tests/test_cu1_suite.py \
  tests/test_events.py \
  tests/test_common.py
```

Result: **45 passed**.

The actual public command was also executed:

```text
python3 -m drpo_reference cu1 \
  --stage source \
  --output /tmp/cu1-source-smoke \
  --device cpu \
  --smoke
```

Observed:

- process exit zero;
- expected manifest, preparation records, checkpoints, aggregate CSV, seed
  record, and terminal audit were created;
- matrix completeness passed;
- `formal_evidence_allowed` remained `false`;
- manifest terminology remained
  `same-distribution held-out-context generalization`;
- environment audit preserved 4 positive and 8 negative contour actions,
  equal-within-role rewards/advantages, and finite near/far geometry.

A whole-package local pytest attempt exceeded the 120-second local timeout after
partial progress. It is not reported as a local full-suite pass. The historical
exact-head GitHub full-pytest and package-validation records remain the full-suite
evidence.

## 4. Shared-kernel correctness findings

The inspected implementation and tests preserve:

- explicit non-negative distance validation;
- detached remoteness coordinates by default;
- Positive-only zero negative weight and Uncontrolled unit negative weight;
- Global, reciprocal-linear, reciprocal-quadratic, exponential-linear, and
  exponential-quadratic formulas without alias ambiguity;
- exact point-retention coefficient solutions;
- C-U1 standardized-distance semantics distinct from D-U1/Countdown surprisal
  distance semantics;
- near mask `distance <= threshold` and far mask as its exact complement;
- raw-gradient L2 accounting across all non-`None` tensors;
- deterministic scalar budget matching with fail-closed zero-source handling;
- separate task-performance, support/probability-boundary, numerical, and
  environment-invalid events.

No concrete shared-kernel defect was reproduced.

## 5. C-U1 correctness findings

The implementation preserves the authoritative C-U1 behavior at the following
levels:

1. **Environment identity**
   - train and held-out states are independent draws from the same `Normal(0,I)`
     distribution;
   - the state-to-base and state-to-task-direction mappings match;
   - positive and negative contour geometry, rewards, advantages, centroids, and
     near/far identities match.
2. **Policy and Gaussian primitives**
   - two-hidden-layer actor initialization and state dictionaries match;
   - Gaussian log probability, standardized distance, and output-score
     decomposition match.
3. **Objective and gradients**
   - Positive-only, local-negative, all-negative, and near/far losses match;
   - fixed-sigma and learnable-sigma raw gradients match;
   - remoteness selection is detached and the legacy denominator is preserved.
4. **Updates and trajectories**
   - first Adam update matches;
   - fixed-seed short Positive-only trajectories and final evaluation match;
   - source diagnostic, all six causal intervention gradients, phase scans,
     far-pressure controls, taper losses, taper gradients, and selected short
     trajectories are covered by exact or frozen-tolerance differential tests.
5. **Public evidence boundary**
   - the public CPU smoke writes expected records and checkpoints;
   - smoke completion remains non-formal;
   - task-performance collapse, support/variance boundary, NaN/Inf numerical
     failure, and invalid environment remain separately reportable.

No C-U1 correctness defect was reproduced in this re-audit.

## 6. Acceptance decision

**Shared kernel:** engineering correctness accepted for the currently migrated
scope.  
**C-U1:** engineering correctness accepted for the currently migrated scope.

This acceptance means that the migrated formulas, losses, masks, detachment,
raw gradients, first updates, selected fixed short trajectories, event
separation, and public CPU smoke are adequately bound to the authoritative
implementation.

It does **not** mean:

- registered full C-U1 reproduction has been rerun;
- convergence or steady state has been established;
- any scientific result status has changed;
- C-U1 is OOD generalization;
- a method ranking has been re-established from this smoke or short-path
  evidence.

## 7. Next authorized validation slice

Proceed to D-U1 revision-4 correctness acceptance using the existing
machine-readable task lock. Do not begin feature development or scientific
execution unless a concrete validation defect is reproduced or the user gives a
separate explicit instruction.
