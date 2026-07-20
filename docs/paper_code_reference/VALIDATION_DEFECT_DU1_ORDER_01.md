# PAPER-CODE-VALIDATION-DU1-ORDER-01

**Parent claim:** `PAPER-CODE-VALIDATION-01`  
**Component:** D-U1 revision-4 reviewer public runner  
**Discovered by:** standalone `drpo-reference-v0.1` package validation  
**Scientific-status impact:** none

## Observed failure

At validation head `89e55e8eed739bfb26adda7fd032e12ee81bd1e2`, Paper Code
Validation run `29741698427` built and verified the exact-head ZIP, installed it,
compiled it, and passed shared, CLI, C-U1, and D-U1 dispatch tests. The D-U1
public smoke test failed.

Diagnostic artifact:

- artifact ID: `8460658889`;
- artifact digest:
  `sha256:2ee069e05f8c3068f89d76b355938e6e23adbdf95e35f8667f9b340a6347c6ff`.

`CHECKPOINT_COMPLETE.json.methods_completed` used lexicographic method order:

```text
all_negative
exponential_quadratic_distance
global_matched
positive_only
reciprocal_linear_distance
reciprocal_quadratic_distance
```

The frozen public protocol order is `FORMAL_METHODS`, beginning with
`positive_only`.

## Root cause

`paper_code/src/drpo_reference/categorical/du1_public.py` sorts trajectories and
summaries by the raw method string before writing per-seed records. The
checkpoint method list is derived from those sorted summaries, so the serialized
order is alphabetic rather than protocol order.

This is an output-contract and audit-order defect. It does not change any
per-method training value, but it can change array/column/report order and makes
the public checkpoint contract disagree with the frozen method coordinate.

## Frozen repair scope

Modify only the existing D-U1 public runner so trajectory and summary ordering
uses the index of each method in `FORMAL_METHODS` rather than the method string.
Unknown method IDs must fail closed.

The repair must not change:

- method membership or names;
- environment or data geometry;
- seeds, budgets, optimizer, loss, gradients, or updates;
- calibration or taper coefficients;
- event definitions, metrics, or numerical values;
- formal/scientific status.

## Required verification

1. existing D-U1 public smoke test passes in the extracted standalone package;
2. `methods_completed == list(FORMAL_METHODS)`;
3. shared, CLI, C-U1, and D-U1 package tests pass;
4. package manifest, installation, compile, Ruff, format, and all CLI gates pass;
5. full repository PR Gate and Evidence Locator pass on the repair head.

A smoke pass remains engineering evidence only and does not complete the
registered D-U1 formal matrix or terminal scientific review.