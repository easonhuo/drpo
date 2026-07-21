# M0 Atomic Development Transaction — Stage 2 Decision

**Claim:** `GOV-DRPO-MAINTENANCE-RUNNER-REPLAYAB-01`  
**Benchmark:** `M0-ATOMIC-REPLAYAB-01`  
**Pre-result identity:** `2489f36990fe703707ec7b94fa73e469c3a0de33`  
**Producer SHA-256:** `05756ff82064248eab0ee71400fde24bc05d555ee6252b3a4b1ec8ac6d82d9ad`  
**Decision:** `NARROW_M0`  
**Scientific impact:** none

## Decision

The general 3–8-file screen passed correctness, failure-boundary, wall-time,
active-operation, no-slowdown, and separate E7/E8 gates. It did not pass the frozen
60% median operator-action reduction gate: the observed median reduction was
**50.0%**.

The separately frozen 7–10-file confirmation bank passed every frozen gate:

- median controlled wall-time improvement: **44.9%**;
- mean controlled wall-time improvement: **45.0%**;
- median active-operation improvement: **44.9%**;
- median operator-action reduction: **68.3%**;
- E7 median wall-time improvement: **41.1%**;
- E8 median wall-time improvement: **48.1%**;
- minimum accepted-case wall-time improvement: **39.4%**;
- correctness/failure-boundary failures: **0**.

Therefore M0 is accepted only for the narrow class below. The smaller-file direct route
remains the appropriate default outside this class.

## Accepted narrow class

- 7–10 complete reviewed UTF-8 regular-file after-images;
- mode `100644`;
- one new dedicated development branch created directly at the final commit;
- exact current-`main` base;
- no deletes, renames, executable-mode changes, binaries, symlinks, gitlinks, LFS,
  workflows, handoff/registry, authority/governance, or scientific execution;
- Draft PR and exact-head checks remain mandatory;
- no automatic merge, retry, rebase, force push, or default-route activation.

This result supports the repository-publication assembly segment only. It does not claim
a 44.9% reduction in scientific design, code generation, test execution, GitHub queue
time, training, evaluation, aggregation, or terminal audit.

## Measurement adapter disposition

The 109-line local adapter passed its bounded local qualification and generated the
accepted controlled evidence. It is **not merged as runtime code**. M0 uses capabilities
already exposed by the GitHub App, so retaining a measurement-only executable would add
maintenance cost without improving the accepted route. The exact implementation and
test patches remain immutable audit evidence.

## Merge rule

The PR may merge after its final exact-head checks pass and the final diff confirms that
only contracts, patches, raw evidence, and decision records are present. The merge records
the bounded decision; it does not activate M0 as the default repository route.
