# M0 Stage 1/2 Final Review

**Verdict:** `PASS_NARROW_M0`  
**Merge recommendation:** `MERGE_EVIDENCE_AND_CONTRACT_ONLY`  
**Pre-result commit:** `2489f36990fe703707ec7b94fa73e469c3a0de33`  
**Base main:** `d3f7d046f948108a3d837bdcff617eed5146a2f0`

## Identity and pre-result freeze

The case inventory, evaluator, thresholds, treatment orientation, yellow-zone decision,
and reproducible patches were committed before the accepted execution. Packet SHA-256
values matched the frozen inventory for all 12 cases. All accepted reports bound the
same producer SHA-256 `05756ff82064248eab0ee71400fde24bc05d555ee6252b3a4b1ec8ac6d82d9ad`.

The pre-result remote transaction created a seven-file commit directly from the exact
base, then created the new development branch directly at that final commit. PR #239 was
opened as Draft. On that exact pre-result head, Evidence Locator Gate run `29843129717`
and PR Gate Log run `29843127542` both completed successfully.

## Stage 1 qualification

- adapter executable change: 109 lines, yellow band, below the 140-line stop boundary;
- new Python paths: 0;
- local measurement harness: 6/6 passed after pre-result freeze;
- Python compilation: passed;
- protected path: rejected;
- NUL content: rejected;
- stale-head: protected ref preserved;
- validation failure: no candidate ref created;
- commands in packets: impossible under the strict field set;
- full repository checkout was unavailable in the local shell environment, so local full
  pytest and Ruff are not claimed for the non-merged adapter.

Because the executable is not part of the merge candidate, final repository acceptance
is performed on the exact evidence-only PR head through GitHub Actions.

## Stage 2 independent acceptance

All 12 frozen cases were retained:

- six general-screen successes;
- two deterministic failure-boundary cases;
- four narrow-confirmation successes.

Two general-screen cases exceeded the frozen 15% first-execution pair-ratio spread rule.
Each received and retained one additional opposite-order execution. No post-result case,
threshold, evaluator, or treatment change was made.

Independent checks found:

- 0 packet/producer identity failures;
- 0 terminal-state failures;
- 0 final-tree/path/mode mismatches;
- 0 Arm-B parent violations;
- 0 Arm-A treatment-contrast violations;
- 0 stale-ref mutations;
- 0 validation-failure ref creations;
- 0 protected-content acceptances.

## Gate disposition

General screen:

- median wall time: 36.31% — PASS;
- mean wall time: 33.99% — PASS;
- median active operation: 36.31% — PASS;
- median operator actions: 50.00% — FAIL;
- E7/E8 separately positive — PASS;
- no slowdown — PASS.

Narrow 7–10-file confirmation:

- median wall time: 44.93% — PASS;
- mean wall time: 44.98% — PASS;
- median active operation: 44.93% — PASS;
- median operator actions: 68.33% — PASS;
- E7/E8 separately positive — PASS;
- no slowdown — PASS.

The only justified decision is `NARROW_M0`. `ADOPT_M0` would overstate the evidence;
`REDESIGN_TO_M1` is not triggered because no residual publisher responsibility was
demonstrated; `REJECT_OPTIMIZATION` is not justified because correctness and the narrow
class passed.

## Merge-content audit requirement

The final PR diff must contain no installed executable/test code, no workflows, no new
Python path, no handoff/registry change, no governance implementation, and no scientific
code. It may contain only the frozen contracts, patch evidence, raw results, this review,
and the decision record.

A later default-policy update, broader file-count class, existing-branch update route,
delete/rename/mode support, or M1 publisher remains separately scoped and unapproved.
