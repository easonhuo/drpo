# CODE-BLOAT-REPAIR-PILOT-01 Decision

## Status

**Decision: REDESIGN before PR #105 may be recommended for merge.**

This is a single-case matched historical repair observation. It is engineering evidence, not a DRPO scientific experiment and not a randomized live same-model A/B result.

## Frozen objects

- ReplayAB development branch before this pilot: `4acdc5855cfe3d110d466166174dac0bf2d93e5a`.
- Proposed automated review under test: PR #105 at `d73d7d8d2bcd604ff3967f8fc5eb654fe95a40ff`.
- Common historical task base: `929142930a3e2efaa7cafc8e4afe3866600027a5`.
- Initial overgrown attempt: PR #98 at `8672e173eb8ac52b9ec973415d17d887b2fba479`.
- Corrected same-base attempt: PR #101 at `f957e7f63c376e328e3d677cb143d526f6937c51`.

Both attempts addressed the paper-aligned E8 correction from `alpha*exp(-c*u**2)` to `alpha*exp(-c*u)` while preserving the existing alpha1-c/high-c execution lineage.

## Result 1 — the old flow did not prevent code bloat

PR #98 passed its existing exact-head PR Gate Log but still added:

- 1,075 total lines across 9 new files;
- 635 production-Python lines;
- 687 production Python plus shell lines;
- 4 new production-Python files;
- a new calibration, common, runtime, auto-runner, launcher and compatibility path for a narrowly scoped correction.

Therefore green compile/test/lint/governance CI alone did not enforce minimal sufficient change.

## Result 2 — rejection and scope feedback produced a much smaller repair

PR #101 used the same base and task but modified the predecessor lineage in place:

- total additions fell from 1,075 to 317: **70.51% fewer additions**;
- total churn fell from 1,075 to 424: **60.56% less churn**;
- production-Python churn fell from 635 to 98: **84.57% less production-Python churn**;
- production Python plus shell churn fell from 687 to 98: **85.74% less implementation churn**;
- new production-Python files fell from 4 to 0;
- exact-head PR Gate Log remained PASS.

This case shows that a model/process can correct an over-expansion error after explicit size and scope feedback. It does not by itself estimate the probability of correction across tasks.

## Result 3 — the current PR #105 gate does not close the repair loop

The proposed gate correctly rejects Arm A because the Python change is large and structural and the initial PR lacks the required internal evidence.

However, it also rejects Arm B. PR #101 legitimately changes or renames existing regression tests because the frozen matrix and formula changed. The current validator treats every changed AST for every pre-existing `test_*` function as forbidden, with no reviewer-frozen task-contract route for authorized test supersession.

Consequently, under the exact PR #105 implementation:

- overgrown attempt rejected: yes;
- corrected implementation accepted: no;
- repair closure for this case: **0/1**.

A gate that rejects both bad and correct implementations is not yet an effective code-bloat optimization. It is a brake without a complete recovery path.

## Required redesign

Replace the unconditional test-rewrite prohibition with a fail-closed, reviewer-owned test-preservation contract:

1. Existing regression tests are immutable by default.
2. A test may be changed, renamed or superseded only when a task contract frozen outside the implementation branch identifies:
   - the exact old test node;
   - why the old assertion is incompatible with the approved behavior change;
   - the replacement test node or nodes;
   - preserved behavior that must remain covered;
   - hidden or independent acceptance evidence.
3. The implementation author may not create or weaken that authorization inside the same PR.
4. Deleted behavior, `assert True`, missing replacement coverage, and unrelated test rewrites remain rejected.

This preserves the anti-cheating objective while allowing legitimate protocol, API, formula and matrix changes.

## Next benchmark required before adoption

Run a live paired coding-agent experiment on at least 6 frozen tasks:

- same base, task contract, model/version, temperature, tool budget and hidden acceptance suite;
- Arm A: existing workflow without the candidate review feedback;
- Arm B: candidate review after attempt 1, with at most two repair rounds;
- compare final task completeness, regressions, production churn, copied files/blocks, attempts, wall time and token/CI cost;
- report unsafe-pass rate and repair-at-1/repair-at-2, not only rejection rate;
- retain failed and timed-out trajectories.

Until that live benchmark passes, the only supported conclusion is:

> Explicit size/scope feedback materially reduced code volume in one matched historical repair, but PR #105 in its current form would reject the correct repair and therefore has not demonstrated an end-to-end improvement over the old workflow.
