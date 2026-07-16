# CODE-BLOAT-REPAIR-PILOT-02 Decision

## Decision

**PROVISIONAL GO for the repaired test-supersession mechanism; HOLD for merge or default-route adoption.**

The first replay iteration found that the old validator rejected both the overgrown attempt and the correct smaller repair. The repaired validator at `7826f5d60c83d8a58a11dc526b487cc09078d818` removes that dead end without allowing silent or trivial test weakening.

## Matched historical result

The same E8 paper-aligned task and common base are retained:

- Arm A — PR #98: 635 production-Python churn and 4 new production-Python files;
- Arm B — PR #101: 98 production-Python churn and 0 new production-Python files;
- production-Python churn reduction: **84.57%**.

Under the repaired gate:

- Arm A result: `FAIL`, as required;
- Arm B result: `PASS`, as required;
- repair closure improves from `0/1` to **`1/1`**.

The Arm-B replay used the historical old and new test ASTs. Six pre-existing tests were legitimately updated or renamed. Every old node was mapped to one new or changed core-evidence node, with preserved production anchors and no reduction in executable checks.

## Rule change

Pre-existing regression tests remain immutable by default. A changed or removed old test is accepted only when the review block exactly covers it and proves all of the following:

1. the replacement test exists at the head;
2. it is new or actually changed, not an unrelated unchanged test;
3. it is declared as core requirement evidence;
4. a named production anchor is preserved, including explicit before/after names for a legitimate rename;
5. no replacement contains a constant or trivial assertion;
6. the aggregate executable-check count is at least the old count;
7. one replacement cannot be reused to justify several old tests.

Missing evidence, copied implementation, public-symbol removal, invented reuse, silent test rewrite, trivial assertion, unchanged replacement reuse, reduced check count, and non-core replacement all remain fail closed.

## Validation actually completed

Focused local replay:

- 13 cases passed their expected outcomes;
- Python compilation passed.

Exact-head GitHub run `29515499237` passed:

- tiered test-plan shadow;
- Python compile;
- shell syntax;
- handoff authority;
- formal execution channel;
- governance inventory;
- governance stage status;
- full pytest;
- Ruff.

## What this result supports

It supports the narrow statement:

> On the frozen matched E8 repair, explicit size/reuse feedback reduced production-Python churn by 84.57%, and the repaired gate now rejects the overgrown implementation while allowing the correct smaller repair.

It does not yet support a universal probability that an arbitrary coding model will repair itself, nor a claim that every legitimate large change will pass.

## Remaining gates before adoption

1. Rebuild the final three-file implementation on current `main`; the development PR is stale and remains Draft.
2. Repeat the matched outcome checks after current-main integration.
3. Run at least six frozen live coding tasks with identical model/version, task contract, tool budget and hidden acceptance tests in both arms.
4. Report unsafe-pass rate, false-rejection rate, repair-at-1, repair-at-2, final production churn, copied files/blocks, task completeness, regressions, wall time and token/CI cost.
5. Retain every failed, timed-out and unrepaired trajectory.

No merge, Ready-for-review transition, or default-route activation is authorized by this pilot.
