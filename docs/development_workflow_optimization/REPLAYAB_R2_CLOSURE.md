# ReplayAB R2 Closure

Claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`

Work ID: `REPLAYAB-R2-SEMANTIC-ACCEPTANCE-01`

Calibration: `REPLAYAB-R2-CALIBRATION-01`

Controlled discrimination: `REPLAYAB-R1-R2-DISCRIMINATION-01`

Closure review: `docs/development_workflow_optimization/REPLAYAB_R2_CLOSURE_REVIEW.md`

Review merge: `main@90067574b8ef915cb052becad0e9ba3ec7b3b5c4`

Status: `CLOSED`

Scientific impact: none

## Authorization

The repository owner explicitly approved continuation of the R2 closure sequence on 2026-07-21. The authorization and the exact-candidate terminal-audit evidence are durably recorded in PR #216.

This record takes effect only when reviewed and merged. It does not retroactively alter R0, R1, the historical R2 implementation record, or any scientific experiment state.

## Decision

ReplayAB R2 is formally closed as a bounded deterministic semantic-acceptance evidence ruler.

R2 supports immutable `AcceptanceContract` and `AcceptanceResult` evidence, independent per-arm acceptance, exact evaluator/contract/case/run/outcome bindings, inclusive frozen metric bounds, mandatory-behavior and forbidden-regression checks, protected-path checks, and evidence-bound semantic pair reporting. Two different implementation trees may both be accepted when both satisfy the same frozen contract. Efficiency comparison remains blocked unless both arms are accepted with all required immutable identities aligned.

ReplayAB R2 ingests evaluator-result evidence and recomputes acceptance. It does not execute evaluator code.

## Closure evidence

### Final merged-main R2 terminal audit

- audited main: `b18aea9186d7e3ccc5d43b456719cafc23761e03`;
- workflow run: `29785274230`;
- artifact: `8478312547`;
- digest: `sha256:b9c962df253fba71cbff6918985bde96c323852a1b7c2dd7593cad622ae027f3`;
- full repository: 1181 passed, 27 skipped, zero failures/errors;
- focused ReplayAB: 86/86;
- frozen R2 subset: 14/14;
- closed R1 non-regression: 21/21;
- Ruff, compilation, handoff authority, formal channel, governance inventory, and governance-stage validation: PASS.

### Controlled R1-versus-R2 discrimination

- benchmark: `REPLAYAB-R1-R2-DISCRIMINATION-01`;
- benchmark head: `bb83fa5d8aaa38649112caa55dc115cfc1a18ff8`;
- workflow run: `29789194545`;
- artifact: `8479682445`;
- digest: `sha256:ab44fb01570c15da6040667887b9bc5d77c4617597880a6f8aa96166b6807fbb`;
- decision: `PASS_CONTROLLED_ADVANTAGE`;
- correct-arm false rejection: R1 `4/20`, R2 `0/20`;
- incorrect-arm false acceptance: R1 `0/12`, R2 `0/12`;
- arm accuracy: R1 `87.5%`, R2 `100%`;
- pair accuracy: R1 `75%`, R2 `100%`;
- efficiency-release coverage: R1 `4/8`, R2 `8/8`;
- efficiency-release precision: both `100%`.

### Closure-review exact-candidate audit

- audited merge candidate: `7f3de734c6f6dc4a9e8ccfb05fbd645cf5507c74`;
- base main: `a6e55ced251280b77ebe8e7d3cd18cc0c172ebbe`;
- review head: `630b238d0ad2ff1f5601ec254c386f6ca13739fd`;
- workflow run: `29835704072`;
- artifact: `8497370186`;
- digest: `sha256:f8753985b1c72d1ab8923e0449f85caf08cda25bdefe30e5c57bf47703cb1e6b`;
- changed path: only `docs/development_workflow_optimization/REPLAYAB_R2_CLOSURE_REVIEW.md`;
- focused ReplayAB: 89/89;
- full repository: 1192 passed, 27 skipped, zero failures/errors;
- Ruff, compilation, handoff authority, formal channel, governance inventory, and governance-stage validation: PASS;
- `review_merge_permitted: true`;
- `r2_closed: false` at review time, as required before this separate record.

## Supported claim

ReplayAB R2 may be described as a bounded C2 semantic-acceptance evidence ruler. On the frozen controlled discrimination bank, it preserves rejection of all predeclared incorrect outcomes while reducing false rejection caused solely by implementation non-identity.

This is controlled evidence, not a population-level semantic-acceptance error estimate.

## Remaining limits

R2 does not provide:

- evaluator execution inside ReplayAB Core;
- a hidden evaluator service or generator-evaluator isolation;
- live coding-agent workers;
- first-attempt and repair-trajectory capture;
- stochastic repeated A/B execution or multi-task statistical inference;
- detection of regressions outside the frozen acceptance contract and exercised calibration dimensions;
- general Candidate 01 efficiency evidence;
- Candidate 01 default-route authorization;
- R3 implementation authorization;
- scientific experiment authorization.

## Next-step boundary

R2 closure permits a documentation-first R3 gap audit and design review only.

R3 implementation, evaluator execution, live workers, stochastic experiments, Candidate 01 default adoption, handoff or registry changes, and scientific execution remain separately gated. Every later stage requires a fresh scope, ROI review, explicit authorization, and its own terminal audit.
