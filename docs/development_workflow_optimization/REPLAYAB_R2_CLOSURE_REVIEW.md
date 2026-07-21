# ReplayAB R2 Closure Review

Claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`

Work ID: `REPLAYAB-R2-SEMANTIC-ACCEPTANCE-01`

Closure review candidate base: `main@ad9bda80796dcf5c48976f5d64ffd79a006c70d5`

R2 semantic evidence base: `main@8e1a0f61f5846fc4133e5de405280e371a96b994`

Calibration: `REPLAYAB-R2-CALIBRATION-01`

Controlled discrimination: `REPLAYAB-R1-R2-DISCRIMINATION-01`

Scientific impact: none

Status: `USER_CLOSURE_APPROVAL_RECORDED; SEPARATE_CLOSURE_RECORD_REQUIRED`

## Approval and decision boundary

The repository owner explicitly approved the sequence that merges the controlled R1-versus-R2 evidence, refreshes and merges this closure review, and then creates a separate immutable R2 closure record. That authorization is recorded in the PR discussion.

This review concludes that R2 satisfies the frozen bounded semantic-acceptance exit gates and that the additional controlled-discrimination evidence supports closure of the bounded R2 work item.

This document does **not** itself close R2. R2 becomes closed only after a separate `REPLAYAB_R2_CLOSURE.md` record is reviewed and merged. This review does not start R3, adopt Candidate 01 as a default route, authorize evaluator execution or live workers, or authorize scientific execution.

## Supported R2 capability

The supported capability is limited to immutable `AcceptanceContract` and `AcceptanceResult` evidence, independent per-arm acceptance, exact identity and content bindings, inclusive frozen metric bounds, mandatory-behavior and forbidden-regression checks, protected-path checks, and evidence-bound semantic pair reporting. Different implementation trees may both be accepted when both satisfy the same frozen contract. Efficiency release remains blocked unless both arms are accepted and all report, run, evidence, timing, evaluator, contract, case, and outcome bindings match.

ReplayAB ingests immutable evaluator-result evidence and recomputes acceptance. It does not execute evaluator code.

## Final evidence

| Evidence | Result |
|---|---|
| R2 gap audit and narrow implementation contract at `main@b01b7715c8a703905a03403e54dfce258e69ba1e` | Documentation-first verdict `NARROW`; evaluator execution, plugins, workers, trajectories, stochastic aggregation, authority, handoff, registry, and science excluded |
| Frozen calibration inventory and expected verdicts | Twelve cases frozen before implementation results; case removal forbidden; failures and rejections retained |
| Initial bounded implementation checkpoint `5d485ee287dddc8a86c9d24b1d9ecc666eaf63aa` | 310 changed production Python lines, zero new Python paths, focused tests and changed-file Ruff passed |
| Post-implementation review | Found one explicit evaluator-to-outcome content-binding gap; status `NARROW_FIX_REQUIRED` rather than premature closure |
| Hardened implementation `5a02ee26830e1a7abd0d94485f91410aa8bb89fa` | `AcceptanceResult.outcome_sha256` added; substituted outcome evidence fails closed; frozen twelve-case bank unchanged; one post-review tamper regression added |
| R2 implementation PR #185, head `74630c6fa8c7be665ca2615308bab1db7a85f28b` | Merged as `cfe43571cd8c6d0909c61d36c4f6e4d07c2d2362` |
| Merged-main focused audit run `29742478179`, artifact `8460956367` | ReplayAB focused suite, frozen R2 calibration, closed R1 non-regression, identity checks, Ruff, and compilation passed; closure correctly remained blocked by unrelated repository health |
| Repository-health repairs #207, #208, #209, #211, and #213 | Merged in approved order as `a4366fb51d377625f66ae0ed05f27b001f0c67a6`, `77cf88cef9e0cf17298b128a7ad6184730eca287`, `2cb3fad70fc37bab40451fdd2562d40b1a6ed861`, `dc09ad1589290c46004418b5e3493dc468096677`, and `b18aea9186d7e3ccc5d43b456719cafc23761e03` |
| Pre-merge combined audit run `29752970308`, artifact `8465556847` | Five repair heads composed cleanly: 1181 passed, 27 skipped, Ruff, compilation, authority, governance, R2, and R1 all passed |
| Final merged-main R2 terminal audit run `29785274230`, artifact `8478312547` | Exact `main@b18aea9186d7e3ccc5d43b456719cafc23761e03`; 1181 passed, 27 skipped, zero failures; focused ReplayAB 86/86; frozen R2 14/14; closed R1 non-regression 21/21; all repository and governance gates passed |
| Frozen R1-versus-R2 benchmark | 16 pairs / 32 arms across API compatibility, numeric tolerance, protected-path safety, and serialization semantics; 20 correct arms, 12 incorrect arms, and 8 efficiency-eligible pairs |
| Controlled-discrimination exact-head audit run `29789194545`, artifact `8479682445` | Exact benchmark head `bb83fa5d8aaa38649112caa55dc115cfc1a18ff8`; benchmark 3/3, R1/R2 terminal non-regression 35/35, focused ReplayAB 89/89, full repository 1184 passed and 27 skipped, all other gates passed |
| Controlled-discrimination PR #218 | Exact head merged into `main` as `8e1a0f61f5846fc4133e5de405280e371a96b994`; decision `PASS_CONTROLLED_ADVANTAGE` |
| First closure-review exact-candidate audit run `29821895759`, artifact `8491826238` | Immutable candidate scope, closure contract, focused ReplayAB 89/89, Ruff, compile, handoff authority, formal channel, and governance inventory passed; fail-closed because current-main governance-stage validation exposed one unrelated authorization-field mismatch |
| Governance repair PR #233 | Repository-owner-approved one-line fix aligned `kind: reopen` with `change_class: reopen`; merged as `ad9bda80796dcf5c48976f5d64ffd79a006c70d5` after full pytest, Ruff, compilation, handoff authority, formal channel, governance inventory, and governance-stage validation passed |

Artifact digests:

- final merged-main R2 terminal audit: `sha256:b9c962df253fba71cbff6918985bde96c323852a1b7c2dd7593cad622ae027f3`;
- pre-merge combined audit: `sha256:ff590dc1536a0dee07522664352e35186c82b6ae3ee4e7ce30eab01b7417275d`;
- merged-main focused audit: `sha256:204a978aa992c8b765d6e34066ee9c6283cb3a1cade67085df04ae30257d7ea7`;
- controlled-discrimination terminal audit: `sha256:ab44fb01570c15da6040667887b9bc5d77c4617597880a6f8aa96166b6807fbb`;
- first closure-review failed audit: `sha256:4c2e94b4a1e1ce087aea7bc5b7c7628eeeb4af064301b97810d5e3dfdccf3496`.

## Controlled-discrimination result

| Metric | R1 exact-artifact | R2 semantic acceptance |
|---|---:|---:|
| Correct-arm false rejection | 4/20 = 20% | 0/20 = 0% |
| Incorrect-arm false acceptance | 0/12 = 0% | 0/12 = 0% |
| Arm-label accuracy | 87.5% | 100% |
| Pair-label accuracy | 75% | 100% |
| Eligible efficiency release coverage | 4/8 = 50% | 8/8 = 100% |
| Efficiency-release precision | 100% | 100% |

The predeclared gate passed with a correct-arm false-rejection reduction of `0.20`, an efficiency-coverage gain of `0.50`, and no increase in incorrect-arm false acceptance. The supported bounded claim is:

> On the frozen controlled discrimination bank, R2 is a strict judge-level capability extension over R1 exact-artifact mode: it preserves rejection of predeclared incorrect outcomes while reducing false rejection caused solely by implementation non-identity.

This result is C2 controlled evidence, not a population-level error estimate.

## Exit-gate assessment

| Requirement | Verdict |
|---|---|
| Calibration inventory frozen before implementation results | PASS, 12 cases |
| Frozen expected-verdict bank retained without post-result case removal | PASS |
| Different implementations may both be accepted under one contract | PASS |
| Missing mandatory behavior rejects the affected arm | PASS |
| Forbidden regression rejects the affected arm | PASS |
| Both identically wrong implementations are rejected | PASS |
| Inclusive lower and upper metric boundaries are accepted | PASS |
| Out-of-bound metric evidence is rejected | PASS |
| Evaluator, contract, case, and run identity mismatch fails closed | PASS |
| Exact outcome evidence SHA-256 is directly bound by `AcceptanceResult` | PASS |
| Outcome-binding tamper regression fails closed | PASS |
| Protected-path failure rejects the affected arm | PASS |
| Efficiency release is blocked unless both arms are accepted | PASS |
| Pair report preserves independent A and B acceptance and exact evidence identities | PASS |
| Existing R1 schema-v2 exact-artifact and failure-boundary behavior remains unchanged | PASS, 21/21 terminal non-regression in the merged-main R2 audit and 35/35 in the controlled-discrimination audit |
| R2 controlled-bank correct-arm false-rejection rate | PASS, 0/20 versus R1 4/20 |
| R2 controlled-bank incorrect-arm false-acceptance rate | PASS, 0/12 and no worse than R1 |
| R2 controlled-bank arm and pair accuracy | PASS, 100% / 100% |
| R2 controlled-bank efficiency-release precision and coverage | PASS, 100% precision and 100% coverage; coverage gain 0.50 over R1 |
| No evaluator execution, plugin, backend registry, sandbox, worker, trajectory, network work, authority, handoff, registry, or scientific change | PASS |
| Production implementation remained within the preferred reviewed budget | PASS |
| Final exact-head focused ReplayAB suite | PASS, 86/86 in the merged-main R2 audit and 89/89 in the controlled-discrimination audit |
| Final exact-head frozen R2 subset | PASS, 14/14 |
| Final repository pytest evidence | PASS, 1181 passed / 27 skipped in the R2 merged-main audit; 1184 passed / 27 skipped in the controlled-discrimination audit |
| Repository-wide Ruff and Python compilation | PASS |
| Handoff authority, formal execution channel, governance inventory, and Stage 5 validation | PASS after the unrelated Stage 5 authorization-field repair merged in PR #233 |
| Explicit user authorization to proceed with R2 closure sequence | PASS, durably recorded in PR discussion |

## Supported description after closure

After approval and merge of the separate closure record, ReplayAB R2 may be described as a bounded deterministic semantic-acceptance evidence ruler: it can independently classify two different implementations against one frozen contract and may compare efficiency only after both implementations are accepted with exact immutable evidence bindings.

The frozen C2 bank additionally supports that, within those controlled cases, R2 reduces exact-identity-induced false rejection without accepting any predeclared incorrect arm.

## Remaining limits

R2 does not provide:

- evaluator execution inside ReplayAB Core;
- a hidden evaluator service or generator-evaluator isolation;
- live coding-agent workers;
- first-attempt and repair-trajectory capture;
- stochastic repeated A/B execution or multi-task statistical inference;
- detection of regressions outside the frozen mandatory, forbidden, tolerance, protected-path, and evaluator evidence contract;
- a population-level semantic-acceptance error rate outside the frozen calibration and controlled-discrimination banks;
- general Candidate 01 efficiency evidence;
- Candidate 01 default-route authorization;
- R3 implementation authorization;
- scientific experiment authorization.

The zero covered failures in the final audits apply only to the frozen calibration, controlled benchmark, and repository test coverage. They are not a population-level error estimate for arbitrary coding tasks or evaluators.

## Next-step boundary

After this refreshed review passes exact-candidate audit and is merged, the approved next action is to create and review a separate immutable `docs/development_workflow_optimization/REPLAYAB_R2_CLOSURE.md` record.

Only after that closure record is merged may a documentation-first R3 gap audit and design review begin. R3 implementation, evaluator execution, live workers, stochastic experiments, Candidate 01 default adoption, and scientific changes remain separately gated.
