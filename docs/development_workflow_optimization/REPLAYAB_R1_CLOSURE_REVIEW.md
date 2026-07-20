# ReplayAB R1 Closure Review

Claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`

Closure base: `main@d2f3a7d0e2533d18f649c089caf1c8dc3b446a7c`

Calibration: `REPLAYAB-R1-C1-CALIBRATION-01`

Scientific impact: none

Status: `AWAITING_USER_CLOSURE_APPROVAL`

## Proposed decision

R1 satisfies the frozen deterministic exit gates and is ready for formal closure review.

The supported capability is bounded to schema-v2 deterministic `exact_artifact` and `failure_boundary` evidence, immutable run and evidence identities, balanced opposite-order pairs, covered invalid-execution detection, and evidence-bound efficiency release.

This review does not itself close R1, start R2, or adopt Candidate 01 as a default route.

## Final evidence

| Evidence | Result |
|---|---|
| Initial formal calibration run `29719538660`, artifact `8451833046` | Ten behavioral verdicts matched; closure remained blocked by an authority digest mismatch |
| Provenance diagnostic run `29719811214`, artifact `8451923028` | The recorded digest matched no repository object; the implementation-contract identity was stable |
| Repair materialization run `29721957880`, artifact `8452716796` | Approved nine-file scope; 60 focused tests passed; no Python file changed |
| PR #180 gate run `29722109039` | Full pytest, Ruff, compile, authority, formal-channel, and governance checks passed |
| Repaired calibration run `29722190304`, artifact `8452787806` | 21 tests passed; 10/10 verdict agreement; both authority checks and runtime guardrail passed |
| Repair merge `d2f3a7d0e2533d18f649c089caf1c8dc3b446a7c` | Narrow authority binding repair merged |
| Final merged-main calibration run `29722525319`, artifact `8452920571` | `R1_FINAL_CALIBRATION_PASS`; 21 tests passed; 10/10 agreement; covered false acceptance 0; covered false rejection 0; authority and runtime passed |

Final calibration artifact digest:

`sha256:b5e716552a7438b17981d0b83868623f992f39fb1be43edbd52f40a221fc0308`

Implementation-contract SHA-256:

`ae7f23134285b5314647bd1068bc3fa1f3935deccdd24536eb8183cb57e11494`

## Live C1 anchors

- C06 run `29693652154`: four arms, opposite-order equivalence, expected `BLOCKED / SOURCE_DRIFT`, protected workspace unchanged.
- C01 run `29693991085`: four arms, opposite-order READY equivalence, exact result identity preserved, Candidate B operator actions reduced from 7 to 1.

The C01 observation is one task and does not establish general efficiency.

## Exit-gate assessment

| Requirement | Verdict |
|---|---|
| Frozen calibration verdict agreement | PASS, 10/10 |
| Covered false acceptance count | PASS, 0 |
| Covered false rejection count | PASS, 0 |
| Interruption and partial mutation block efficiency release | PASS |
| Both arms identically wrong are rejected | PASS |
| Timing is bound to exact run and evidence identities | PASS |
| Opposite-order schedule is deterministic | PASS |
| Authority digest matches the immutable contract | PASS |
| Runtime guardrail | PASS |
| Exact-head repository checks | PASS |
| Repair changed no production Python, V1, authority owner, handoff, registry, or science | PASS |
| R1 production size remains within the reviewed yellow-zone boundary | PASS |

## Supported description after closure

After approval and merge of a closure record, ReplayAB may be described as a bounded C1 deterministic evidence ruler for exact artifacts and expected failure boundaries.

## Remaining limits

R1 does not provide:

- semantic acceptance of different correct implementations;
- a hidden semantic evaluator;
- complete worker repair trajectories;
- isolated live coding-agent workers;
- stochastic multi-task inference;
- general Candidate 01 efficiency evidence;
- Candidate 01 default-route authorization;
- R2 implementation authorization;
- scientific execution authorization.

The zero covered error counts apply only to the frozen ten-case bank.

## Next-step boundary

User approval of formal R1 closure would permit a documentation-first R2 gap audit and design review only.

R2 code, live workers, stochastic execution, Candidate 01 default adoption, and scientific changes remain separately gated.
