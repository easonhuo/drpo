# ReplayAB R0 Identity-Split Closure

**Project:** DRPO A/B Replay Engine  
**Claim:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Roadmap stage:** `R0 -- Identity split and planning freeze`  
**Development branch:** `dev/gov-dev-workflow-optimization-benchmark-01`  
**Roadmap commit:** `5aba3eaf3fe705bc5306e2d187622577add38e4d`  
**Candidate-plan commit:** `18470b0e2975a167d2b4d1870dba3debb594140b`  
**Current main observed during closure:** `64001f2e7d8636642cf30e57bf6ffc57882bf6ac`  
**Status:** `R0 complete`; documentation and responsibility split only; R1 implementation not authorized by this closure

## 1. Closure decision

R0 is complete.

The earlier roadmap recorded R0 as `active` because Candidate 01 still lacked an independent plan and the current reading order did not identify a durable closure record. Those gaps are now closed by:

- `REPLAYAB_ENGINE_ROADMAP.md` for ReplayAB Core capability development;
- `V1_SUBMISSION_WORKFLOW_OPTIMIZATION_PLAN.md` for Candidate 01;
- this closure record for the R0 gate, reading order, supersession rule, and next permitted action.

The original combined Stage 0--7 history remains unchanged. This closure separates future responsibilities without deleting or rewriting historical implementation records.

## 2. Current document map

Use the following roles.

| Document | Current role |
|---|---|
| `README.md` | durable project history, motivation, existing owners, and historical optimization context |
| `REPLAYAB_ENGINE_ROADMAP.md` | current capability roadmap, validity model, comparison modes, confidence grades, calibration requirements, and definition of done for ReplayAB Core |
| `V1_SUBMISSION_WORKFLOW_OPTIMIZATION_PLAN.md` | Candidate 01 definition, risks, case-bank requirements, metrics, thresholds, rollback, and adoption evidence |
| `REPLAY_BENCHMARK_PROTOCOL.md` | first-iteration exact-artifact historical replay protocol and its frozen timing/adoption rules |
| `IMPLEMENTATION_PLAN.md` | non-destructive historical record of the original combined ReplayAB-plus-Candidate-01 Stage 0--7 sequence |
| `REPLAYAB_R0_CLOSURE.md` | R0 completion record and transition authority into a possible R1 gap audit |

No document in this directory replaces `docs/handoff.md` as the unique research master.

## 3. Current reading order

A future session working on ReplayAB must read:

1. repository-root `AGENTS.md`;
2. Section 0 of `docs/handoff.md`;
3. `experiments/registry.yaml`, if present;
4. `docs/development_workflow_optimization/REPLAYAB_R0_CLOSURE.md`;
5. `docs/development_workflow_optimization/REPLAYAB_ENGINE_ROADMAP.md`;
6. `docs/development_workflow_optimization/V1_SUBMISSION_WORKFLOW_OPTIMIZATION_PLAN.md` only when Candidate 01 is relevant;
7. `docs/development_workflow_optimization/REPLAY_BENCHMARK_PROTOCOL.md` for the original deterministic protocol;
8. `docs/development_workflow_optimization/IMPLEMENTATION_PLAN.md` only for historical Stage provenance;
9. the current branch, exact commit, related code, tests, and any later reviewed closure or supersession record.

The old README reading order is historical. This later closure record supplies the current order without destructively rewriting that history.

## 4. Stable object separation

R0 freezes four distinct objects.

### 4.1 ReplayAB Core

The measuring instrument. It may own:

- immutable case contracts;
- run and evidence identity;
- evidence ingestion and normalization;
- independent correctness and safety evaluation;
- paired comparison and aggregation;
- confidence grades and decision reports;
- calibration of the ruler itself.

It does not own candidate behavior, V1, authority, scientific execution, or GitHub merge policy.

### 4.2 Candidate optimization

One workflow change being measured. Candidate 01 is the existing thin V1 one-click composition. A future candidate must have its own plan, failure hypotheses, metrics, and adoption decision.

Candidate-specific rules may not be hidden inside ReplayAB Core.

### 4.3 Execution backend

A mechanism that produces run evidence. Historical-artifact ingestion, deterministic local-command execution, and a future isolated Regeneration Runner are backend classes.

A backend executes. ReplayAB judges.

### 4.4 Evaluator

The frozen acceptance mechanism for a case. It must be independent of the generator where hidden evaluation is required and must support multiple correct implementations in semantic mode.

The evaluator may not be created or changed after inspecting candidate outcomes merely to obtain a desired verdict.

## 5. Comparison modes frozen at R0

ReplayAB future work must preserve explicit per-case comparison modes:

- `exact_artifact`: deterministic accepted output and protected semantic equality;
- `semantic_acceptance`: different implementations judged independently against one frozen acceptance contract;
- `failure_boundary`: correct fail-closed terminal behavior;
- `stochastic_generation`: repeated isolated live runs measuring an outcome distribution.

The current implementation is strongest on fixture-level deterministic exact-artifact and failure-boundary primitives. R0 does not claim the other modes are implemented.

## 6. Validity layers frozen at R0

A ReplayAB conclusion is valid only when the required layers pass:

1. **case validity** -- frozen task, treatment, controls, evaluator, budgets, metrics, and exclusions;
2. **execution validity** -- declared paths, comparable controls, isolation, no leakage, and retained failures;
3. **evaluator validity** -- frozen independent acceptance, false-pass and false-rejection visibility, and unauthorized-behavior detection;
4. **statistical validity** -- predeclared tasks, balanced order, retained failures, uncertainty, and claim-appropriate sample size.

A lower-layer success may not be described with a higher confidence grade.

## 7. Confidence grades frozen at R0

- `C0`: schema or fixture only;
- `C1`: deterministic real replay;
- `C2`: independently accepted semantic replay;
- `C3`: isolated live paired run;
- `C4`: repeated multi-task paired evidence;
- `C5`: post-adoption observation and rollback evidence.

The current ReplayAB branch has implementation starting points and narrow historical observations. R0 does not upgrade them automatically to C1 or higher.

## 8. Current implementation classification

The existing files are classified prospectively as follows.

### ReplayAB Core starting points

- `src/drpo/workflow_replay/model.py`;
- `src/drpo/workflow_replay/execute.py`;
- `src/drpo/workflow_replay/compare.py`;
- their focused tests and generic fixtures.

These are R1 inputs, not proof that R1 passed.

### Candidate 01 implementation

- `src/drpo/workflow_replay/orchestrate.py`;
- Candidate 01 behavior exposed through `scripts/run_workflow_replay.py`;
- Candidate 01 integration tests and fixtures.

The code location under `workflow_replay` is historical and does not make V1-specific orchestration generic Core logic.

### Existing evidence

The code-bloat repair pilots are matched historical observations assembled from frozen evidence. They are useful diagnostics but are not randomized live same-model A/B, automatic end-to-end historical ingestion, or a probability estimate for coding-agent behavior.

### Current-main adjacent capability

Current `main@64001f2e7d8636642cf30e57bf6ffc57882bf6ac` includes the opt-in `GOV-CODE-PAIRED-REPAIR-01` A0-to-B1 recorder. Future ReplayAB planning must assess it for adapter reuse rather than reimplementing first-attempt and one-repair evidence. It remains a same-worker before/after observation, not a two-worker randomized A/B backend.

## 9. Unsupported claims after R0

R0 does not establish that ReplayAB:

- is production-ready;
- automatically executes real historical tasks end to end;
- accepts multiple semantically correct implementations;
- records complete first-attempt and repair trajectories across backends;
- detects all unknown candidate regressions;
- isolates live coding-agent workers;
- prevents evaluator or treatment leakage in a live system;
- controls server-side model build, routing, cache, or sampling state;
- estimates coding-agent error probabilities;
- supports a general external repository without adapters;
- validates Candidate 01 for adoption;
- is equivalent to a platform-internal model A/B system.

These limits must remain visible in future reports.

## 10. R0 deliverable audit

| R0 requirement | Evidence | Verdict |
|---|---|---|
| Separate ReplayAB Core from Candidate 01 | Roadmap Sections 1, 4, 16; Candidate 01 plan | PASS |
| Separate execution backend and evaluator roles | Roadmap Sections 4 and 9; this closure | PASS |
| Preserve historical Stage provenance | Roadmap and Candidate 01 plan explicitly retain `IMPLEMENTATION_PLAN.md` | PASS |
| Freeze terminology and comparison modes | Roadmap Sections 4 and 8 | PASS |
| Freeze validity layers and confidence grades | Roadmap Sections 7 and 12 | PASS |
| Map current implementation to capabilities | Roadmap Section 5; this closure Section 8 | PASS |
| State unsupported claims | Roadmap Section 6.3; this closure Section 9 | PASS |
| Create candidate-specific plan | `V1_SUBMISSION_WORKFLOW_OPTIMIZATION_PLAN.md` | PASS |
| Preserve anti-framework constraints | Roadmap Section 15; Candidate 01 ownership boundaries | PASS |
| Avoid code, default-route, or scientific change | R0 commits are documentation-only | PASS |

R0 therefore closes as `complete`.

## 11. Review record

The R0 closure received four review passes before being declared complete.

### Pass 1 -- Responsibility consistency

Result: `PASS`.

- Core, candidate, backend, and evaluator are distinct.
- Candidate 01 no longer defines ReplayAB's roadmap.
- Regeneration Runner remains a possible backend, not a hidden Core requirement for deterministic R1.

### Pass 2 -- Evidence and claim discipline

Result: `PASS`.

- Implemented starting points, historical observations, and validated capabilities are not conflated.
- Deterministic and stochastic claims remain separate.
- Candidate adoption remains unproven.

### Pass 3 -- Historical preservation

Result: `PASS`.

- No old Stage, commit, pilot, threshold, or failure is deleted or retroactively renamed.
- The old roadmap `active` statement is preserved as the state at that earlier freeze and is superseded prospectively by this later closure record.

### Pass 4 -- Governance and next-step boundary

Result: `PASS`.

- No handoff, registry, scientific variable, default workflow, PR readiness, or merge authority changes.
- R0 completion does not authorize R1 implementation.
- The next action is a bounded R1 gap audit only.

## 12. Next permitted action

The next permitted ReplayAB action is a documentation-and-code-inspection **R1 gap audit**, not implementation.

The audit must:

1. start from the then-current authoritative `main`;
2. inspect current ReplayAB code, tests, and current-main adjacent tools such as paired repair;
3. map each existing capability to the R1 exit gates;
4. identify the minimum missing deterministic ingestion, run identity, opposite-order pairing, and calibration work;
5. estimate production code, tests, runtime overhead, adapter reuse, and maintenance cost;
6. decide `GO`, `NARROW`, `REDESIGN`, or `STOP` before writing R1 behavior code.

Because the historical development branch is substantially behind current main, R1 behavior code must not be appended automatically to this branch. A future authorized R1 implementation should use a new pure-Core development branch from the then-current exact `main` SHA and preserve this branch as provenance.

## 13. Authorization boundary

This closure authorizes no code change and no benchmark execution.

It does not authorize:

- R1 implementation;
- Candidate 01 Stage 5 or later evaluation;
- Regeneration Runner development;
- live coding-agent A/B;
- default-route activation;
- Ready-for-review transition;
- merge;
- scientific experiment execution or status change.

Any such action requires a new explicit instruction and a fresh current-main inspection.
