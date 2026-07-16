# DRPO Development Workflow Optimization Project

**Claim:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Status:** documentation and validation design only; no optimizer implementation is authorized  
**Repository base:** `main@7d0ecfbee3b9e44bbad97fb806c8806b604f75f6`

## 1. Purpose and authority boundary

This directory is the durable project hub for improving the DRPO repository-development and scientific-pilot integration workflow. It records the problem history, existing mechanisms, proposed optimizations, validation method, decisions, and rollback criteria so that a later session can recover the full reasoning without relying on chat history.

This document is subordinate to `AGENTS.md`, `docs/handoff.md`, `experiments/registry.yaml`, and the accepted component contracts. It is not a second research master and does not change any scientific experiment, result, seed, threshold, horizon, or execution priority.

Any future session working on development-workflow optimization must read:

1. this document;
2. `REPLAY_BENCHMARK_PROTOCOL.md`;
3. the relevant existing component contract;
4. the applicable incident records under `docs/development_workflow_incidents/`.

## 2. Problem statement

The repository already contains multiple correct and useful mechanisms, but repeated workflow improvements risk becoming a patch stack: each local problem produces another rule, script, gate, temporary workflow, or recovery convention. A locally reasonable feature can still be globally poor when its maintenance cost exceeds the time or risk it removes.

The optimization objective is therefore not maximal automation. It is:

> Reduce end-to-end human and model effort while preserving or improving every existing correctness, provenance, authority, and scientific-safety guarantee.

A workflow change is valuable only when measured benefit clearly exceeds implementation and maintenance cost.

## 3. Existing components and the problems they solve

### 3.1 Existing safety and correctness kernel

The following components remain independent owners of their current responsibilities:

- **Connected GitHub development route:** branch, Draft PR, exact-head CI, review, and explicit user approval before merge.
- **Code-first pilot-registration fastpath:** compiles one reviewer-authored `DEV_PILOT_REGISTRATION_SPEC.yaml` into deterministic `PREPARED_INPUTS`.
- **V1 dev-branch integration transaction:** owns `plan → prepare → normalize → gate → finalize → local READY`.
- **Handoff authority:** owns schema-v3 delta normalization and materialization of `docs/handoff.md`, registry changes, and generated views.
- **RunSpec/lane runner:** owns registered execution identity and supervised experiment launch.
- **Results-repository delivery and evidence locator:** own durable result delivery and immutable result discovery.

These components answer whether a candidate is correctly specified, provenance-bound, authorized, gated, and safe to publish. A later orchestration layer must call them rather than reimplement them.

### 3.2 Proven value already observed

The current safety kernel has demonstrated real value. In the E7 frozen-critic trajectory-GAE pilot, real-data preparation stopped before any of 192 actor branches started and exposed two integration defects: canonical RunSpec placeholder handling and a mixed-precision validation mismatch. No held-out seeds were accessed and no scientific result was claimed.

This is evidence that preflight and fail-closed gates can prevent expensive invalid execution. It is not yet evidence that the overall workflow is faster or easier.

### 3.3 Problems not yet solved

Recent E7/E8 work still shows some of the following:

- manual transfer of adapter outputs into V1 inputs;
- multiple commands and stage-specific recovery knowledge;
- temporary source-export or importer workflows;
- empty, stale, or rebuilt PRs;
- long-lived development branches used as final integration candidates;
- incomplete machine evidence showing whether the default fastpath was actually used;
- pre-run registration and post-run closure remaining operationally disconnected;
- no reliable measurement of time saved by workflow changes.

These are coordination and adoption problems. They must not be confused with scientific-code bugs, missing data, GPU failures, numerical collapse, or method underperformance.

## 4. Root-cause model

The main architectural hypothesis is:

> The repository has the required domain components, but lacks one low-friction executable composition path across their boundaries.

The fastpath produces deterministic inputs. V1 produces a local `READY` candidate. Today, an operator or coding agent must understand and manually connect their intermediate layouts, command order, retry rules, and branch-promotion conventions. This makes the safe path harder to use than ad hoc alternatives.

The proposed remedy is not another authority or state machine. A possible thin orchestration layer would only:

- invoke existing commands in the accepted order;
- place already generated deterministic inputs at the required locations;
- use existing transaction states to continue, stop, or require a new attempt;
- maintain one persistent workspace per case;
- derive a compact status and timing summary from existing records.

Whether this hypothesis is correct must be established by historical replay before any implementation is adopted.

## 5. Optimization policy: evidence before code

No workflow optimization may become a repository default merely because it appears architecturally cleaner.

Before implementation:

1. identify repeated incidents or measurable loss;
2. define the affected task classes;
3. select representative historical replay cases;
4. freeze correctness and efficiency acceptance criteria;
5. set a production-code and maintenance budget;
6. define rollback and stop conditions.

After a disposable prototype exists, compare it with the current accepted path under the paired replay protocol. The prototype may enter `main` only when it demonstrates correctness equivalence, per-case non-regression, meaningful aggregate time reduction, and bounded complexity.

## 6. Historical replay as the default validation framework

Workflow optimization is treated as an engineering experiment.

For each historical case, freeze:

- historical `main` base;
- frozen implementation identity;
- reviewer-approved input/specification;
- expected changed paths and final semantic state;
- existing gates;
- replay environment and cache policy.

Run two paths:

- **A — accepted baseline:** current documented manual/component path;
- **B — candidate optimization:** proposed wrapper, orchestrator, or other optimization while calling the same owners and gates.

A and B must produce equivalent repository and authority outcomes. The candidate is evaluated on paired elapsed time, active operation time, command count, file-transfer count, retries, temporary artifacts, and failure recovery.

The detailed procedure is defined in `REPLAY_BENCHMARK_PROTOCOL.md`.

## 7. No-regression rule

The desired result is that every in-scope case becomes faster. Because wall-clock measurements contain scheduler and filesystem noise, the formal rule is:

- no case may show a **material regression** greater than `max(60 seconds, 5% of baseline controlled-replay time)`;
- a result within that tolerance is a tie, not an improvement;
- every material slowdown must block universal adoption unless the case was explicitly declared out of scope before implementation;
- post-hoc exclusions are forbidden;
- correctness, safety, or provenance regression has zero tolerance.

A selective path for only some task classes is allowed only when routing is deterministic, predeclared, and materially simpler than maintaining a universal solution. The default preference is one uniform path with no material per-case regression.

## 8. Adoption thresholds

A candidate workflow optimization may be recommended only when all hard conditions pass:

### Correctness and safety

- all replay cases pass semantic and protected-tree equivalence;
- no gate is removed, weakened, skipped, or reinterpreted;
- no scientific variable or result status changes;
- no failure is misreported as `READY`;
- no task-specific hard-coded repair is introduced.

### Efficiency

- paired median controlled-replay time decreases by at least 30%;
- paired mean controlled-replay time also decreases;
- no in-scope case has a material regression;
- manual command count decreases by at least 60%;
- manual copying of intermediate files falls to zero;
- temporary workflow or temporary PR use falls to zero for covered cases.

### Complexity

For the first orchestration prototype:

- production code target: 250–450 lines;
- hard review trigger: more than 500 production lines;
- no new third-party dependency;
- no modification of V1 core, handoff authority, registry schema, scientific code, or GitHub merge behavior;
- no automatic push, PR creation, approval, or merge;
- no new domain state beyond a derived, rebuildable summary.

Crossing a complexity boundary triggers redesign or cancellation, not silent scope expansion.

## 9. Measurement vocabulary

Reports must distinguish:

- **historical real wall time:** first relevant historical action or PR to final accepted outcome;
- **controlled replay wall time:** start of A or B replay to terminal replay state in the same environment;
- **active operation time:** time requiring operator/model actions, excluding unattended machine execution;
- **machine gate time:** test, normalization, and CI execution time;
- **time reduction:** `(baseline - candidate) / baseline`;
- **throughput gain:** `baseline / candidate - 1`.

Example: 30 minutes to 15 minutes is a 50% time reduction and a 100% throughput gain.

Historical real time and controlled replay time must never be merged into one estimate.

## 10. Iteration model

The optimization project proceeds in bounded stages:

1. **Documentation freeze:** problem history, component ownership, replay protocol, thresholds, and stop conditions.
2. **Replay inventory:** identify 6–10 representative historical tasks and recover available timing/evidence.
3. **Baseline replay:** execute the accepted path and record paired baselines.
4. **Disposable prototype:** implement on an isolated dev branch with no default-policy change.
5. **Candidate replay:** run the same cases under the candidate path.
6. **Decision:** adopt, narrow, redesign, or discard based on the frozen thresholds.
7. **Production observation:** if adopted, monitor at least three real tasks and compare with replay estimates.

Each later optimization starts a new bounded iteration. It may reuse the benchmark framework but must not continuously expand one orchestrator to absorb unrelated responsibilities.

## 11. Monitoring the existing system

Existing mechanisms should continue to be evaluated independently of any orchestrator.

For each real use, record when available:

- issue detected or blocked;
- detecting component and stage;
- severity and avoided cost;
- false-positive status;
- whether the safe route was used or bypassed;
- manual fallback reason;
- temporary workflow or PR use;
- final terminal state.

This answers whether the safety kernel continues to catch valuable defects. A future orchestrator is evaluated separately on whether it makes the correct path faster and easier to use.

## 12. Current project state

As of the base commit:

- the safety and registration components remain active;
- a unified lifecycle orchestrator does not exist;
- no orchestrator implementation is authorized by this documentation claim;
- historical GitHub PR, commit, and Actions timing are partially available;
- historical local V1 stage timing is incomplete and will be supplemented by controlled replay;
- the next authorized activity after document review is replay-case inventory and baseline design, not production implementation.

## 13. Related records

- `docs/dev_branch_integration_protocol.md`
- `docs/dev_pilot_registration_fastpath.md`
- `docs/development_workflow_incident_and_improvement_log.md`
- `docs/development_workflow_incidents/README.md`
- `docs/development_workflow_incidents/DEVOPT-2026-07-14-PILOT-REGISTRATION-MERGE-01.md`
- `docs/development_workflow_transitions/GOV-DEV-PILOT-REGISTRATION-FASTPATH-TRANSITION-01.md`
- `docs/scopes/GOV-DEV-PILOT-REGISTRATION-FASTPATH-01.md`
- `docs/scopes/GOV-DEV-PILOT-REGISTRATION-FASTPATH-ACTIVATION-01.md`

## 14. Update discipline

This document is append-preserving. Later decisions should update the current-state and iteration sections while retaining earlier problem statements, rejected alternatives, benchmark results, and rollback records. Do not rewrite history to make a later solution appear inevitable.
