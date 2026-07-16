# Development Workflow Historical Replay Benchmark Protocol

**Claim:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Protocol status:** frozen design candidate; no optimization implementation authorized

## 1. Question under test

For a proposed repository-workflow optimization, determine whether it:

1. produces the same accepted repository, authority, provenance, and gate outcome as the current path;
2. reduces elapsed and active operation time across representative historical tasks;
3. creates no material per-case slowdown;
4. removes more recurring complexity than it adds.

The protocol tests engineering workflow efficiency. It does not produce scientific evidence about DRPO methods.

## 2. Experimental unit

One replay case is one immutable workflow task reconstructed from repository history.

A valid case manifest freezes:

```yaml
case_id: <stable ID>
task_class: <code_only|add_registration|replace_registration|result_closure|stale_recovery|gate_failure>
historical_base_sha: <40-char SHA>
frozen_implementation_sha: <40-char SHA or null for code-only fixture>
source_prs: [<numbers>]
source_commits: [<SHAs>]
input_spec_sha256: <SHA-256>
expected_changed_paths: [<paths>]
expected_final_tree_or_semantic_hashes: {...}
required_gates: [<commands or gate IDs>]
replay_environment_id: <environment snapshot>
cache_policy: <cold|warm|fixed-shared>
replayability: <complete|reconstructed|partial>
predeclared_exclusions: [<external unavailable elements>]
```

The manifest must be frozen before candidate implementation results are inspected.

## 3. Case inventory and sampling

The first benchmark uses 6–10 cases. It must include at least:

- one code-only integration;
- two new experiment registrations;
- one replacement or protocol update;
- one result closure or evidence-locator update;
- one stale-main or failed-attempt recovery;
- both an E7-derived case and an E8-derived case.

Cases are selected for task-class coverage, not for expected candidate success. Post-hoc removal of a slow or failed case is forbidden.

### Replayability classes

- **Complete:** original immutable inputs and necessary repository objects are available.
- **Reconstructed:** a faithful input spec is derived from immutable PR/commit/delta history and then shared unchanged by A and B.
- **Partial:** external data, credentials, or local artifacts are unavailable; only the registration/integration segment is replayed.

A reconstructed spec-generation step occurs before A/B timing and is not charged to either arm.

## 4. Compared paths

### Arm A — accepted baseline

Execute the current documented component path without the proposed optimization. For the pilot-registration case this normally includes:

1. run the existing preparation adapter;
2. place the exact repository overlay;
3. run V1 plan;
4. run V1 prepare;
5. place the exact registration intent and approval inputs;
6. run normalize;
7. run required gates;
8. run finalize.

No intentionally inefficient delays may be introduced.

### Arm B — candidate optimization

Execute the proposed optimization while using the same underlying owners, inputs, gates, environment, and expected output. A thin orchestrator may invoke and connect the existing stages, but may not reimplement or weaken them.

## 5. Controlled environment

For each pair:

- use the same machine class and dependency environment;
- use isolated workspaces;
- use the same historical base and frozen implementation;
- use the same source blobs and reviewer inputs;
- use the same test selection and gate commands;
- use the same network policy;
- use the same cache policy;
- prevent concurrent unrelated load when practicable.

To reduce order and cache bias, alternate execution order across cases:

- odd-numbered cases: A then B;
- even-numbered cases: B then A.

When tests depend heavily on warm caches, run one unmeasured warm-up shared by both arms, then measure both under the same fixed-warm policy. Do not compare a cold baseline with a warm candidate.

## 6. Timing boundaries

Each replay emits monotonic-clock timestamps.

### Controlled replay wall time

Starts immediately before the first workflow command for the arm and ends when the arm reaches its terminal replay state: `READY`, correctly blocked, or correctly stale.

### Active operation time

Sum of intervals requiring an operator or agent to:

- invoke a command;
- interpret a stage result;
- locate or copy intermediate inputs;
- choose a documented recovery action;
- repair only replay-environment setup.

Unattended command execution is excluded. The measurement method must be identical for A and B.

### Machine stage time

Record separately when available:

- preparation adapter;
- plan;
- prepare;
- normalize;
- gate;
- finalize;
- CI or equivalent external gate.

### Historical real wall time

Derived separately from historical PR, commit, comment, and Actions timestamps. It describes past operational experience but is not substituted for controlled A/B time.

## 7. Operation and complexity counters

For every arm record:

- commands explicitly initiated;
- manual file-copy or placement actions;
- workspaces or attempts created;
- full-path restarts;
- temporary branches;
- temporary workflows;
- temporary or empty PRs;
- CI runs;
- unique blocking errors;
- recovery decisions requiring human/model interpretation;
- candidate production lines, tests, dependencies, and modified core components.

## 8. Correctness equivalence

Efficiency is considered only after correctness equivalence passes.

For successful cases, compare as applicable:

- final Git tree SHA;
- changed-path set and file modes;
- registry experiment semantic hash;
- handoff materialized content hash;
- schema-v3 delta semantic content;
- authority verification result;
- selected gate plan;
- individual and aggregate gate conclusions;
- V1 terminal state;
- provenance and source-lock identities.

Commit SHA equality is preferred but not mandatory when timestamps or commit metadata differ. Tree and protected semantic equality are mandatory.

For failure cases, both arms must:

- fail at an equivalent safety boundary;
- preserve required diagnostics;
- avoid a `READY` state;
- avoid partial authority or scientific-state mutation;
- prescribe an equivalent safe recovery class.

## 9. Metrics

For case `i`:

```text
wall_time_reduction_i = (A_wall_i - B_wall_i) / A_wall_i
active_time_reduction_i = (A_active_i - B_active_i) / A_active_i
throughput_gain_i = A_wall_i / B_wall_i - 1
```

Report:

- every paired case result;
- arithmetic mean;
- median;
- minimum and maximum reduction;
- count of improvements, ties, and material regressions;
- task-class breakdown;
- historical-real-time context separately.

Do not report only a mean. A single severe slowdown must remain visible.

## 10. Per-case no-regression rule

A candidate has a material time regression when:

```text
B_controlled_wall > A_controlled_wall + max(60 seconds, 5% of A_controlled_wall)
```

A difference within that tolerance is reported as a tie. The tolerance accounts only for unavoidable measurement noise; it is not an optimization success.

Any material regression in an in-scope case blocks recommendation as the universal default. A case may be out of scope only when declared before implementation and justified by component ownership, not because the candidate performs poorly.

Correctness and safety regressions have zero tolerance.

## 11. Adoption decision

### Recommend universal adoption only when

- all cases pass correctness equivalence;
- zero in-scope cases show material time regression;
- median controlled wall-time reduction is at least 30%;
- mean controlled wall time also improves;
- median active-operation-time reduction is at least 30%;
- explicit command count falls by at least 60%;
- intermediate manual file copies are zero;
- temporary workflows and temporary PRs are zero for covered cases;
- implementation remains within the frozen complexity budget.

### Recommend a narrower task-class path only when

- routing can be decided from immutable input metadata before execution;
- the narrowed path has no material regression within that class;
- the routing logic is simpler than the time it saves;
- baseline remains the explicit path for excluded classes;
- no duplicated authority or validation logic is introduced.

### Reject or redesign when

- any correctness equivalence check fails;
- any safety gate is weakened;
- material regressions occur without a predeclared scope boundary;
- benefits depend on task-specific hard-coded branches;
- production code exceeds 500 lines without new user-approved scope;
- V1 core, handoff authority, registry schema, scientific code, or GitHub merge behavior must change;
- measured time reduction is below the adoption threshold.

## 12. Failure-injection cases

At least three negative replay scenarios are required:

### Main drift

Advance the controlled main reference after preparation. Both arms must reject stale provenance. The candidate must not silently rebase, reuse stale green results, or continue to `READY`.

### Registration before-image mismatch

Provide an incorrect semantic before-image hash. Both arms must fail before authority materialization and publish no partial registration.

### Gate failure

Introduce a deterministic lint or focused-test failure. Both arms must preserve logs, avoid `READY`, and not auto-repair or relax the gate.

Optional additional cases include interrupted normalization, conflicting preparation output, and mutated reviewer approval.

## 13. Prototype development cost accounting

Prototype cost is part of ROI.

Record:

- design and implementation elapsed time;
- production and test line counts;
- files added and modified;
- new dependencies;
- existing component cores touched;
- review defects found;
- expected maintenance owner and rollback cost.

A candidate that saves replay time but adds comparable recurring maintenance cost is not successful.

A simple break-even estimate should be reported:

```text
break_even_tasks = implementation_and_review_time / median_time_saved_per_task
```

This estimate is supplementary; correctness and no-regression gates remain primary.

## 14. Benchmark artifacts

Each benchmark iteration should retain:

```text
docs/development_workflow_optimization/benchmarks/<iteration-id>/
  BENCHMARK_PLAN.yaml
  CASE_INVENTORY.yaml
  ENVIRONMENT.json
  BASELINE_RESULTS.json
  CANDIDATE_RESULTS.json
  PAIRED_COMPARISON.json
  REVIEW.md
  DECISION.md
```

Raw transaction workspaces may remain persistent-local when too large, but hashes and locators must be recorded. Benchmark artifacts are engineering evidence and must not be added to the scientific experiment registry.

## 15. Review sequence

Before implementation:

1. scope and ownership review;
2. case-selection and anti-cherry-picking review;
3. timing-boundary and cache-fairness review;
4. correctness-equivalence review;
5. adoption-threshold and stop-condition review.

After replay:

1. verify all case manifests were frozen before treatment results;
2. verify A/B inputs and environments match;
3. verify every output-equivalence result;
4. inspect every slowdown and failure;
5. calculate paired metrics independently;
6. review code-size and maintenance cost;
7. issue one of `ADOPT`, `NARROW`, `REDESIGN`, or `REJECT`.

## 16. Current authorization boundary

This protocol authorizes documentation, historical-case inventory, and benchmark planning only. It does not authorize an orchestration implementation, changes to default workflow, changes to existing gates, or merge of any candidate optimization.
