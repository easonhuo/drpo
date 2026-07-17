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
historical_task:
  base_sha: <40-char SHA>
  frozen_implementation_sha: <40-char SHA or null>
  source_prs: [<numbers>]
  source_commits: [<SHAs>]
  historical_real_time_evidence: [<locators>]
benchmark:
  toolchain_sha: <40-char SHA>
  input_spec_sha256: <SHA-256>
  expected_changed_paths: [<paths>]
  expected_final_tree_or_semantic_hashes: {...}
  required_gates: [<commands or gate IDs>]
  environment_id: <environment snapshot>
  cache_policy: <cold|fixed_warm>
  replayability: <complete|reconstructed|partial>
  predeclared_exclusions: [<external unavailable elements>]
```

The manifest must be frozen before candidate implementation results are inspected.

## 3. Historical task versus benchmark toolchain

Two different snapshots are required:

- **Historical task snapshot:** the old base, implementation, PRs, commits, and outcome being replayed.
- **Benchmark toolchain snapshot:** the accepted repository tooling used identically by Arm A and Arm B.

The primary A/B comparison isolates the candidate optimization. Therefore both arms use the same benchmark toolchain, validators, gates, dependencies, and environment. Arm A omits only the candidate optimization; Arm B includes it.

Historical real time remains valuable context, but it is not substituted for the controlled baseline. Comparing an old, cold, failure-prone real PR directly against a clean, cached candidate replay would overstate the gain.

A separate benchmark may compare two generations of the safety kernel, but it must declare that different toolchains are the treatment. It cannot be mixed with the orchestration benchmark.

## 4. Case inventory and sampling

The first benchmark uses 6–10 cases. It must include at least:

- one code-only integration;
- two new experiment registrations;
- one replacement or protocol update;
- one result closure or evidence-locator update;
- one stale-main or failed-attempt recovery;
- both an E7-derived case and an E8-derived case.

Cases are selected for task-class coverage, not for expected candidate success. Post-hoc removal of a slow, failed, or inconvenient case is forbidden.

### Replayability classes

- **Complete:** original immutable inputs and necessary repository objects are available.
- **Reconstructed:** a faithful input spec is derived from immutable PR/commit/delta history and then shared unchanged by A and B.
- **Partial:** external data, credentials, or local artifacts are unavailable; only the declared registration/integration segment is replayed.

A reconstructed-spec generation step occurs before A/B timing and is not charged to either arm.

## 5. Compared paths

### Arm A — accepted baseline

Execute the current documented component path without the proposed optimization. For a pilot-registration case this normally includes:

1. run the existing preparation adapter;
2. place the exact repository overlay;
3. run V1 plan;
4. run V1 prepare;
5. place the exact registration intent and approval inputs;
6. run normalize;
7. run required gates;
8. run finalize.

No intentionally inefficient delay, redundant check, or artificial mistake may be added to make the baseline slower.

### Arm B — candidate optimization

Execute the candidate while using the same underlying owners, inputs, gates, environment, and expected output. A thin orchestrator may invoke and connect existing stages, but may not reimplement, skip, reinterpret, or weaken them.

## 6. Controlled environment

For each case:

- use the same machine class and dependency environment;
- use isolated workspaces;
- use the same historical task inputs;
- use the same benchmark toolchain SHA;
- use the same source blobs and reviewer inputs;
- use the same test selection and gate commands;
- use the same network policy;
- use the same cache policy;
- prevent unrelated concurrent load when practicable;
- record CPU, memory, filesystem, Python, Git, and dependency identity.

### Cache policy

Use either:

- **cold:** clean comparable caches before each measured arm; or
- **fixed warm:** perform one unmeasured shared warm-up, then preserve the same declared cache state for both arms.

Do not compare a cold baseline with a warm candidate.

## 7. Repetition and execution order

Time is a primary outcome, so one run per arm is insufficient.

For every case, perform at least two measured pairs:

1. Pair 1: A then B;
2. Pair 2: B then A.

Use the per-arm median across repetitions as the case time. This cancels first-run and order effects better than alternating order only across different cases.

Run a third repetition for each arm when either condition holds:

- the two measurements for one arm differ by more than `max(60 seconds, 5% of their median)`;
- a transient external event, resource contention, or cache inconsistency is documented.

A repetition may be discarded only for a predeclared environmental invalidation such as machine interruption. The invalidation and replacement run must remain in the raw log. Candidate failure or slowness is not an environmental invalidation.

## 8. Timing boundaries

Every measured arm uses a monotonic clock.

### Controlled replay wall time

Starts immediately before the first workflow command and ends when the arm reaches its declared terminal replay state:

- `READY` for a successful case;
- correctly `BLOCKED` for a deterministic failure case;
- correctly `STALE` for a drift case.

The primary benchmark ends at local transaction state because the candidate does not own publication, user approval, or merge. GitHub PR and Actions time are recorded separately as operational context.

### Active operation time

Active time is measured from an event log, not estimated after the fact. Each operator or agent action records:

```json
{
  "event": "invoke_command|inspect_result|place_input|choose_recovery|environment_repair",
  "start_monotonic_ns": 0,
  "end_monotonic_ns": 0,
  "case_id": "...",
  "arm": "A|B",
  "repetition": 1,
  "note": "..."
}
```

Unattended command execution is excluded. The same event categories and capture method apply to both arms.

### Machine stage time

Record separately:

- preparation adapter;
- plan;
- prepare;
- normalize;
- gate;
- finalize;
- optional external CI.

### Historical real wall time

Derive historical time from immutable PR, commit, comment, workflow, and job timestamps. Record the start/end definition for every case. Historical time describes the real past workflow; it is not the controlled causal estimate.

## 9. Operation and complexity counters

For every arm record:

- commands explicitly initiated;
- manual file-copy or placement actions;
- workspaces or attempts created;
- full-path restarts;
- temporary branches;
- temporary workflows;
- temporary or empty PRs;
- local gate runs and external CI runs;
- unique blocking errors;
- recovery decisions requiring human/model interpretation.

For the candidate also record:

- production and test lines;
- files added or modified;
- new dependencies;
- existing component cores touched;
- task-specific branches;
- persistent artifacts introduced.

## 10. Correctness equivalence

Efficiency is considered only after correctness equivalence passes.

For successful cases compare, as applicable:

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

For failure cases both arms must:

- fail at an equivalent safety boundary;
- preserve required diagnostics;
- avoid `READY`;
- avoid partial authority or scientific-state mutation;
- prescribe an equivalent safe recovery class.

## 11. Metrics

For case `i`, use the per-arm medians:

```text
wall_time_reduction_i = (median_A_wall_i - median_B_wall_i) / median_A_wall_i
active_time_reduction_i = (median_A_active_i - median_B_active_i) / median_A_active_i
throughput_gain_i = median_A_wall_i / median_B_wall_i - 1
```

Report:

- every raw repetition;
- every paired case result;
- arithmetic mean across cases;
- median across cases;
- minimum and maximum reduction;
- count of improvements, ties, and material regressions;
- task-class breakdown;
- historical-real-time context separately.

Do not report only a mean. A single severe slowdown must remain visible.

## 12. Per-case no-regression rule

A candidate has a material time regression when:

```text
median_B_controlled_wall > median_A_controlled_wall + max(60 seconds, 5% of median_A_controlled_wall)
```

A difference within that tolerance is a tie, not an improvement. The desired outcome remains that every in-scope case is faster.

Any material regression blocks recommendation as the universal default. A case may be out of scope only when declared before candidate implementation and justified by component ownership, not because the candidate performs poorly.

Correctness, safety, and provenance regressions have zero tolerance.

## 13. Adoption decision

### Recommend universal adoption only when

- all cases pass correctness equivalence;
- zero in-scope cases show material time regression;
- median case-level controlled wall-time reduction is at least 30%;
- mean controlled wall time also improves;
- median active-operation-time reduction is at least 30%;
- explicit command count falls by at least 60%;
- intermediate manual file copies are zero;
- temporary workflows and temporary PRs are zero for covered cases;
- implementation remains within the frozen complexity budget.

### Recommend a narrower task-class path only when

- scope was declared before implementation;
- routing is decided from immutable input metadata before execution;
- the narrowed path has no material regression within that class;
- routing logic is simpler than the time it saves;
- the baseline remains explicit for excluded classes;
- no duplicated authority or validation logic is introduced.

### Reject or redesign when

- any correctness equivalence check fails;
- any safety gate is weakened;
- material regressions occur without a predeclared scope boundary;
- benefits depend on task-specific hard-coded branches;
- production code exceeds 500 lines without new user-approved scope;
- V1 core, handoff authority, registry schema, scientific code, or GitHub merge behavior must change;
- measured time reduction is below the adoption threshold.

## 14. Failure-injection cases

At least three negative replay scenarios are required.

### Main drift

Advance the controlled main reference after preparation. Both arms must reject stale provenance. The candidate must not silently rebase, reuse stale green results, or continue to `READY`.

### Registration before-image mismatch

Provide an incorrect semantic before-image hash. Both arms must fail before authority materialization and publish no partial registration.

### Gate failure

Introduce a deterministic lint or focused-test failure. Both arms must preserve logs, avoid `READY`, and not auto-repair or relax the gate.

Optional cases include interrupted normalization, conflicting preparation output, and mutated reviewer approval.

## 15. Prototype development cost and ROI

Record:

- design and implementation elapsed time;
- review and repair elapsed time;
- production and test line counts;
- files added and modified;
- new dependencies;
- existing component cores touched;
- review defects found;
- expected maintenance owner and rollback cost.

A simple break-even estimate is:

```text
break_even_tasks = implementation_and_review_time / median_time_saved_per_task
```

Report break-even tasks and expected task frequency. A candidate that saves replay time but adds comparable recurring maintenance cost is not successful.

## 16. Minimal benchmark artifacts

Avoid turning measurement into another framework. Each iteration retains only:

```text
docs/development_workflow_optimization/benchmarks/<iteration-id>/
  CASE_INVENTORY.yaml       # frozen plan, cases, environment, thresholds
  RAW_RESULTS.jsonl         # repetitions, events, stage timings, counters
  PAIRED_COMPARISON.json    # derived metrics and equivalence checks
  DECISION.md               # review, defects, ROI, adoption decision
```

Large transaction workspaces may remain persistent-local, but their hashes and locators must be recorded. These are engineering artifacts and must not enter the scientific experiment registry.

No database, dashboard, service, or blocking CI is authorized for the first iteration.

## 17. Review sequence

Before candidate implementation:

1. scope and ownership review;
2. case-selection and anti-cherry-picking review;
3. historical-task and benchmark-toolchain separation review;
4. timing, repetition, cache, and order fairness review;
5. correctness-equivalence review;
6. adoption-threshold and stop-condition review.

After replay:

1. verify all manifests and thresholds were frozen before candidate results;
2. verify A/B inputs, toolchain, environment, cache, and gates match;
3. verify all repetitions and invalidations are present;
4. verify every output-equivalence result;
5. inspect every slowdown and failure;
6. independently calculate paired metrics;
7. review code size, complexity, and break-even;
8. issue one of `ADOPT`, `NARROW`, `REDESIGN`, or `REJECT`.

## 18. Current authorization boundary

This protocol authorizes documentation, historical-case inventory, timing recovery, and benchmark planning only. It does not authorize an orchestration implementation, a telemetry system, changes to default workflow, changes to existing gates, or merge of any candidate optimization.
