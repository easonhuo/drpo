# DRPO Development Workflow Incident and Improvement Log

中文名：**DRPO 日常开发流程问题与优化记录**

## 1. Document purpose

This document records engineering-process incidents, root-cause analyses, and proposed workflow improvements in chronological order.

It is intentionally separate from scientific experiment status:

- `docs/handoff.md` remains the unique research master;
- `experiments/registry.yaml` remains the experiment registry;
- this log does not change scientific claims, experiment status, seeds, thresholds, or execution priority;
- entries describe development-process evidence and proposed optimizations only;
- an optimization listed here is not active until it is implemented, tested, reviewed, and merged through the normal repository route.

The primary audience is a later development session that will improve the repository workflow without having to reconstruct the incident from chat history.

## 2. Entry format

Every new entry should contain:

1. date and incident ID;
2. affected task, experiment, or claim;
3. intended change and expected complexity;
4. observed elapsed time and critical-path delays;
5. chronological incident record;
6. root causes;
7. scientific or engineering impact;
8. immediate fixes already completed;
9. proposed systemic improvements;
10. acceptance tests and success metrics;
11. unresolved questions and implementation status.

Do not rewrite or delete old entries. Add a later correction entry when an earlier diagnosis changes.

## 3. Workflow metrics to record

Future sessions should record these timestamps or counts whenever practical:

- `task_start_utc`;
- `first_science_patch_utc`;
- `command_contract_pass_utc`;
- `real_data_liveness_pass_utc`;
- `server_launch_ready_utc`;
- `draft_pr_open_utc`;
- `merge_ready_utc`;
- `merged_utc`;
- number of code commits before launch readiness;
- number of CI reruns;
- number of defects caught locally, in CI, and on the server;
- time spent on scientific implementation, execution plumbing, governance, CI queue, and rework;
- whether the final result used the exact reviewed commit.

The main optimization target is not “fewest tests.” It is shorter time to a correctly reviewed launch while preserving scientific and provenance gates.

---

# 2026-07-14 — DEVOPT-2026-07-14-E7-SQUARED-EXP-01

## 4. Context

Affected experiment:

```text
EXT-H-E7-SQUARED-EXP-NIGHT-01
```

User-observed development duration:

```text
53 minutes 33 seconds
```

The scientific delta appeared small:

1. replace the linear-distance EXP coordinate with squared remoteness;
2. add analytic Gaussian KL diagnostics and KL-triggered old-policy refresh;
3. define several `c` settings;
4. run all branches for 1M updates.

The user correctly questioned why this required almost one hour before the experiment was ready.

## 5. What the task actually expanded into

The implementation did not remain a four-line scientific change. It included:

- a dedicated successor runner;
- a squared-EXP kernel installation context;
- three actor paths: A2C, fixed-K4 PPO, and adaptive-KL PPO;
- KL diagnostics and reference-refresh accounting;
- geometry diagnostics;
- branch generation and unique branch identities;
- 1M/500k/late-window reporting;
- terminal aggregation;
- one-click, liveness, resume, and resource-autotune entry points;
- RunSpecs;
- schema-v3 registration preparation;
- result deposition and terminal-audit integration.

Some of this work was necessary for a durable 126-branch experiment. The process error was that nearly all of it was placed on the critical path before the first trustworthy launch.

## 6. Chronological incident record

### 6.1 Launch semantics were initially coupled to registration

The intended workflow was code-first:

1. finish the reviewed development branch;
2. pass liveness;
3. start the development pilot;
4. materialize registration in parallel.

The first implementation instead treated missing registration as a launch blocker. This inverted the intended dependency and caused a preventable rewrite of the one-click path.

### 6.2 The generated trainer variant was invalid

The branch command used:

```text
--variant iqlv_squared_exp_night
```

The canonical trainer supported:

```text
--variant iqlv_exp_rank
```

Module-level tests passed because they did not execute the complete path:

```text
runner
  -> generated branch command
  -> bootstrap
  -> canonical trainer argparse
```

The server-local run exposed the defect. The source was corrected and a regression test was later added.

### 6.3 Too many work streams were serialized

The science path, runner, aggregator, runtime autotune, registration, RunSpec, and archive preparation were developed largely as one serial block.

A frozen interface between these components would have allowed:

- science implementation and command-contract testing to finish first;
- server liveness to start;
- aggregator, registration, and archive work to proceed in parallel while the experiment ran.

### 6.4 Commit and CI granularity became too fine

Repeated small changes triggered repeated PR checks. This increased elapsed time and made it harder to identify the true final reviewed head.

The stable integration layout ultimately became:

1. one implementation commit containing code, config, tests, and RunSpec pins;
2. one immutable registration/result commit containing the schema-v3 delta, materialization report, materialized handoff, registry update, and compact evidence.

That structure should have been designed before the first registration attempt.

### 6.5 Stage-5 merge-history verification rejected a normal PR integration

The authority verifier originally required the side-branch delta-add commit itself to appear on the `main` first-parent chain. A normal GitHub merge commit integrates the side branch without placing the side-branch add commit directly on the first-parent chain.

This produced a false rejection during registration. The issue was repaired separately in PR #56 by mapping a unique side-branch add commit to its first-parent merge-integration commit while preserving delta and materialization-report immutability.

This was a pre-existing governance defect exposed by the experiment. It was not caused by squared-EXP or KL logic, but it extended the closure path.

### 6.6 Materialization-report immutability was not reflected in the build helper

The Stage-5 authority contract requires the schema-v3 delta and its `MATERIALIZATION_REPORT.json` to be immutable. Early attempts added or modified the report in a later commit, which correctly failed authority verification.

The final solution rebuilt the branch so that:

- the delta and report were added together in one registration commit;
- neither file was touched afterward;
- the materialized handoff and generated views corresponded to that exact registration image.

The repository lacked a routine helper that made this correct layout the easy default.

### 6.7 RunSpec commit pins conflicted with branch reconstruction

When the branch was rebuilt onto the new `main`, the old RunSpecs still pointed to a previous implementation commit that was no longer an ancestor of the reconstructed head.

The final layout required:

1. create the implementation commit first;
2. write its exact SHA into both RunSpecs;
3. create the immutable registration commit on top.

The dependency is deterministic but was not encoded in a canonical builder.

### 6.8 A temporary validation workflow lacked PyTorch

One temporary registration workflow installed `pytest` and `ruff` but not `torch`. E7 tests therefore failed during test collection with `ModuleNotFoundError: torch`.

This was an environment-definition defect, not a scientific test failure. It still consumed an additional workflow cycle.

### 6.9 Final clean integration succeeded

The final branch used a clean two-commit structure, passed the exact-head PR Gate, and merged as:

```text
db663872564547f73a20b633bd231f76785a2a2d
```

The experiment result was registered and the compact evidence was deposited. This successful endpoint does not erase the avoidable rework above.

## 7. Root-cause analysis

### RC-1: no explicit “launch-ready minimum slice”

The task had no written critical path separating what was necessary to begin a trustworthy development run from what was necessary for merge and archival closure.

### RC-2: tests validated modules, not the executable command contract

The most likely integration fault—an invalid CLI variant—was outside the tested boundary.

### RC-3: launch, registration, merge, and archival states were insufficiently decoupled

The repository distinguishes these states conceptually, but the implementation workflow did not consistently optimize them as separate milestones.

### RC-4: schema-v3 integration invariants were enforced by validators but not generated by a canonical builder

The system could reject a bad registration but did not yet make a correct registration layout simple and deterministic.

### RC-5: commit-SHA dependencies were discovered late

RunSpecs require an exact implementation SHA, while immutable registration must be a later commit. The two-commit dependency should be part of the standard design template.

### RC-6: CI dependencies and test tiers were not declared per workflow

A temporary workflow guessed its Python dependencies instead of consuming a canonical E7 test environment definition.

### RC-7: repeated micro-commits repeatedly paid full validation cost

The process lacked a debounce or local-finalization phase before launching the expensive exact-head gate.

## 8. Impact

The incident caused:

- longer time to first trustworthy server launch;
- avoidable server-side discovery of a command-generation defect;
- repeated CI cycles;
- temporary workflows and branch reconstruction;
- higher risk of provenance confusion between reviewed, launched, and registered commits;
- reviewer effort spent on workflow repair rather than scientific analysis.

No evidence currently indicates that the final scientific matrix or completed result was corrupted. The result remains capped at a development `pilot` for its independently documented two-seed and source-provenance limitations.

## 9. Immediate fixes already completed

The following fixes are already in the repository:

1. trainer plumbing uses `variant=iqlv_exp_rank`;
2. a variant-plumbing regression test covers that exact correction;
3. Stage-5 authority recognizes schema-v3 deltas integrated through normal GitHub merge commits;
4. the final experiment integration uses an implementation commit followed by an immutable registration/result commit;
5. the exact-head full PR Gate passed before merge;
6. compact result evidence and terminal audit are deposited.

These fixes solve the observed instance. They do not yet implement the full workflow optimization below.

## 10. Proposed systemic optimizations

### DEVOPT-A: launch-ready minimum slice

For a successor that preserves the canonical trainer and changes only a bounded scientific dimension, define the critical path as:

1. frozen proposal and matrix;
2. minimal science code;
3. branch-command generation;
4. end-to-end command-contract test;
5. short real-data liveness;
6. server launch.

Aggregator enhancements, full documentation, registration materialization, and result archival may proceed in parallel after liveness, provided the run is explicitly labelled as a development pilot and the launch snapshot is recorded.

Target for this class of task:

```text
server launch-ready within 20 minutes, excluding external CI queue time
```

This is a process target, not permission to skip required checks.

### DEVOPT-B: canonical command-contract gate

Add a reusable test helper that verifies:

```text
runner config
  -> branch expansion
  -> exact command list
  -> bootstrap parser
  -> canonical trainer parser
  -> bounded model construction/update
  -> required diagnostic file
```

At minimum it must catch:

- unsupported `--variant` values;
- missing or duplicated flags;
- placeholder fields that survive expansion;
- incorrect branch-specific KL or weight parameters;
- invalid dataset or output paths;
- mismatched step/evaluation values.

The gate should run before a server launch command is shown to the user.

### DEVOPT-C: standard two-commit experiment integration builder

Create a canonical helper for this structure:

#### Commit 1 — implementation commit

Contains:

- code;
- config;
- tests;
- scripts;
- RunSpecs pinned to this commit.

#### Commit 2 — immutable registration/result commit

Contains:

- registry change;
- one schema-v3 delta;
- `MATERIALIZATION_REPORT.json` added in the same commit as the delta;
- materialized `docs/handoff.md`;
- refreshed generated views;
- compact result evidence when available.

The helper must refuse to proceed when the delta or report already has history, when a RunSpec SHA is not an ancestor, or when the trusted current-main normalizer is unavailable.

### DEVOPT-D: registration preflight

Before opening the final PR Gate, automatically verify:

- exact current `main` SHA;
- implementation commit ancestry;
- RunSpec pins;
- exactly one schema-v3 delta for the update;
- delta/report same-add-commit requirement;
- report and delta immutability;
- registry assertion coverage;
- handoff replay and generated-view refresh;
- no temporary workflow or failure log remains in the final tree.

The preflight should produce one machine-readable report attached to the PR or saved under a temporary CI artifact, not committed as new scientific evidence.

### DEVOPT-E: tiered CI with final-head debounce

Use three tiers:

1. **Edit gate:** targeted tests, command contract, compile, and Ruff for changed files;
2. **Launch gate:** real-data liveness and required environment checks;
3. **Merge gate:** authority, governance, full pytest, full Ruff, and final diff review.

The full merge gate should run on a deliberate finalization commit rather than every exploratory micro-commit. A changed head after finalization must rerun it, but ordinary edits should not repeatedly pay the full cost.

### DEVOPT-F: canonical test dependency profiles

Define reusable dependency profiles, for example:

- `governance-minimal`;
- `e7-cpu-tests` including PyTorch;
- `countdown-tests`;
- `full-repository`.

Temporary or specialized workflows should import a profile rather than manually guessing packages.

### DEVOPT-G: parallel work streams after interface freeze

After the branch schema and output contract are frozen, split work into:

1. science/runner/liveness;
2. aggregation/terminal audit;
3. registration/provenance;
4. runtime-resource and resume tooling.

Only the first stream blocks initial development launch. The other streams must converge before result closure and merge.

### DEVOPT-H: workflow telemetry

Persist development-process metrics in a small machine-readable CI artifact or PR comment:

```json
{
  "task_start_utc": "...",
  "command_contract_pass_utc": "...",
  "liveness_pass_utc": "...",
  "launch_ready_utc": "...",
  "merge_ready_utc": "...",
  "ci_reruns": 0,
  "server_found_defects": 0,
  "rework_commits": 0
}
```

This data should be used to decide whether expensive gates are catching real problems or creating low-value delay. It must not be used to pressure scientific runs into skipping terminal audits.

## 11. Acceptance criteria for the optimization task

A later session implementing these proposals should demonstrate all of the following on a low-risk real experiment successor or synthetic fixture:

1. an invalid trainer variant is rejected before server launch;
2. a correct minimal successor reaches launch-ready state without requiring registry materialization first;
3. the final branch can be represented as one implementation commit plus one immutable registration commit;
4. RunSpecs point to the implementation commit and remain valid ancestors of the final head;
5. schema-v3 delta and materialization report are added together and never modified;
6. normal GitHub merge-commit integration passes authority verification;
7. the E7 focused test workflow has all required dependencies, including PyTorch;
8. no temporary maintenance workflow remains in the final diff;
9. the final exact-head gate runs once unless a real failure or head change requires another run;
10. scientific gates, held-out seed rules, terminal audit, and explicit merge approval remain intact.

## 12. Non-goals

The optimization must not:

- weaken handoff or registry authority;
- permit direct editing of protected `docs/handoff.md`;
- remove full pytest or governance checks without evidence from logged gate value;
- auto-merge a PR;
- allow unreviewed held-out execution;
- convert a smoke test into a scientific result;
- hide task collapse, support/variance boundary events, or NaN/Inf failures;
- combine GLM implementation responsibility with ChatGPT scientific review responsibility.

## 13. Recommended implementation order for the next session

1. add the reusable command-contract test helper;
2. add the two-commit registration preflight/builder in dry-run mode;
3. add canonical dependency profiles;
4. add final-head CI tiering and telemetry;
5. test the workflow on a low-risk successor with no scientific-variable changes;
6. review interception value and elapsed-time data before removing any existing gate.

## 14. Current status

```text
Documented only
Not implemented
Not registered as a governance change
No workflow behavior changed by this file
```

The next session should treat this entry as an engineering handoff and independently review the proposed design against current `main`, `AGENTS.md`, `docs/handoff.md` Section 0, and governance stage status before changing code.

---

# 2026-07-15 — DEVOPT-2026-07-15-E7-PREDECESSOR-CLOSURE-GAP-01

## Context

Affected experiments:

```text
EXT-H-E7-SQEXP-ACTOR-DECISION-01
EXT-H-E7-SQEXP-HIGHC-BOUNDARY-01
```

The predecessor was a 192-branch development pilot covering A2C and PPO-K4-KL, Positive-only, linear `c=12`, squared-EXP `c={4,8,16,32,64,128}`, three datasets, and seeds `200--203`. The successor added only squared-EXP `c={256,512}` over the same actor, dataset, and development-seed axes.

The immediate symptom appeared when the successor result was reviewed: exact `c64/c128` values could not be recovered from `docs/handoff.md`, `experiments/registry.yaml`, the successor result package, or the currently available repository tree. Chat history contained rounded summaries, but not the predecessor raw aggregate needed for exact cross-experiment comparison.

This entry records a workflow incident only. It does not add or revise any scientific result.

## Intended workflow and expected complexity

The user explicitly approved a code-first workflow:

```text
review and freeze implementation
-> launch development pilot on the server
-> terminal-audit and package the result
-> register and deposit compact evidence afterward
```

Code-first launch was intentional and valid. The missing step was the mandatory post-run closure.

The intended successor comparison was not merely `c512-c256`. The scientific decision depended on the full historical boundary:

```text
Positive-only -> c64 -> c128 -> c256 -> c512
```

Expected repair complexity:

- **artifact recovery and one-off result closure:** low to medium, if the predecessor raw-complete package still exists;
- **systemic dependency/closure gate:** medium, because it must preserve fast code-first launch while preventing a dependent successor from relying on chat-only historical context.

## Observed duration and critical-path delay

No trustworthy end-to-end timestamps were recorded for the post-run closure phase. That absence is itself part of the incident.

The delay became visible only after the successor had completed. At that point, exact predecessor comparison required reconstructing an older artifact rather than reading a compact repository deposition. The critical path moved from ordinary result interpretation to artifact forensics.

## Chronological incident record

### 1. The predecessor was intentionally implemented without registration

Draft PR #69 implemented `EXT-H-E7-SQEXP-ACTOR-DECISION-01` on frozen implementation/result identity:

```text
d1afb5ff094f69986e0ecc3bf7f9385485add62b
```

The PR explicitly excluded `docs/handoff.md`, `experiments/registry.yaml`, and schema-v3 registration material. That was consistent with the approved code-first launch policy.

### 2. The predecessor aggregator produced the required exact data

The predecessor implementation defines and writes:

```text
aggregate/branch_results.csv
aggregate/group_summary.csv
aggregate/actor_comparisons.csv
aggregate/ppo_retention_gate.json
aggregate/terminal_audit.json
aggregate/aggregate_summary.json
```

`group_summary.csv` contains exact dataset-by-actor-by-control four-seed aggregates, including late mean, seed standard deviation, final mean, best score, BEST-to-FINAL drop, late slope, temporal variability, and effective negative mass.

Therefore the numerical information was produced by the experiment design. The confirmed failure is not “the experiment never computed c64/c128.” The failure is that the output was not deposited into durable repository-accessible state.

### 3. The run completed, but the promised post-run closure did not occur

The predecessor was subsequently discussed as a completed 192-branch result, including a failed PPO-retention gate and a high-`c` A2C trend. However, no corresponding compact result deposition, registry closure, or handoff result entry is present on the reviewed `main` tree.

The workflow effectively stopped at:

```text
implementation frozen
-> server result completed
-> result discussed in chat
```

instead of continuing through:

```text
terminal audit
-> raw-complete package
-> durable delivery
-> compact deposition
-> registry/handoff closure
```

### 4. A dependent successor was approved before predecessor closure

The high-`c` successor was designed because the then-current optimum remained at the search boundary. Its initial code-first implementation was bound to:

```text
6795aa6f086c44e8073c5a995a1612f334a3a067
```

The successor correctly reran only the new `c256/c512` branches, avoiding wasteful repetition of existing Positive-only, `c64`, and `c128` branches.

However, predecessor closure was not made a launch, aggregation, review, or merge dependency for this successor.

### 5. The successor described historical joining but did not implement it

The successor aggregator performs exact within-package `c512-c256` comparisons. It also states that predecessor `c128` and Positive-only results must be joined as separately identified historical context.

That requirement remained prose. The aggregator had no input contract for:

- predecessor artifact path;
- predecessor experiment ID;
- predecessor implementation/run SHA;
- predecessor terminal-audit hash;
- predecessor `group_summary.csv` hash;
- exact join-key validation;
- missing-predecessor fail-closed behavior.

Consequently, the successor package was internally complete but scientifically incomplete for the intended boundary decision.

### 6. Existing tests passed because they validated the narrower implemented contract

The high-`c` tests and terminal audit checked:

- all 48 new branches;
- all 12 dataset-by-actor-by-control groups;
- six A2C/PPO comparisons;
- 24 paired `c512-c256` rows;
- held-out-seed exclusion;
- NaN/Inf separation;
- no unauthorized common-`c`, PPO, GAE, or `c1024` selection.

They did not check that the historical `c128`/Positive-only context required by the interpretation text was actually supplied and hash-bound.

The tests therefore passed the implemented narrow contract while missing the wider scientific dependency.

### 7. The missing closure was discovered only during result interpretation

When exact `c64/c128` values were requested, the current handoff and repository did not contain them. The initially available response used rounded chat summaries and, in one case, an inferred value derived from a pooled mean.

That was incorrect analytical behavior. Missing predecessor evidence should have caused a fail-closed comparison, not numerical reconstruction from rounded conversation context. Those approximate cross-package values are withdrawn and must not be treated as evidence.

## Root causes

### RC-1: code-first was implemented as “registration optional” rather than “registration deferred but mandatory”

The workflow correctly removed registration from the pre-launch critical path, but it did not create a mandatory post-run transition or owner.

### RC-2: no machine-readable post-run closure state machine

The repository distinguishes concepts such as raw-complete, terminal-audited, packaged, delivered, registered, and applied-to-repository, but this pilot path did not enforce the sequence as a dependency graph.

### RC-3: successor dependency was expressed in prose rather than an executable contract

“Join predecessor context later” did not identify exact required files, hashes, join keys, or failure semantics.

### RC-4: compact result deposition was not a required predecessor deliverable

The raw server result could exist while the repository retained no small, durable decision table. The system therefore depended on a large external artifact and chat memory for routine comparison.

### RC-5: no durable artifact locator and checksum binding

The reviewed tree did not provide a canonical predecessor artifact URI/location, SHA-256, or recovery instruction sufficient to retrieve the exact result later.

### RC-6: no fail-closed evidence rule for cross-experiment analysis

The reviewer path did not explicitly forbid replacing missing aggregate data with rounded summaries, remembered values, or algebraic inference.

### RC-7: closure ownership was ambiguous

The user had assigned implementation and execution to the development path and registration/review closure to the reviewer path. The closure task was not persisted as an accountable blocker after the run finished.

## Impact

Confirmed engineering impact:

- exact `c64/c128` values were unavailable from the authoritative repository state during successor review;
- `c128->c256` paired comparisons, confidence summaries, final-score comparisons, and stability diagnostics could not be recomputed exactly;
- the successor's main boundary interpretation was delayed by artifact recovery work;
- chat summaries temporarily became a de facto secondary result store;
- an approximate value was initially inferred where the correct behavior was to stop.

Scientific boundary:

- there is no evidence that the predecessor training run or its original aggregate computation was numerically corrupted;
- there is no evidence that `c256/c512` branches are corrupted;
- the incident prevents a trustworthy exact cross-package ranking until the predecessor artifact is recovered;
- it does not authorize rerunning the predecessor, changing seeds, changing thresholds, or filling missing values from memory;
- task-performance collapse, support/variance boundary, and NaN/Inf remain separate event classes.

## Immediate fixes completed in this review

1. The missing-handoff symptom was traced to a skipped post-run closure rather than to absent aggregation code.
2. The predecessor aggregator outputs required for recovery were identified.
3. The successor's historical-join statement was verified to be non-executable prose.
4. Approximate or inferred `c64/c128` values were explicitly withdrawn as evidence.
5. The correct next step was frozen as artifact recovery and exact joining, not automatic rerun.
6. This incident and proposed remedy were added to the durable engineering log.

No workflow behavior, registry state, handoff state, scientific claim, or experiment status is changed by this entry.

## Proposed systemic optimization

Proposed implementation claim:

```text
GOV-DEV-POSTRUN-CLOSURE-DEPENDENCY-GATE-01
```

This claim is a proposal only and is not authorized or active through this log entry.

### DEVOPT-I: mandatory deferred-closure state machine

Preserve code-first launch, but require the following post-run milestones to be recorded independently:

```text
RUN_COMPLETE
-> TERMINAL_AUDITED
-> RAW_COMPLETE_PACKAGED
-> DURABLY_DELIVERED
-> COMPACT_RESULT_DEPOSITED
-> REGISTERED_OR_EXPLICITLY_CLOSED_AS_UNREGISTERED_PILOT
```

A pilot may launch before registration. It may not silently remain indefinitely between `RUN_COMPLETE` and closure.

### DEVOPT-J: dependent-successor gate

Every successor must declare one of:

```text
predecessor_dependency: none
predecessor_dependency: required_for_execution
predecessor_dependency: required_for_interpretation
predecessor_dependency: required_for_merge_or_result_closure
```

For an interpretation-dependent successor such as the high-`c` boundary experiment, launch may proceed when scientifically justified, but final aggregation/reviewer decision must fail closed until the required predecessor evidence is available.

### DEVOPT-K: executable historical-result input contract

A successor that consumes predecessor results must require and validate:

```yaml
predecessor_experiment_id: EXT-H-E7-SQEXP-ACTOR-DECISION-01
predecessor_implementation_sha: <full sha>
predecessor_result_artifact_sha256: <sha256>
predecessor_terminal_audit_sha256: <sha256>
predecessor_group_summary_sha256: <sha256>
join_keys:
  - dataset
  - actor_update_mode
  - control
  - seed
```

The aggregator must reject missing files, identity mismatches, duplicate keys, incomplete seed sets, altered metric definitions, and incompatible horizons/windows.

### DEVOPT-L: mandatory compact result deposition

For matrix pilots, repository closure should retain a compact, human- and machine-readable set even when raw trajectories remain external:

```text
branch_results.csv or a complete per-branch compact summary
group_summary.csv
paired_comparisons.csv
selection_gate.json, when applicable
terminal_audit.json
aggregate_summary.json
ARTIFACT_INDEX.json
```

The compact set must be sufficient to reproduce every reported table without reopening full trajectories.

### DEVOPT-M: durable artifact locator

`ARTIFACT_INDEX.json` should record:

- experiment ID;
- implementation SHA and run SHA;
- result-package SHA-256;
- durable storage location or recovery procedure;
- compact-file SHA-256 values;
- package kind;
- whether raw trajectories/checkpoints are intentionally external;
- whether the artifact has been delivered and independently verified.

A local path that existed only on the execution server is not a durable locator.

### DEVOPT-N: fail-closed cross-experiment analysis rule

When exact predecessor evidence is absent, the reviewer must report:

```text
comparison unavailable: predecessor evidence not recovered
```

The reviewer must not substitute:

- chat memory;
- rounded prose summaries;
- values inferred from pooled means;
- screenshots without raw values;
- a different experiment with similar controls.

Approximate descriptive context may be quoted only when clearly labelled non-authoritative and must not enter a result table, paired comparison, confidence interval, or method-selection decision.

### DEVOPT-O: explicit closure owner and automatic reminder/blocker

The result-review owner must be recorded when a code-first pilot launches. On `RUN_COMPLETE`, the system should create or update a persistent closure item containing:

- exact result directory/package;
- terminal-audit status;
- missing closure milestones;
- responsible reviewer;
- successor experiments currently depending on the result.

The item closes only after compact deposition and registry/handoff action, or after an explicit reviewed decision that the pilot remains unregistered with a durable artifact index.

### DEVOPT-P: closure telemetry

Add timestamps:

```text
run_complete_utc
terminal_audit_pass_utc
raw_package_created_utc
durable_delivery_verified_utc
compact_deposition_commit_utc
registration_commit_utc
```

Track:

- time from run completion to compact deposition;
- number of successors opened before predecessor closure;
- number of analyses blocked by missing artifacts;
- number of results reconstructed from non-authoritative context, with a target of zero.

## Acceptance criteria

A later implementation of `GOV-DEV-POSTRUN-CLOSURE-DEPENDENCY-GATE-01` must demonstrate:

1. a code-first pilot can still pass liveness and launch without prior registry materialization;
2. `RUN_COMPLETE` creates a persistent, review-owned closure state rather than ending the workflow;
3. a predecessor with missing compact evidence cannot be marked fully closed;
4. a successor declaring `required_for_interpretation` may run but cannot produce a final cross-experiment decision without the predecessor contract;
5. missing predecessor files fail closed with a precise diagnostic;
6. a wrong experiment ID, implementation SHA, package SHA, terminal-audit hash, or group-summary hash fails closed;
7. duplicate or incomplete dataset/actor/control/seed join keys fail closed;
8. an exact predecessor plus successor fixture produces the full Positive-only/`c64`/`c128`/`c256`/`c512` table and paired differences without manual editing;
9. every number in the combined report can be traced to a compact file and checksum;
10. chat-only or rounded values cannot enter machine-readable result output;
11. task collapse, support/variance boundary, NaN/Inf, persistent drift, and fixed-horizon inconclusive states remain separately represented;
12. no held-out seeds are accessed and no frozen scientific variable is changed by the closure tooling;
13. handoff and registry authority remain unchanged and are updated only through their accepted transaction path;
14. a recovered historical artifact can be closed without rerunning the scientific experiment;
15. tests include a faithful replay of this E7 predecessor/successor incident.

Success metrics after deployment:

- zero dependent successors reaching final review with an undeclared predecessor evidence gap;
- zero result tables populated from inferred or remembered values;
- 100% of completed matrix pilots have a compact result index or an explicit reviewed closure exception;
- median `RUN_COMPLETE -> COMPACT_RESULT_DEPOSITED` time is measured and reduced without delaying trustworthy server launch.

## Remaining uncertainties

1. The durable location of the predecessor 192-branch raw-complete artifact is not currently known in this review context.
2. The exact result-package SHA-256 and compact-file hashes remain to be recovered.
3. It is not yet confirmed whether the server run directory, an earlier uploaded ZIP, or another branch contains the original compact aggregate.
4. The exact predecessor scientific table must not be reconstructed until that artifact is verified.
5. The proposed governance claim and tooling scope require a separate user-approved implementation task before code changes begin.

## Status

```text
Incident documented
Root cause identified
Approximate predecessor values withdrawn
Artifact recovery pending
Systemic optimization proposed only
No workflow behavior changed
No scientific state changed
No registry or handoff change
```

---

# Future entry template

## YYYY-MM-DD — DEVOPT-YYYY-MM-DD-<SHORT-ID>

### Context

### Intended change and expected complexity

### Observed duration

### Chronological incident record

### Root causes

### Impact

### Immediate fixes

### Proposed systemic optimization

### Acceptance criteria

### Remaining uncertainties

### Status