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
