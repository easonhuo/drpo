# Code Regeneration Replay Protocol

Claim: `GOV-CODE-REGENERATION-REPLAY-01`

Candidate treatment: `GOV-CODE-CHANGE-BUDGET-02`

Protocol status: **frozen before regenerated outputs**.

## 1. Primary estimand

Estimate the effect of exposing a coding agent to the candidate code-change-budget
review and repair loop while holding the historical coding task, repository base,
model product, tools, budget, and acceptance criteria fixed.

The primary unit is one paired historical task. The benchmark reports each case
before aggregate statistics.

## 2. Arms

### Arm A — baseline regeneration

The agent receives the frozen task packet and ordinary repository/test feedback.
It does not receive:

- the candidate large-change verdict;
- candidate reuse/file-necessity diagnostics;
- the historical later/smaller implementation;
- the other regenerated arm.

It may use up to two ordinary repair rounds after its first complete patch.

### Arm B — candidate regeneration

The agent receives the identical task packet. After each complete patch, the
frozen candidate gate is evaluated externally. Its exact verdict and diagnostics
are returned to the agent. Arm B may use up to two repair rounds.

Arm B does not receive the historical later/smaller implementation or post-task
bug explanations.

## 3. Fresh-context requirement

Every arm runs in a fresh model context. The orchestrator may know both arms, but
the generating context may know only its own task packet and its own feedback.

A single conversation pretending to forget the previous arm is invalid.

The two arm prompts must be byte-identical except for an arm-neutral run identity.
Candidate feedback is introduced only after Arm B submits a complete candidate.

## 4. Model and tool controls

Record for every arm:

- product-visible model name;
- date and session identifier;
- reasoning/effort setting when exposed;
- repository base SHA;
- task-packet SHA-256;
- allowed tools;
- network policy;
- token or message budget;
- wall-time limit;
- repair-round limit.

Use the same visible model and settings for both arms. If the provider does not
expose exact build identity or token counts, preserve that limitation in the
result and classify the pair as quasi-controlled.

## 5. Historical reconstruction

A task packet is reconstructed only from information available before the
historical implementation was reviewed:

- user requirement or accepted planning document;
- repository state at the frozen base;
- frozen scientific/governance variables;
- expected outputs and execution responsibilities;
- pre-existing tests and authority rules.

The packet must not include:

- the historical implementation diff;
- later reviewer instructions that reveal the preferred architecture;
- bugs discovered only after historical execution;
- later compact implementation details;
- candidate gate recommendations.

The orchestrator records every omitted later fact in `LEAKAGE_EXCLUSIONS.md`.

## 6. Hidden acceptance contract

The evaluator freezes hidden acceptance checks before either arm runs. They may
include:

- exact scientific matrix expansion;
- protected-variable equality;
- required public symbols and CLI behavior;
- focused unit/integration tests;
- full pytest and Ruff;
- liveness reaching a declared update or runtime state;
- artifact and critic/hash immutability;
- terminal audit outputs;
- forbidden duplicate implementations.

The generator sees the task contract, not the complete hidden-test code.

## 7. Candidate gate snapshot

The candidate gate is frozen at commit
`7826f5d60c83d8a58a11dc526b487cc09078d818`.

It is evaluated as an external benchmark tool against both arm diffs. The
historical task checkout is not modified to include the gate. This separates the
historical repository base from the benchmark evaluator.

The evaluator records:

- automatic/small-change path;
- PASS or FAIL;
- every diagnostic;
- gate wall time;
- structured-evidence size;
- whether a correct final output remains rejected.

## 8. Attempt lifecycle

Each arm has at most three complete attempts:

1. initial candidate;
2. repair 1;
3. repair 2.

An attempt is complete only when it contains a patch/tree and a self-declared
terminal implementation. Partial scratch work is preserved but is not gate
scored.

After each attempt:

1. freeze diff and tree identity;
2. measure code and file counters;
3. run candidate gate externally for both arms;
4. run the same acceptance tests;
5. return ordinary test feedback to both arms;
6. additionally return candidate-gate feedback only to Arm B;
7. stop when accepted or after attempt 3.

No extra rescue round is allowed for an inconvenient failure.

## 9. Correctness precedence

An implementation is `accepted` only if all mandatory task checks pass. Code
reduction never compensates for missing functionality, weakened tests, scientific
drift, runtime failure, or absent required artifacts.

Terminal classes:

- `ACCEPTED`;
- `REJECTED_INCOMPLETE`;
- `REJECTED_UNSAFE`;
- `REJECTED_GATE_FALSE_POSITIVE`;
- `TIMEOUT`;
- `ENVIRONMENT_INVALID`.

Only predeclared environment invalidations may be rerun without charging another
attempt. Every invalidated run remains in the raw log.

## 10. Code-size accounting

Measure base-to-attempt diff:

- production Python additions;
- production Python deletions;
- production Python churn;
- test Python churn;
- non-Python production/config/document churn;
- total changed files;
- added production Python files;
- renamed/deleted production Python files;
- copied-file and copied-block findings.

Generated files, adapters, runners, launchers, aggregators, validators, recovery
systems, and compatibility layers count as production code when they execute or
control execution. Tests and frozen protocol documents are reported separately.

## 11. Time and process accounting

Record separately:

- model generation wall time;
- active operator/orchestrator time;
- candidate-gate runtime;
- evidence-authoring time;
- focused test runtime;
- full CI runtime;
- liveness runtime;
- human/reviewer adjudication time;
- total time from prompt release to terminal state.

Also record:

- number of complete attempts;
- number of gate rejections;
- number of unique errors;
- number of new branches/workspaces;
- token usage when exposed;
- external CI runs;
- manual recovery decisions.

## 12. ROI interpretation

The candidate's cost is not its source lines alone. Charge:

- validator execution;
- structured evidence;
- repair rounds caused by the validator;
- extra review and maintenance.

Benefits include:

- reduced production churn;
- fewer duplicate modules;
- fewer integration bugs and failed executions;
- smaller review surface;
- lower future maintenance cost.

Report both the immediate end-to-end time and the avoided historical failure
cost. Do not infer saved engineering time solely from line-count reduction.

## 13. Aggregate decision rules

Phase 1 closes only after all six frozen cases terminate.

Required thresholds:

- unsafe pass = 0;
- false rejection of a correct final implementation = 0;
- accepted implementation task completeness = 100%;
- Arm-B repair-at-2 >= 80% among repairable initial failures;
- median production-Python churn reduction >= 30%;
- median total wall-time regression <= 20%;
- scientific/governance scope drift = 0;
- required liveness regression = 0.

Report medians and every individual case. Do not use only the mean, and do not
remove outliers after seeing outcomes.

## 14. Rule-change freeze

The candidate gate cannot be changed during the six-case Phase-1 batch. A defect
found during the batch is recorded. The batch continues when safe, or stops under
a predeclared safety stop. Any repaired gate starts a new versioned batch; old
results remain preserved.

Safety stops:

- candidate gate allows a known unsafe/incomplete implementation;
- evaluator changes a frozen scientific variable;
- arm isolation is breached;
- hidden acceptance contract changes after an arm starts.

A false rejection does not disappear through post-hoc rule editing; it remains a
Phase-1 failure and may motivate Phase 2.

## 15. Publication and adoption boundary

Protocol setup, one successful case, or descriptive historical compression does
not authorize merge or default activation.

After Phase 1:

- passing thresholds permits a clean current-main rebuild and shadow/canary
  proposal;
- mixed results require targeted redesign and a new frozen batch;
- unsafe pass, persistent false rejection, or poor ROI rejects default adoption.

No scientific claim or experiment launch is authorized by this benchmark.
