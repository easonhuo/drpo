# GOV-CODE-REGENERATION-REPLAY-01

## Status

- class: governance engineering benchmark;
- repository: `easonhuo/drpo`;
- base branch: `main`;
- base commit: `dd355642df2ba2479715c9620749c1e9f75f76ba`;
- development branch: `dev/gov-code-regeneration-replay-01`;
- candidate rule: `GOV-CODE-CHANGE-BUDGET-02`;
- candidate gate snapshot: `7826f5d60c83d8a58a11dc526b487cc09078d818`;
- scientific experiment status impact: none;
- execution status: protocol and case freeze in progress; no regenerated coding arm has completed.

## Question

Determine whether the candidate code-change-budget rule causes a coding agent to
produce a smaller, reuse-first, task-complete implementation without increasing
unsafe acceptance, false rejection, scientific drift, or end-to-end development
cost beyond the declared tolerance.

This benchmark evaluates engineering workflow behavior. It cannot create DRPO
scientific evidence, change an experiment result status, or authorize a scientific
launch.

## Evidence ladder

The benchmark keeps three evidence classes separate:

1. **descriptive historical comparison** — old and later implementations are
   compared, but generation conditions differ;
2. **deterministic gate replay** — historical outputs are passed through the
   frozen candidate gate and expected accept/reject behavior is checked;
3. **regeneration replay** — both arms are freshly generated from the same
   historical base and frozen task contract in isolated contexts.

Only class 3 is the primary evidence for workflow adoption. Classes 1 and 2 are
supporting diagnostics and cannot be promoted to a causal claim.

## Treatment

For each case, both arms receive:

- the same historical repository base;
- the same frozen task contract;
- the same repository authority documents available at that base;
- the same coding-model label and settings;
- the same tool and token budget;
- the same hidden acceptance tests and liveness contract;
- at most two repair rounds after the first complete candidate.

Arm A follows the baseline coding path and does not receive the candidate
code-change-budget verdict or structured reuse feedback.

Arm B receives the frozen candidate gate verdict after each complete candidate.
It may repair only in response to that verdict and the same ordinary test output
available to Arm A.

No arm may inspect the historical accepted implementation, the other regenerated
arm, or post-task bug explanations before its terminal state is frozen.

## Isolation

A valid pair requires two fresh model contexts. Sequential role-play inside one
conversation is not accepted as a regeneration pair because knowledge of one arm
can contaminate the other.

Each context records:

- visible model label and session timestamp;
- base SHA and task-contract hash;
- all prompts and feedback;
- token usage when exposed by the platform;
- tool calls and elapsed time;
- every generated patch and repair attempt;
- final terminal state.

If exact backend model-build identity is unavailable, the pair is classified as
`quasi_controlled_same_product_model`, not fully randomized.

## Frozen outcomes

Correctness is evaluated before code size.

Required outcomes include, as applicable:

- task-contract completeness;
- focused tests;
- full repository pytest and Ruff;
- protected scientific-variable equality;
- RunSpec and authority validation;
- real liveness for tasks whose historical failure occurred only at runtime;
- terminal audit separating task-performance, support/boundary, and NaN/Inf
  events when the task owns those diagnostics.

A smaller but incomplete implementation is a failure.

## Metrics

Per arm, record:

- production-Python additions, deletions, and total churn;
- test-Python churn separately;
- total changed files and new production-Python files;
- copied files or copied implementation blocks;
- reused base modules and symbols;
- first-pass correctness;
- repair-at-1 and repair-at-2;
- unsafe pass and false rejection;
- active generation time;
- candidate-gate runtime;
- evidence-authoring time;
- CI runtime;
- reviewer time;
- total wall time to terminal accepted or terminal rejected state;
- token and tool cost when available.

## Phase-1 adoption thresholds

The frozen Phase-1 thresholds are:

- unsafe pass: `0`;
- correct final implementation false rejection: `0`;
- task completeness among accepted outputs: `100%`;
- repair-at-2 for initially rejected but repairable Arm-B outputs: at least `80%`;
- median production-Python churn reduction: at least `30%`;
- median end-to-end wall-time regression: no worse than `20%`;
- scientific-variable or experiment-responsibility drift: `0`;
- required real-liveness regression: `0`.

These thresholds remain fixed until the Phase-1 batch is closed. A failed or
inconvenient case cannot be removed after generation begins.

## ROI accounting

The candidate is not judged by validator complexity in isolation. Its overhead
is charged explicitly:

`gate runtime + evidence writing + repair retries + extra review + maintenance`.

Its benefit is measured as:

`avoided production churn + avoided duplicate modules + avoided debugging and
runtime failures + lower future review and maintenance surface`.

The gate is considered too heavy if it meets code-reduction targets only by
causing repeated dead ends, excessive evidence work, or more than the allowed
end-to-end slowdown.

## Phase-1 case strata

Phase 1 contains six frozen cases:

1. large canonical extension with historical duplicate-stack failure;
2. small in-place scientific formula and matrix correction;
3. legitimate wrapper and RunSpec integration over byte-preserved code;
4. legitimate large new runtime subsystem;
5. small governance activation with preserved neighboring text;
6. stale-lineage/current-main integration that should reuse reviewed blobs.

The exact inventory is stored in
`docs/development_workflow_optimization/regeneration_replay/CASE_INVENTORY.yaml`.

## Execution order

1. freeze this scope, the protocol, inventory, task contract, hidden acceptance
   contract, and evaluator snapshot;
2. run the GAE case first because it has the highest observed code-bloat and
   runtime-failure cost;
3. do not modify the candidate rule during the GAE pair;
4. complete the remaining five Phase-1 cases;
5. publish all raw attempts, including failures and timeouts;
6. audit the frozen thresholds;
7. only then decide whether to rebuild the candidate gate on current `main`, run
   shadow/canary adoption, redesign it, or reject it.

## Prohibitions

- no scientific launch;
- no modification of frozen datasets, seeds, coefficients, horizons, thresholds,
  convergence rules, or experiment responsibilities;
- no deletion of inconvenient historical evidence;
- no post-hoc task-contract expansion to favor either arm;
- no claim of universal superiority from one case;
- no merge or default-route activation from protocol setup alone.
