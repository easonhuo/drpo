# DRPO Maintenance Runner + ReplayAB implementation contract

**Claim:** `GOV-DRPO-MAINTENANCE-RUNNER-REPLAYAB-01`  
**Parent measurement claim:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Document version:** `0.2-stage0-review`  
**Initial branch base:** `main@d7e82201159a736f0d2e48403aae15ea07e178a7`  
**Latest main reviewed:** `main@d5029d05696382ce02cc0eb1e4c44291b00f8a7d`  
**Status:** Stage 0 active and under repeated review; no behavior code, write-capable workflow, default-route change, scientific execution, or merge is authorized by this document alone  
**Scientific impact:** none

## 1. Executive decision

Develop one small shared Maintenance Runner candidate, qualify thin E7 and E8 validation adapters on top of it, and judge the complete candidate with the existing ReplayAB Core before any adoption decision.

The implementation is deliberately divided into separate owners:

- **Maintenance Runner:** the workflow candidate under test. It owns only repository transaction mechanics.
- **ReplayAB Core:** the measuring instrument. It owns frozen cases, independent acceptance, paired comparison, and decision evidence.
- **E7/E8 adapters:** fixed validation selections and scientific-invariant checks. They do not choose scientific values.
- **Handoff Authority:** a separate authority validator. It is not part of the first Maintenance Runner task class.

Implementation, liveness, replay success, and default adoption are separate states. No earlier state implies a later one.

## 2. Problem and success condition

Recent E7/E8 successor tasks repeatedly spent time on the same mechanical work:

1. resolving an exact current base;
2. staging an approved change payload;
3. checking branch identity and allowed paths;
4. applying the change in an isolated workspace;
5. selecting and running focused validation;
6. checking frozen scientific values and generated matrices;
7. creating and pushing a development-branch commit;
8. collecting exact-head evidence for review.

The candidate is useful only when it reduces those repeated operations without reducing correctness, safety, scientific control, or provenance.

It is not intended to make training, full CI, scientific review, or stochastic coding-model reasoning intrinsically faster.

## 3. Non-goals and first-version exclusions

The first accepted version must not:

- choose experiment IDs, parameters, seeds, datasets, budgets, thresholds, anchors, stopping rules, or claim boundaries;
- edit `docs/handoff.md`, `experiments/registry.yaml`, schema-v3 deltas, authority after-images, or governance-stage ledgers;
- modify `.github/workflows/**` as a task payload;
- launch liveness, pilot, formal training, aggregation, terminal audit, or result publication;
- create or update a scientific result status;
- write to `main`, create an undeclared branch, force-push, merge, auto-merge, or mark a PR ready;
- accept arbitrary shell, Python, command, test, or path expressions from task input;
- install a new dependency;
- become the default route before the final ReplayAB decision and separate user approval.

Profile data-driven migration is related but separate. The runner must first prove useful on frozen historical payloads. Moving E7/E8 constants out of Python requires a later, independently reviewed claim.

## 4. Existing owners that must be reused

The candidate must compose existing repository components rather than duplicate them:

- the connected GitHub App dev-branch route defined in `AGENTS.md`;
- existing test-selection and PR-gate commands;
- existing E7/E8 loaders, matrix builders, RunSpec checks, and focused tests;
- existing ReplayAB exact-artifact, failure-boundary, and R2 semantic-acceptance capabilities;
- existing formal-execution, governance-stage, and handoff-authority validators when their paths are in scope;
- existing Git and GitHub branch protections.

The candidate may invoke an owner. It may not reimplement that owner's authority or scientific semantics.

## 5. Four-stage development map

The previously separate planner, apply, and commit phases are merged into one implementation stage. E7 and E8 qualification are also merged into one adapter stage.

### Stage 0 — design, threat model, and replay contract

No behavior-changing code.

Stage 0 freezes:

- candidate responsibility and non-goals;
- task-manifest and payload boundaries;
- write and branch safety model;
- exact file plan and code/runtime budgets;
- internal checkpoints for Stage 1;
- E7/E8 adapter boundaries;
- ReplayAB case-selection, ordering, metrics, and decision rules;
- rollback and stop conditions;
- the document-dependency protocol in Section 6.

Stage 0 does not close until every review pass in Section 7 is resolved and the user explicitly approves Stage 1.

### Stage 1 — shared runner closed loop

One development stage implements and validates the complete shared transaction:

```text
validate manifest and payload
→ plan deterministically
→ apply in isolated workspace
→ run fixed validation profile
→ re-audit final diff
→ create one commit
→ non-force push to the same pre-existing dev branch
→ emit exact-head evidence
```

Stage 1 has internal checkpoints but only one major stage acceptance:

- **Checkpoint 1A — read-only planner:** parsing, identity, path, plan hash, no mutation.
- **Checkpoint 1B — isolated apply/validate:** payload verification, apply check, fixed tests, failure retention, no push.
- **Checkpoint 1C — write liveness:** one commit and same-branch non-force push after all checks.

A checkpoint pass permits work on the next checkpoint inside Stage 1. It is not a stage acceptance and does not authorize routine use.

### Stage 2 — E7 and E8 thin-adapter qualification

Reuse the Stage 1 transaction without copying its mechanics.

Stage 2 adds two separate fixed adapters:

- E7 validation selection and invariant checks;
- E8 validation selection and invariant checks.

The adapters may validate frozen values and generated outputs. They may not choose, infer, optimize, or alter those values.

Stage 2 does not include profile data migration, training, handoff materialization, or scientific interpretation.

### Stage 3 — final ReplayAB evaluation

Run the frozen paired case bank and issue exactly one engineering verdict:

- `ADOPT`;
- `NARROW`;
- `REDESIGN`;
- `REJECT`.

No verdict changes repository defaults without a separate explicit approval and reviewed change.

## 6. Document-as-contract dependency protocol

Every implementation stage is strongly dependent on this document rather than merely inspired by it.

### 6.1 Plan identity

Before a stage starts, its durable stage packet must record:

- this document path;
- the exact Git commit containing the approved document version;
- the document SHA-256;
- the stage number and approved stage status;
- the exact current-main SHA reviewed for that stage;
- the authorized paths, responsibilities, commands, tests, budgets, and exit gates.

A stage run with a different or missing plan identity is invalid.

### 6.2 No silent deviation

Implementation must stop before proceeding when it requires any of the following:

- a path not authorized by the stage packet;
- a new Python path without exact-path oral approval;
- a new dependency;
- broader GitHub permissions;
- a new task input field or executable input;
- a relaxed safety, test, time, complexity, or replay threshold;
- a changed scientific responsibility;
- a different ReplayAB case, label, exclusion, or acceptance rule.

The document must be revised and reviewed first. Code is never allowed to redefine the plan after implementation difficulty or benchmark results are observed.

### 6.3 Stage output binding

Every stage report must bind:

- approved plan commit and SHA-256;
- implementation base and head;
- changed paths and line-count method;
- executed commands and observed outcomes;
- unresolved items and deviations;
- explicit `PASS`, `HOLD`, `REDESIGN`, or `STOP` decision.

A later stage may start only from a `PASS` report that cites the same approved document lineage.

### 6.4 Change classes

Document changes are classified as:

- **clarification:** no responsibility, permission, threshold, file, or acceptance change;
- **contract amendment:** changes scope, file plan, permissions, tests, thresholds, cases, or responsibilities;
- **result record:** records evidence without changing the frozen contract.

A contract amendment after Stage 1 behavior results requires a new pre-result review. Adoption thresholds and case labels cannot be relaxed post hoc.

### 6.5 Engineering-plan boundary

This document controls only the Maintenance Runner engineering project. It does not replace `docs/handoff.md`, change scientific experiment state, or become a second research master.

## 7. Stage 0 repeated-review protocol

Stage 0 is reviewed in separate passes. A single broad read-through is insufficient.

### Pass A — objective and ROI

Questions:

- Is the problem repeated enough to justify a new workflow?
- Does the proposed transaction remove real manual work rather than move it into setup?
- Can the gain be measured independently from child-test time and GitHub queue time?
- Is a smaller reuse of an existing owner sufficient?

Failure action: narrow or cancel before behavior code.

### Pass B — architecture and duplication

Questions:

- Are Maintenance Runner, ReplayAB, E7/E8 adapters, V1, and Handoff Authority still separate owners?
- Is existing repository behavior composed rather than copied?
- Does any abstraction exist only for hypothetical future experiments?
- Can the first version be implemented without a service, plugin system, backend registry, or generalized framework?

Failure action: simplify before implementation.

### Pass C — security and mutation safety

Questions:

- Can task input execute arbitrary code or select arbitrary tests?
- Can it target `main`, another branch, a protected path, a symlink, a submodule, or Git metadata?
- What happens on stale head, concurrent update, partial apply, test failure, cancellation, timeout, or push rejection?
- Can secrets or payload contents leak through logs or artifacts?
- Can a GitHub-token push be mistaken for independently triggered PR CI?

Failure action: fail closed or remove the task class.

### Pass D — governance and scientific safety

Questions:

- Are handoff, registry, workflow, authority, governance, and formal-execution paths excluded from V1 payloads?
- Are frozen scientific fields checked by existing owners?
- Are smoke, liveness, replay, and formal results kept distinct?
- Does any task implicitly change scientific status or execution order?

Failure action: stop and revise scope.

### Pass E — ReplayAB validity

Questions:

- Are both arms given identical frozen inputs except for the treatment?
- Are case selection, labels, exclusions, environment, cache policy, and thresholds frozen before candidate results?
- Are both arms independently accepted before efficiency is compared?
- Are order effects, failures, timeouts, and per-family outcomes retained?
- Is the conclusion limited to deterministic repository-transaction performance?

Failure action: no adoption claim.

### Pass F — implementation and rollback feasibility

Questions:

- Are exact files and responsibilities identified?
- Can focused tests cover the dangerous boundaries?
- Can routine use be disabled with one reviewed change?
- Does rollback preserve history and leave the current route intact?
- Is the production-code budget realistic without minification or responsibility mixing?

Failure action: redesign or cancel.

Stage 0 closure requires a written result for all six passes and no unresolved blocker.

## 8. First-version task contract

Every run consumes one immutable, reviewable manifest and one approved unified-diff payload. The manifest is data, not executable code.

V1 supports only `unified_diff`. Complete-file generation, Base64 chunk protocols, binary payloads, and arbitrary archive extraction are excluded until replay evidence proves they are necessary.

Illustrative manifest:

```yaml
task_id: E7-PROFILE-EXAMPLE-01
schema_version: 1
task_type: patch_apply
plan_commit: 40-character SHA
base_commit: 40-character SHA
target_branch: dev/example
expected_head: 40-character SHA
payload_path: .github/maintenance-payload/E7-PROFILE-EXAMPLE-01/update.patch
payload_sha256: 64-character digest
allowed_paths:
  - configs/example.json
  - runspecs/templates/example.yaml
validation_profile: e7_focused
expected_change:
  min_files: 1
  max_files: 3
  allow_delete: false
  allow_rename: false
  allow_mode_change: false
```

The schema must reject:

- unknown keys;
- non-full or uppercase Git SHAs;
- mutable refs in place of SHAs;
- `main`, the default branch, missing branches, or wildcard branch patterns;
- absolute paths, `..`, empty segments, control characters, or duplicate normalized paths;
- paths outside the explicit allowlist;
- overlapping payload and output paths;
- executable command, environment, expression, or script fields;
- binary, submodule, symlink, rename, deletion, or file-mode changes in V1;
- oversized manifest or payload;
- a payload hash, plan identity, base, or expected head mismatch.

## 9. V1 transaction sequence

The shared runner owns only the following sequence:

1. parse the manifest strictly;
2. verify plan identity and the stage authorization packet;
3. resolve the repository default branch and reject it as target;
4. verify `base_commit` and `expected_head` are reachable commit objects;
5. verify the remote target branch exists and equals `expected_head`;
6. verify manifest and payload size limits;
7. verify payload SHA-256;
8. parse the patch without applying it and derive its changed paths and change kinds;
9. reject forbidden change kinds and paths;
10. create an isolated workspace from `expected_head`;
11. run `git apply --check` and apply only in that workspace;
12. verify the actual diff exactly matches the allowed change contract;
13. run one fixed repository-owned validation profile;
14. recheck remote target head immediately before commit;
15. create one commit with task, plan, manifest, payload, base, and parent identities;
16. verify the committed tree and changed paths;
17. non-force push only to the same target branch;
18. emit an append-only report and exact final commit SHA.

Any failure before push must leave the remote target unchanged. A non-fast-forward push rejection is a safe stale-head result, not a retry invitation.

## 10. Fixed validation profiles

Task input selects a profile by exact ID. It cannot provide commands.

### `generic_small_code`

- Python compile for changed Python scope when applicable;
- Ruff for the repository-owned selected scope;
- repository test selector and focused pytest;
- changed-path and file-mode audit;
- code-only handoff-authority no-op verification when required by current repository policy.

### `e7_focused`

- all `generic_small_code` requirements;
- existing E7 profile/matrix/RunSpec focused tests selected by repository code;
- frozen scientific-field and expected-branch-count checks owned by E7 code.

### `e8_focused`

- all `generic_small_code` requirements;
- existing E8 profile/cell/RunSpec focused tests selected by repository code;
- frozen scientific-field and expected-cell-count checks owned by E8 code.

V1 does not include `authority_materialization`, `workflow_change`, `paper_release`, or `formal_experiment` profiles.

## 11. Edge-case and failure matrix

Stage 1 tests and liveness must cover at least the following classes.

| Class | Expected result |
|---|---|
| valid manifest and patch | validate, test, commit, and non-force push |
| malformed or unknown manifest field | reject before workspace creation |
| stale `base_commit` contract | reject with no mutation |
| stale remote `expected_head` at start | reject with no mutation |
| remote head changes during validation | reject before commit or fail non-force push safely |
| target is `main` or default branch | reject |
| undeclared or missing target branch | reject |
| path traversal or absolute path | reject |
| case-folding or normalization collision | reject |
| changed path outside allowlist | reject |
| workflow, handoff, registry, authority, governance, or Git metadata path | reject |
| binary, submodule, symlink, delete, rename, or mode change | reject in V1 |
| payload hash mismatch | reject before patch parse |
| payload over size limit | reject before workspace creation |
| patch does not apply cleanly | reject with retained diagnostics |
| patch applies partially only | impossible through atomic apply; otherwise reject and discard workspace |
| focused test failure | no commit and no push |
| test command timeout | no commit and no push; retain timeout evidence |
| runner cancellation before push | no remote change |
| cancellation after successful push | commit remains valid; report recovery state explicitly |
| commit succeeds but push is rejected | local/disposable commit only; remote unchanged |
| artifact upload fails after push | do not claim complete evidence; recover from commit identity and rerun report-only step |
| secret-like value in payload or log | redact or block according to frozen rule |
| concurrent task on same branch | serialize or reject; never interleave |
| concurrent tasks on different branches | may run independently |
| flaky focused test | retain failure; no automatic retry unless both arms' replay contract predeclares the same retry |
| GitHub-token push does not retrigger PR workflows | do not claim PR CI ran; Stage 1 validation must stand on its own exact head |
| current `main` advances during a task | does not alter frozen task base; report divergence for reviewer |
| generated file differs nondeterministically | independent acceptance fails unless variability was frozen in advance |

No failure may be converted to success by silently broadening paths, rerunning a different test set, changing the payload, or moving the expected head.

## 12. Stage 1 acceptance

Stage 1 may be accepted only when all internal checkpoints pass and one final major acceptance verifies:

- strict positive and adversarial manifest tests;
- isolated-workspace mutation only;
- every failure class required by Section 11 fails closed;
- exact-head recheck immediately before commit and push;
- no commit or push after failed validation;
- one successful low-risk documentation/fixture liveness;
- one stale-head liveness with no push;
- one test-failure liveness with no push;
- final commit provenance independently re-read from GitHub;
- full repository pytest, Ruff, compile, handoff-authority no-op, formal-execution, and governance-stage checks on the Stage 1 candidate head;
- production code and self-overhead budgets pass.

The successful liveness must not touch scientific code, experiment configuration, handoff, registry, workflows, or governance authority.

## 13. Stage 2 E7/E8 adapter contract

Stage 2 reuses the Stage 1 runner unchanged except for fixed profile selection and explicit scientific-invariant checks.

### E7 adapter may validate

- exact experiment/profile identity and parent identity;
- frozen datasets and order;
- frozen seeds;
- frozen control points, anchors, exclusions, and expected branch count;
- deterministic matrix expansion;
- RunSpec consistency;
- existing focused E7 tests.

### E8 adapter may validate

- exact experiment/profile identity and parent identity;
- frozen dataset/corpus and initialization identity;
- frozen seeds and seed offsets;
- frozen parameter grid, controls, exclusions, and expected cell count;
- deterministic matrix expansion;
- RunSpec consistency;
- existing focused E8 tests.

### Adapter prohibitions

- no automatic parameter or seed proposal;
- no inference from previous results;
- no profile migration or common E7/E8 scientific schema;
- no training or evaluator execution;
- no handoff or registry write;
- no ranking, convergence, steady-state, or result-status decision.

Stage 2 acceptance requires at least three frozen E7 and three frozen E8 successful-path tasks plus one failure-boundary task per family. Every arm must independently satisfy the same frozen acceptance contract.

## 14. ReplayAB final protocol

### 14.1 Claim boundary

ReplayAB measures the deterministic repository transaction layer:

- correctness and safety of the produced dev-branch commit;
- active interventions and operation count;
- controlled end-to-end time;
- candidate self-overhead;
- failure-boundary behavior;
- maintenance complexity.

It does not estimate how often a future coding agent invents a correct patch. Both arms receive the same approved task packet and payload identity.

### 14.2 Arms

- **Arm A — current route:** frozen reconstruction of the accepted GitHub-App/manual plus existing-script sequence.
- **Arm B — candidate route:** Maintenance Runner using the same base, payload, branch policy, test requirements, and acceptance contract.

Neither arm may consume the other arm's workspace, outputs, diagnostics, or timing results.

### 14.3 Frozen case bank

Before Stage 1 write-capable liveness results are used for any efficiency claim, freeze eight replayable cases:

- four E7 cases;
- four E8 cases.

Each family should include:

1. config/RunSpec-only success;
2. small code-plus-test success;
3. a historical task with temporary workflow or repeated integration work;
4. a failure-boundary task.

The inventory freezes case ID, task family, source history, base, payload hash, allowed paths, treatment, evaluator, expected terminal state, replayability class, cache policy, exclusions, and ground-truth acceptance facts.

Cases, labels, orientations, exclusions, and thresholds may not be removed or changed after candidate results.

### 14.4 Ordering and repetitions

For each successful-path case:

- one paired repetition in A→B order;
- one paired repetition in B→A order;
- isolated workspaces;
- identical cache policy and environment fingerprint;
- all failures and timeouts retained.

A third paired repetition is mandatory when paired wall-time spread exceeds 15%, a cache/environment mismatch occurs, or one repetition crosses the predeclared timeout boundary.

Failure-boundary cases require repeated terminal agreement; their speed is diagnostic only.

### 14.5 Independent acceptance

Use R2 semantic acceptance when different commit metadata or implementation trees may both be correct. Use exact-artifact mode only for genuinely deterministic outputs. Use failure-boundary mode when stopping is the correct result.

Efficiency is hidden until both arms are independently accepted.

Each arm must pass:

- mandatory behavior and output completeness;
- forbidden-regression checks;
- path, mode, branch, and protected-value checks;
- relevant focused and governance tests;
- task, evaluator, plan, payload, base, and run identity bindings;
- expected terminal state;
- no unauthorized partial mutation.

### 14.6 Metrics

Primary:

- independently accepted case rate;
- incorrect-arm false-acceptance count;
- safety-boundary agreement;
- median active intervention count;
- median repository commands/tool actions;
- median time to reviewable dev-branch commit;
- median controlled end-to-end wall time;
- candidate self-overhead excluding child validation;
- failed-attempt and retry count;
- production lines and changed-path complexity.

Secondary:

- planning time;
- patch verification/apply time;
- test-selection and child-test time;
- commit/push time;
- cache/order effects;
- per-case and per-family outcomes.

### 14.7 Decision gate

`ADOPT` is allowed only when all conditions pass:

1. both arms are independently accepted on every successful-path case;
2. every failure-boundary case stops at the correct boundary without unauthorized mutation;
3. Arm B has zero false acceptance and zero protected-path regression;
4. median active interventions fall by at least 40%;
5. median time to reviewable dev commit falls by at least 25%;
6. Arm B controlled wall time does not regress by more than 5% in either E7 or E8;
7. median candidate self-overhead is at most 2 seconds and p95 at most 5 seconds, excluding child validation;
8. no duplicate validator or full-test execution is used only to manufacture telemetry;
9. code and complexity budgets pass;
10. E7 and E8 results are both positive when reported separately.

If correctness passes but only one family benefits, return `NARROW`. A correctness or security failure returns `REJECT` unless the failure clearly precedes benchmark validity, in which case return `REDESIGN` and rerun only under a newly frozen contract.

## 15. Code, runtime, and complexity budgets

Preferred cumulative production budget through Stage 1:

| Component | Preferred lines |
|---|---:|
| Workflow YAML | 80–140 |
| Runner production Python | 180–300 |
| Fixed validation mapping and shell glue | 30–80 |
| **Total** | **290–520** |

Rules:

- yellow review above 520 production lines;
- hard stop above 650 production lines before Stage 2;
- tests and fixtures are outside the production budget but must remain targeted;
- no service, daemon, database, dashboard, queue, scheduler, plugin system, backend registry, or dependency;
- no responsibility may be hidden in generated code, minified code, shell heredocs, or unrelated modules to fit the budget;
- Stage 2 adapter additions above 150 production lines per family trigger redesign;
- static planning target: median ≤250 ms and p95 ≤1 s;
- candidate self-overhead target: median ≤2 s and p95 ≤5 s, excluding child validation;
- a candidate that saves manual actions but materially increases controlled wall time is a negative optimization.

## 16. Proposed file plan and approval gates

Stage 0 proposes, but does not yet authorize creation of:

- `.github/workflows/drpo-maintenance-runner.yml` — manual, branch-scoped workflow entrypoint;
- `scripts/run_drpo_maintenance_runner.py` — strict manifest, patch, isolated validation, commit, and same-branch push transaction;
- `tests/test_drpo_maintenance_runner.py` — positive, adversarial, failure-injection, and liveness-contract tests;
- non-Python manifests, fixtures, and reports under `docs/development_workflow_optimization/maintenance_runner/`.

The two proposed `.py` paths are new governed Python destinations. Before Stage 1 creates either path, the repository owner must explicitly approve each exact path and stated responsibility under `GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02`.

Why existing Python files are insufficient:

- ReplayAB files judge outcomes and must not own candidate mutation behavior;
- V1 files own pilot-registration transactions and must not become a generic patch runner;
- E7/E8 files own scientific domain behavior and must not own repository-wide write mechanics;
- placing the runner in an unrelated existing file would violate responsibility boundaries and make rollback harder.

If exact-path approval is not given, Stage 1 must redesign around existing approved paths or stop. It may not hide Python behavior in another file type.

## 17. GitHub Actions and CI semantics

The write-capable workflow must use least privilege:

- `contents: write` only in the job that performs the guarded same-branch push;
- no `pull-requests: write` in V1;
- no secret-bearing environment input;
- `persist-credentials: false` except for the explicit guarded push mechanism;
- branch-scoped concurrency;
- exact expected-head recheck before write.

A push performed with `GITHUB_TOKEN` is not assumed to trigger another workflow. Therefore:

- Stage 1 must run its declared validation before push;
- it must publish only evidence it actually produced;
- it must not claim PR Gate, Handoff Authority, or any other workflow ran recursively;
- an exact-head external check, when required, must be explicitly triggered or observed separately.

Automatic Draft PR creation is excluded from V1 and may be considered only after the runner is accepted and the additional permission is separately reviewed.

## 18. Relationship to Handoff Authority

The pending Handoff Authority PR is a separate read-only checker.

The first Maintenance Runner task class forbids handoff and registry changes, so it neither invokes nor depends on remote cross-workflow authority triggering.

A future authority-aware task class would need to invoke the existing trusted normalizer and authority verifier inside one explicitly designed transaction. That is a separate scope and cannot be inferred from this plan.

## 19. Rollback and kill switch

Before adoption, rollback is immediate:

- disable or remove the candidate workflow on its development branch;
- stop candidate runs;
- retain all commits, manifests, logs, reports, failures, and ReplayAB evidence;
- keep the current GitHub App/manual route unchanged.

After a later default adoption, rollback must be one reviewed commit that:

- disables the candidate trigger or removes the default-route reference;
- restores the previously documented route;
- preserves historical evidence;
- does not modify scientific artifacts or experiment state;
- requires explicit user approval.

Stage work stops immediately when any of the following occurs:

- false acceptance or protected-path mutation;
- write to the wrong branch;
- force push or unauthorized branch creation;
- silent scientific-variable change;
- benchmark case or threshold changed after results;
- production code exceeds the hard stop;
- controlled wall time indicates a negative optimization;
- required new path, permission, dependency, or responsibility lacks approval.

## 20. Current uncertainties that Stage 0 must resolve

Stage 0 remains open until these are resolved and durably recorded:

1. exact eight-case ReplayAB inventory and artifact availability;
2. final payload-size and manifest-size limits based on historical task distribution;
3. exact repository-owned command lists for `generic_small_code`, `e7_focused`, and `e8_focused`;
4. whether the proposed new Python paths receive explicit approval;
5. how exact-head evidence is retained when artifact upload fails after a successful push;
6. whether the first liveness should use a documentation-only or fixture-only payload;
7. refresh of the implementation branch to the then-current `main` before Stage 1 starts.

No uncertainty may be resolved after seeing benchmark results merely to improve the decision.

## 21. Stage ledger

| Stage | Status | Behavior impact | Required next decision |
|---|---|---|---|
| 0 — design, threat model, replay contract | `active_under_review` | none | complete six review passes, freeze inventory and command profiles, user approves Stage 1 |
| 1 — shared runner closed loop | `not_started` | dev-branch candidate only | major safety/CI/runtime acceptance |
| 2 — E7/E8 thin adapters | `not_started` | validation only; no training | joint adapter qualification |
| 3 — ReplayAB final evaluation | `not_started` | evidence only | `ADOPT/NARROW/REDESIGN/REJECT` |

Later-stage authorization is never inferred from an earlier-stage pass.

## 22. Stage 0 review record

Current review cycle: `stage0-review-1`.

- Pass A — objective and ROI: `provisionally_pass`; benefit is measurable, but exact historical case inventory remains open.
- Pass B — architecture and duplication: `pass_with_boundary`; candidate, ReplayAB, adapters, V1, and Handoff Authority remain separate.
- Pass C — security and mutation safety: `pass_with_v1_exclusions`; binaries, symlinks, deletes, renames, mode changes, protected paths, branch creation, and arbitrary commands are excluded.
- Pass D — governance and scientific safety: `pass_with_approval_gate`; no scientific or authority paths are in V1, and new Python paths still require exact approval.
- Pass E — ReplayAB validity: `hold`; exact case inventory, evaluator bindings, and historical artifact availability must be frozen before closure.
- Pass F — implementation and rollback feasibility: `provisionally_pass`; line budget appears feasible but command profiles and evidence-recovery behavior remain open.

Stage 0 is not closed by this review cycle. The next review must resolve the `hold` and all open items in Section 20 before requesting Stage 1 authorization.
