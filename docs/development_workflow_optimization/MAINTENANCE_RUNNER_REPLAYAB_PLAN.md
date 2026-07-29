# DRPO Maintenance Runner + ReplayAB implementation contract

**Claim:** `GOV-DRPO-MAINTENANCE-RUNNER-REPLAYAB-01`  
**Measurement authority:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Document version:** `0.3-stage0-review-2`  
**Initial branch base:** `main@d7e82201159a736f0d2e48403aae15ea07e178a7`  
**Latest main reviewed:** `main@d5029d05696382ce02cc0eb1e4c44291b00f8a7d`  
**Status:** `stage0_active_under_review`  
**Scientific impact:** none  
**Authorization boundary:** this document authorizes documentation and review only. It does not authorize behavior code, a write-capable workflow, a new Python path, a closed-governance-stage reopen, routine use, default-route activation, scientific execution, or merge.

The previous Stage 0 draft remains preserved in Git history. This revision replaces its current after-image because Stage 0 is still a design phase, not accepted historical evidence.

## 1. Decision

Develop one small shared **Maintenance Runner** candidate, qualify separate thin E7 and E8 validation adapters on top of it, then judge the completed candidate with the existing **ReplayAB Core**.

The following owners remain separate:

- **Maintenance Runner:** repository transaction mechanics only;
- **ReplayAB Core:** frozen cases, independent acceptance, paired comparison, and decision evidence;
- **E7/E8 adapters:** fixed validation selection and scientific-invariant checking only;
- **V1 pilot registration:** authoritative pilot-registration transaction, not a generic patch runner;
- **Handoff Authority:** handoff/registry verification, excluded from the first Maintenance Runner task class.

Implementation, liveness, stage acceptance, ReplayAB acceptance, routine use, and default adoption are different states. No state implies the next one.

## 2. Problem being measured

Repeated E7/E8 successor tasks spend time on the same mechanical sequence:

1. resolve exact base and target-head identities;
2. stage an already approved change payload;
3. audit paths and protected scientific values;
4. apply in an isolated workspace;
5. choose and run repository-owned focused validation;
6. audit the final diff;
7. create and non-force push one dev-branch commit;
8. retain exact-head evidence for review.

The candidate is useful only if it reduces those operations without reducing correctness, security, scientific control, provenance, or controlled end-to-end performance.

It is not intended to improve model reasoning, training time, full-CI runtime, scientific review time, or formal experiment execution.

## 3. Authority hierarchy and governance preconditions

This plan is subordinate to:

1. repository-root `AGENTS.md`;
2. `docs/handoff.md` Section 0;
3. `experiments/registry.yaml` for scientific state;
4. `docs/governance_pipeline_stage_status.yaml`;
5. the accepted ReplayAB contracts and measurement thresholds under `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`.

The parent measurement claim does **not** authorize this candidate's GitHub workflow changes or automatic dev-branch push. Its existing scope explicitly excludes those behaviors. Before Stage 1 behavior code starts, this candidate therefore requires:

- a dedicated reviewed scope for `GOV-DRPO-MAINTENANCE-RUNNER-REPLAYAB-01`;
- explicit user authorization for the Stage 1 implementation scope;
- a current-ledger determination of which closed governance stage is affected;
- the required `reopen` authorization record and rollback plan for that stage;
- any required `large-code-change-approval` environment review;
- exact-path oral approval for every proposed new Python destination.

Current evidence indicates governance pipeline Stage 1 is the likely affected closed stage because it owns update integration, but the final determination must be made from the then-current ledger immediately before implementation. This document does not self-authorize a reopen.

## 4. Non-goals and V1 exclusions

The first accepted version must not:

- choose or modify experiment IDs, datasets, seeds, parameters, anchors, budgets, thresholds, stopping rules, expected counts, or claim boundaries;
- edit handoff, registry, schema-v3 deltas, authority after-images, governance ledgers, or formal-execution policy;
- accept `.github/workflows/**`, authority, governance, handoff, registry, or Git metadata as task payload paths;
- launch liveness, pilot, formal training, aggregation, terminal audit, or result publication;
- create or update a scientific status;
- write to `main`, create a branch, force-push, merge, auto-merge, or mark a PR ready;
- accept arbitrary commands, scripts, expressions, environment variables, test lists, or path globs from task input;
- add a dependency;
- extract archives or accept binary, submodule, symlink, delete, rename, or mode-change payloads;
- automatically create or update a Draft PR;
- become the default route before final ReplayAB acceptance and a separate explicit adoption change.

Profile data-driven migration is a separate later claim. The runner must first prove useful on frozen historical payloads while preserving current E7/E8 implementations.

## 5. Four-stage development map

### Stage 0 — design, threat model, and replay contract

Documentation only. Stage 0 freezes:

- responsibility boundaries and non-goals;
- manifest, payload, branch, permission, and failure semantics;
- exact file and approval plan;
- code, effort, runtime, and complexity budgets;
- Stage 1 internal checkpoints and major acceptance;
- E7/E8 adapter boundaries;
- ReplayAB case selection, ordering, metrics, thresholds, and verdict rules;
- document-dependency, rollback, and stop rules.

Stage 0 remains open until every review pass in Section 7 is complete, the historical case inventory is frozen, command profiles are exact, current `main` is refreshed, and the user explicitly approves Stage 1.

### Stage 1 — shared runner closed loop

One development stage implements:

```text
strict manifest/payload validation
→ deterministic read-only plan
→ isolated apply and fixed validation
→ final-diff audit
→ one commit
→ same-branch non-force push
→ exact-head evidence
```

Internal checkpoints:

- **1A — planner:** strict parsing, identity, path, and plan hash; no mutation;
- **1B — isolated apply/validate:** patch verification, isolated mutation, fixed tests; no push;
- **1C — write liveness:** one guarded commit and same-branch non-force push.

The checkpoints are implementation controls, not separate large acceptance projects. Stage 1 has one major acceptance after all three pass.

### Stage 2 — E7/E8 thin-adapter qualification

Reuse the Stage 1 transaction unchanged. Add separate E7 and E8 fixed validation adapters. The adapters validate pre-approved values and deterministic outputs; they do not select, infer, tune, migrate, train, or interpret.

Stage 2 has one joint major acceptance after both adapters qualify.

### Stage 3 — final ReplayAB evaluation

Run the frozen paired case bank and issue one engineering verdict:

- `ADOPT`;
- `NARROW`;
- `REDESIGN`;
- `REJECT`.

No verdict changes repository defaults without a separate explicit change and merge approval.

## 6. Document-as-contract dependency

Every later stage must be mechanically and procedurally bound to this plan.

### 6.1 Stage packet

Before a stage starts, its durable packet must record:

- this document path;
- approved plan commit and document SHA-256;
- stage ID and approved status;
- current authoritative `main` SHA and resolution method;
- authorized paths and responsibilities;
- exact commands and validation profiles;
- code/runtime/effort budgets;
- exit gates, stop rules, and rollback;
- all required oral and governance approvals.

A run with a missing or mismatched plan identity is invalid.

### 6.2 No silent deviation

Work stops before implementation when it requires:

- an unlisted path or responsibility;
- a new Python path without exact approval;
- a new dependency;
- broader permissions;
- a new input field or executable input;
- a changed test command;
- a relaxed correctness, security, time, complexity, or replay threshold;
- a changed case, label, orientation, exclusion, or evaluator;
- any scientific responsibility or value change.

The plan must be amended and reviewed first. Code, failing tests, schedule pressure, or benchmark results cannot redefine the plan retrospectively.

### 6.3 Stage result binding

Each stage report must bind:

- plan commit and SHA-256;
- implementation base and head;
- changed paths and line-count method;
- commands actually executed and exact outcomes;
- active engineering time and unattended machine time separately;
- defects, deviations, unresolved items, and rollback state;
- one explicit `PASS`, `HOLD`, `REDESIGN`, or `STOP` decision.

A later stage may start only from a `PASS` report in the same plan lineage.

### 6.4 Change classes

- **Clarification:** no scope, permission, file, threshold, case, or responsibility change.
- **Contract amendment:** changes any of those items and requires pre-result review.
- **Result record:** records evidence without altering the frozen contract.

Adoption thresholds and case labels may never be relaxed after candidate results.

### 6.5 Boundary

This is an engineering implementation contract, not a second research master. It cannot change scientific status or override `docs/handoff.md`.

## 7. Repeated Stage 0 review

Stage 0 uses six independent review passes.

### Pass A — objective and ROI

Check that the workflow removes repeated work rather than moving it into setup, and that benefit can be measured independently from child-test and queue time. Cancel when a smaller existing command or documentation fix solves the same problem.

### Pass B — architecture and duplication

Check that Maintenance Runner, ReplayAB, V1, E7/E8, and Handoff Authority remain separate owners; existing validators are composed rather than copied; no hypothetical framework, plugin system, service, daemon, queue, database, or scheduler appears.

### Pass C — security and mutation safety

Check arbitrary-input rejection, target-branch identity, protected paths, path normalization, payload identity, stale-head races, concurrency, partial failure, cancellation, timeout, push rejection, log redaction, and GitHub-token trigger semantics.

### Pass D — governance and scientific safety

Check current closed-stage rules, exact approvals, new-Python path gate, code-change-budget gate, exclusion of authority/scientific paths, and separation of smoke/liveness/replay/formal evidence.

### Pass E — ReplayAB validity

Check identical controls except treatment, pre-result case and threshold freeze, evaluator independence, opposite-order repetitions, failure retention, per-family reporting, and independent correctness before efficiency release.

### Pass F — implementation and rollback feasibility

Check exact file responsibilities, realistic code/runtime budgets, dangerous-boundary tests, one-step disablement, preserved history, and an intact current route.

Stage 0 closes only when all six passes are `PASS` and no unresolved blocker remains.

## 8. V1 task contract

Each run consumes one immutable manifest and one approved unified diff. V1 does not support complete-file generation, Base64 chunk reconstruction, archives, or binary payloads.

Illustrative manifest:

```yaml
task_id: E7-PROFILE-EXAMPLE-01
schema_version: 1
task_type: patch_apply
plan_commit: 40-character lowercase SHA
base_commit: 40-character lowercase SHA
target_branch: dev/example
expected_head: 40-character lowercase SHA
payload_path: .github/maintenance-payload/E7-PROFILE-EXAMPLE-01/update.patch
payload_sha256: 64-character lowercase digest
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

Strict rejection applies to:

- unknown or missing fields;
- mutable refs, malformed identities, or identity mismatch;
- default branch, `main`, missing branch, or wildcard target;
- absolute paths, `..`, empty/control segments, normalization or case-fold collisions;
- undeclared paths or overlapping payload/output paths;
- executable command, script, expression, environment, or test input;
- workflow, handoff, registry, authority, governance, formal-execution, or Git metadata paths;
- binary, submodule, symlink, delete, rename, or mode changes;
- payload or manifest over the frozen size limits.

## 9. V1 transaction

The runner may only:

1. parse the manifest strictly;
2. verify plan and authorization packet identities;
3. resolve and reject the repository default branch as target;
4. verify base and expected-head commit objects;
5. verify the remote pre-existing target branch equals expected head;
6. verify manifest/payload size and SHA-256;
7. derive patch paths and change kinds without applying;
8. reject forbidden paths/change kinds;
9. create an isolated workspace from expected head;
10. run `git apply --check` and apply only there;
11. verify the resulting diff exactly matches the contract;
12. invoke one fixed repository-owned validation profile;
13. recheck remote head immediately before commit;
14. create one commit containing task, plan, manifest, payload, base, and parent identities;
15. re-audit committed tree and changed paths;
16. non-force push only to the same target branch;
17. emit append-only evidence and the exact final SHA.

Any failure before push leaves the remote unchanged. A non-fast-forward rejection is a safe stale-head result and must not trigger an automatic rebase or retry.

## 10. Fixed validation profiles

Task input selects an exact profile ID; it cannot provide commands.

### `generic_small_code`

- changed-scope compile when applicable;
- repository-owned Ruff selection;
- repository test selector and focused pytest;
- diff/path/mode audit;
- code-only handoff-authority no-op verification when current policy requires it.

### `e7_focused`

- `generic_small_code`;
- exact existing E7 profile/matrix/RunSpec tests;
- frozen scientific-field and branch-count checks owned by E7 code.

### `e8_focused`

- `generic_small_code`;
- exact existing E8 profile/cell/RunSpec tests;
- frozen scientific-field and cell-count checks owned by E8 code.

The exact commands must be frozen before Stage 1. V1 has no authority-materialization, workflow-change, paper-release, or formal-experiment profile.

## 11. Required failure matrix

| Case | Required outcome |
|---|---|
| valid manifest and patch | validate, test, commit, non-force push |
| unknown/malformed field | reject before workspace creation |
| stale base contract | reject |
| stale head at start | reject |
| head changes during validation | reject before commit or fail push safely |
| default/main/missing/undeclared branch | reject |
| path traversal, normalization, or case collision | reject |
| path outside allowlist | reject |
| protected or Git metadata path | reject |
| binary/submodule/symlink/delete/rename/mode change | reject |
| payload hash or size failure | reject before apply |
| patch apply failure | reject and discard isolated workspace |
| focused test failure or timeout | no commit and no push |
| cancellation before push | no remote change |
| cancellation after push | commit remains; recovery state reported precisely |
| commit succeeds but push rejects | remote unchanged |
| evidence upload fails after push | no complete-evidence claim; recover from commit identity through a report-only step |
| concurrent run on same branch | serialize or reject; never interleave |
| different branches | may run independently |
| flaky test | retain failure; no unplanned retry |
| `GITHUB_TOKEN` push does not retrigger PR CI | never claim recursive CI ran |
| `main` advances during frozen task | task base unchanged; divergence reported |
| nondeterministic generated output | independent acceptance fails unless predeclared |
| secret-like payload/log content | block or redact under a frozen rule |

No failure may be converted to success by widening paths, changing tests, changing payload, moving expected head, or altering the task contract.

## 12. Stage 1 major acceptance

Stage 1 passes only when:

- all 1A/1B/1C checkpoint tests pass;
- required failure classes fail closed;
- only isolated workspace mutates before guarded push;
- no commit/push follows failed validation;
- exact remote head is rechecked immediately before write;
- one low-risk documentation/fixture success liveness passes;
- one stale-head liveness and one test-failure liveness stop without push;
- the final commit is independently re-read from GitHub;
- exact-head full pytest, Ruff, compile, handoff-authority no-op, formal-execution, and governance-stage checks pass;
- code, effort, runtime, permission, and approval budgets pass.

The liveness may not touch scientific code/config, handoff, registry, workflows, authority, or governance files.

## 13. Stage 2 adapter contract

### E7 may validate

- exact experiment/profile and parent identities;
- frozen datasets/order, seeds, controls, anchors, exclusions, and expected branches;
- deterministic matrix expansion and RunSpec consistency;
- existing focused E7 tests.

### E8 may validate

- exact experiment/profile and parent identities;
- frozen corpus/init identities, seeds/offsets, parameter grid, controls, exclusions, and expected cells;
- deterministic matrix expansion and RunSpec consistency;
- existing focused E8 tests.

### Both are forbidden to

- propose or alter scientific values;
- infer from results;
- migrate profiles or impose a shared scientific schema;
- train, evaluate models, materialize handoff/registry, rank methods, or decide convergence/status.

Stage 2 acceptance requires at least three frozen E7 and three frozen E8 successful tasks plus one failure-boundary task per family. Every arm is independently judged under the same frozen contract.

## 14. ReplayAB protocol

### 14.1 Claim boundary

ReplayAB measures deterministic repository-transaction correctness, safety, active work, controlled wall time, candidate self-overhead, failure behavior, and maintenance complexity. It does not estimate future coding-agent patch-generation success.

### 14.2 Arms

- **Arm A:** frozen reconstruction of the accepted GitHub-App/manual plus existing-script route.
- **Arm B:** Maintenance Runner with identical task packet, payload, base, branch policy, validation, and evaluator.

No arm consumes the other arm's workspace, output, diagnostics, or timing.

### 14.3 Case bank

Freeze eight replayable cases before candidate efficiency results:

- four E7;
- four E8;
- in each family: config/RunSpec success, small code+test success, historically integration-heavy success, and failure-boundary case.

Freeze case ID, source history, base, payload hash, paths, treatment, evaluator, expected terminal state, replayability, environment/cache policy, exclusions, and ground-truth acceptance facts. No post-result removal, relabeling, orientation change, threshold change, or exclusion change.

### 14.4 Repetition

For every successful case:

- one A→B pair;
- one B→A pair;
- isolated workspaces and identical cache/environment policy;
- all failures/timeouts retained.

A third pair is mandatory when wall-time spread exceeds 15%, environment/cache identity differs, or a timeout boundary is crossed. Failure-boundary speed is diagnostic only.

### 14.5 Acceptance modes

- `semantic_acceptance` when multiple outcomes can be correct;
- `exact_artifact` only for truly deterministic outputs;
- `failure_boundary` when stopping is correct.

Efficiency is hidden until both arms are independently accepted with correct identity, behavior, outputs, paths, protected values, tests, provenance, terminal state, and no partial unauthorized mutation.

### 14.6 Controlling decision thresholds

The parent measurement claim is controlling. `ADOPT` requires all of the following:

1. every successful-path arm independently passes;
2. every failure-boundary case stops correctly without unauthorized mutation;
3. Arm B has zero false acceptance and zero protected-path regression;
4. no in-scope case is slower by more than `max(60 seconds, 5% of Arm-A median controlled replay time)`;
5. no slowdown comes from avoidable candidate overhead or duplicate work;
6. median controlled wall time decreases by at least 30%;
7. mean controlled wall time decreases;
8. median active operation time decreases by at least 30%;
9. command/tool-action count decreases by at least 60%;
10. manual intermediate copies and temporary workflow/PR use fall to zero for the accepted task class;
11. candidate self-overhead and code budgets pass;
12. E7 and E8 remain positive when reported separately.

Time to reviewable dev commit and intervention count remain useful diagnostics, but they cannot replace the controlling thresholds.

Correctness with benefit in only one family yields `NARROW`. A correctness or security failure yields `REJECT` unless benchmark validity itself failed before candidate judgment, in which case the result is `REDESIGN` under a newly frozen contract.

## 15. Code, runtime, and effort budgets

The parent measurement budget is controlling.

Preferred total new production code for the candidate:

| Component | Preferred lines |
|---|---:|
| Workflow YAML | 70–120 |
| Runner Python | 200–280 |
| Fixed mapping/glue | 30–50 |
| **Total** | **300–450** |

Rules:

- `451–500` production lines: yellow review and Stage 2 blocked;
- `>500` production lines: hard redesign or cancellation;
- tests/fixtures are separate but targeted;
- no minification, hidden heredoc implementation, unrelated-file placement, or artificial splitting;
- adapter addition above 120 production lines per family triggers redesign;
- no new dependency, service, plugin system, database, daemon, queue, dashboard, or scheduler;
- static planning target: median `<=250 ms`, p95 `<=1 s`;
- successful candidate self-overhead target: median `<=1 s`;
- self-overhead above `max(2 s, 2% of Arm-A median)` enters yellow review;
- self-overhead above `5 s`, duplicate scans/validators, or candidate-only network work triggers redesign;
- active engineering center: 15–24 hours; more than 27 hours requires an ROI restart decision.

## 16. Proposed files and approvals

Stage 0 proposes, but does not authorize:

- `.github/workflows/drpo-maintenance-runner.yml` — manual, branch-scoped workflow entrypoint;
- `scripts/run_drpo_maintenance_runner.py` — strict manifest, patch, isolated validation, commit, and same-branch push transaction;
- `tests/test_drpo_maintenance_runner.py` — positive, adversarial, failure-injection, and liveness-contract tests;
- non-Python manifests, fixtures, and reports under `docs/development_workflow_optimization/maintenance_runner/`.

The two proposed `.py` paths require separate explicit exact-path approval before creation.

Existing Python files are insufficient because ReplayAB judges outcomes, V1 owns pilot registration, E7/E8 own scientific behavior, and unrelated-file reuse would mix responsibilities and weaken rollback. If exact-path approval is not granted, redesign or stop; do not hide Python behavior in another file type.

The current code-change-budget gate is expected to trigger for Stage 1 and must not be bypassed.

## 17. GitHub permissions and trigger semantics

The write job must use least privilege:

- `contents: write` only in the guarded write job;
- no `pull-requests: write` in V1;
- no secret-bearing task input;
- branch concurrency lock;
- exact expected-head check before write;
- no credential persistence outside the explicit push step.

A `GITHUB_TOKEN` push is not assumed to retrigger PR workflows. Stage 1 must run its own declared exact-head validation, publish only evidence it produced, and never claim another workflow ran recursively. Additional CI must be explicitly triggered or independently observed.

## 18. Rollback and stop conditions

Before adoption, disable/remove the candidate on its dev branch and retain all commits, inputs, logs, reports, failures, and ReplayAB evidence. The current route remains unchanged.

Stop immediately on:

- false acceptance or protected-path mutation;
- wrong-branch write, branch creation, or force push;
- silent scientific change;
- post-result case/threshold alteration;
- missing scope, reopen, environment, or exact-path approval;
- production code above 500 lines;
- duplicate owner logic or negative controlled runtime;
- need for a service, dependency, arbitrary command, automatic merge, or broader task class.

Any later default-adoption rollback requires one reviewed commit and explicit approval; scientific artifacts and experiment state remain untouched.

## 19. Stage 0 unresolved items

Stage 0 remains open until these are frozen:

1. exact eight-case inventory and artifact availability;
2. exact command lists for all three validation profiles;
3. payload/manifest size limits from historical distribution;
4. durable evidence recovery after successful push plus artifact-upload failure;
5. first liveness payload selection;
6. exact new-Python path approvals;
7. current-ledger closed-stage reopen scope and authorization;
8. refresh/rebase to then-current `main` and reread startup sources.

No item may be resolved post hoc to improve benchmark results.

## 20. Stage ledger and review record

| Stage | Status | Next decision |
|---|---|---|
| 0 — design/threat/replay contract | `active_under_review` | resolve all holds, freeze inventory/commands, obtain Stage 1 approvals |
| 1 — shared runner closed loop | `not_started` | major safety/CI/runtime acceptance |
| 2 — E7/E8 thin adapters | `not_started` | joint adapter qualification |
| 3 — ReplayAB final evaluation | `not_started` | `ADOPT/NARROW/REDESIGN/REJECT` |

Current review cycle: `stage0-review-2`.

- Pass A — objective/ROI: `provisionally_pass`; exact case inventory remains open.
- Pass B — architecture/duplication: `pass`; owner boundaries are explicit and profile migration is excluded.
- Pass C — security/mutation: `pass_with_v1_exclusions`; dangerous change kinds and protected paths are excluded.
- Pass D — governance/scientific safety: `hold`; dedicated scope, closed-stage reopen determination, large-code approval, and exact Python-path approvals are not yet complete.
- Pass E — ReplayAB validity: `hold`; exact case/evaluator inventory is not yet frozen.
- Pass F — feasibility/rollback: `provisionally_pass`; command profiles, size limits, and evidence-recovery behavior remain open.

Stage 0 is not closed. Stage 1 behavior code must not begin from this review state.
