# DRPO Maintenance Runner + ReplayAB phased plan

**Claim:** `GOV-DRPO-MAINTENANCE-RUNNER-REPLAYAB-01`  
**Parent measurement claim:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Base:** `main@d7e82201159a736f0d2e48403aae15ea07e178a7`  
**Status:** design frozen for staged implementation; no workflow implementation, write permission, default-route change, scientific execution, or merge is authorized by this document alone  
**Scientific impact:** none

## 1. Decision

Develop one small shared maintenance-runner candidate, then judge it with the existing ReplayAB Core before considering adoption.

The objects remain separate:

- **Maintenance Runner** is the candidate workflow being tested.
- **ReplayAB Core** is the measurement instrument and must not be rewritten into the candidate.
- **E7/E8 adapters** are thin domain validators added only after the shared runner passes its own safety and cost gates.
- **Handoff Authority** remains a separate authority validator; it is not the shared runner.

No candidate becomes the default repository route merely because it is implemented or because one liveness run succeeds.

## 2. Problem being solved

Recent E7/E8 successor-profile tasks repeatedly spent substantial time on mechanical repository work:

1. exact-base and branch checks;
2. applying or reconstructing approved changes;
3. selecting and running focused validation;
4. checking changed paths and frozen scientific fields;
5. creating a commit on a development branch;
6. obtaining exact-head CI evidence;
7. preparing a reviewable Draft PR.

These steps are repeated across sessions and are often rebuilt as temporary workflows. The target is to reduce this repeated mechanical work without automating scientific decisions.

The candidate does **not** attempt to make model training, full CI, or scientific review intrinsically faster.

## 3. Existing components that must be reused

The implementation must compose existing owners rather than duplicate them:

- connected GitHub App development route from `AGENTS.md`;
- existing PR gates and test-selection mechanisms;
- existing ReplayAB Core exact-artifact, failure-boundary, and R2 semantic-acceptance capabilities;
- existing RunSpec and pilot-registration tools where applicable;
- existing handoff authority and governance validators;
- experiment-specific E7/E8 loaders and validators until a separately reviewed data-driven migration is approved.

The candidate may call these components. It may not reimplement their authority or scientific semantics.

## 4. First-version scope

The first useful version supports one narrow transaction:

```text
frozen task manifest + approved patch or complete text payload
→ exact base/head validation
→ allowed-path audit
→ apply in isolated development workspace
→ run declared focused validation
→ verify final changed paths and protected values
→ create one commit on the declared dev branch
→ emit logs, hashes, timing, and final commit SHA
```

The first version does not automatically:

- choose E7/E8 parameter values, seeds, budgets, thresholds, or experiment IDs;
- generate scientific claims;
- change `docs/handoff.md` or `experiments/registry.yaml`;
- create a ready RunSpec or upgrade its state;
- launch liveness, pilot, or formal training;
- push to `main`, force-push, merge, auto-merge, or mark a PR ready;
- accept arbitrary shell commands;
- create branches not predeclared by the task;
- modify workflow, governance, authority, or protected scientific files unless a later phase explicitly authorizes that task class.

## 5. Candidate task contract

Every run must consume one immutable, reviewable manifest. The manifest is data, not executable code.

Minimum fields:

```yaml
task_id: E7-PROFILE-EXAMPLE-01
task_type: patch_apply
base_commit: 40-character SHA
target_branch: dev/example
expected_head: 40-character SHA
payload:
  kind: unified_diff
  path: .github/agent-payload/E7-PROFILE-EXAMPLE-01/update.patch
  sha256: 64-character digest
allowed_paths:
  - configs/example.json
  - runspecs/templates/example.yaml
forbidden_paths:
  - docs/handoff.md
  - experiments/registry.yaml
validation_profile: e7_focused
expected_change:
  min_files: 1
  max_files: 3
  require_clean_result: true
```

The schema must reject unknown keys, non-full SHAs, wildcard branch escalation, path traversal, overlapping allowed/forbidden paths, arbitrary commands, and mutable remote references.

## 6. Shared runner responsibilities

The shared runner owns only the transaction mechanics:

1. validate manifest and payload identities;
2. prove the branch is not `main` and matches the declared development branch;
3. prove the checked-out head equals `expected_head`;
4. reconstruct and hash-check the payload when chunking is used;
5. run `git apply --check` before mutation for patch payloads;
6. audit changed paths before and after application;
7. invoke a fixed, repository-owned validation profile;
8. stop without commit or push on any failure;
9. create one non-force commit only after all declared checks pass;
10. emit append-only evidence containing exact inputs, commands selected by repository code, timestamps, outcomes, and final SHA.

It does not own E7/E8 matrix semantics, handoff authority, or scientific interpretation.

## 7. Thin validation profiles

A validation profile is a repository-owned fixed mapping, not user-supplied shell.

Initial profile set:

- `generic_small_code`: compile, Ruff, selected pytest, diff check;
- `e7_focused`: existing E7 focused tests plus matrix/RunSpec checks already owned by E7 code;
- `e8_focused`: existing E8 focused tests plus matrix/RunSpec checks already owned by E8 code.

Handoff/registry materialization is explicitly excluded from the first implementation. A later `authority_materialization` profile requires its own design and authorization because it writes authoritative after-images.

## 8. Profile data-driven migration boundary

Moving E7 or E8 profile data out of Python is related but separate.

The maintenance runner can be evaluated before profile migration by applying frozen historical patches. Profile migration is considered only after the shared transaction proves useful.

A later migration must preserve, byte-for-byte or semantically under a frozen evaluator:

- experiment IDs;
- datasets and order;
- seeds;
- control points and anchors;
- expected branch counts;
- RunSpec fields;
- output naming and provenance;
- all frozen scientific exclusions and claim boundaries.

No migration may combine E7 and E8 scientific schemas merely for architectural uniformity.

## 9. Phased development

### Stage 0 — design and replay contract

Deliverables:

- this plan;
- frozen candidate responsibility;
- code/runtime budgets;
- replay validity and adoption gates;
- no behavior-changing code.

Exit gate:

- current-main review complete;
- no scientific or authority files changed;
- ReplayAB reuse path identified;
- user approves proceeding to Stage 1.

### Stage 1 — read-only manifest planner

Implement strict manifest parsing and deterministic planning without applying files, committing, or pushing.

The planner must output:

- resolved exact base/head;
- candidate changed paths from the payload;
- selected fixed validation profile;
- forbidden-path result;
- deterministic plan hash.

Exit gate:

- focused positive and adversarial tests pass;
- no repository mutation occurs;
- median planner overhead is at most 250 ms and p95 at most 1 s in fixture replay;
- production implementation remains within the Stage 1 line budget.

### Stage 2 — isolated apply and validation

Add payload reconstruction, SHA-256 verification, `git apply --check`, isolated application, and fixed validation-profile execution.

Still forbidden:

- commit;
- push;
- PR creation;
- workflow write permission.

Exit gate:

- success, malformed payload, stale head, forbidden path, failing test, partial apply, and interruption cases all retain evidence and fail closed;
- no source workspace mutation outside the isolated worktree;
- candidate self-overhead excludes child-test time and remains within budget.

### Stage 3 — dev-branch commit liveness

Add one-commit behavior for a pre-existing declared dev branch. No automatic PR creation yet.

Safety requirements:

- `main` and default branch rejected;
- no new branch creation;
- no force push;
- exact expected head rechecked immediately before commit and push;
- tests must pass before commit;
- final tree and changed paths re-audited after commit;
- push only to the same declared dev branch;
- concurrent runs for the same branch serialized.

The first real liveness uses a disposable low-risk documentation or fixture-only task, not scientific code and not handoff/registry.

Exit gate:

- exact-head liveness succeeds;
- stale-head and test-failure liveness stop without push;
- final commit provenance is independently verified.

### Stage 4 — E7 adapter qualification

Add only the minimum E7-specific adapter needed to validate already-approved successor-profile material.

The adapter may validate:

- exact profile ID and parent;
- frozen datasets, seeds, controls, anchors, and exclusions;
- deterministic branch expansion and expected count;
- RunSpec consistency;
- focused E7 tests.

It may not choose or alter those values.

Exit gate:

- at least three frozen E7 historical tasks can be replayed by both arms;
- every arm independently satisfies the same semantic acceptance contract;
- no E7 scientific value changes between accepted A and B outcomes.

### Stage 5 — E8 adapter qualification

Reuse the shared transaction and add a separate E8 validator. Do not force E8 into the E7 schema.

Exit gate:

- at least three frozen E8 historical tasks can be replayed by both arms;
- expected cells, seeds, profile identity, and RunSpec outputs match the frozen contracts;
- no scientific execution occurs.

### Stage 6 — ReplayAB paired evaluation

Run the frozen replay bank described below. This stage produces an engineering decision only.

Possible decisions:

- `ADOPT`: candidate qualifies for a separately approved default-route change;
- `NARROW`: useful only for a smaller task class;
- `REDESIGN`: benefit exists but safety, complexity, or runtime fails;
- `REJECT`: no sufficient net benefit.

No decision automatically changes `main` or repository defaults.

## 10. ReplayAB protocol

### 10.1 What is measured accurately

The replay measures the deterministic repository transaction layer:

- correctness and safety of the resulting development commit;
- elapsed transaction time under a controlled environment;
- active interventions and command count;
- failure and recovery behavior;
- candidate overhead;
- output consistency under frozen scientific constraints.

It does not estimate the probability that a future coding model invents the correct patch. Both arms receive the same frozen task packet and approved payload or equivalent source material.

### 10.2 Arm definitions

- **Arm A — current route:** the currently accepted manual/GitHub-App plus existing script sequence reconstructed as a frozen command plan.
- **Arm B — candidate route:** the Maintenance Runner using the same base, payload, branch policy, tests, and acceptance contract.

Neither arm may use the other arm's outputs.

### 10.3 Frozen case bank

Before candidate results, select 8 representative replayable tasks:

- 4 E7 tasks;
- 4 E8 tasks.

Each family should include, when historical evidence permits:

1. a config/RunSpec-only change;
2. a small code-plus-test change;
3. a task that previously required temporary workflow plumbing;
4. a failure-boundary case such as stale base, forbidden path, or failing validation.

The final case IDs, bases, payload hashes, expected outcomes, and exclusions must be frozen in a separate inventory before Stage 2 behavior results are observed. Cases may not be removed after an unfavorable result.

### 10.4 Repetition and ordering

For every successful-path case:

- run two paired repetitions;
- one repetition uses A→B;
- the other uses B→A;
- use isolated workspaces and the same cache policy;
- retain all failures and timeouts.

Run a third paired repetition when the paired wall-time spread exceeds 15% or a cache/environment mismatch is detected.

Failure-boundary cases require deterministic repeated agreement rather than speed comparison.

### 10.5 Independent acceptance

Use ReplayAB R2 semantic acceptance when different commit metadata or implementation trees can still be correct. Use exact-artifact mode only when the frozen task truly requires identical outputs.

Each arm must independently pass:

- declared behavior and output completeness;
- changed-path and protected-path rules;
- frozen scientific-value checks;
- relevant focused tests;
- repository governance checks applicable to the task;
- exact input and provenance bindings;
- expected terminal state.

Efficiency results are hidden until both arms are accepted.

### 10.6 Metrics

Primary metrics:

- independently accepted task rate;
- safety-boundary agreement rate;
- median active intervention count;
- median repository commands/tool actions;
- median time to reviewable dev-branch commit;
- median controlled end-to-end wall time;
- candidate self-overhead excluding child validation;
- failed-attempt and retry count;
- changed production lines and maintenance complexity.

Secondary diagnostics:

- test-selection time;
- payload reconstruction time;
- commit/push time;
- cache/order effects;
- per-task rather than aggregate-only outcomes.

### 10.7 Adoption gate

`ADOPT` is allowed only if all conditions pass:

1. both arms are independently accepted on every successful-path replay;
2. all failure-boundary cases stop at the correct boundary without partial unauthorized mutation;
3. Arm B introduces no false acceptance or protected-path regression;
4. median active interventions fall by at least 40%;
5. median time to reviewable dev commit falls by at least 25%;
6. Arm B controlled wall time does not regress by more than 5% on any task family;
7. median candidate self-overhead is at most 2 seconds and p95 at most 5 seconds, excluding child tests;
8. no hidden rerun of full tests or validators is used to manufacture telemetry;
9. preferred production-line budget is met, or a pre-result redesign review approves a yellow-zone exception;
10. the result remains positive when E7 and E8 are reported separately.

If correctness passes but only one family benefits, the decision is `NARROW`, not `ADOPT`.

## 11. Code and runtime budgets

The candidate must remain small.

Preferred cumulative production budget through Stage 3:

| Component | Preferred lines |
|---|---:|
| Workflow YAML | 80–140 |
| Shared non-Python runner/helper | 160–260 |
| Fixed validation-profile mapping | 40–80 |
| **Total** | **280–480** |

Rules:

- yellow review above 500 production lines;
- hard stop above 650 production lines before E7/E8 adapters;
- tests and fixtures are outside the production budget but must remain targeted;
- no service, daemon, database, dashboard, queue, scheduler, or new dependency;
- no new Python file without exact-path oral approval under repository policy;
- adapter code above 200 production lines per experiment family triggers redesign rather than framework expansion.

A candidate that saves manual actions but increases controlled wall time or maintenance complexity beyond these gates is a negative optimization.

## 12. Security and governance boundaries

The write-capable phase must enforce:

- least-privilege `contents: write` only where required;
- no `pull-requests: write` in the first implementation;
- no arbitrary command input;
- no secret disclosure in artifacts or logs;
- payload and manifest SHA-256 binding;
- exact base and exact expected head;
- same-branch non-force push only;
- branch-level concurrency lock;
- fixed path allowlist and denylist;
- failure before mutation whenever possible;
- no commit or push after failed validation;
- complete provenance and retained failure evidence.

Any future automatic Draft PR creation, authority materialization, workflow-file modification, or protected-environment approval is a separate scope expansion.

## 13. Relationship to existing Handoff Authority PR

The pending Handoff Authority workflow is a separate read-only checker. It is neither required for Stage 1 nor a substitute for the Maintenance Runner.

The first Maintenance Runner version forbids handoff/registry changes. Therefore it does not depend on cross-workflow triggering.

A later authority-aware task class must invoke the existing authority validator inside the same transaction and requires a separately approved design.

## 14. Rollback

Before default adoption, rollback is simply:

- stop the candidate branch and workflows;
- keep the current GitHub App/manual route unchanged;
- retain ReplayAB evidence and failed cases;
- do not delete historical candidate commits or reports.

After any later default adoption, rollback must:

- disable the candidate trigger or remove the default-route reference in one reviewed commit;
- restore the previously documented route;
- preserve all task manifests, logs, and provenance;
- require explicit user approval.

No scientific artifact or experiment status is affected by rollback.

## 15. Stage ledger

| Stage | Status | Main impact | Next gate |
|---|---|---|---|
| 0 — design and replay contract | `active` | none | document review and user approval |
| 1 — read-only planner | `not_started` | none | focused adversarial tests and overhead gate |
| 2 — isolated apply/validate | `not_started` | none | fail-closed fixture replay |
| 3 — dev-branch commit liveness | `not_started` | no default change | three real safety liveness cases |
| 4 — E7 adapter | `not_started` | none | frozen E7 replay qualification |
| 5 — E8 adapter | `not_started` | none | frozen E8 replay qualification |
| 6 — paired ReplayAB | `not_started` | none | `ADOPT/NARROW/REDESIGN/REJECT` decision |

Development stops at every stage boundary for review. Later-stage authorization is not inferred from an earlier-stage pass.
