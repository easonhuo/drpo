# DRPO M0 Atomic Development Transaction + ReplayAB controlling plan

**Claim:** `GOV-DRPO-MAINTENANCE-RUNNER-REPLAYAB-01`  
**Measurement authority:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Document version:** `1.1-stage0-review-A`  
**Initial design branch base:** `main@d7e82201159a736f0d2e48403aae15ea07e178a7`  
**Current main reviewed:** `main@ad9bda80796dcf5c48976f5d64ffd79a006c70d5`  
**Status:** `stage0_review_A_applied`  
**Scientific impact:** none  
**Implementation authorization:** none

This document supersedes the implementation direction in
`docs/development_workflow_optimization/MAINTENANCE_RUNNER_REPLAYAB_PLAN.md`.
The superseded M2 design remains intact in Git history and as a historical document.
No earlier review or abandoned design is deleted.

## 1. Controlling decision

The first iteration will **not** implement a new patch runner, integration engine,
GitHub Actions workflow, E7 adapter, or E8 adapter.

The first candidate is **M0: a documented, hash-bound atomic GitHub development
transaction using capabilities already provided by the connected GitHub App and the
repository's existing validators**.

M0 is evaluated before any new runtime code is authorized.

The previously proposed M2 transaction

```text
patch parse
→ isolated apply
→ validation
→ commit construction
→ branch push
→ transaction evidence
```

is rejected for this iteration because it materially duplicates responsibilities
already owned by V1 integration, Candidate 01 orchestration, existing validators, and
GitHub's native Git-object transaction.

M1, a thin publisher for an already verified commit or V1 `READY` identity, is not
pre-authorized. It may be proposed only when frozen M0 evidence proves one specific
publication gap that cannot be removed by a smaller documentation or connector-use
change.

## 2. Exact problem class

M0 addresses one narrow recurring task:

```text
reviewed complete file after-images
+ exact repository base
+ fixed validation contract
→ one atomic commit on a dedicated development branch
→ one Draft PR
→ independently observed exact-head checks
```

The treatment begins only after the desired file contents are already known and
reviewed. M0 does not generate, repair, or scientifically review those contents.

M0 may receive credit only for reducing:

- repeated per-file remote write operations;
- partial intermediate branch states;
- unnecessary multi-commit reconstruction;
- repeated branch/base/diff bookkeeping;
- repeated manual assembly of the same validation and evidence checklist.

M0 may not receive credit for reducing:

- scientific design or implementation time;
- E7/E8 profile hard-coding;
- model-agent reasoning or patch-generation errors;
- V1 registration or authority materialization work;
- training, evaluation, aggregation, or terminal audit time;
- GitHub queue time outside its treatment;
- hardware, dependency, data, credential, or network failures.

## 3. Existing-owner capability map

| Owner | Existing responsibility | M0 relationship |
|---|---|---|
| Connected GitHub App | read current main, create branches, create blobs/trees/commits, non-force ref updates, Draft PRs, inspect CI | M0 standardizes these existing operations |
| Direct contents-file route | simple single-file writes and sequential commits | Arm-A baseline where historically used |
| Git object transaction | one tree and one commit containing multiple reviewed after-images | Arm-B M0 treatment |
| V1 dev integration | reviewed dev snapshot to locally verified `READY`, including authority and gates | excluded; never reimplemented |
| Candidate 01 | one-click preparation plus V1 composition | excluded from M0 benefit |
| Existing E7/E8 code and tests | profile, matrix, RunSpec, and scientific-invariant validation | invoked through frozen commands only |
| Handoff Authority | schema-v3 handoff/registry authority | protected and excluded |
| ReplayAB Core | case contracts, independent acceptance, paired comparison, decision evidence | measures M0; not embedded in M0 |

A capability being available does not prove the end-to-end M0 procedure is sufficient.
A Stage 0 non-publishing check confirmed that the current connector can construct a
tree update from a commit base and existing blob without creating a ref, branch, commit,
or PR. Stage 1 must still verify the complete remote procedure under the exact task
class below.

## 4. M0 task-class boundary

### 4.1 Included

The first qualification covers small, reviewed development changes that:

- start from an authoritative full `main` SHA;
- use one new dedicated `dev/<claim>` branch, or an existing branch whose exact expected
  head is pinned;
- consist only of complete UTF-8 file after-images;
- change repository code, tests, configs, RunSpecs, or non-authoritative documentation;
- use existing repository validators without changing their implementation;
- do not require handoff/registry materialization;
- do not launch an experiment;
- can be reviewed through one Draft PR.

### 4.2 Excluded

M0 does not handle:

- `docs/handoff.md`, `experiments/registry.yaml`, schema-v3 deltas, generated authority
  views, governance-stage ledgers, or authority implementation;
- `.github/workflows/**` or branch-protection/default-policy changes;
- formal-execution policy or result-status changes;
- binary files, symlinks, gitlinks, Git LFS pointers, archive extraction, or secret
  material;
- file deletion, rename, or executable-mode change in the first qualification;
- automatic merge, auto-merge, ready-for-review transition, approval, or experiment launch;
- arbitrary user-supplied commands or test expressions;
- profile migration or a shared E7/E8 scientific schema.

A task outside this boundary follows its existing owner. It is not a failed M0 case.

## 5. M0 atomic transaction contract

### 5.1 Frozen input packet

Every attempt binds:

- `task_id`;
- authoritative `base_commit`;
- target branch name;
- target mode: `new_branch_from_base` or `existing_branch_exact_head`;
- `expected_head` when the branch already exists;
- complete reviewed after-images;
- SHA-256 for every after-image;
- exact allowed paths;
- expected file modes;
- fixed validation profile ID;
- expected changed-file inventory;
- stage-plan commit and plan-file SHA-256;
- reviewer decision identity;
- rollback and recovery class.

The packet contains data only. It contains no shell command, Python expression,
environment override, branch wildcard, mutable ref, or test list.

### 5.2 Pre-write checks

Before any ref is created or moved:

1. resolve current `main` through the GitHub repository API;
2. require the resolved SHA to equal the packet base;
3. validate target-branch policy and reject `main` or the default branch;
4. normalize and audit every repository path;
5. reject protected paths and unsupported file kinds;
6. verify all after-image hashes and expected modes;
7. verify the exact validation profile and expected changed-file inventory;
8. verify all required approvals and the plan identity.

### 5.3 Atomic Git-object construction

M0 uses one immutable object sequence:

1. create one blob per reviewed after-image;
2. create one tree from the exact parent tree plus the approved entries;
3. create one commit whose unique parent is the frozen base or expected head;
4. independently inspect the commit tree and parent before publication;
5. create or non-force update only the declared dev-branch ref;
6. independently re-read the remote ref and commit from GitHub.

Creating blobs, a tree, or a commit object does not publish a branch. The only
publication boundary is the branch-ref creation or non-force update.

### 5.4 Draft PR and CI

After successful ref publication:

1. open one Draft PR to `main`, or reuse the already declared Draft PR for an existing
   branch;
2. verify PR base, head branch, and exact head SHA;
3. observe applicable GitHub checks on that exact head;
4. independently inspect changed paths and diff statistics;
5. retain the PR URL, exact head, workflow-run identities, and conclusions.

M0 never infers that a token-originated push retriggered CI. A check is reported only
when an actual exact-head run is observed.

### 5.5 Terminal states

The permitted terminal states are:

- `PLANNED_NO_WRITE`;
- `BLOCKED_PREWRITE`;
- `OBJECTS_CREATED_REF_UNCHANGED`;
- `PUBLISHED_PR_PENDING`;
- `PUBLISHED_CHECKS_FAILED`;
- `PUBLISHED_CHECKS_PASSED`;
- `PUBLISHED_EVIDENCE_INCOMPLETE`;
- `STALE_HEAD`;
- `CANCELLED_NO_REF_CHANGE`.

No state is called complete merely because a commit object or branch ref exists.

## 6. Failure and recovery semantics

| Boundary | Required behavior |
|---|---|
| base or expected-head mismatch | stop before object/ref mutation |
| invalid or protected path | stop before object/ref mutation |
| blob/tree/commit object creation fails | branch ref remains unchanged |
| final tree or parent audit fails | branch ref remains unchanged |
| branch creation/update is non-fast-forward or rejected | no retry, rebase, or force push |
| PR creation fails after ref publication | retain commit; report-only recovery may open the same PR later |
| exact-head checks fail | preserve branch and evidence; no automatic repair |
| evidence collection fails after publication | label `PUBLISHED_EVIDENCE_INCOMPLETE`; recover by re-reading immutable GitHub identities |
| cancellation before ref publication | remote branch unchanged |
| cancellation after ref publication | remote commit remains; recovery resumes from immutable ref identity |
| current main advances after task freeze | report divergence; do not rewrite task base post hoc |
| flaky test | preserve failure; no unplanned retry |
| unknown terminal condition | fail closed and stop qualification |

Report-only recovery may add or repair evidence records. It may not alter the code
commit, move the branch, change the validation contract, or replay the content write.

## 7. Validation profiles

M0 does not implement validators. It references exact repository-owned commands frozen
in `maintenance_runner/VALIDATION_PROFILE_MATRIX.yaml`.

The initial profile IDs are:

- `generic_small_update`;
- `e7_small_update`;
- `e8_small_update`.

Each profile must specify:

- pre-publication checks that can run on the reviewed after-images or an exact checkout;
- exact-head PR checks that must be independently observed;
- expected no-op governance checks;
- protected scientific fields or generated-count checks;
- timeout and failure semantics.

No task packet may supply or modify commands.

## 8. Three-stage execution map

### Stage 0 — contract and evidence freeze

Documentation only.

Required deliverables:

- this controlling plan;
- `M0_ATOMIC_DEV_TRANSACTION_SPEC.md`;
- `VALIDATION_PROFILE_MATRIX.yaml`;
- `STAGE_PACKET_TEMPLATE.yaml`;
- `TRANSACTION_EVIDENCE_TEMPLATE.yaml`;
- `CASE_INVENTORY.yaml`;
- `EVALUATOR_BINDINGS.yaml`;
- independent Stage 0 review records.

Exit requires:

- current-main reread;
- exact M0 owner and task boundary;
- no unresolved M2 ownership overlap;
- exact validation commands;
- exact eight-case inventory and artifact availability;
- independent evaluator bindings;
- frozen decision thresholds;
- no required new Python path, workflow, dependency, or closed-stage reopen for M0;
- all review passes `PASS`;
- explicit user approval to begin Stage 1.

### Stage 1 — M0 qualification

Stage 1 adds no repository runtime code.

It performs:

- one low-risk E7-derived success transaction;
- one low-risk E8-derived success transaction;
- one stale-head failure transaction;
- one validation-failure transaction.

Stage 1 passes only when:

- every success produces exactly one approved dev-branch commit and one Draft PR;
- both failure cases stop at the expected boundary;
- no protected or scientific state changes;
- exact-head checks and evidence are independently observed;
- operation count, active effort, and failure recovery are recorded;
- M0 remains simpler than the recurring work it replaces.

Stage 1 does not authorize routine use or adoption.

### Stage 2 — frozen ReplayAB evaluation and verdict

Stage 2 has two deliberately separated evidence layers:

1. **controlled local-git replay:** the existing ReplayAB `local-git-v1` machinery
   executes equivalent historical sequential operations and M0 atomic operations with
   monotonic timing and isolated repositories;
2. **remote operational context:** Stage 1 GitHub ref, Draft PR, and exact-head CI
   observations validate remote semantics but are not mixed into the controlled causal
   timing estimate.

No new ReplayAB backend is authorized by default. If the frozen M0 operations cannot be
represented by the current local-git instrument, Stage 0 returns to `HOLD`; implementation
does not silently expand ReplayAB.

Run the eight frozen cases in balanced opposite order.

Possible verdicts:

- `ADOPT_M0`;
- `NARROW_M0`;
- `REDESIGN_TO_M1`;
- `REJECT_OPTIMIZATION`.

A verdict does not alter repository defaults by itself.

## 9. M1 contingency gate

M1 may be designed only when all conditions hold:

1. M0 correctness and safety pass;
2. a repeated, measured publication step remains materially operator-heavy or unsafe;
3. the missing responsibility is exactly:
   `verified commit identity → same-branch non-force publication → publication evidence`;
4. existing GitHub App operations cannot remove the loss through a smaller contract or
   template change;
5. the gap occurs in at least two frozen successful cases;
6. the projected M1 implementation and maintenance cost is below the measured recurring
   loss;
7. a new pre-result contract, scope, rollback, code budget, and approvals are frozen.

M1 may not parse or apply patches, construct scientific contents, rerun V1, own
authority, open or merge PRs, or become a general workflow engine.

A triggered M1 is a new reviewed contract amendment. Stage 1 code cannot begin merely
because M1 appears convenient.

## 10. ReplayAB design

### 10.1 Arms

- **Arm A:** the exact accepted or historically used route frozen separately for each
  case, including its actual file, commit, and review operations without artificial
  delay.
- **Arm B:** the M0 atomic transaction using the same file after-images, base, validation,
  evaluator, environment policy, and expected result.

A case whose historical Arm A already used an equivalent single atomic transaction has
no treatment contrast and cannot be selected as a successful efficiency case.
Arm B receives no better scientific content than Arm A.

### 10.2 Case bank

Freeze eight cases:

- four E7-derived;
- four E8-derived;
- in each family:
  - one config/RunSpec-only success;
  - one code-plus-test success;
  - one historically integration-heavy but M0-in-scope success;
  - one failure-boundary case.

Each entry freezes:

- source PR/commit and provenance;
- base and complete after-images;
- per-file SHA-256 and modes;
- changed paths;
- validation profile;
- evaluator identity;
- expected terminal state;
- environment/cache policy;
- original task limitations;
- replayability and missing-evidence classification.

No case may include handoff/registry materialization as M0 treatment benefit.

### 10.3 Independent acceptance

ReplayAB independently re-reads:

- final branch ref and commit parent;
- final tree and modes;
- changed-path inventory;
- exact-head check conclusions;
- protected-path and scientific-value invariants;
- expected terminal state;
- absence of partial unauthorized mutation.

Candidate-produced summaries are evidence inputs, not the acceptance oracle.

### 10.4 Evidence-layer separation

Controlled timing, child-operation timing, and operator-action counts come from the
existing local-git ReplayAB execution record. GitHub PR and Actions timestamps are
reported separately as operational context.

Remote GitHub qualification may confirm:

- commit parent/tree and branch-ref semantics;
- Draft PR identity;
- exact-head check existence and conclusion;
- recovery after ref or evidence failures.

It may not replace controlled timing or be combined with it into one estimate.

### 10.5 Repetition

For each successful case:

- one A→B pair;
- one B→A pair;
- isolated branches/workspaces;
- identical cache and environment policy;
- retain all failures, invalidations, and timeouts.

A third pair is required when paired wall-time spread exceeds 15%, environment/cache
identity differs, or a timeout boundary is crossed.

Failure-boundary cases are repeated for deterministic agreement; speed is diagnostic.

## 11. Controlling decision thresholds

The parent ReplayAB thresholds remain controlling.

`ADOPT_M0` requires:

1. every successful arm independently passes;
2. every failure-boundary case stops correctly;
3. zero false acceptance, protected-path regression, or scientific-state mutation;
4. no in-scope case slows by more than
   `max(60 seconds, 5% of Arm-A median controlled time)`;
5. median controlled wall time improves by at least 30%;
6. mean controlled wall time improves;
7. median active-operation time improves by at least 30%;
8. command/tool-action count falls by at least 60%;
9. manual intermediate copies and temporary workflow/PR use are zero for the accepted
   task class;
10. E7 and E8 remain positive when reported separately;
11. M0 introduces no production code, dependency, service, or new durable state.

Correctness with benefit in only one predeclared family yields `NARROW_M0`.
A correctness or security failure yields `REJECT_OPTIMIZATION`.
A valid M0 with a specifically measured remaining publisher gap may yield
`REDESIGN_TO_M1`.

## 12. Document-as-contract binding

Every stage packet records:

- this plan path;
- approved plan commit and file SHA-256;
- current-main SHA and resolution method;
- stage ID and authorization status;
- exact task/case identities;
- allowed paths and expected after-image hashes;
- validation profile and evaluator binding;
- thresholds, terminal states, rollback, and stop rules;
- commands actually executed and results observed.

Any mismatch invalidates the run.

The plan must be amended before work when a new path, permission, command, case,
threshold, evaluator, responsibility, or scientific value is required.

No implementation difficulty or benchmark result may relax the frozen contract.

## 13. Review protocol

Stage 0 requires at least three separately recorded reviews after this rewrite:

### Review R-A — ownership and necessity

Falsify whether M0 still duplicates V1, Candidate 01, ReplayAB, or E7/E8 domain owners.
Confirm that zero-code M0 is the smallest candidate.

### Review R-B — failure, security, and governance

Audit partial object creation, ref publication, stale-head races, PR/CI failures,
evidence recovery, permissions, protected paths, scientific exclusions, and current
governance requirements.

### Review R-C — replay validity and stability

Audit Arm-A fairness, treatment isolation, evaluator independence, case availability,
timing validity, thresholds, M1 trigger conditions, rollback, and cross-session
continuation.

A review may return `PASS`, `HOLD`, `REDESIGN`, or `STOP`.
Stage 0 closes only when all three reviews pass and every deliverable is frozen.

## 14. Rollback and stop conditions

Before adoption, rollback is to stop using the M0 packet/template and continue the
existing direct route. No repository runtime component must be disabled or migrated.

Stop or redesign immediately when:

- M0 cannot be distinguished from Arm A;
- the atomic route requires a workflow, new Python code, dependency, service, or second
  integration engine;
- final correctness cannot be independently evaluated;
- a protected path or scientific state is touched;
- cases or thresholds need post-result changes;
- the task class is too narrow to repay documentation and replay cost;
- M1 would absorb patch application, validation ownership, V1, authority, PR management,
  or scientific logic.

## 15. Stage ledger

| Stage | Status | Next gate |
|---|---|---|
| 0 — M0 contract and evidence freeze | `rewrite_under_review` | complete artifacts and R-A/R-B/R-C |
| 1 — M0 qualification | `blocked_not_started` | four qualification transactions |
| 2 — ReplayAB and verdict | `blocked_not_started` | `ADOPT_M0/NARROW_M0/REDESIGN_TO_M1/REJECT_OPTIMIZATION` |

## 16. Current uncertainties

Before Stage 0 closure:

1. freeze exact validation commands from current repository owners;
2. select eight cases and prove after-image/artifact availability;
3. freeze operation/time recording rules for connector actions;
4. determine whether a real M0 liveness can use low-risk existing tasks without creating
   disposable repository clutter;
5. prove the frozen operations fit the existing `local-git-v1` ReplayAB instrument
   without adding a backend;
6. record exact plan SHA-256 from an exact byte stream;
7. refresh or rebuild the implementation branch from current main before any non-document
   work.

These are design-review tasks. They do not authorize M1 or any runtime implementation.
