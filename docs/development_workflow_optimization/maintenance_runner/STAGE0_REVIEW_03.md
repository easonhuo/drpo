# Maintenance Runner Stage 0 Review 03

**Claim:** `GOV-DRPO-MAINTENANCE-RUNNER-REPLAYAB-01`  
**Review type:** pre-behavior architecture, threat, governance, and ReplayAB review  
**Decision:** `REDESIGN_BEFORE_STAGE1`  
**Scientific impact:** none  
**Experiment execution:** none

## 1. Locked review identity

- repository: `easonhuo/drpo`;
- default branch: `main`;
- current main verified for this review: `ad9bda80796dcf5c48976f5d64ffd79a006c70d5`;
- plan branch: `dev/gov-maintenance-runner-replayab-01`;
- reviewed plan commit: `257949e4fe63e514967202dd17867ccd5fcbb5eb`;
- reviewed plan path: `docs/development_workflow_optimization/MAINTENANCE_RUNNER_REPLAYAB_PLAN.md`;
- reviewed plan Git blob: `53aa8874acb860a995dc0698926e52eb907e1cf8`;
- branch relation at review start: 5 commits ahead of and 18 commits behind current `main`;
- startup sources reread: root `AGENTS.md`, `docs/handoff.md` Section 0, `experiments/registry.yaml`, and `docs/governance_pipeline_stage_status.yaml`.

The exact file SHA-256 required by the plan-dependency protocol has not yet been computed from an exact local byte stream. Git shell network resolution was unavailable in the review environment. The Git blob and commit identities are durable review anchors, but Stage 1 remains blocked until the plan SHA-256 is recorded.

## 2. Review criteria

This review uses the six passes frozen in the implementation contract:

1. objective and ROI;
2. architecture and duplication;
3. security and mutation safety;
4. governance and scientific safety;
5. ReplayAB validity;
6. implementation and rollback feasibility.

A user preference to proceed is authorization to review, not evidence that the design is sound. The review therefore treats cancellation or narrowing as valid outcomes.

## 3. Pass A — objective and ROI

**Verdict:** `HOLD`.

The recurring problem is real. Historical DRPO work contains repeated temporary source-import/export workflows, rebuilt PRs, manual file placement, and repeated stage coordination. A bounded optimization remains worth evaluating.

The current candidate, however, does not yet isolate its unique treatment from mechanisms that already exist:

- the connected GitHub App route already owns development branches, commits, Draft PRs, and exact-head CI;
- the GitHub App exposes atomic Git blob/tree/commit/ref operations, so multi-file development commits do not inherently require a new repository workflow;
- V1 already owns reviewed dev snapshot to local `READY`, including source locking, exact tree construction, authority, gates, freshness, diagnostics, and terminal state;
- Candidate 01 already implements a one-click composition of preparation plus V1, although its candidate evaluation is still pending;
- the pilot-registration fastpath already compiles reviewer-authored registration specifications into deterministic V1 inputs.

The proposed Maintenance Runner currently mixes at least three different losses:

1. getting an approved multi-file change onto a dev branch;
2. coordinating the accepted V1 integration path;
3. avoiding Python edits when E7/E8 profiles are currently hard-coded.

One new workflow cannot receive credit for all three. The third problem is profile architecture, not repository transport. The second already has Candidate 01. Stage 0 must therefore narrow the candidate to one unique missing transaction before ROI thresholds are meaningful.

## 4. Pass B — architecture and duplication

**Verdict:** `HOLD_WITH_REDIRECT`.

The current plan proposes a new runner that parses manifests and patches, validates paths and identities, creates an isolated workspace, constructs a commit, runs gates, records evidence, and pushes a branch.

That design overlaps materially with existing owners:

- `scripts/dev_integration_write_path.py` already implements strict Git identities, path normalization, forbidden paths, blob/mode validation, locks, isolated commit construction, diagnostics, and fail-closed behavior;
- `scripts/dev_integration_finalize.py` and its core already own normalization, selected gates, provenance, and local `READY` finalization;
- `scripts/run_workflow_replay.py` and `src/drpo/workflow_replay/orchestrate.py` already implement Candidate 01 command composition and run evidence;
- ReplayAB already owns comparison and must not be embedded into the candidate.

The new candidate is allowed to solve a different responsibility, but it may not create a second generic integration transaction merely because its input is a patch rather than an audited dev snapshot.

The next contract revision must draw the boundary as follows:

```text
approved payload -> tested dev-branch commit
```

and explicitly exclude:

```text
reviewed dev snapshot -> local READY
registration materialization
handoff/registry authority
Draft PR creation and merge
scientific profile generation
```

Before behavior code, the document must compare three candidate shapes:

### Option M0 — no new repository workflow

Use the connected GitHub App's atomic blob/tree/commit/non-force-ref path plus existing exact-head PR CI. Standardize the operation contract and evidence in documentation only.

### Option M1 — thin publisher over an existing local candidate

Consume an already verified commit or V1 `READY` identity and perform only exact-head, same-branch, non-force publication. Do not apply a patch or rerun V1 ownership.

### Option M2 — new patch-apply workflow

Accept one hash-bound unified diff, apply and validate it, create one commit, and push one pre-existing dev branch. This is the current proposal and is acceptable only if M0 and M1 cannot solve the frozen task class and the unavoidable duplicated safety logic remains inside the code budget.

The smallest sufficient option must win. Architectural elegance is not evidence for M2.

## 5. Pass C — security and mutation safety

**Verdict:** `HOLD`.

The V1 exclusions in the current plan are directionally correct:

- unified diff only;
- no arbitrary commands or user-selected tests;
- no default-branch target;
- no force push or branch creation;
- no workflow, handoff, registry, authority, governance, or formal-execution payload paths;
- no binary, gitlink, symlink, rename, deletion, or mode change;
- exact expected head and same-branch publication;
- no commit or push after failed validation.

Three blocking issues remain.

### 5.1 Workflow bootstrap and liveness

Official GitHub Actions behavior requires a `workflow_dispatch` workflow to exist on the default branch before it can receive dispatch events. Therefore a production manual workflow cannot receive its first realistic dispatch solely from this unmerged development branch.

The contract must choose one predeclared liveness route. It may not rediscover a temporary scheduler, PR-edit trigger, or broad `pull_request_target` bridge during implementation.

Acceptable directions for further review are:

- local/bare-remote write simulation plus a separately approved post-merge canary;
- an exact same-repository PR liveness workflow with a branch- and PR-locked permission boundary, followed by removal before production design;
- no Actions workflow at all under Option M0.

No direction is approved by this report.

### 5.2 Recursive CI claims

A push performed with the repository `GITHUB_TOKEN` does not normally trigger another workflow run. The candidate must run the checks it claims inside the same run, or separately trigger and observe an explicitly authorized workflow. It may never describe a token-originated push as independently exact-head-CI validated unless that separate run actually exists.

### 5.3 Evidence failure after push

A successful push followed by report/artifact publication failure creates a valid remote commit with incomplete workflow evidence. The current plan names this state but does not freeze its recovery protocol.

The revised contract must specify:

- the minimal evidence embedded in the commit message or committed report before push;
- the immutable information recoverable from GitHub after push;
- a report-only recovery action that cannot modify the code commit;
- the exact terminal labels for `PUSHED_EVIDENCE_INCOMPLETE` and recovered evidence;
- prohibition on reapplying the payload merely to regenerate logs.

## 6. Pass D — governance and scientific safety

**Verdict:** `HOLD`.

The scientific boundary is currently sound: no E7/E8 training, no parameter choice, no experiment-state transition, no handoff or registry write, and no method ranking.

The engineering authorization is incomplete:

- the parent ReplayAB claim explicitly excludes GitHub workflow changes and automatic push;
- the proposed candidate therefore needs its own approved scope and rollback record;
- the update-integration responsibility intersects the closed governance pipeline and requires an exact determination of whether Stage 1 must be reopened;
- proposed new Python paths require exact-path oral approval before creation;
- the code-change-budget approval path is expected to trigger and may not be bypassed;
- the current development branch is materially behind `main` and cannot be the implementation base.

The latest main change corrected `GOV-NEW-PYTHON-FILE-ORAL-APPROVAL-02` so `kind: reopen` and `change_class: reopen` agree. It changes no scientific or ReplayAB behavior, but the eventual candidate authorization must use the corrected governance form.

## 7. Pass E — ReplayAB validity

**Verdict:** `HOLD`.

ReplayAB can judge the deterministic repository transaction, but the case bank cannot be frozen before the candidate treatment is unambiguous.

The current eight-case shape remains reasonable as a sampling target:

- four E7-derived cases;
- four E8-derived cases;
- per family: config/RunSpec success, small code-plus-test success, integration-heavy success, and one failure boundary.

The exact inventory is not yet valid because:

- M0, M1, and M2 do not have the same input and output boundary;
- Candidate 01 and V1 work must not be counted as Maintenance Runner benefit;
- the candidate preflight adapter and independent acceptance evaluator are not yet separated;
- the exact artifact availability, payload hashes, original approvals, expected terminal states, and evaluator identities have not been frozen;
- a workflow bootstrap method could contaminate wall-time and command-count comparisons.

The revised protocol must use a frozen benchmark toolchain separate from the candidate branch. Candidate-side checks are implementation evidence; ReplayAB acceptance must independently re-read the final commit and failure evidence.

No efficiency result is visible until both arms independently pass.

## 8. Pass F — implementation and rollback feasibility

**Verdict:** `REDESIGN`.

The current 300--450 production-line target is plausible only for a narrowly bounded publisher or patch transaction. It is not credible for a second complete integration engine plus workflow, adapters, evidence recovery, and security hardening.

The historical 103-minute successor task also shows that repository transport was only one source of delay. Hard-coded profile semantics, runner and aggregator changes, RunSpec construction, and authority materialization were separate costs. The first candidate must not claim that a transport workflow solves profile hard-coding or registration design.

Rollback remains simple only while the current route is unchanged and the candidate owns no durable domain state. That boundary is retained.

## 9. Review-cycle decision

Stage 0 does **not** pass in the current form.

The decision is:

```text
REDESIGN_BEFORE_STAGE1
```

This is not a rejection of workflow optimization. It is a rejection of beginning behavior code before the candidate's unique responsibility and liveness route are isolated from mechanisms already present in the repository.

## 10. Required next Stage 0 revision

Before another closure review, the implementation contract must:

1. replace the single presumed M2 architecture with an explicit M0/M1/M2 decision table;
2. freeze the exact recurring task class that each option can and cannot solve;
3. identify the minimal reusable existing-owner calls for every option;
4. freeze the workflow bootstrap/liveness route or choose no workflow;
5. freeze evidence recovery after a successful push and failed artifact publication;
6. separate candidate preflight from the independent ReplayAB evaluator;
7. select the eight cases only after the treatment boundary is frozen;
8. freeze validation commands and payload limits from those selected cases;
9. obtain the required scope, reopen determination, and exact Python-path approvals only for the selected option;
10. rebuild the implementation branch from then-current `main` before Stage 1.

## 11. Current stage state

| Stage | State after this review |
|---|---|
| Stage 0 | `active_redesign_required` |
| Stage 1 | `blocked_not_started` |
| Stage 2 | `blocked_not_started` |
| Stage 3 | `blocked_not_started` |

No workflow code, runner code, test code, experiment code, handoff, registry, or governance authority file is changed by this review.
