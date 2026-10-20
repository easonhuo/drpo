# M0 Stage 0 Review R-B — failure, security, and governance

**Claim:** `GOV-DRPO-MAINTENANCE-RUNNER-REPLAYAB-01`  
**Reviewed plan:** `docs/development_workflow_optimization/M0_ATOMIC_DEV_TRANSACTION_PLAN.md`  
**Reviewed plan commit:** `eccd3a1c49744f7db222de0ee5b0f369e0d72ba8`  
**Current main:** `ad9bda80796dcf5c48976f5d64ffd79a006c70d5`  
**Review verdict:** `HOLD_CORRECTIONS_REQUIRED`  
**Scientific impact:** none

## 1. Review scope

This pass audits the irreversible boundaries of the zero-code M0 transaction:

- content exposure through Git object creation;
- stale-main and stale-head races;
- new-branch and existing-branch publication;
- PR and CI evidence;
- cancellation and report recovery;
- Python-file and large-code approvals;
- closed-stage and default-route governance.

## 2. Finding: object creation is already an irreversible content exposure

The draft correctly treats branch-ref publication as the visible code-state boundary, but
it understates the security boundary.

`create_blob` stores content in the repository object database before any branch exists.
An unreferenced blob is not part of the branch tree, but its SHA can still identify the
stored object. Therefore secret-bearing or unreviewed content must be rejected before
the first blob call, not merely before ref publication.

Required correction:

- rename the earliest gate to `PREOBJECT`;
- complete content review, secret-like-content rejection, path audit, hash verification,
  mode verification, approval checks, and size limits before `create_blob`;
- distinguish `BLOCKED_PREOBJECT` from `OBJECTS_CREATED_REF_UNCHANGED`;
- never claim that object creation is equivalent to no repository-side effect.

## 3. Finding: freshness must be rechecked at the publication boundary

The draft checks current main and expected head before object construction, but object
construction and audit may take enough time for refs to move.

Required correction immediately before branch-ref creation/update:

- re-resolve current `main`;
- require it to remain equal to the frozen base for the first qualification;
- for a new branch, prove the branch still does not exist;
- for an existing branch, prove it still equals `expected_head`;
- verify the created commit has the declared unique parent;
- stop with the ref unchanged on any mismatch.

A non-fast-forward failure is terminal for the attempt. No rebase, moved expected head,
or automatic retry is allowed.

## 4. Finding: new branches must appear only at the final commit

For `new_branch_from_base`, creating a branch at base and later moving it would expose an
intermediate branch state and add an unnecessary action.

The only accepted sequence is:

```text
create reviewed blobs
→ create final tree
→ create final commit with parent=base
→ recheck main and branch absence
→ create branch directly at final commit
```

For an existing branch, use one non-force update from `expected_head` to the final commit.

## 5. Finding: approval gates remain task-level obligations

M0 introduces no Python implementation, but an M0 payload may add or modify Python.

The packet must bind, when applicable:

- exact-path oral approval for every new Python destination;
- the durable approval record;
- code-change-budget classification;
- any required large-code approval;
- current scope authorization for the task itself.

M0 cannot turn a missing approval into a post-publication CI discovery.

## 6. Finding: Stage 1 failure tests should not create remote clutter

The stale-head and validation-failure qualification cases do not need real GitHub
branches or PRs. Their safety boundary can be tested in the controlled local/bare-remote
replay layer.

Stage 1 remote publication should be limited to the two approved low-risk success
transactions. Failure-boundary evidence remains mandatory but is produced without
creating disposable remote branches.

## 7. Finding: adoption and qualification have different governance effects

Documentation and the opt-in Stage 1 qualification do not alter the default route.
They do not by themselves require a closed-stage production reopen.

A later `ADOPT_M0` implementation that changes `AGENTS.md`, default policy, or routine
repository behavior requires:

- a separately reviewed default-policy change;
- current-ledger closed-stage determination;
- explicit user approval;
- rollback;
- exact-head governance validation.

No ReplayAB verdict self-authorizes that change.

## 8. Evidence recovery

The draft's `PUBLISHED_EVIDENCE_INCOMPLETE` state is sound with one restriction:
report-only recovery may re-read immutable GitHub refs, commit metadata, PR identity,
diff, and workflow runs, but may not add a new code commit.

If a durable report is later committed, it must be a separate evidence-only change and
must not be represented as part of the original atomic code commit.

## 9. Verdict

The zero-code M0 direction remains acceptable, but the plan must be amended before this
review can pass.

Required changes:

1. add the PREOBJECT security boundary;
2. recheck main and target branch immediately before ref publication;
3. create new branches directly at the final commit;
4. bind Python/code-budget approvals in every relevant task packet;
5. keep failure qualification local and remote qualification success-only;
6. separate opt-in qualification from any later default-policy adoption.

No runtime code, workflow, scientific path, handoff, registry, or governance file is
authorized by this review.
