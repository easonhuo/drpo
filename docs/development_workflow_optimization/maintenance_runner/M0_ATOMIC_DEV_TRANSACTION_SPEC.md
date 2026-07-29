# M0 atomic development transaction specification

**Claim:** `GOV-DRPO-MAINTENANCE-RUNNER-REPLAYAB-01`  
**Controlling plan:** `docs/development_workflow_optimization/M0_ATOMIC_DEV_TRANSACTION_PLAN.md`  
**Status:** Stage 0 contract artifact; no implementation authorization

## 1. Purpose

M0 is a zero-production-code procedure for publishing already reviewed complete file
after-images as one atomic development-branch commit through the connected GitHub App.

M0 does not replace V1, Candidate 01, ReplayAB Core, E7/E8 scientific code, handoff
authority, the formal execution channel, or GitHub PR gates.

## 2. Accepted input

One immutable stage packet containing:

- authoritative full base SHA;
- one declared development branch;
- reviewed UTF-8 after-images and SHA-256 identities;
- exact paths and modes;
- task scope and reviewer identity;
- fixed validation profile and evaluator binding;
- new-Python and code-change approvals when applicable;
- expected terminal state and rollback.

The packet is data only. It cannot contain executable commands, environment overrides,
mutable refs, branch wildcards, or arbitrary test selection.

## 3. PREOBJECT gate

Before storing the first Git blob, verify:

1. current `main` equals the packet base;
2. target is not the default branch;
3. path normalization and protected-path exclusions;
4. UTF-8, file-size, mode, and after-image hashes;
5. no secret-bearing or unreviewed content;
6. exact expected changed-file inventory;
7. task scope, reviewer, plan, new-Python, and code-budget approvals;
8. fixed validation and evaluator identities.

Failure is `BLOCKED_PREOBJECT` and creates no Git object or ref.

## 4. Object and publication sequence

```text
create reviewed blobs
→ create final tree from exact parent tree
→ create one commit with one declared parent
→ independently inspect parent/tree/paths/modes
→ re-resolve main and target branch
→ create new branch directly at final commit, or non-force update exact existing head
→ re-read remote ref and commit
→ open/reuse Draft PR
→ observe actual exact-head checks
```

A blob, tree, or commit object is a repository-side effect even when unreferenced. Ref
publication is a later and distinct boundary.

## 5. Prohibited behavior

- no patch application or content generation;
- no branch creation at base followed by a later move;
- no automatic retry, rebase, force push, merge, or ready-for-review transition;
- no workflow, handoff, registry, authority, governance-ledger, or formal-policy payload;
- no binary, symlink, gitlink, deletion, rename, or mode change in qualification V1;
- no arbitrary commands or task-provided test lists;
- no experiment launch or scientific-status change.

## 6. Evidence

The transaction evidence must bind:

- plan, packet, base, branch, parent, tree, and final commit identities;
- every after-image hash and Git blob identity;
- preobject decisions;
- object, ref, PR, and check observations;
- exact changed paths and modes;
- terminal state and recovery class;
- active actions and controlled timing, reported in their proper evidence layer.

Candidate-produced evidence is not the independent acceptance oracle.

## 7. Recovery

After ref publication, recovery may re-read immutable GitHub objects, open the already
declared Draft PR, or restore missing evidence. It may not alter the code commit, move
the ref, regenerate contents, or reclassify a failed check.

## 8. Qualification boundary

Remote Stage 1 qualification publishes only two explicitly approved low-risk success
transactions. Stale-head and validation-failure qualification remain in isolated
local/bare-remote replay and create no disposable GitHub branch.

M0 has zero production-code budget. The separately governed ReplayAB measurement adapter
is an instrument change and is not part of this specification's runtime behavior.
