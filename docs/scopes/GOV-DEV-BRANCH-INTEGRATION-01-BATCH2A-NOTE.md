# GOV-DEV-BRANCH-INTEGRATION-01 Batch 2A acceptance note

This note narrows the current implementation step under the already approved V1 scope.

## Base and claim

- Base commit: `e4aa36bf5ce03794c0d935b4570e276a0703d93e`
- Claim: `GOV-DEV-BRANCH-INTEGRATION-01`
- Phase: `batch_2a_local_source_commit`
- Research experiment impact: none

## Authorized work

- consume an immutable Batch 1 `REVIEWED` transaction;
- re-resolve and fetch the locked main and dev commits;
- re-audit approved file operations, blob identities, and Git modes;
- build the exact candidate tree in an isolated Git index;
- create a local source commit with the locked main as its unique parent;
- verify the checked-out worktree and committed scope;
- persist a prepare report and advance the transaction to `PREPARED`;
- add direct tests and update protocol documentation.

## Excluded work

- no handoff or registry mutation;
- no schema-v3 delta creation or normalization;
- no repository gate execution by the integration tool;
- no final ready commit;
- no network push, pull-request creation, CI polling, or merge by the integration tool;
- no experiment code, configuration, variables, seeds, thresholds, results, or execution-order changes;
- no Stage 1, Stage 2, or Stage 5 protected-file changes.

## Acceptance

Batch 2A is acceptable only when targeted tests and the existing repository PR gates pass, the diff remains inside this note's scope, and review confirms that the source commit has exactly one parent and an exact approved file after-image.

Passing Batch 2A does not complete V1. Registry/handoff normalization, required gates, final ready commit, and both real shadow cases remain separately gated.

## Rollback

Revert the single squash-merge commit for Batch 2A. Batch 1 remains the accepted read-only planner, existing transaction records remain readable, and no scientific or governance authority state requires migration.
