# M0 Stage 0 Review R-A — ownership, necessity, and capability

**Claim:** `GOV-DRPO-MAINTENANCE-RUNNER-REPLAYAB-01`  
**Reviewed plan:** `docs/development_workflow_optimization/M0_ATOMIC_DEV_TRANSACTION_PLAN.md`  
**Reviewed plan commit:** `d4d6ac05401926f1e9b7176cc48fbbe78bb68431`  
**Current main:** `ad9bda80796dcf5c48976f5d64ffd79a006c70d5`  
**Review verdict:** `HOLD_CORRECTIONS_REQUIRED`  
**Scientific impact:** none

## 1. Questions

1. Does M0 still duplicate V1, Candidate 01, ReplayAB, or E7/E8 domain owners?
2. Is zero-code M0 the smallest candidate that can address the frozen problem?
3. Can the current GitHub connector actually construct one tree/commit without
   sequential per-file commits?
4. Can the existing ReplayAB instrument measure the treatment without being rewritten?

## 2. Ownership result

The rewritten boundary is materially better than the superseded M2 design.

M0 begins only after complete reviewed file after-images exist. It does not:

- parse or apply a patch;
- construct a V1 `READY` transaction;
- prepare pilot registration;
- normalize handoff/registry state;
- implement E7/E8 scientific validation;
- select scientific values;
- generate candidate contents;
- merge or launch experiments.

Therefore M0 no longer duplicates the accepted V1 or Candidate 01 responsibilities.
It standardizes an existing GitHub Git-object publication operation rather than adding
a second repository transaction engine.

## 3. Direct connector capability check

A non-publishing capability check was performed against current main:

- base commit supplied to `create_tree`:
  `ad9bda80796dcf5c48976f5d64ffd79a006c70d5`;
- existing `AGENTS.md` blob:
  `2232f0da56dee675bb84c3779cc7bdb9ff1d59fc`;
- returned unreferenced tree object:
  `982fc86d39904a9ac0f384f64d027a317aeeee70`.

No commit, branch, ref, PR, workflow, or repository file was created by this check.
The branch-visible repository state did not change.

This confirms that the currently exposed GitHub connector can use the current commit as
the base of a tree update and is not limited to sequential Contents-API file commits.
M0 is therefore technically representable without a new repository workflow.

The check does not yet prove end-to-end safety. Final commit parent/tree inspection,
non-force publication, exact-head PR checks, and failure recovery remain Stage 1 gates.

## 4. Remaining measurement mismatch

The current ReplayAB real-pair implementation records local subprocess commands and
monotonic timing. It does not directly execute or time conversational GitHub connector
actions.

The plan currently mixes two evidence layers:

- real GitHub qualification of remote ref, PR, and CI semantics;
- controlled ReplayAB timing and action-count comparison.

They must be separated before Stage 0 closes:

1. **remote qualification layer:** one small E7-derived and one E8-derived real M0
   transaction verify GitHub object/ref/PR/CI behavior;
2. **controlled replay layer:** existing local-git ReplayAB replays equivalent sequential
   and atomic Git operations using the same after-images;
3. GitHub queue/PR wall time is reported separately as operational context;
4. local controlled time is the causal timing estimate;
5. no new ReplayAB backend is authorized unless current `local-git-v1` cannot express the
   frozen operations.

## 5. Arm-A fairness correction

Arm A cannot be generically labelled “the sequential Contents API route.” Each case must
freeze the route that was actually accepted or used for that historical task.

If a historical case already used one atomic commit, it provides no M0 treatment
contrast and must not be selected as a success case merely to fill the inventory.

## 6. Verdict

M0 remains the smallest viable candidate and the ownership redirect is accepted.

Stage 0 remains on hold until the controlling plan is amended to:

- separate remote qualification from local controlled replay;
- freeze case-specific Arm-A routes;
- prohibit a new ReplayAB backend by default;
- state that M0 may be rejected if the selected case bank lacks a real treatment contrast.

No return to M2 is justified by this review.
