# GOV-EXPERIMENT-ORIGIN-MAIN-GATE-RETIREMENT-01 — Retire `origin/main` as an Experiment Launch Gate

**Status:** user-authorized for implementation and review  
**Base commit:** `9fdd72dc9364525dcc8f26257a74b3bc09bd1050`  
**Authorization record:** `docs/governance_stage_authorizations/GOV-EXPERIMENT-ORIGIN-MAIN-GATE-RETIREMENT-2026-08-26.yaml`  
**Scientific experiment impact:** none

## 1. Purpose

Permanently retire equality with `origin/main` as a permission check for DRPO experiment launch, recovery packaging, or experiment artifact packaging. This closes the recurring failure mode where an exact reviewed frozen commit on a canonical clean checkout is rejected only because `origin/main` points at another commit.

This is a governance/provenance correction only. It does not change any task, dataset, seed, coefficient, optimizer, loss, taper, training horizon, evaluator, result status, or experiment ordering.

## 2. New experiment source contract

For formal experiment launch, the authoritative source identity is the explicitly frozen full Git commit SHA.

The public formal experiment path must enforce all of the following:

1. the caller supplies one full `--expected-commit`;
2. local `HEAD` equals that exact commit;
3. the checkout is clean;
4. the repository remote identity is the canonical `easonhuo/drpo` repository where the experiment-specific launcher already checks it;
5. source/config snapshots and run provenance bind to that launch commit;
6. end-of-run HEAD/worktree mutation remains a provenance failure.

`HEAD == origin/main` is not part of this contract and must not block an otherwise valid frozen experiment.

## 3. `origin/main` remains useful outside launch authorization

This retirement does not delete remote-main resolution from the repository. `origin/main` may still be resolved and recorded for development-base freshness, rebase/merge review, code-delivery freshness, or descriptive provenance.

Those uses must remain distinct from experiment launch permission. A later movement of `origin/main` does not invalidate or block a separately frozen experiment commit merely because the two SHAs differ.

## 4. Compatibility and active E8 cleanup

Historical experiment runners and documents are preserved. Some historical runners may still pass `--require-origin-main-match`; destructive history-wide rewrites are not required.

The public hardened guard/package entry points must treat that legacy option as retired compatibility input rather than a blocking requirement. Formal guard calls without an explicit full `--expected-commit` must fail closed instead of falling back to an `origin/main` equality check.

The active E8 lambda-completion path must stop explicitly requesting the retired check in its formal guard, delivery preflight, and recovery package path. No E8 scientific code or scientific configuration may change as part of this cleanup.

## 5. Regression requirements

Acceptance requires tests that deliberately create `HEAD != origin/main` while keeping `HEAD == expected_commit` and a clean checkout, then verify that:

- the public formal guard passes source preflight even when a legacy caller still supplies `--require-origin-main-match`;
- the public package entry point does not reject the same exact frozen commit because remote main differs;
- an incorrect `--expected-commit` still fails;
- dirty formal checkout and end-of-run source mutation protections remain intact;
- development/base-freshness resolution still detects remote-main movement, proving that only launch authorization was retired.

## 6. Rollback

Rollback must be reviewed and non-destructive:

1. restore the public guard/package origin-main behavior from the pre-change base commit;
2. restore any active-launcher flags removed by this scope only if a new explicit governance decision requires them;
3. restore the matching AGENTS policy text and Stage-2 protected-file fingerprints;
4. preserve this scope, authorization record, tests, and Git history as provenance;
5. do not modify scientific experiment results or frozen scientific configuration during rollback.

## 7. Merge boundary

Implementation stays on the development branch and Draft PR until separately reviewed. This authorization permits implementing and testing this exact retirement scope; it does not authorize merging the branch to `main` or launching the GPU workload.
