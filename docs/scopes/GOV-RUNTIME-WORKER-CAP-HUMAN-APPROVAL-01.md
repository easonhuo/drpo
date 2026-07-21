# GOV-RUNTIME-WORKER-CAP-HUMAN-APPROVAL-01

## Approval and parent

The repository owner explicitly approved the following correction on 2026-07-21:

- `MAX_WORKERS` remains available as a hard safety ceiling;
- its default is unset;
- an AI agent may recommend a value but may not set or change it without explicit user approval;
- every approved value must be durable and exact-run bound.

Parent implementation: `GOV-E7-RUNTIME-AUTOTUNE-ADAPTIVE-SEARCH-V3-01`.

## Problem being corrected

The previous V3 draft correctly made `max_workers` optional, but still treated an environment or CLI value as sufficient authority. That allowed an executor to choose a conservative value such as 24 and silently censor autotune's search space. This confused three responsibilities:

1. resource-pool placement and capacity;
2. autotune concurrency selection;
3. human risk tolerance.

## Locked replacement rule

1. The normal value is `unset`.
2. Unset means autotune selects concurrency within measured resource limits.
3. A non-null cap is a repository-owner policy decision, not an executor tuning decision.
4. A non-null cap requires a committed approval JSON under `docs/runtime_worker_cap_authorizations/`.
5. The approval binds the exact experiment, work directory, cap, CPU affinity, contract hash, run-spec hash, grid hash, and approved runtime-code commit.
6. The canonical wrapper writes an immutable `USER_APPROVED_WORKER_CAP.json` into the work directory for both unset and capped modes.
7. Any value or mode change in an existing work directory fails closed and requires a new approval plus a new run/work directory.
8. A cap-censored result may not be reported as the uncapped autotune optimum.

## Authorized files

Existing files modified:

- `scripts/run_e7_squared_exp_night_one_click.sh`;
- `scripts/run_e7_squared_exp_night_resume_one_click.sh`;
- `tests/test_e7_squared_exp_night_runspecs.py`;
- the existing V3 repair files carried by the parent claim.

New non-Python files authorized by this scope:

- `scripts/validate_user_approved_worker_cap.sh`;
- `docs/runtime_worker_cap_approval.md`;
- `docs/runtime_worker_cap_authorizations/README.md`;
- this scope document.

No new Python path is created.

## Scientific boundary

This is a runtime-governance bugfix. It must not change datasets, methods, coefficients, seeds, training horizons, evaluation cadence, GAE, optimizer behavior, scientific thresholds, convergence rules, terminal audits, or result status.

No E7 run, resume, probe, or selection-only server shadow is authorized by the code change itself.

## Acceptance

- no cap and no approval succeeds and records immutable unset mode;
- approval without a cap fails;
- cap without approval fails;
- untracked, dirty, malformed, mismatched, or stale approval fails;
- exact approved cap succeeds;
- changing, deleting, or replacing the cap in the same work directory fails;
- both canonical one-click and resume wrappers invoke the gate before the auto runner;
- the helper and approval policy are protected by the next RunSpec provenance binding;
- focused tests, shell syntax, compile, Ruff, full pytest, authority no-op, and governance validators pass on the exact review head;
- a server selection-only shadow passes before merge or scientific use.

## Rollback

If the approval gate causes an unintended compatibility regression, revert this child scope while retaining the parent V3 low-first search and probe-cost repair. Do not weaken approval checks in a server-local patch.
