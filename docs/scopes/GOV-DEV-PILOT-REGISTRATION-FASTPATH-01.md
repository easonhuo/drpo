# GOV-DEV-PILOT-REGISTRATION-FASTPATH-01 — PR-A Scope Contract

**Status:** authorized for PR-A implementation and real V1 shadow  
**Authorization:** explicit user approval in the 2026-07-14 development-workflow review  
**Base commit:** `c0ff38b51b0062b26a20771421f62b08eaaa0d12`  
**Target branch:** `dev/gov-dev-pilot-registration-fastpath-01`  
**Scientific experiment impact:** none

## 1. Problem

Recent code-first pilot integrations exposed avoidable registration and merge amplification:

- implementation and registration work were interleaved before the implementation SHA was frozen;
- generated authority files were debugged through repeated remote CI cycles;
- per-file writes produced many micro-commits and repeatedly invalidated exact-head checks;
- stale-main recovery required branch reconstruction and repeated RunSpec/provenance updates;
- the existing test selector was observed but not yet used to reduce exploratory feedback cost.

The incidents are recorded under `docs/development_workflow_incident_and_improvement_log.md` and the companion annex `docs/development_workflow_incidents/DEVOPT-2026-07-14-PILOT-REGISTRATION-MERGE-01.md`.

## 2. PR-A objective

Provide one thin, deterministic preparation layer for code-first development pilots. It compiles a reviewer-authored Pilot Registration Spec into the existing V1 integration inputs and produces a preflight report before the existing V1 transaction is invoked.

```text
frozen implementation snapshot
→ reviewer-authored Pilot Registration Spec
→ strict validation and deterministic input compilation
→ existing V1 plan / prepare / normalize / gate / finalize
→ local READY
```

PR-A reduces manual assembly and remote trial-and-error. It does not weaken or replace the existing V1 transaction, schema-v3 authority, reviewer decision, or final merge gate.

## 3. Allowed files

PR-A may add or modify only:

- `docs/dev_pilot_registration_fastpath.md`;
- `docs/development_workflow_incidents/**`;
- this scope contract and `docs/scopes/GOV-DEV-PILOT-REGISTRATION-FASTPATH-01-REAL-SHADOW-NOTE.md`;
- `docs/templates/DEV_PILOT_REGISTRATION_SPEC.yaml`;
- `scripts/prepare_dev_pilot_registration.py`;
- `tests/test_prepare_dev_pilot_registration*.py`;
- the narrowly branch-scoped summary hook in `tests/conftest.py`;
- narrowly required documentation links or test fixtures that remain inside this claim's scope.

The wildcard expansion for isolation tests was reviewed after the first exact-head CI pass exposed no test failure but independent review identified an uncovered output-root/worktree-pollution boundary. The real-shadow scope was then explicitly added to test the adapter against immutable real E7 refs and the accepted V1 transaction without publication.

Any additional path requires a scope review before modification.

## 4. Explicit non-goals

PR-A must not:

- edit `docs/handoff.md` or `experiments/registry.yaml`;
- add a handoff renderer, registry authority, test-impact map, or second transaction state machine;
- modify `scripts/handoff_authority.py`, the formal experiment channel, update-package tooling, or other Stage 1/2/5 protected files;
- change any scientific experiment variable, seed, threshold, horizon, state, result, or execution priority;
- infer scientific evidence, approve a reviewer decision, or create approval tokens;
- push a generated candidate, open or refresh a PR from inside the adapter or shadow, poll CI, merge, or delete a branch;
- modify the repository's default GitHub Actions policy;
- create an experiment-specific temporary workflow;
- remove or weaken any existing gate.

PR-B tiered CI and telemetry are separately scoped future work and are not authorized by this PR-A contract.

## 5. Authority and reuse rules

The new tool is an input compiler and preflight only. It must reuse:

- `validate_dev_integration.py` for existing request and reviewer-decision semantics;
- existing `REGISTRATION_INTENT.yaml` and `REGISTRATION_APPROVAL.yaml` contracts;
- the accepted V1 `plan`, `prepare`, `normalize`, `gate`, and `finalize` stages;
- the trusted current-main schema-v3 authority and test selector when V1 later runs.

It must not copy their semantic logic into a parallel implementation. A compiled output is not `READY`; only the existing V1 transaction may produce local `READY`.

The real shadow may derive exact operations mechanically from immutable locked refs, but that derivation is restricted to acceptance. Production integrations still require an explicit reviewer-authored Pilot Registration Spec.

## 6. Failure and mutation boundary

Preparation must be fail-closed and side-effect bounded:

- parse and validate all inputs before writing output files;
- reject unknown keys, unsafe paths, malformed SHAs, duplicate operations, unreviewed experiment targets, and inconsistent add/replace semantics;
- require `--output-root` to resolve outside the repository worktree;
- never overwrite a non-empty output directory unless the exact prior manifest proves byte-identical idempotent output;
- serialize publication with an exclusive preparation lock;
- write to a temporary sibling directory and atomically publish only after every generated file and hash has been verified;
- on failure, leave the repository and source inputs unchanged and emit a structured diagnostic only outside tracked scientific evidence;
- do not invoke Git network operations, authority normalization, repository mutation, or CI during adapter compilation.

The branch-scoped acceptance test may invoke the existing V1 network/ref lock and gates after compilation. It must never publish the resulting candidate.

## 7. PR-A acceptance

PR-A is acceptable only when tests demonstrate:

1. a valid Pilot Registration Spec deterministically produces existing V1-compatible request, review, registration-intent, and approval-binding inputs;
2. generated hashes and reviewer bindings are internally consistent without auto-approving the review;
3. a code-only pilot omits registration inputs rather than inventing them;
4. add and replace experiment modes are strictly distinguished;
5. unknown fields, malformed full SHAs, unsafe paths, duplicate operations, target mismatch, stale expected-before hash, and invalid result classifications fail closed;
6. output inside the repository is rejected before mutation and concurrent publication is lock-protected;
7. preparation is idempotent for identical inputs and rejects conflicting pre-existing output;
8. a failed preparation publishes no partial candidate directory;
9. generated outputs pass the existing request/reviewer validators with an explicit reviewer-approved fixture;
10. a historical registration fixture can be replayed without changing scientific state;
11. the real locked E7 registration shadow uses adapter-generated inputs and reaches local V1 `READY`;
12. the READY candidate preserves all non-target registry entities and all pre-existing target fields, changes no scientific status, and is not published;
13. exact-head compile, Ruff, governance checks, handoff authority verification, formal-channel validation, full pytest, and the repository-selected required tests pass.

## 8. Rollback

Rollback is additive and requires no data migration:

1. stop using `prepare_dev_pilot_registration.py`;
2. preserve any generated diagnostics and real-shadow logs as engineering evidence;
3. continue assembling the existing V1 inputs manually;
4. revert only the PR-A implementation commit if necessary;
5. leave V1 transaction records, handoff/registry history, scientific artifacts, and closed governance stages unchanged.

## 9. Stop conditions

Development must stop and return for a new authorization if implementation would require:

- changing a Stage 1/2/5 protected file or responsibility;
- changing the schema-v3 authority contract;
- automatically publishing or merging candidates;
- modifying formal CI defaults;
- creating a second registration or transaction engine;
- altering a scientific experiment or its evidence classification.
