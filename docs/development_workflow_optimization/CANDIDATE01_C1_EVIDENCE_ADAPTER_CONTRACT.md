# Candidate 01 C1 Common-Evidence Adapter Contract

**Claim:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Candidate:** `Candidate 01 -- V1 One-Click Integration`  
**Base:** `main@cd770f47b89f8971923945c19caec49720c0e139`  
**Branch:** `dev/candidate01-c1-evidence-adapter-contract-01`  
**Dependency:** readiness audit PR #143, head `7c6fd28e9b8daa9f9f138d969a2a9aeed7d60ba5`  
**Decision:** `NARROW_IMPLEMENTATION_CONTRACT`  
**Scientific impact:** none

## 1. Objective

Add one bounded deterministic execution adapter that runs the explicit V1 path as Arm A and the
existing Candidate 01 path as Arm B, records both arms in the already-merged ReplayAB R1 evidence
schema, and supports the two-case liveness gate before any full nine-case evaluation.

This work measures Candidate 01. It must not change Candidate 01, V1, R1 judgment rules, scientific
execution, repository defaults, or merge authority.

## 2. Current facts

- `scripts/run_workflow_replay.py` exposes only the Candidate 01 command.
- `src/drpo/workflow_replay/execute.py` records deterministic fixture callbacks, not isolated real
  repository runs.
- `src/drpo/workflow_replay/evidence.py` already validates R1 contracts, opposite-order identities,
  content-addressed evidence, run artifacts, pair equivalence, and timing release.
- `src/drpo/workflow_replay/orchestrate.py` is the frozen Arm-B treatment.
- PR #143 froze a nine-case historical-first bank and selected `C01-CODE-ONLY` plus
  `C06-STALE-MAIN` for liveness.

## 3. Allowed paths

Behavior code may modify only:

- `scripts/run_workflow_replay.py`;
- `src/drpo/workflow_replay/execute.py`.

Existing tests may be extended only in:

- `tests/test_workflow_replay_execute.py`;
- `tests/test_workflow_replay_orchestrate.py`;
- `tests/test_workflow_replay_r1.py`.

New non-Python evidence may be added only under:

- `tests/fixtures/workflow_replay/candidate01_c1/**`;
- `docs/development_workflow_optimization/candidate01_c1/**`.

No new Python path is authorized.

## 4. Frozen non-modification boundary

The implementation must not modify:

- `src/drpo/workflow_replay/evidence.py`;
- `src/drpo/workflow_replay/orchestrate.py`;
- `src/drpo/workflow_replay/model.py` or `compare.py`;
- preparation or V1 owner scripts;
- GitHub workflows, `AGENTS.md`, handoff authority, `docs/handoff.md`, or the registry;
- Candidate 01 behavior, command order, placement rules, or terminal interpretation;
- any scientific code, configuration, result, seed, threshold, or execution order.

`evidence.py` is already at the frozen R1 yellow-zone upper boundary. Needing to change it is a stop
condition, not permission to expand R1.

## 5. CLI surface

Keep the existing `candidate` command backward compatible. Add one command:

```text
python3 scripts/run_workflow_replay.py real-pair \
  --contract <R1_CASE_CONTRACT.yaml> \
  --case-packet <CASE_PACKET.yaml> \
  --source-repo <FULL_LOCAL_GIT_REPOSITORY> \
  --output-root <NEW_EMPTY_DIRECTORY> \
  --backend-id local-git-v1 \
  --json
```

The command must:

1. validate the R1 contract and require its `input_spec_sha256` to equal the exact case-packet file
   SHA-256;
2. resolve the exact base and source commits from the supplied local Git repository;
3. build the existing R1 schedule `pair-0: A -> B`, `pair-1: B -> A`;
4. create a separate isolated workspace and evidence directory for every run;
5. execute the declared arm once;
6. write one existing-schema R1 Run Artifact per run;
7. load all four artifacts through `load_run_artifact`;
8. compare each pair through `compare_normalized_runs`;
9. release timing and operation metrics only for an equivalent pair;
10. write a compact terminal report and exit nonzero on invalid evidence or mismatch.

No network access is permitted inside `real-pair`. Missing local commit objects fail before run start.

## 6. Isolation and source identity

Each run must start from a new isolated clone or worktree at the case's exact historical base.

- Arm workspaces may not be reused.
- Preparation and transaction roots are run-local.
- The source repository is read-only input.
- Local temporary refs may be created only inside the run-local repository.
- Cache policy is taken from the frozen case contract.
- No run may read another arm's output directory.
- Existing output roots, symlinked roots, dirty initial workspaces, missing commits, or identity
  mismatches fail closed.

Workspace identity is the canonical SHA-256 of:

- `HEAD` commit;
- index tree identity;
- deterministic porcelain status;
- path, mode, and SHA-256 for every untracked entry reported by Git.

The same function is used before and after execution. Failure-boundary cases require byte-stable
workspace identity; READY cases require the registered expected change.

## 7. Arm definitions

### Arm A — explicit path

Arm A must not call `run_candidate`. It explicitly performs, in order:

1. preparation adapter;
2. deterministic repository-overlay placement;
3. V1 plan;
4. V1 prepare;
5. optional transaction-input placement;
6. V1 normalize;
7. V1 gate;
8. V1 finalize.

It may call only the already-owned checked-in commands. It must not reimplement authority,
normalization, gate, or finalization logic.

### Arm B — Candidate 01

Arm B calls the existing `run_candidate` exactly once with the same subprocess recorder used by Arm
A. `orchestrate.py` remains unchanged. The adapter may observe commands, outputs, placements,
timing, and failure classes; it may not alter them.

## 8. Evidence contract

Every run writes under an exclusive run directory:

- append-only event journal;
- frozen case packet copy;
- normalized outcome JSON when terminal is `READY`, `BLOCKED`, or `STALE`;
- exact result JSON only for `READY`;
- existing-schema R1 Run Artifact.

The event journal must record:

- run identity and frozen contract digest;
- arm and order position;
- workspace-before identity;
- every child-command start, finish, exit code, and elapsed time;
- every deterministic placement path;
- command, placement, and operator-action counts;
- workspace-after identity;
- exactly one matching terminal event.

The exact READY result binds:

- ready commit and tree identity;
- changed paths and modes;
- expected artifact hashes;
- authority and required-gate report identities.

Failure-boundary evidence contains no result artifact. Unexpected exceptions become
`INTERRUPTED` or `INVALIDATED`, remain visible, and cannot enter efficiency release.

Operation metrics are derived only from the content-addressed event journal after pair equivalence.
They are therefore bound by the existing event-log locator and pair evidence digests without
changing the R1 schema.

The metric definitions are frozen before execution:

- `child_command_count`: actual checked-in commands invoked by the arm;
- `placement_path_count`: individual files deterministically placed;
- `operator_action_count`: user-visible actions required by the compared workflow, not adapter
  internals. Arm A counts each stage command plus one repository-overlay placement action and, when
  present, one transaction-input placement action. Arm B is exactly one invocation of the existing
  Candidate 01 command;
- `total_ns`, `child_ns`, and `self_overhead_ns`: the existing R1 timing fields.

Truth-calibration time, case reconstruction, source acquisition, test execution, and report review
are excluded from arm efficiency. They remain visible as evaluation cost, not Candidate speedup.

## 9. Failure classification

The initial liveness cases freeze these classes:

- `C01-CODE-ONLY`: expected `READY`, exact artifact, authority `PASS`, all required gates `PASS`;
- `C06-STALE-MAIN`: expected `STALE`, diagnostic `STALE_MAIN`, recovery class
  `refresh_main_and_regenerate_packet`, no result artifact, no workspace mutation.

A stale case that places files before stopping is an execution-invalid partial mutation, not an
acceptable expected stop. A false `READY`, missing diagnostic, mismatched recovery class, changed
workspace, missing event, or substituted evidence fails the pair.

## 10. Truth freeze and liveness

Before measured runs, independently reconstruct and review the exact Arm-A truth for C01 and C06.
Freeze the case packet, R1 contract, expected paths, modes, hashes, gates, diagnostics, recovery
class, evaluator digest, and evidence-schema digest before Candidate output is inspected.

Then run exactly:

- C01 pair 0 `A -> B` and pair 1 `B -> A`;
- C06 pair 0 `A -> B` and pair 1 `B -> A`.

This is eight arm runs and four pair comparisons. Liveness passes only when all eight runs have
valid immutable evidence and all four pairs match the frozen truth. C06 contributes no efficiency
conclusion. Any Candidate mismatch stops the full nine-case evaluation and becomes a Candidate
finding; it does not authorize an in-place Candidate repair.

## 11. Tests and gates

Focused tests must cover:

- exact schedule and isolated workspace creation;
- local-commit preflight with no network;
- deterministic workspace identity, including untracked files and file modes;
- Arm-A command order and Arm-B single invocation;
- append-only events, command timing, placements, interruption, and duplicate-output rejection;
- READY evidence accepted by the unchanged R1 loader;
- STALE evidence with unchanged workspace accepted;
- stale-after-placement classified as partial mutation and rejected;
- evidence digest, run identity, and order substitution rejected;
- operator-action definitions and operation metrics withheld until correctness equivalence;
- existing `candidate` CLI and orchestration tests unchanged.

Required exact-head gates:

```text
python3 -m pytest tests/test_workflow_replay_execute.py \
  tests/test_workflow_replay_orchestrate.py \
  tests/test_workflow_replay_r1.py -q
python3 -m pytest -q
python3 -m ruff check .
python3 scripts/handoff_authority.py verify --repo-root .
python3 scripts/validate_formal_execution_channel.py --repo-root .
python3 scripts/validate_governance_pipeline_stage_status.py --repo-root .
```

The two-case real liveness is a separate engineering gate after exact-head CI. Unit fixtures or a
smoke callback cannot substitute for it.

## 12. Size and runtime budget

Production changed-line count covers nonblank, non-comment Python changes in the two allowed
behavior files.

- preferred: `120--220`;
- yellow architecture review: `221--300`;
- above `300`: stop and redesign;
- no new dependency;
- no network work during execution;
- no repository-wide content scan outside Git-reported tracked or untracked identities.

Child Git and V1 commands count as child time. Recorder, hashing, serialization, and comparison
self-overhead must have median `<=1 s`, p95 `<=2 s`, and no observation above `5 s` on the frozen
liveness environment.

## 13. Stop conditions

Stop without expanding scope when:

- any new Python file appears necessary;
- `evidence.py`, `orchestrate.py`, V1 owners, workflows, handoff, or registry would need changes;
- production changes exceed 300 lines;
- Arm A cannot remain independent of `run_candidate`;
- exact historical commits cannot be resolved locally;
- liveness exposes partial mutation, order sensitivity, evidence substitution, or a Candidate
  correctness mismatch;
- timing or operation metrics cannot be traced to content-addressed evidence.

## 14. Rollback and authorization

Rollback is one revert of the later adapter implementation PR plus deletion of unmerged generated
liveness outputs. There is no database, migration, default-route activation, scientific state, or
persistent service to unwind.

This contract authorizes no behavior code by itself. After this document passes exact-head review,
implementation still requires explicit user approval. Merge, Candidate repair, full nine-case
execution, adoption, default-route activation, R2 work, and scientific execution remain forbidden.
