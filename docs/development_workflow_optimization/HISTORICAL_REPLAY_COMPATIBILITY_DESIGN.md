# Historical Replay Compatibility Design

Parent claim: `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`

Base: `main@85b0a68d77ed085a7f6e67771fb0f7672c43da09`

Status: implementation authorized by the user on 2026-07-19; formal benchmark execution remains separately gated.

## Problem

ReplayAB intends to compare two workflow paths on the same immutable historical task while using one frozen benchmark toolchain. Current execution still derives part of its validation behavior from the historical base. This makes some pre-repair READY tasks fail before Candidate 01 is evaluated.

C06 still passes because it stops before normalization. A controlled task based on repaired main reaches READY and is exactly equivalent across A and B. A task whose base predates the repair still uses historical validation behavior. The remaining blocker is therefore historical replay compatibility, not Candidate 01 result equivalence.

## Proposed model

Keep three identities separate:

1. Historical task identity: old main, old source commit, reviewed blobs and expected task output.
2. Frozen benchmark identity: one toolchain SHA used identically by A and B.
3. Result identity: a commit based on the historical main that contains only the historical task changes and approved generated outputs.

The benchmark tooling may validate the historical result, but benchmark implementation files must not become part of that result.

## Invariants

- A and B receive identical historical inputs and identical benchmark controls.
- Candidate 01 remains the only treatment difference.
- Historical parent SHA, changed paths, modes and result semantics remain unchanged.
- V1 core, Stage 5 authority, handoff, registry, scientific code and production defaults are not modified.
- The benchmark toolchain is bound to one full commit SHA and recorded in every run.
- Missing objects, incompatible history, source drift or result drift fail closed.
- Failed compatibility probes are preserved.

## Eligibility

The first iteration is limited to tasks based on repositories at or after Stage 5 schema-v3 authority activation. Earlier tasks belong to a different control regime and are outside this orchestration benchmark.

Before any Candidate-B execution, Arm A alone must establish that a historical task is replayable under the frozen benchmark toolchain. The complete pool, accepted cases and rejected cases with reasons must be frozen before B is run.

Eligible cases are selected by predeclared task-class coverage and chronological order, never by Candidate outcome or speed. Fewer than six representative cases keeps the formal efficiency benchmark blocked.

## Allowed implementation scope

The implementation may modify only the existing replay adapter and an existing replay test file. It may not add another Python file, dependency, protected governance change or scientific change. It must use existing V1 and authority owners rather than reimplementing their rules.

Target maximum scope:

- one production file;
- one existing test file;
- at most 250 nonblank production lines;
- at most 350 nonblank test lines.

## Required validation

The implementation must prove:

- exact frozen toolchain identity;
- identical benchmark controls for A and B;
- unchanged historical parent and result paths;
- no benchmark implementation file entering the historical result;
- C06 still passes;
- controlled current-main C01 still passes;
- at least one post-Stage-5 historical READY task reaches Arm-A READY before B is run;
- incompatible histories fail with retained diagnostics.

Static checks, unit tests and one-case liveness do not constitute the formal efficiency result.

## Interpretation

This design asks whether Candidate 01 reduces orchestration effort under one current accepted benchmark toolchain. It does not compare historical and current validation systems, recreate historical real time, change old repository history or produce scientific evidence.

## Assessment

The design is feasible without protected-core changes. Governance risk is low-to-medium only if implementation remains confined to the replay adapter. If a historical READY task cannot pass without changing V1 core, Stage 5 authority, handoff or registry, implementation must stop and the benchmark must be redesigned.
