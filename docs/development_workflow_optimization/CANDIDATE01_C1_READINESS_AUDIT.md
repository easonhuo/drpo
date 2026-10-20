# Candidate 01 C1 Evaluation-Readiness Audit

**Claim:** `GOV-DEV-WORKFLOW-OPTIMIZATION-BENCHMARK-01`  
**Candidate:** `Candidate 01 -- V1 One-Click Integration`  
**Base:** `main@bb637503e1289f24f7a28e587f50665afb20e0de`  
**Branch:** `dev/candidate01-c1-readiness-audit-01`  
**Decision:** `NARROW_NOT_READY`  
**Scientific impact:** none

## 1. Decision

Candidate 01 should be evaluated with ReplayAB R1, but the full evaluation must not start yet.

The question is deterministic: does the one-click path produce the same accepted repository
artifact, or the same safe stop, as the explicit multi-step path while reducing operator work?

The repository already has the Candidate 01 Arm-B implementation and the R1 deterministic
ruler. It does not yet have one common real-run evidence path that executes both arms and emits
R1-compatible Run Artifacts, event journals, workspace identities, outcomes, and bound timing.

The case bank can be frozen now. A bounded evidence adapter and a two-case liveness gate are
required before the full bank runs.

## 2. Case policy

Use real history as the backbone and synthetic cases only for missing safety boundaries.
Synthetic cases do not contribute to efficiency claims. Historical scientific PRs are used only
as repository integration tasks; no training or scientific result is replayed.

## 3. Nine-case bank

| ID | Historical source | Task | Replay class | Mode | Terminal | Efficiency |
|---|---|---|---|---|---|---|
| `C01-CODE-ONLY` | PR #119 | code-only integration | reconstructed historical | `exact_artifact` | `READY` | yes |
| `C02-E7-ADD` | PR #52 | new E7 registration | reconstructed historical | `exact_artifact` | `READY` | yes |
| `C03-E8-ADD` | PR #49 | new E8 registration | reconstructed historical | `exact_artifact` | `READY` | yes |
| `C04-PROTOCOL-REPLACE` | PR #122 | protocol replacement | reconstructed historical | `exact_artifact` | `READY` | yes |
| `C05-RESULT-CLOSURE` | PR #86 | compact result closure | reconstructed historical | `exact_artifact` | `READY` | yes |
| `C06-STALE-MAIN` | PR #74 | stale-main attempt | reconstructed historical failure | `failure_boundary` | `STALE` | no |
| `C07-BEFORE-IMAGE` | PRs #54/#56 incident | stale replacement before-image | reconstructed historical failure | `failure_boundary` | `BLOCKED` | no |
| `C08-OVERLAY-CONFLICT` | Candidate regression | conflicting destination | synthetic boundary | `failure_boundary` | `BLOCKED` | no |
| `C09-GATE-FAILURE` | frozen failing gate | required gate failure | synthetic boundary | `failure_boundary` | `BLOCKED` | no |

### Historical identities

- `C01`: base `2f677f4b00954ea71d0efa7def552a1ea3daa565`, head
  `4bdc1fa80bafd997b6358c51a83f5e57dd77ed16`, registration mode `none`.
- `C02`: base `7350ab0161f39d1ccabfab8726b57cb14400071e`, head
  `6ac6d4cc8bc245eaa48907fb38a73059eef8906f`, subject
  `EXT-H-E7-W0-HIGHC-ACTOR-01`.
- `C03`: base `9c49824558b1eb7f697f299b246a135ff35a2017`, head
  `4b4366991c305e6da9e5e91b0f333c17174ce96d`, subject
  `EXT-C-E8-ORACLE-OFFLINE-V2-CONTINUOUS-EXP-GRID-0.5B-01`.
- `C04`: base `531506b4be6967012c5d01aa4d112deaaccf1c49`, head
  `e17fbedcdf326cda2780bfd075ff6ee3f9f09c79`; integration only, no GAE run.
- `C05`: base `0da7bd7bea1684c922fe7d4b25890be3c0327666`, head
  `8394c277d7df71fc6e2e105885297c45e92556db`; compact evidence only.
- `C06`: frozen implementation `6795aa6f086c44e8073c5a995a1612f334a3a067` versus
  newer main `d0b028bf438d32e550a281a79246cf828b36450e`.
- `C07`: use a genuine replacement packet with the historical stale before-image hash. Both
  arms must stop before authority mutation.
- `C08`: preserve the conflicting file, stop before V1 plan, and leave no partial placement.
- `C09`: reach a real required gate failure, stop before finalize, and never report `READY`.

The existing real V1 shadow in `tests/test_prepare_dev_pilot_registration_real_shadow.py` is a
calibration reference, not a measured Candidate 01 case.

## 4. Truth freeze

Expected truth must be independent of Candidate 01 output. For each reconstructed case:

1. resolve the exact historical base and source head;
2. reconstruct the reviewer task packet from accepted repository evidence;
3. run the explicit Arm-A path once in an isolated calibration workspace;
4. independently review paths, modes, gates, authority, terminal state, diagnostics, and tree
   identities;
5. freeze the R1 contract and expected verdict;
6. discard the calibration workspace;
7. only then run measured opposite-order pairs.

## 5. Arms

Arm A explicitly runs preparation, overlay placement, V1 plan, V1 prepare, optional transaction
input placement, normalize, gate, and finalize.

Arm B invokes the existing Candidate 01 path and must call the same owners once in the same
order. It may automate only sequencing and deterministic file placement.

No artificial delay or deliberate mistake may be added to Arm A.

## 6. Readiness

| Requirement | Status |
|---|---|
| Candidate 01 Arm B | `PASS` |
| explicit Arm-A protocol | `PASS_AS_PROTOCOL` |
| R1 exact/failure evaluator | `PASS` |
| historical refs | `PASS` |
| nine-case coverage | `PASS_AS_DESIGN` |
| immutable case packets | `MISSING` |
| common Arm-A/Arm-B Run Artifact producer | `MISSING` |
| real append-only event journal | `MISSING` |
| isolated opposite-order execution | `MISSING` |
| before/after workspace digest | `MISSING` |
| bound operation metrics | `MISSING` |
| two-case liveness | `NOT_RUN` |

`src/drpo/workflow_replay/execute.py` currently records fixture-callback execution. Candidate 01
runs real child commands, but it does not emit the complete R1 evidence contract. R1 can judge
evidence once produced; it does not create the isolated workspaces or run both arms.

## 7. Minimal closure before evaluation

A later separately approved implementation may add only:

- one common real-run evidence path for Arm A and Arm B;
- isolated workspaces from a frozen base;
- R1 event, outcome, result, and before/after identity evidence;
- command, placement, wall, child, self-overhead, and operator-action counts;
- consumption of existing R1 opposite-order identities and comparison.

It must not add a service, database, scheduler, plugin registry, generic agent backend, automatic
repair, semantic acceptance, Candidate adoption logic, or scientific execution.

No new Python path is justified. First extend existing `scripts/run_workflow_replay.py` and
`src/drpo/workflow_replay/execute.py`.

Production changed-line budget:

- preferred `120--220`;
- yellow review `221--300`;
- above `300`: stop and redesign.

## 8. Liveness gate

Before the full bank, run only:

1. `C01-CODE-ONLY` as a positive exact-artifact case;
2. `C06-STALE-MAIN` as a historical failure-boundary case.

Each runs one `A -> B` pair and one `B -> A` pair. All four runs must have immutable evidence,
match the frozen truth, use separate workspaces, and preserve correctness under order reversal.
Efficiency remains blocked for the stale case.

This is engineering liveness, not Candidate adoption evidence.

## 9. Final verdict

The verdict is **`NARROW_NOT_READY`**:

- the nine-case historical-first bank is suitable;
- R1 is sufficient for the deterministic claim;
- Candidate 01 can serve as Arm B;
- full evaluation waits for immutable case packets, the common evidence adapter, and two-case
  liveness;
- R2 is not required.

## 10. Authorization boundary

This document does not authorize adapter code, new Python files, case generation, A/B execution,
Candidate adoption, R2 work, merge, or any scientific change. The next permitted action is a
separate implementation contract naming exact existing files, budget, tests, and rollback, then
explicit user approval before behavior code changes.
