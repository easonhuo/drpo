# M0 Atomic Development Transaction — Stage 1/2 Execution Contract

**Claim:** `GOV-DRPO-MAINTENANCE-RUNNER-REPLAYAB-01`  
**Benchmark:** `M0-ATOMIC-REPLAYAB-01`  
**Authoritative base:** `main@d3f7d046f948108a3d837bdcff617eed5146a2f0`  
**Scientific impact:** none  
**Status:** frozen before accepted Stage 2 execution

## Decision inherited from Stage 0

The superseded M2 patch/apply/test/commit/push runner is rejected as duplicative. M0 has no production runtime component. It standardizes the existing GitHub Git-object route for already reviewed complete UTF-8 after-images.

The only Stage 1 executable change is a bounded local measurement command added to the existing `scripts/run_workflow_replay.py`. It cannot contact GitHub, modify handoff or registry authority, execute experiments, or become an E7/E8 adapter.

ReplayAB R2 is already closed on the base main. The new command is a raw controlled-transaction evidence producer; independent acceptance remains outside the treatment and is bound by `EVALUATOR_CONTRACT.yaml`, exact-head tests, and the final decision record. No R2 Core behavior is modified.

## Frozen implementation scope

Allowed executable path:

- `scripts/run_workflow_replay.py`

Allowed test path:

- `tests/test_workflow_replay_execute.py`

Allowed non-Python fixtures and evidence:

- `tests/fixtures/workflow_replay/m0_atomic/CASE_BANK.json`
- `docs/development_workflow_optimization/benchmarks/M0_ATOMIC_REPLAYAB_01/**`
- `docs/scopes/GOV-DRPO-MAINTENANCE-RUNNER-REPLAYAB-01-STAGE12.md`

No new Python file, workflow, dependency, backend, service, publisher, E7 adapter, E8 adapter, handoff/registry change, scientific variable, experiment execution, default-route activation, or automatic merge is authorized.

The executable diff is in the yellow review band: more than 100 but no more than 140 added/changed executable lines. This requires an explicit pre-publication yellow-zone review; it does not relax any correctness or safety gate.

## Treatment

Arm A models sequential reviewed after-image publication: one file placement and commit per file, followed by one development-ref publication.

Arm B models the M0 Git-object transaction: one reviewed after-image tree operation, one commit construction, and one development-ref publication. The local implementation uses isolated Git repositories to reproduce final-tree and parent/ref semantics. GitHub remote timing is reported separately and is not mixed into controlled timing.

Both arms receive identical packet content. Neither arm receives scientific logic, test commands, environment overrides, or repair behavior from the packet.

## Packet boundary

Every packet is data-only and binds:

- schema version, case ID, E7/E8 scenario family;
- expected terminal state and optional fixed failure class;
- one to 32 complete UTF-8 files;
- safe repository-relative paths;
- mode `100644` only;
- content SHA-256 for every file.

Packets cannot include commands. They reject NUL content, files above 1 MiB, duplicate or unsafe paths, `.github/**`, handoff/registry and schema-v3 delta paths, handoff authority paths, governance stage ledgers, and governance authorization records.

## Evidence layers

1. **Controlled local Git:** A/B/B/A execution in new isolated repositories; final tree, parent, file mode, terminal state, protected ref, operation count, and monotonic wall time.
2. **Independent acceptance:** exact packet and producer hashes, final-tree equivalence, parent/ref semantics, failure-boundary preservation, and protected-path rejection.
3. **Remote operational qualification:** one exact-head implementation PR built by an atomic Git-object commit, with its parent/tree/diff and GitHub Actions observed separately.

A local result does not prove GitHub CI or remote semantics. A remote PR timestamp does not enter the controlled timing estimate.

## Frozen cases and sequencing

The general screen contains eight cases: four E7-labeled and four E8-labeled, with three successful file-count/path-role scenarios and one deterministic failure boundary per family. They are retained regardless of result.

A separate four-case narrow confirmation bank is frozen before its accepted execution. It contains successful E7 and E8 scenarios with 7–10 files. It exists because the general screen may support a size-bounded treatment even when the general task class fails the action-count gate. The screen cases are not removed or relabeled.

Each command runs two opposite-order pairs. If paired wall-time spread exceeds 15%, at least one additional opposite-order execution is required and retained.

## Frozen gates

General `ADOPT_M0` requires:

- every successful run passes independent acceptance;
- both failure cases stop at the frozen boundary with the protected ref preserved;
- zero false acceptance or protected/scientific mutation;
- no accepted case is slower by more than `max(60 seconds, 5% of Arm-A median)`;
- median controlled wall-time improvement at least 30%;
- mean controlled wall time improves;
- median active-operation-time improvement at least 30%;
- median operator/tool-action reduction at least 60%;
- E7 and E8 are separately positive;
- M0 production code remains zero.

If the general screen fails only because small file-count cases cannot meet the action-count threshold, the permitted next verdict is `NARROW_M0`. Narrow confirmation then applies the same gates to the frozen 7–10-file task class. No small-case result is discarded.

Correctness or security failure yields `REJECT_OPTIMIZATION`. A specific residual verified-commit publication gap may yield `REDESIGN_TO_M1`; M1 is not authorized by this contract.

## Merge rule

Stage 1 is not merged before Stage 2. The implementation, screen, narrow confirmation, exact-head tests, and final decision use one exact PR head.

Merge is permitted only for:

- `ADOPT_M0`; or
- `NARROW_M0` after narrow confirmation and exact-head review pass.

`REDESIGN_TO_M1` and `REJECT_OPTIMIZATION` are not merged. Merge does not activate M0 as the repository default route; a later default-policy change remains separately gated.

## Pre-freeze activity

Local smoke and implementation-qualification runs performed while shaping the command are excluded from accepted Stage 2 evidence. Accepted raw results must be rerun after this contract, case inventory, evaluator contract, and packet bytes are frozen.
