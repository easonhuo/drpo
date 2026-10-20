# M0 Stage 0 Review R-C — ReplayAB validity and development stability

**Claim:** `GOV-DRPO-MAINTENANCE-RUNNER-REPLAYAB-01`  
**Reviewed plan:** `docs/development_workflow_optimization/M0_ATOMIC_DEV_TRANSACTION_PLAN.md`  
**Reviewed plan commit:** `421c8024de95dafb8161b921d815f973275e5d34`  
**Current main:** `ad9bda80796dcf5c48976f5d64ffd79a006c70d5`  
**Review verdict:** `HOLD_MEASUREMENT_ADAPTER_REQUIRED`  
**Scientific impact:** none

## 1. Review question

Can the existing ReplayAB implementation execute and evaluate the M0 treatment without
silently turning the measurement instrument into another candidate or requiring a
mid-Stage-1 redesign?

## 2. What is already generic

`src/drpo/workflow_replay/execute.py` already provides reusable measurement primitives:

- strict `CommandSpec` and `ExecutionPlan`;
- deterministic Arm-A and Arm-B plan hashing;
- paired plan construction;
- append-only monotonic event recording;
- command count, child time, total time, and self-overhead separation;
- fail-closed fixture execution.

The case manifest and evidence layers already bind historical base, toolchain,
environment, cache policy, changed paths, outcome hashes, gates, and terminal state.

These parts do not need to be rewritten.

## 3. What is still Candidate-01-specific

The real measured CLI in `scripts/run_workflow_replay.py` is not a generic command-pair
runner.

Its `real-pair` path currently:

- prepares Candidate 01 inputs;
- runs V1 plan/prepare/normalize/gate/finalize;
- imports Candidate 01 orchestration;
- expects `READY_COMMIT.json`, `GATE_REPORT.json`, and V1 transaction directories;
- labels the producer as Candidate 01.

Therefore the current CLI cannot execute the M0 sequential-versus-atomic Git treatment
merely by adding new YAML cases.

The Stage 0 plan's statement that the existing `local-git-v1` instrument can be used
without any behavior change is not yet true.

## 4. Smallest sufficient measurement change

The smallest acceptable correction is not a new backend or new Python package.

Stage 1 may propose one bounded extension of the existing measurement entrypoint:

- modify existing `scripts/run_workflow_replay.py`;
- add one fixed `git-object-pair` command;
- load frozen Arm-A and Arm-B command plans from a strict non-executable case packet;
- create isolated local/bare Git repositories;
- invoke existing `ExecutionPlan`, event, evidence, schedule, and comparison primitives;
- independently inspect final parent, tree, changed paths, modes, terminal state, and
  untouched protected ref;
- emit the existing normalized run-artifact and pair-report shapes.

The change must not:

- call GitHub;
- implement M0 remote publication;
- parse arbitrary shell from a user task;
- encode E7/E8 branches;
- change Candidate 01 behavior;
- alter ReplayAB decision thresholds;
- add a Python file or dependency;
- create another evaluator.

Preferred production change: at most 100 lines in the existing script.
A 101--140 line change requires a yellow review.
More than 140 production lines, or a need to modify Core semantics, triggers redesign.

Focused coverage should extend existing tests rather than create a new Python path:

- `tests/test_workflow_replay_execute.py`;
- existing evidence/comparison tests when needed;
- non-Python fixtures under `tests/fixtures/workflow_replay/m0_atomic/`.

## 5. Development-order correction

To prevent a mid-stage pivot, Stage 1 must use this internal order:

1. **1A — measurement-adapter qualification:** prove the existing generic Core can run
   one success and one failure pair through the bounded CLI extension;
2. **1B — controlled eight-case readiness:** freeze and validate all command plans and
   evaluators without viewing efficiency results;
3. **1C — remote M0 success liveness:** perform only the two approved low-risk GitHub
   success transactions;
4. **1D — Stage 1 acceptance:** audit local failure boundaries, remote semantics, code
   size, and operation evidence together.

No remote M0 liveness begins before 1A passes. No eight-case efficiency output is
released before independent acceptance.

## 6. Case-bank stability

The historical PRs are payload sources, not timing baselines. Historical iteration
commits include design and repair work outside M0's treatment.

For every case:

- final reviewed after-images are reconstructed and frozen;
- Arm A is a predeclared accepted sequential Git-object/file-write plan;
- Arm B is the atomic tree/commit/ref plan;
- both receive identical after-images;
- scientific creation time and historical repair time are excluded;
- a single-file payload is not eligible as a success efficiency case;
- authority-materialization and workflow-change PRs are excluded;
- a case without complete after-images or independent expected outcome is classified
  partial or excluded before results.

This resolves the conflict between case-specific historical provenance and a controlled
common treatment.

## 7. M1 trigger stability

`REDESIGN_TO_M1` remains a possible Stage 2 verdict only for a measured remote
publication gap after M0 and the measurement adapter are correct.

A failure of the measurement adapter is not evidence for M1.
A failure to reconstruct historical payloads is not evidence for M1.
A weak M0 speedup is not permission to broaden M1.

## 8. Verdict

The M0 architecture remains stable, and M2 remains rejected.

However, the plan must explicitly include the bounded existing-file ReplayAB CLI
extension as instrument work. Continuing to say “no runtime code of any kind” would
defer a known requirement into Stage 1 and recreate the exact mid-development redesign
risk the user asked to avoid.

Stage 0 remains open until:

- the controlling plan incorporates this instrument scope and budgets;
- exact case and evaluator data files are frozen;
- exact validation commands are frozen;
- the revised plan passes a final closure-readiness review.
