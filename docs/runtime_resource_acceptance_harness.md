# Runtime Resource Acceptance Harness

**Claim:** `GOV-RUNTIME-RESOURCE-ACCEPTANCE-HARNESS-01`  
**Scientific impact:** none  
**Execution model:** one foreground command, engineering evidence only

## What the server AI does

The server AI does not write Python or shell code. It performs four bounded operator
steps:

1. check out the reviewed exact harness commit;
2. copy `configs/runtime_resource_acceptance_server.example.json` outside the
   repository;
3. replace the marked absolute paths and the temporary non-overlapping CPU/GPU pools;
4. run one foreground command and wait for terminal state.

Profile editing is limited to machine-specific paths and explicit resource IDs. The
harness rejects unknown fields, relative paths, placeholders, overlapping pools, and
unavailable CPUs before starting hardware work.

The compatibility field below must remain unchanged:

```json
"test": "/dev/null"
```

It is not a test-data path. The pinned GPU selection-only entrypoint neither emits a
`--test` argument nor hashes or opens a test split. Its immutable selection records
`test_split_access=not_accessed_selection_only` and `test_sha256=null`.

## One command

```bash
bash scripts/run_runtime_resource_acceptance_one_click.sh \
  --profile /absolute/path/runtime_resource_acceptance_server.json
```

The command must remain in the foreground. It creates a new output root below the
profile's `output_parent`, polls every owned process to terminal state, terminates only
its own timed-out process groups, performs final cleanup, and prints the final package
path and SHA-256.

A profile can be checked without touching hardware:

```bash
python3 scripts/run_runtime_resource_acceptance.py \
  --profile /absolute/path/runtime_resource_acceptance_server.json \
  --validate-profile
```

## Fixed stages

### Stage 0 — topology and safety preflight

Records the exact harness and GPU commits, inherited affinity, `lscpu`, optional NUMA
information, `nvidia-smi`, GPU topology, cgroup/memory evidence, and current processes.
Configured process patterns block rather than kill a conflicting E7/E8 run.

The CPU pools in the example are illustrations only. Replace them with two
non-overlapping subsets of the server's inherited affinity while retaining an OS and
existing-workload buffer. The harness does not make the temporary split permanent.

### Stage 1 — Resource Pool V1

Exercises exact identity reuse, CPU/GPU mismatch rejection, and real child-process
inheritance through `scripts/run_with_resource_pool.py`.

### Stage 2 — measured-CPU E7

Runs the PPO-family resource `plan`, freezes `RUNTIME_SELECTION.json` and
`RUN_IDENTITY.json`, performs validate-only three-sample revalidation, and then runs a
short real-D4RL representative liveness at the exact selected worker count. It invokes
the existing probe/candidate implementation and never constructs the full 186-branch
or Stage-A matrix.

The supplied contract, RunSpec, and grid must be the existing reviewed E7 inputs. The
server AI must not synthesize or edit them.

### Stage 3 — GPU placement

Creates a detached worktree at the exact `gpu_selection_commit` and invokes the
phase-aware Countdown envelope with `--selection-only`. Calibration, training peak,
maximum-shape validation evaluation, bounded candidates, clean exits, and immutable
selection are retained. The scientific slot runtime is never called and the test split
is not accessed.

### Stage 4 — E8 thread envelope

Sequentially repeats the same one-GPU/one-slot engineering envelope under the explicit
profile candidates: inherited/default environment, 4, 8, and 16 threads. It reports
raw evidence and does not activate a permanent thread policy.

### Stage 5 — concurrent isolation

Only after independent E7 and E8 stages both pass, launches one selected-count E7
liveness and one one-GPU E8 envelope concurrently through non-overlapping resource
pools. Every sampled process affinity must remain within its declared pool.

## Status semantics

Every stage is one of:

```text
PASS
FAIL
BLOCKED
INCONCLUSIVE
NOT_RUN
```

No majority vote is used. A resource-limited run that cannot exercise a candidate
above one is `INCONCLUSIVE` for the multi-worker/multi-slot claim. It is never rewritten
as a PASS. A partial acceptance package is still produced.

## Outputs

The output root contains stage-local logs and structured evidence plus:

```text
ACCEPTANCE_SUMMARY.json
SERVER_ACCEPTANCE_REPORT.md
COMMANDS_EXECUTED.jsonl
FINAL_PROCESS_AUDIT.json
FILE_MANIFEST.sha256
```

The final `.tar.gz` contains text/JSON/CSV/Markdown evidence only. Repository
worktrees, datasets, model weights, adapters, checkpoints, and optimizer states are
excluded.

## Interpretation boundary

This harness validates runtime measurement, placement, process cleanup, and resource
isolation. It is not a formal experiment and cannot establish task performance,
method ranking, convergence, steady state, controlled mechanism identification, or
OOD generalization.
