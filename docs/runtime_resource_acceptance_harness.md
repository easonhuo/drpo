# Runtime Resource Acceptance Harness

**Claim:** `GOV-RUNTIME-RESOURCE-ACCEPTANCE-HARNESS-01`  
**Scientific impact:** none  
**Execution model:** one foreground shared-host command, engineering evidence only

## Current contract

The default route is a bounded shared-host capacity check, not a CPU-reservation or
exclusive-resource guarantee.

It combines four controls:

1. explicit CPU/GPU hard ceilings for DRPO-owned processes;
2. pool-local measurement of currently available CPU, RAM, GPU, and VRAM capacity;
3. configured worker/slot caps plus launch-time revalidation;
4. owned-process supervision, cleanup, and terminal audit.

ResearchBench, AIDE, joblib/loky, and other permanent server workloads may remain alive.
Their names and process metadata are recorded, but their presence alone is not a
readiness gate. Their actual use of a declared CPU pool is reflected by the measured
capacity available inside that pool.

A CPU pool limits where DRPO-owned processes may execute. It does not reserve those CPUs
or prevent unrelated workloads from sharing them.

The historical exclusive-cgroup route remains an optional diagnostic for compatible
cgroup v2 hosts. It is not required by the default target-server route. See
`docs/runtime_resource_acceptance_server_correction_05.md`.

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
python3 scripts/run_runtime_resource_acceptance_shared_host.py \
  --profile /absolute/path/runtime_resource_acceptance_server.json \
  --validate-profile
```

No operator-side process-count-zero or exclusive-partition gate may be inserted before
the command.

## Hard ceilings

The profile supplies two non-overlapping CPU pools. Every delegated E7/E8 child must
inherit the exact requested affinity, so DRPO-owned work cannot escape to the rest of
the machine.

Additional hard caps remain:

- E7: `max_workers`, task count, and growth bound;
- E8: `max_devices` and `max_slots_per_gpu`;
- GPU IDs: explicit launcher-enforced pool;
- E8 thread envelopes: explicit inherited/default, 4, 8, and 16-thread candidates;
- CPU/RAM/GPU/VRAM headroom and per-worker safety factors.

These are ceilings on DRPO-owned execution. They are not a claim that the selected CPUs
or GPUs are idle or exclusive.

## Fixed stages

### Stage 0 — topology and bounded preflight

Records the exact harness and GPU commits, inherited affinity, `lscpu`, optional NUMA
information, `nvidia-smi`, GPU topology, cgroup/memory evidence, and current processes.
Configured process patterns are observation-only provenance. Stage 0 records matching
external workloads but does not block merely because they exist.

Stage 0 still blocks on true prerequisites such as missing inputs, dirty or mismatched
checkout, or requested CPUs outside inherited affinity.

The CPU pools in the example are illustrations only. Replace them with two
non-overlapping subsets of the server's inherited affinity while retaining an OS and
existing-workload buffer. The harness does not make the temporary split permanent.

### Stage 1 — Resource Pool V1

Exercises exact identity reuse, CPU/GPU mismatch rejection, and real child-process
inheritance through `scripts/run_with_resource_pool.py`.

### Stage 2 — measured-capacity E7

Runs the PPO-family resource `plan` inside the E7 CPU pool. CPU accounting covers only
the active affinity CPUs. The selector combines measured external occupancy in that
pool, measured per-worker demand, RAM headroom, and configured worker caps.

It freezes `RUNTIME_SELECTION.json` and `RUN_IDENTITY.json`, performs validate-only
three-sample revalidation, and then runs a short real-D4RL representative liveness at
the exact selected worker count. It invokes the existing probe/candidate implementation
and never constructs the full 186-branch or Stage-A matrix.

If current capacity cannot safely support one worker, the stage is `BLOCKED`. If only a
single worker can be exercised for a multi-worker claim, the stage is `INCONCLUSIVE`.
The supplied contract, RunSpec, and grid must be the existing reviewed E7 inputs. The
server AI must not synthesize or edit them.

### Stage 3 — measured-capacity GPU placement

Creates a detached worktree at the exact `gpu_selection_commit` and invokes the
phase-aware Countdown envelope with `--selection-only` inside the E8 CPU/GPU ceilings.
Calibration, training peak, maximum-shape validation evaluation, bounded candidates,
clean exits, and immutable selection are retained. The scientific slot runtime is never
called and the test split is not accessed.

CPU/RAM and GPU/VRAM measurements determine the safe slot count below
`max_devices * max_slots_per_gpu`. External process names are not a separate gate.

### Stage 4 — E8 thread envelope

Sequentially repeats the same one-GPU/one-slot engineering envelope under the explicit
profile candidates: inherited/default environment, 4, 8, and 16 threads. It reports raw
evidence and does not activate a permanent thread policy.

Every candidate remains confined to the E8 CPU pool, so even an inherited/default
thread library cannot consume CPUs outside the configured ceiling.

### Stage 5 — concurrent bounded execution

Only after independent E7 and E8 stages are usable, launches one selected-count E7
liveness and one one-GPU E8 envelope concurrently through non-overlapping DRPO CPU
pools. Every sampled DRPO-owned process affinity must remain within its declared pool.

This proves containment between the two DRPO workloads. It does not claim isolation
from unrelated shared-host workloads.

## Status semantics

Every stage is one of:

```text
PASS
FAIL
BLOCKED
INCONCLUSIVE
NOT_RUN
```

No majority vote is used.

- `PASS`: the required engineering checks completed within the configured ceilings.
- `INCONCLUSIVE`: only a single worker/slot was exercised for a multi-worker claim.
- `BLOCKED`: current safe measured capacity or another prerequisite was unavailable.
- `FAIL`: identity, implementation, process supervision, cleanup, OOM, or numerical
  failure.

A capacity-limited `BLOCKED` result is an environment observation for that run, not a
request to terminate permanent workloads and not evidence that the scientific experiment
is impossible. A partial acceptance package is still produced.

## Outputs

The output root contains stage-local logs and structured evidence plus:

```text
ACCEPTANCE_SUMMARY.json
SERVER_ACCEPTANCE_REPORT.md
COMMANDS_EXECUTED.jsonl
FINAL_PROCESS_AUDIT.json
FILE_MANIFEST.sha256
```

Stage 0 additionally writes the shared-host capacity contract and the observed external
workload inventory.

The final `.tar.gz` contains text/JSON/CSV/Markdown evidence only. Repository worktrees,
datasets, model weights, adapters, checkpoints, and optimizer states are excluded.

## Interpretation boundary

This harness validates runtime measurement, bounded placement, process cleanup, and
DRPO-owned resource containment. It is not a formal experiment and cannot establish
task performance, method ranking, convergence, steady state, controlled mechanism
identification, or OOD generalization.
