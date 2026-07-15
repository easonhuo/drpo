# GOV-RUNTIME-RESOURCE-ACCEPTANCE-HARNESS-01 — one-command server acceptance

**Approval:** user explicitly authorized implementation on 2026-07-15.  
**Stacked base:** `aacaa5cb2425ebbcd5f5666b331694a37545d27a` from Draft PR `#67`.  
**Dependencies:** `GOV-RUNTIME-RESOURCE-AUTOTUNE-CPU-V2-01`,
`GOV-RUNTIME-RESOURCE-POOL-01`, and the separately pinned GPU-placement
selection-only child commit.  
**Scientific impact:** none.  
**Default-policy impact:** none.

## Objective

Replace ad hoc server-agent judgment with one repository-owned acceptance command.
The local server AI may supply machine-specific absolute paths and explicit CPU/GPU
pools through one validated JSON profile, but it must not write code, invent commands,
change test stages, alter runtime/scientific parameters, or decide how to interpret a
partial result.

The harness must exercise only engineering evidence:

1. topology and current-process audit;
2. resource-pool dry runs and mismatch rejection;
3. E7 measured-CPU plan selection;
4. E7 revalidation without starting the full pilot;
5. short real-data E7 liveness at the exact selected worker count;
6. GPU placement selection-only shadow on an exact pinned commit;
7. four-condition E8 thread-envelope scan;
8. concurrent E7/E8 liveness under non-overlapping CPU pools;
9. terminal process/orphan audit and compact artifact packaging.

No stage may start a complete E7 or E8 scientific sweep.

## Operator contract

The server operator performs only:

```text
1. checkout the reviewed exact harness commit;
2. copy the example profile and fill absolute server paths plus explicit CPU/GPU IDs;
3. run one shell command in the foreground;
4. wait for terminal state;
5. return the generated tar.gz and SHA-256.
```

Editing the profile is configuration, not code development. Every field is schema-
validated and echoed into provenance. The harness refuses unknown fields.

## Files authorized

- `scripts/run_runtime_resource_acceptance_one_click.sh`;
- `scripts/run_runtime_resource_acceptance.py`;
- `src/drpo/runtime_resource_acceptance.py`;
- `configs/runtime_resource_acceptance_server.example.json`;
- `docs/runtime_resource_acceptance_harness.md`;
- focused tests;
- the minimum E7 auto-entrypoint extension required for revalidation-only and
  selected-count liveness commands;
- append-only history entry in `docs/runtime_resource_autotune_evolution.md`.

No handoff, registry, scientific config, formal channel, or current experiment state
may change.

## Exact dependency model

The harness checkout contains measured-CPU V2 and Resource Pool because it is stacked
on PR `#67`. GPU placement remains a separate exact worktree created from the pinned
selection-only child commit. The harness must:

- verify its own checkout is clean;
- verify the GPU commit is a full 40-character SHA and available as a Git object;
- create a detached isolated GPU worktree under the output root;
- verify exact HEAD and clean status;
- remove the temporary worktree during terminal cleanup;
- never use a moving branch name as execution identity.

## Profile contract

The JSON profile has a closed schema with these groups:

```text
schema_version
output_parent
expected_harness_commit (optional exact-check guard)
gpu_selection_commit
resource_pools
E7 input paths and bounded engineering durations
E8 model/bank/validation/config paths and bounded probe durations
thread-scan candidates
concurrent-shadow policy
```

Absolute paths are required for all external model/data/config inputs. Output paths
must be outside every repository worktree. CPU pools must parse as Linux CPU lists,
be subsets of inherited affinity, and be pairwise disjoint. GPU IDs must be explicit
ordered unique identifiers.

The example profile contains unmistakable `REPLACE_WITH_ABSOLUTE_PATH` strings. The
harness fails before hardware work when any placeholder remains.

## Stage state machine

Every stage has exactly one terminal state:

```text
PASS
FAIL
BLOCKED
INCONCLUSIVE
NOT_RUN
```

A failed prerequisite prevents dependent stages and records `BLOCKED`; it does not
silently skip them. `continue_after_failure=false` is the default. Terminal cleanup
always runs.

Overall status is never inferred from a majority. It is:

- `PASS` only when every enabled required stage passes;
- `INCONCLUSIVE` when no required stage fails but at least one required capacity claim
  is empirically unresolved;
- `BLOCKED` when required inputs or safe capacity are unavailable;
- `FAIL` on contract, process, identity, cleanup, OOM, or numerical failure.

## Stage 0 — preflight and topology

Required checks:

- exact/clean harness checkout;
- external input existence and regular-file/directory boundaries;
- output root outside the repository;
- no tracked-file overlap;
- process inventory without killing or altering unrelated processes;
- CPU affinity, `/proc`, cgroup, memory, `lscpu`, optional `numactl`, `nvidia-smi`,
  and GPU topology evidence;
- declared pools are valid and non-overlapping;
- enabled stages do not conflict with live processes identified by configured patterns.

A live conflict yields `BLOCKED`; the harness never kills it.

## Stage 1 — resource-pool contract

Use only `scripts/run_with_resource_pool.py` to validate:

- E7 CPU identity;
- E8 CPU/GPU identity;
- exact immutable reuse;
- changed CPU rejection;
- changed GPU rejection;
- real child-process affinity and exported provenance.

Mismatch tests use isolated identity paths and harmless Python commands.

## Stage 2 — E7 measured-CPU acceptance

### Plan

Run the existing PPO-family measured-CPU `plan` inside the E7 pool. It may run the
registered non-scientific representative probe and bounded candidate grid. It must
write immutable selection and run identity.

### Validate

A new opt-in `validate` command must load the frozen selection, verify run identity,
perform three-sample revalidation, write attempt-local evidence, and exit without
calling the full pilot runner.

### Selected-count liveness

A new opt-in `liveness` command must:

- revalidate first;
- run the existing real D4RL representative command at the exact selected worker count;
- use a non-scientific seed namespace and bounded engineering steps/time;
- use the existing candidate CPU/RAM validity and process-group cleanup code;
- write `SELECTED_LIVENESS.json`;
- preserve immutable selection and run identity;
- never construct or execute the full branch matrix.

If revalidation cannot safely support the frozen count, it fails closed and never
runs fewer workers.

## Stage 3 — GPU placement selection-only

Invoke the exact pinned GPU child commit with `--selection-only` under the E8 resource
pool. The harness must not call the historical full-run mode. It verifies immutable
selection, required phases, zero exits, no controller termination, no OOM, no orphan,
and whether a candidate above one was actually exercised.

No candidate above one because measured resources forbid it is `INCONCLUSIVE` for the
multi-slot claim, not a code failure.

## Stage 4 — E8 thread-envelope scan

Run the same one-GPU, one-slot selection-only envelope sequentially under:

```text
unbounded
4
8
16
```

for OMP/MKL/OpenBLAS/NumExpr thread variables. The profile may disable candidates only
by editing the explicit candidate list before launch. The harness records wall time,
CPU, RSS, VRAM, GPU utilization/probe evidence, return status, OOM, NaN/Inf, and orphan
status. It does not select or activate a permanent thread policy.

## Stage 5 — concurrent pool shadow

After the independent E7 and E8 liveness stages pass, launch exactly one bounded E7
selected-count liveness and one one-GPU/one-slot E8 selection-only envelope
concurrently through `run_with_resource_pool.py`. Sample owned process trees at a
fixed interval and verify every observed affinity is a subset of the declared pool.
No unowned process is signalled or modified.

The stage records independent and concurrent wall time but does not impose an
unregistered throughput-regression threshold.

## Process supervision

Every owned command runs in a new process group with:

- command and environment allowlist record;
- stdout/stderr log;
- foreground polling;
- explicit timeout;
- SIGTERM then bounded SIGKILL only for the owned process group;
- final descendant/orphan audit;
- return code, timeout, controller intervention, peak RSS, CPU time, and sampled
  affinity evidence.

The harness itself must handle SIGINT/SIGTERM by terminating only owned groups and
writing a terminal failure report.

## Artifact contract

The output root is new or empty and includes:

```text
ACCEPTANCE_SUMMARY.json
SERVER_ACCEPTANCE_REPORT.md
COMMANDS_EXECUTED.jsonl
FINAL_PROCESS_AUDIT.json
FILE_MANIFEST.sha256
stage0_topology/
stage1_resource_pool/
stage2_e7_cpu_v2/
stage3_gpu_placement/
stage4_e8_thread_scan/
stage5_concurrent_pool/
```

The final tar.gz includes only JSON/JSONL/CSV/Markdown/text logs/small config copies and
checksums. It excludes model, dataset, checkpoint, adapter, optimizer, and worktree
payloads. The package path, size, and SHA-256 are printed at terminal state.

## Fail-closed boundaries

The harness refuses:

- dirty or unexpected checkout;
- placeholders or relative external paths;
- output inside a repository;
- unknown profile fields;
- overlapping or unavailable CPU pools;
- duplicate/malformed GPU IDs;
- moving GPU branch refs in place of a full SHA;
- full-sweep commands;
- changed E7 selection digest or worker count;
- second E7 probe/grid during validate;
- controller cleanup on an otherwise accepted GPU candidate;
- OOM, NaN/Inf, orphan process groups, or repository mutation;
- missing terminal evidence;
- packaging of forbidden large/model artifacts.

## Explicit exclusions

- central scheduler or reservation service;
- dynamic scaling, preemption, migration, or online pool negotiation;
- permanent CPU/GPU pool selection;
- automatic NUMA optimization;
- scientific thread-policy activation;
- Slurm/Kubernetes/Ray integration;
- full scientific sweeps;
- result ranking, convergence, steady-state, OOD, or method-superiority claims;
- merge/default activation of dependency PRs.

## Deterministic acceptance

Tests must cover profile validation, placeholder rejection, path isolation, pool
non-overlap, exact commit/worktree checks, owned-process supervision, timeout cleanup,
stage dependency transitions, E7 validate/liveness no-full-run behavior, package
allowlist, checksum generation, and final status aggregation. Exact-head Python
compile, shell syntax, focused tests, full pytest, Ruff, handoff authority, formal
channel, and governance gates must pass.

## Remaining empirical uncertainty

Only the target server can establish:

- actual E7 selected count inside its pool;
- whether that count remains safe at run time;
- actual GPU slots per H20;
- actual effect of thread candidates;
- concurrent E7/E8 isolation and wall-time behavior;
- server-specific cgroup and topology details.

Those are resolved by the harness output, not by changing code during server execution.

## Rollback

Stop using the harness and dependency opt-in flags. All historical launchers remain
available. Preserve acceptance artifacts and failure evidence; never reinterpret them
as scientific results.