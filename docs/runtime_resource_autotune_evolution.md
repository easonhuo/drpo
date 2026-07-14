# DRPO Runtime Resource Autotune Evolution and Current Design

**Claim:** `GOV-RUNTIME-RESOURCE-AUTOTUNE-EVOLUTION-01`  
**Document base:** `c0ff38b51b0062b26a20771421f62b08eaaa0d12`  
**Status:** engineering design and append-only evolution ledger  
**Scientific impact:** none  
**Default-policy impact:** none

## 1. Purpose and authority

This document is the long-lived engineering history and current design record for
DRPO runtime resource autotuning.

It has two responsibilities:

1. state the currently proposed or effective resource-selection contract in one
   place; and
2. preserve every material design iteration, incident, replacement decision,
   hardware-shadow finding, and rollback boundary.

This document is **not** the research master, an experiment registry, a formal result,
or a second scheduler specification. `docs/handoff.md` remains the unique research
master and `experiments/registry.yaml` remains the experiment registry. Resource
probes are engineering evidence only and must never upgrade a scientific result.

The chronological history in Section 13 is append-only. A later correction adds a
new entry that identifies the superseded statement; it does not erase the old entry.
The current-design sections may be updated only together with an explicit history
entry identifying what changed and why.

Engineering statuses used here are:

```text
proposed
implemented
ci_validated
hardware_shadow_validated
active_opt_in
superseded
rejected
```

## 2. Repository state at this design baseline

At the document base, the authoritative `main` behavior is the opt-in V1 path:

- E7 measures representative process-tree peak RSS and selects active subprocess
  count from logical CPU count, one-minute load average, host/cgroup memory, task
  count, a configured cap, and a bounded growth factor;
- E8 selects visible, sufficiently idle GPUs with a free-VRAM floor and host-memory
  limit, with one process per GPU;
- fixed launchers remain available and unchanged;
- the selected runtime schedule is provenance and must not change the scientific
  matrix.

The following later designs or observations are not yet authoritative `main`
behavior at this base:

- same-GPU multi-process placement and phase-aware GPU envelope work in Draft PR
  `#53`;
- the measured-CPU GPU selector at reviewed commit
  `a378c4359777d7ae6202b001d9318241373f23a8`;
- the E7 Stage A CPU self-feedback incident observed on the server while exercising
  the code-first Stage A branch;
- the measured-CPU E7 V2 design specified in this document.

Each is recorded because it provides engineering evidence, but none is promoted to
active behavior merely by being described here.

## 3. Locked design boundary

The runtime autotune layer may choose only execution resources already declared as
runtime-variable by the workload adapter. For the current CPU scope, that variable
is the number of independent subprocesses.

It may not change:

- dataset, environment, model, method, seed, coefficient, or control family;
- optimizer, learning rate, batch size, sequence length, precision, or thread
  environment;
- training horizon, stopping rule, evaluation frequency, evaluation episode count,
  or result-selection rule;
- scientific branch identities or matrix membership;
- formal-result status, convergence classification, or method ranking.

The measured-CPU V2 implementation is a single-node capacity and bounded-throughput
preflight. It is not:

- a process scheduler or supervisor;
- dynamic online scaling;
- worker migration;
- CPU affinity or NUMA optimization;
- Slurm, Kubernetes, Ray, Dask, or provider integration;
- multi-node placement;
- automatic batch, thread, dataloader, or precision tuning;
- a cross-project portable policy package;
- a claim of global throughput optimality.

## 4. Current decision: measured-CPU V2 for DRPO E7

### 4.1 Objective

Replace the raw-load-average CPU capacity model with a measured model that uses:

1. effective process-visible CPU capacity;
2. actual system CPU execution occupancy;
3. measured representative worker CPU demand;
4. existing host/cgroup memory evidence;
5. the existing bounded throughput candidate search where a workload already uses
   it; and
6. immutable `plan` selection followed by lightweight `run` revalidation.

The design must solve both known failure classes:

- inaccurate capacity decisions caused by interpreting Linux load average as occupied
  CPU cores; and
- self-feedback where a `plan` benchmark raises load average and a subsequent `run`
  invocation recalculates a radically smaller worker count.

### 4.2 Scope of reuse

The shared measured-CPU arithmetic is intended for existing E7 CPU auto paths that
ultimately delegate to the same runtime selection implementation, including:

- canonical E7 CPU/RAM selection;
- PPO w(0) grid selection;
- squared-EXP night selection;
- high-c actor selection;
- squared-EXP KL Stage A selection after that development branch is synchronized.

Thin workload wrappers continue to own representative-branch identity and scientific
fingerprints. They must not copy the capacity arithmetic.

### 4.3 Required components and necessity test

| Component | Direct purpose | Why it is required |
|---|---|---|
| affinity-aware CPU count | observe process-visible CPUs | `os.cpu_count()` may include unusable CPUs |
| cgroup CPU quota | avoid quota overestimation | affinity may exceed the container's executable quota |
| `/proc/stat` busy-core sample | measure current CPU execution | load average includes history and uninterruptible waits |
| process-tree CPU demand | distinguish light and heavy workers | free cores alone do not imply a worker count |
| CPU safety reserve | cover sampling and workload variance | short probes are estimates, not exact future demand |
| existing RAM/cgroup memory gate | prevent memory oversubscription | CPU capacity alone is insufficient |
| workload-specific throughput grid | avoid selecting a safe but inefficient ceiling | already valuable for PPO-family E7 paths |
| immutable plan selection | remove probe self-feedback and identity drift | run must not silently choose a different schedule |
| lightweight run revalidation | block genuinely unsafe later starts | immutable does not mean blindly reusable |
| schema/policy versioning | reject old load-average selections | old evidence has incompatible semantics |
| attempt-local revalidation record | preserve allow/block evidence without mutating selection | the authoritative plan artifact must remain immutable |

No additional production module is required for these responsibilities. The first
implementation should extend the existing runtime resource core and adapters.

## 5. Machine CPU capacity contract

### 5.1 Affinity capacity

The process-visible CPU set is obtained from `os.sched_getaffinity(0)` on Linux.
Its cardinality is:

```text
affinity_capacity_cores
```

The exact CPU IDs are also used to select the per-CPU rows read from `/proc/stat`.
When affinity is unavailable, the implementation may use `os.cpu_count()` only as an
explicitly recorded compatibility fallback. Production Linux shadows must exercise
the affinity path.

### 5.2 Cgroup quota capacity

A finite cgroup quota is converted to equivalent CPU cores:

```text
quota_capacity_cores = quota_microseconds / period_microseconds
```

V2 must support:

- cgroup v2 `cpu.max`; and
- direct-file cgroup v1 compatibility through `cpu.cfs_quota_us` and
  `cpu.cfs_period_us` when those files are visible under the configured cgroup root.

`max`, a negative v1 quota, or absent quota files mean unlimited quota. Malformed,
zero-period, or contradictory values fail closed rather than being guessed.

### 5.3 Effective capacity

```text
effective_cpu_capacity_cores =
    min(affinity_capacity_cores, finite_quota_capacity_cores)
```

When no finite quota exists, effective capacity equals affinity capacity.
Fractional quota capacity is preserved as a float for arithmetic and recorded
verbatim. It is not rounded up.

The capacity available to the autotuned workload is:

```text
cpu_capacity_ceiling_cores =
    effective_cpu_capacity_cores * cpu_fraction
```

The existing default `cpu_fraction=0.85` remains the global execution headroom unless
an approved adapter policy explicitly supplies another value.

## 6. Actual CPU occupancy contract

### 6.1 Why load average is diagnostic only

Linux load average is an exponentially smoothed count of runnable and
uninterruptible tasks. It is not a count of CPU cores currently executing work. It
can remain high after a benchmark exits and can include I/O waits that do not consume
CPU execution capacity.

V2 therefore records one-, five-, and fifteen-minute load average only as diagnostic
provenance. No load-average field participates in worker-capacity arithmetic or cache
acceptance.

### 6.2 `/proc/stat` sampling

System execution occupancy is measured over a bounded monotonic interval using the
per-CPU rows for the current affinity set.

Busy time includes:

```text
user + nice + system + irq + softirq + steal
```

Idle compute capacity includes:

```text
idle + iowait
```

Guest fields are not double-counted because Linux already includes guest time in
user/nice accounting.

For an interval with valid positive tick deltas:

```text
system_busy_cores =
    affinity_capacity_cores * busy_tick_delta / total_tick_delta
```

The default sampling interval is one second. It must be configurable for tests and
hardware shadows but is not a scientific parameter.

Missing per-CPU rows, a non-positive total delta, or a changing affinity set makes
the measured auto decision unavailable. The auto path fails closed; operators may
use the unchanged fixed launcher rather than receiving an invented capacity.

### 6.3 Aligned measurement windows

Worker CPU demand and system busy cores must be measured over aligned monotonic
windows. A system sample from before or after the worker interval must not be mixed
with worker CPU demand from another interval.

The selection artifact records interval start/end timestamps, elapsed seconds, CPU
set, system tick deltas, and process-tree CPU deltas.

## 7. Representative worker CPU demand

### 7.1 Measurement

The existing representative resource probe is extended to record cumulative user and
system CPU seconds for the probe process tree while it is alive. Average worker CPU
cores are:

```text
measured_cpu_cores_per_worker =
    process_tree_cpu_seconds / aligned_elapsed_seconds
```

The probe must:

- use the workload's existing representative branch and thread environment;
- remain in the dedicated non-scientific seed namespace;
- produce a positive RSS peak and positive CPU-demand measurement;
- terminate or be cleaned up according to the existing probe contract;
- leave no process-group descendants;
- preserve logs and small resource evidence while removing generated model payload.

The first V2 implementation does not add a new phase framework to E7. E7 workers are
long-lived, so cumulative process-tree sampling is sufficient for the registered
workloads. Very short-lived descendants between polls remain a known limitation;
the global CPU reserve and bounded concurrent benchmark are the safety controls.

### 7.2 Reservation

Proposed defaults:

```text
per_worker_cpu_safety_factor = 1.25
minimum_cpu_cores_per_worker = 1.0
```

The reserved demand is:

```text
reserved_cpu_cores_per_worker =
    max(minimum_cpu_cores_per_worker,
        measured_cpu_cores_per_worker * per_worker_cpu_safety_factor)
```

Both values are runtime safety policy and are recorded in the resource fingerprint.
They do not alter scientific execution fields.

### 7.3 External occupancy

Because the system busy sample includes the probe itself, external occupancy is:

```text
external_busy_cores =
    max(0,
        system_busy_cores - measured_probe_cpu_cores)
```

The values must come from the same aligned interval. The result is an estimate and is
therefore never allowed to increase available capacity beyond the measured ceiling.

## 8. Capacity and throughput selection

### 8.1 CPU worker limit

```text
worker_cpu_budget_cores =
    max(0, cpu_capacity_ceiling_cores - external_busy_cores)

cpu_worker_limit =
    floor(worker_cpu_budget_cores / reserved_cpu_cores_per_worker)
```

If the limit is below one, auto selection fails closed.

### 8.2 Memory worker limit

The existing memory contract remains:

```text
reserved_memory_per_worker =
    max(1, ceil(peak_process_tree_rss * memory_safety_factor))

usable_memory =
    floor(effective_memory_available * (1 - memory_headroom_fraction))

memory_worker_limit =
    floor(usable_memory / reserved_memory_per_worker)
```

Memory and CPU measurements must refer to the same representative workload identity.

### 8.3 Safe capacity ceiling

```text
safe_capacity_ceiling = min(
    cpu_worker_limit,
    memory_worker_limit,
    total_tasks,
    configured_max_workers when present,
    bounded_growth_limit
)
```

The existing bounded growth limit remains useful as a policy-level blast-radius cap.
It is not evidence that the selected worker count is optimal.

### 8.4 Throughput candidate search

Workloads that already implement the bounded PPO-family throughput search retain it.
The measured model supplies only the safe ceiling. It does not create a new generic
throughput engine.

The existing candidate principle remains:

- benchmark a small set containing the verified fallback and representative points
  near 50%, 75%, and 100% of the safe ceiling;
- require every candidate worker to exit successfully without timeout;
- measure aggregate completed optimizer updates per second;
- select the smallest successful candidate reaching the configured fraction of the
  peak aggregate throughput;
- never benchmark above the safe capacity ceiling.

The current default throughput-retention fraction remains `0.97` for adapters that
already use it. E7 adapters without a throughput grid may select the safe capacity
ceiling directly and must record that limitation.

## 9. Plan, run, resume, and revalidation

### 9.1 Plan creates one immutable selection

`plan` is the only operation that may perform:

- representative memory/CPU probing;
- bounded throughput candidate benchmarking;
- automatic worker-count selection; and
- creation of the authoritative `RUNTIME_SELECTION.json`.

A work directory with an existing valid selection is not silently replanned. The
operator must use a new work directory for a new automatic decision. This preserves
failed evidence and avoids mixed probe attempts.

### 9.2 Run never reselects

`run` must not call automatic selection, launch the representative resource probe, or
execute the throughput grid.

It must:

1. load and verify the immutable selection;
2. verify workload, source, adapter, policy, and scientific fingerprints;
3. verify that the run identity fixes the same selected worker count and selection
   digest;
4. discover current affinity, cgroup quota, host/cgroup memory, and actual busy cores;
5. project the stored per-worker CPU and memory reservations for the selected count;
6. write an attempt-local revalidation record; and
7. either start with the exact planned worker count or fail closed.

Run may never silently change `112` to `80`, `20`, `1`, or any other value.

### 9.3 Revalidation arithmetic

```text
projected_total_cpu_cores =
    current_external_busy_cores
    + selected_workers * reserved_cpu_cores_per_worker

cpu_revalidation_ok =
    projected_total_cpu_cores <= current_cpu_capacity_ceiling_cores

memory_revalidation_ok =
    selected_workers * reserved_memory_per_worker
    <= current_usable_memory
```

Revalidation also requires unchanged affinity/resource binding where the adapter
marks it as identity-bearing. A quota reduction, changed workload fingerprint,
changed policy, insufficient memory, or unsafe CPU projection blocks launch with:

```text
RUNTIME_CAPACITY_CHANGED_REPLAN_REQUIRED
```

The original selection is never mutated to make the attempt pass.

### 9.4 Revalidation artifact

Each run attempt writes under an attempt-specific directory:

```text
_runtime_resource_attempts/<attempt_id>/RUNTIME_REVALIDATION.json
```

It records:

- selection digest and selected workers;
- current source and workload identity checks;
- current affinity and cgroup quota;
- current system busy and external busy cores;
- stored worker CPU/RAM reservation;
- projected CPU/RAM totals;
- allow/block decision and structured reason;
- timestamp and elapsed sampling interval.

This file is runtime provenance, not a second selection authority.

## 10. Selection identity, schema, and cache

### 10.1 Versioning

The implementation must introduce an explicit measured-CPU selector policy version.
Selections created by the raw-load-average policy are incompatible and must be cache
misses.

At minimum the selection records:

```text
document_schema_version
selector_policy_version
adapter_id and adapter implementation identity
source commit/worktree state
workload and scientific fingerprints
machine/resource binding
selected_workers
selection_digest
```

### 10.2 Stable identity versus dynamic evidence

Stable identity includes:

- selected worker count;
- workload and scientific fingerprints;
- adapter and policy versions;
- representative probe policy;
- CPU/RAM safety parameters;
- affinity or resource binding when declared by the adapter.

Dynamic evidence is recorded but excluded from the stable digest:

- load average diagnostics;
- current busy cores;
- current free memory;
- attempt timestamp;
- evidence path;
- cache or probe route.

Changing dynamic evidence triggers revalidation, not a new scientific identity.
Changing selected resources, workload/scientific fingerprints, adapter version, or
policy version requires a new selection in a new work directory.

### 10.3 No new generic cache engine

V2 does not add a separate cache service or cache abstraction. Existing adapter-owned
measurement reuse may continue only when workload, source, probe policy, selector
policy, and representative branch identity match exactly. The authoritative
selection file is not a mutable measurement cache.

## 11. Implementation surface and cost

### 11.1 Production files

The preferred implementation modifies existing responsibilities:

```text
src/drpo/runtime_resource_autotune.py
src/drpo/runtime_resource_adapters.py
src/drpo/e7_ppo_w0_runtime_autotune.py
relevant E7 auto runner plan/run entrypoints
```

Thin high-c, squared-EXP night, and KL Stage A adapters should inherit the shared
behavior. They should need only fingerprint/version plumbing, not copied arithmetic.
The fixed launchers and scientific runners remain unchanged.

The implementation also updates:

```text
docs/runtime_resource_autotuning_v1.md
or a versioned successor usage document
docs/scopes/GOV-RUNTIME-RESOURCE-AUTOTUNE-01.md
this evolution ledger
existing focused tests
```

It must not modify `docs/handoff.md`, `experiments/registry.yaml`, formal scientific
configuration, or the closed formal execution channel.

### 11.2 Estimated size

| Area | Estimate |
|---|---:|
| production code | 260–400 lines |
| deterministic tests | 350–550 lines |
| usage/scope updates | 80–160 lines |
| focused engineering implementation | 2–3.5 engineer-days |
| CI and real-server shadow | 0.5–1.5 engineer-days |

The estimate excludes the much larger project-neutral runtime-policy architecture in
Draft PR `#50`. This iteration deliberately reuses the existing DRPO core and does
not add a portable package, generic contract engine, scheduler backend, or Stage-2
formal integration.

## 12. Acceptance, risk, and rollback

### 12.1 Deterministic acceptance matrix

Required tests include:

1. affinity 384 and finite cgroup quota 376 produces effective capacity 376;
2. unlimited quota falls back to affinity without rounding up;
3. malformed or zero-period quota fails closed;
4. high load average with low measured busy cores does not collapse capacity;
5. low load average with high measured busy cores blocks unsafe concurrency;
6. `iowait` is not charged as CPU execution and guest time is not double-counted;
7. aligned worker/system windows subtract probe demand exactly once;
8. non-positive `/proc/stat` delta fails closed;
9. process-tree CPU measurement includes long-lived descendants;
10. CPU-bound, memory-bound, task-bound, configured-cap, and growth-bound cases;
11. throughput candidates never exceed the measured safe ceiling;
12. invalid or timed-out candidate cannot be selected;
13. plan writes one immutable selection and a stable digest;
14. run does not invoke memory probe, CPU probe, or throughput benchmark;
15. plan-selected worker count remains unchanged when load average rises afterward;
16. genuinely unsafe run revalidation blocks without silently downshifting;
17. run identity mismatch or changed workload/scientific fingerprint blocks;
18. raw-load-average policy selections are invalidated;
19. failed or blocked attempts preserve the original selection and write revalidation
    evidence;
20. all probe process groups are cleaned and no orphan remains;
21. high-c, squared-EXP night, PPO w(0), and Stage A wrappers use the shared policy
    rather than copied formulas;
22. fixed launchers and scientific branch matrices remain unchanged.

### 12.2 CI gates

Before server shadow:

- Python compilation;
- focused tests for runtime resource core, adapters, and affected runners;
- full pytest;
- Ruff;
- handoff authority no-op verification;
- formal execution-channel validation;
- governance inventory and stage-status validation;
- exact diff review confirming no scientific field change.

### 12.3 Real CPU shadow

The exact reviewed commit must be tested in a new work directory on the target E7
server.

The shadow must:

1. record affinity, cgroup quota, load averages, busy-core samples, worker CPU demand,
   RAM evidence, safe ceiling, every throughput candidate, and final selection;
2. actually exercise at least one candidate above one when measured capacity permits;
3. prove that the plan benchmark may raise load average without changing the frozen
   worker count;
4. invoke run in selection-consumption mode and prove that no second throughput grid
   starts;
5. verify lightweight CPU/RAM revalidation and unchanged selection digest;
6. verify no orphan process or stale process group;
7. avoid starting the full scientific sweep during the selection-only shadow;
8. then complete a separately approved small real-data liveness using the exact
   selected worker count before the 150-branch Stage A run is resumed.

Static tests, CI, a single-worker probe, or a selection-only report do not by
themselves establish full runtime readiness.

### 12.4 Risks and controls

| Risk | Control |
|---|---|
| overestimating worker capacity | global CPU fraction, per-worker reserve, RAM gate, bounded throughput validation |
| short-window occupancy noise | one-second aligned sample, safety factors, run-time revalidation |
| representative branch mismatch | adapter-owned hard fingerprint and representative identity |
| self-feedback from benchmark | plan-only selection and run prohibition on reselection |
| stale selection after environment change | lightweight measured revalidation and fail-closed replan requirement |
| legacy cache reuse | selector policy version and source/fingerprint checks |
| shared-core regression | focused wrapper coverage plus full repository tests |
| scientific drift | immutable scientific fingerprints and exact diff audit |

### 12.5 Rollback

1. Stop invoking the affected `*_auto.py` entrypoints.
2. Use the unchanged fixed launcher or the last separately verified fixed schedule.
3. Preserve the failed work directory, selection, revalidation records, benchmark
   summaries, and logs.
4. Revert the measured-CPU maintenance commit as one reviewed change.
5. Do not reinterpret a failed resource shadow as a scientific result.

## 13. Evolution ledger

### 13.1 `AUTOTUNE-2026-07-12-V1-MINIMAL`

**Status:** `active_opt_in` on the document base.  
**Claim:** `GOV-RUNTIME-RESOURCE-AUTOTUNE-01`.

**Design:**

- E7 representative process-tree RSS probe;
- CPU limit from logical CPU count, `cpu_fraction`, and one-minute load average;
- host/cgroup memory limit;
- task, configured, fallback, and bounded-growth limits;
- E8 visible/idle/free-VRAM filtering with one process per GPU;
- machine-readable `RUNTIME_SELECTION.json`;
- fixed launchers preserved.

**Known limitation:** load average was treated as occupied CPU capacity; E7 capacity
was not workload-CPU-demand aware. E8 did not measure training/evaluation VRAM peaks.

### 13.2 `AUTOTUNE-2026-07-13-E7-PROBE-HORIZON`

**Status:** `implemented`; real-server re-acceptance remained required by its scope
record.

**Previous behavior:** the representative E7 probe could request fewer steps than the
canonical trainer evaluation interval.

**Incident:** a 20,000-step probe finished before a 50,000-step evaluation interval,
leaving evaluation history empty and failing while printing terminal metrics.

**Replacement:** derive an effective probe horizon covering at least two frozen
evaluation intervals while retaining a bounded wall-clock sampler. Formal horizons
and evaluation rules remain unchanged.

**Lesson:** a resource probe must preserve the lifecycle needed for its worker to
remain valid; a short numeric step count is not automatically a valid envelope.

### 13.3 `AUTOTUNE-2026-07-13-GPU-LIVENESS-ONLY`

**Status:** `superseded`; development branch evidence only.

**Previous behavior:** same-GPU workers were accepted when they loaded the model and
remained alive through a bounded interval.

**H20 finding:** eight workers could remain alive without reaching a real optimizer
update or maximum-shape evaluation. The resulting `slots_per_gpu=8` did not establish
full-workload capacity.

**Replacement:** require explicit training and maximum-evaluation phase completion.

### 13.4 `AUTOTUNE-2026-07-14-GPU-PHASE-AWARE`

**Status:** `superseded` as a complete placement solution; its single-worker envelope
remains useful engineering evidence.

**Design:** require `model_loaded`, `training_peak_completed`,
`evaluation_peak_completed`, and `probe_complete`.

**Second H20 finding:**

- the single worker completed the training and pass@64 resource envelope;
- one-minute load average `387.5` on a 384-logical-CPU host was subtracted as worker
  capacity, reducing an otherwise idle eight-H20 pool to one selected GPU;
- completed phase markers were accepted while the worker return code was still null,
  after which controller cleanup terminated it.

**Replacement:** measured CPU occupancy and worker demand; phase-complete workers must
also exit zero and leave no process-group descendants.

### 13.5 `AUTOTUNE-2026-07-14-GPU-MEASURED-CPU-DRAFT`

**Status:** `ci_validated` on Draft PR `#53`; hardware shadow pending and not active on
`main` at this document base.

**Design:**

- `/proc/stat` execution occupancy;
- process-tree worker CPU seconds;
- external-occupancy estimate;
- phase-complete and clean-exit candidate contract;
- old GPU placement cache invalidation.

**Boundary:** this work is specific to the GPU placement branch. Its measured-CPU
ideas motivate the E7 shared-core design but do not make that CPU implementation
complete.

### 13.6 `AUTOTUNE-2026-07-14-E7-SELF-FEEDBACK-INCIDENT`

**Status:** confirmed engineering defect; repository deposit of the server report was
pending at this document base.

**Observed sequence:**

```text
plan benchmarks several worker counts
  -> plan selects 112 workers
  -> the benchmark itself raises one-minute load average above CPU count
  -> immediately following run calls automatic selection again
  -> cache validation rejects the 112-worker selection
  -> load-average arithmetic reduces the CPU limit to one
  -> run replans and starts one worker
```

**Root causes:**

1. load average was not a valid execution-core measurement;
2. plan and run both owned automatic selection;
3. the plan selection was not immutable runtime identity;
4. cache validation used dynamic load average as a hard acceptance field.

**Scientific status:** no scientific result was produced by this failure.

### 13.7 `AUTOTUNE-2026-07-14-E7-MEASURED-CPU-V2`

**Status:** `proposed`; this document is the design gate.

**Selected solution:** Sections 4–12.

**Direct benefits:**

- removes raw-load-average capacity arithmetic;
- accounts for affinity and cgroup CPU quota;
- adapts to measured worker CPU demand;
- retains existing memory and throughput evidence;
- prevents plan/run self-feedback;
- prevents silent worker-count downshift;
- gives every run a reviewable allow/block revalidation record;
- provides one shared implementation for current E7 CPU auto wrappers.

**Explicitly rejected additions:**

- a new scheduler or process service;
- a separate generic cache engine;
- a new project-neutral Python package;
- dynamic resizing, migration, NUMA tuning, affinity tuning, multi-node support, or
  batch/thread autotuning;
- Stage-2 formal execution integration;
- automatic GPU/CPU policy unification in this maintenance change.

## 14. Design review record

### Review 1 — responsibility cohesion

**Question:** does V2 remain a resource preflight rather than becoming a scheduler?

**Result:** pass. Selection arithmetic, bounded probes, immutable plan, and
revalidation remain in the resource layer. Existing runners continue to own process
execution, resume, heartbeat, packaging, and scientific outputs.

### Review 2 — necessity and anti-overengineering

**Question:** does every proposed component solve a demonstrated failure or a required
safety gap?

**Result:** pass after removing a proposed standalone CPU-capacity module, generic
cache engine, cross-project package, dynamic scaling, NUMA/affinity tuning, and
scheduler/provider abstractions. The remaining components are individually justified
in Section 4.3.

### Review 3 — measurement correctness

**Questions:**

- can quota exceed affinity?
- can I/O wait be mistaken for execution?
- can guest time be double-counted?
- can probe CPU be charged twice?
- can unmatched windows corrupt external occupancy?

**Result:** pass after locking the effective-capacity minimum, excluding `iowait` from
busy execution, avoiding guest double-counting, subtracting probe demand once, and
requiring aligned monotonic windows.

### Review 4 — lifecycle and self-feedback

**Question:** can the probe invalidate its own result or can run silently choose a new
worker count?

**Result:** pass after assigning all automatic selection to plan, prohibiting run from
probing or benchmarking, freezing the selection digest, and requiring fail-closed
replan rather than downshift.

### Review 5 — identity, cache, and provenance

**Question:** are stable resource identity and dynamic capacity evidence separated?

**Result:** pass. Selected resources, workload/scientific fingerprints, adapter, and
policy are stable identity. Busy cores, free memory, load diagnostics, timestamps,
and evidence routes remain dynamic and are checked through revalidation. Old policy
selections are invalid.

### Review 6 — failure and cleanup

**Question:** can failed probes, timed-out candidates, or blocked run attempts be
misreported as success or erase evidence?

**Result:** pass. Candidates require successful completion; process groups must be
cleaned; failed attempts preserve logs; run writes a structured block record; the
original selection is never rewritten.

### Review 7 — integration and scientific isolation

**Question:** can the maintenance change alter scientific execution or require a
closed governance-stage expansion?

**Result:** pass. The design changes only active subprocess count, uses existing
opt-in entrypoints, leaves fixed launchers available, excludes formal-channel
integration, and requires handoff/registry no-op verification.

## 15. Remaining uncertainties before implementation

The design is ready for implementation review, but the following remain empirical:

- the actual E7 representative worker CPU demand on the target server;
- the amount of short-window busy-core variance during the throughput grid;
- whether the proposed `1.25` per-worker CPU safety factor is sufficiently
  conservative on all current E7 workloads;
- the selected worker count under the measured model;
- real plan/run revalidation behavior after the benchmark raises load average;
- wrapper compatibility after the Stage A development branch is synchronized with
  the shared-core fix.

These uncertainties are resolved by deterministic tests and the exact-head CPU
shadow in Section 12. They are not reasons to add unproven scheduler or dynamic
scaling features.
