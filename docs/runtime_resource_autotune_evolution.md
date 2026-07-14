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
- dynamic online scaling or worker migration;
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
2. actual system and current-cgroup CPU execution occupancy;
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
| current-cgroup path and quota | avoid quota overestimation | affinity may exceed the executable quota |
| `/proc/stat` busy-core sample | measure host/affinity execution | load average includes history and uninterruptible waits |
| cgroup CPU-usage sample | measure quota already consumed by other same-cgroup work | host load alone cannot protect a tight shared quota |
| process-tree CPU demand | distinguish light and heavy workers | idle cores alone do not imply a worker count |
| CPU safety reserve | cover sampling and workload variance | short probes are estimates |
| existing RAM/cgroup memory gate | prevent memory oversubscription | CPU capacity alone is insufficient |
| workload-specific throughput grid | avoid a safe but inefficient ceiling | already valuable for PPO-family E7 paths |
| candidate resource validation | reject a fast candidate that violates CPU/RAM bounds | throughput alone is not safety |
| immutable plan selection | remove probe self-feedback and identity drift | run must not silently choose another schedule |
| lightweight run revalidation | block genuinely unsafe later starts | immutable does not mean blindly reusable |
| schema/policy versioning | reject old load-average selections | old evidence has incompatible semantics |
| attempt-local revalidation record | preserve allow/block evidence without mutating selection | the plan artifact must remain immutable |

No additional production module is required. The implementation extends the existing
runtime resource core and adapters.

## 5. Machine CPU capacity contract

### 5.1 Affinity capacity

The process-visible CPU set is obtained from `os.sched_getaffinity(0)` on Linux. Its
cardinality is:

```text
affinity_capacity_cores
```

The exact sorted CPU IDs form an affinity binding fingerprint and select per-CPU rows
from `/proc/stat`. A changed affinity set requires replan even when its cardinality is
unchanged, because V2 does not model NUMA/topology equivalence.

When sched-affinity is unavailable, `os.cpu_count()` is allowed only as an explicitly
recorded compatibility fallback. Production Linux shadows must exercise affinity.

### 5.2 Current-cgroup path, quota, and usage

The implementation resolves the current process cgroup rather than assuming the
configured mount root is itself the active cgroup:

- cgroup v2: resolve `0::<relative_path>` from `/proc/self/cgroup`, then read
  `cpu.max` and `cpu.stat` under the configured mount root;
- cgroup v1: use the `cpu` or `cpu,cpuacct` controller path and directly visible
  `cpu.cfs_quota_us`, `cpu.cfs_period_us`, and `cpuacct.usage` files;
- reject paths that escape the configured mount root;
- support a cgroup namespace that exposes the current cgroup at the mount root.

A finite quota is converted to equivalent CPU cores:

```text
quota_capacity_cores = quota_microseconds / period_microseconds
```

Aligned current-cgroup CPU usage is derived from:

```text
v2: delta(cpu.stat usage_usec) / elapsed_microseconds
v1: delta(cpuacct.usage nanoseconds) / elapsed_nanoseconds
```

`max`, a negative v1 quota, or absent quota files in an environment with no detected
CPU controller mean unlimited quota. Malformed values, zero periods, missing usage
accounting under a detected finite quota, contradictory controller evidence, or an
unresolvable active path make auto selection unavailable rather than silently
ignoring the constraint.

Fractional quota capacity is preserved and never rounded up. V2 `nr_throttled` and
`throttled_usec`, when present, are recorded as diagnostics but are not a standalone
hard gate in the first implementation; candidate throughput and measured budgets
remain authoritative.

### 5.3 Reported effective capacity

For provenance:

```text
effective_cpu_capacity_cores =
    min(affinity_capacity_cores, finite_quota_capacity_cores)
```

When quota is unlimited, effective capacity equals affinity capacity. This minimum is
not by itself the worker budget: host/affinity occupancy and same-cgroup quota usage
are independent constraints defined in Section 8.

The existing default `cpu_fraction=0.85` remains global headroom unless an approved
adapter policy explicitly supplies another value.

## 6. CPU occupancy contract

### 6.1 Why load average is diagnostic only

Linux load average is an exponentially smoothed count of runnable and
uninterruptible tasks. It is not current CPU execution. It can remain high after a
benchmark exits and include I/O waits.

V2 records one-, five-, and fifteen-minute load average only as diagnostic provenance.
No load-average field participates in capacity arithmetic or cache acceptance.

### 6.2 `/proc/stat` system sampling

System execution occupancy is measured over an aligned monotonic interval using the
per-CPU rows for the current affinity set.

Busy time includes:

```text
user + nice + system + irq + softirq + steal
```

Idle compute capacity includes:

```text
idle + iowait
```

Only standard fields through `steal` participate in the total; guest and guest-nice
are not added because guest execution is already included in user/nice.

```text
system_busy_cores =
    affinity_capacity_cores * busy_tick_delta / total_tick_delta
```

Missing rows, non-positive deltas, or a changing affinity set makes auto measurement
unavailable.

### 6.3 Current-cgroup sampling

The same monotonic window measures current-cgroup usage:

```text
cgroup_busy_cores = cgroup_cpu_time_delta / elapsed_time
```

This is required when quota is finite because other processes in the same cgroup
consume the same quota even when the host has many idle CPUs. When quota is unlimited,
cgroup usage remains useful provenance but only the affinity constraint is binding.

A finite quota without reliable aligned cgroup usage fails closed. The implementation
must not assume that all external processes belong to another cgroup.

### 6.4 Sampling windows

Plan and candidate measurements use their aligned worker interval. Pre-launch run
revalidation uses three consecutive one-second samples and conservatively takes the
maximum observed system busy cores and maximum observed cgroup busy cores. The sample
count and duration are test-configurable runtime policy, not scientific parameters.

Every evidence record includes monotonic start/end, elapsed seconds, affinity set,
`/proc/stat` deltas, cgroup usage deltas, and process-tree CPU deltas.

## 7. Representative worker CPU demand

### 7.1 Measurement

The existing representative resource probe records cumulative user and system CPU
seconds for its process tree:

```text
measured_cpu_cores_per_worker =
    process_tree_cpu_seconds / aligned_elapsed_seconds
```

Process-level CPU accounting includes threads. Long-lived descendants are included
through process-tree sampling. V2 does not add an E7 phase framework or exact recovery
of descendants that start and exit entirely between polls; global reserve and
concurrent candidate validation cover that residual limitation.

The probe must:

- use the existing representative branch and thread environment;
- remain in the non-scientific probe seed namespace;
- produce positive RSS and CPU-demand evidence;
- follow the bounded probe termination contract;
- leave no process-group descendants;
- preserve logs/small evidence and remove model payload.

### 7.2 Reservation

Proposed defaults:

```text
per_worker_cpu_safety_factor = 1.25
minimum_cpu_cores_per_worker = 1.0
```

```text
reserved_cpu_cores_per_worker =
    max(minimum_cpu_cores_per_worker,
        measured_cpu_cores_per_worker * per_worker_cpu_safety_factor)
```

These runtime safety fields are included in the resource fingerprint and do not alter
scientific execution.

### 7.3 External occupancy during plan

The aligned system and cgroup samples include the probe itself:

```text
external_system_busy_cores =
    max(0, system_busy_cores - measured_probe_cpu_cores)

external_cgroup_busy_cores =
    max(0, cgroup_busy_cores - measured_probe_cpu_cores)
```

Probe demand is subtracted exactly once from each accounting domain. Neither estimate
may create capacity beyond its independent budget.

## 8. Capacity and throughput selection

### 8.1 Two independent CPU constraints

External host load consumes affinity capacity; same-cgroup external load consumes
quota capacity. V2 computes:

```text
affinity_budget_cores = affinity_capacity_cores * cpu_fraction

affinity_worker_budget_cores =
    max(0, affinity_budget_cores - external_system_busy_cores)

quota_worker_budget_cores =
    infinity when quota is unlimited
    otherwise max(
        0,
        quota_capacity_cores * cpu_fraction
        - external_cgroup_busy_cores
    )

worker_cpu_budget_cores =
    min(affinity_worker_budget_cores, quota_worker_budget_cores)

cpu_worker_limit =
    floor(worker_cpu_budget_cores / reserved_cpu_cores_per_worker)
```

This handles:

- tight quota on a mostly idle large host;
- other work sharing that tight quota;
- generous quota on a saturated affinity set; and
- unrelated host load outside the current cgroup.

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

CPU and memory evidence must refer to the same representative workload identity.

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

The growth limit remains a blast-radius cap, not throughput evidence.

### 8.4 Throughput candidates and resource validity

Only workloads already using the bounded PPO-family throughput search retain it. V2
does not create a generic throughput engine.

Candidate rules:

- benchmark the verified fallback and points near 50%, 75%, and 100% of the safe
  ceiling;
- never launch above the safe ceiling;
- require every worker to exit successfully without timeout;
- measure aggregate completed optimizer updates per second;
- measure aggregate worker CPU, system CPU, current-cgroup CPU, and host RSS over the
  aligned candidate window;
- select the smallest resource-valid candidate reaching the configured fraction of
  peak aggregate throughput.

Resource validity requires:

```text
candidate_external_system_busy_cores
+ candidate_worker_cpu_cores
    <= affinity_budget_cores

when quota is finite:
candidate_external_cgroup_busy_cores
+ candidate_worker_cpu_cores
    <= quota_capacity_cores * cpu_fraction
```

and observed/projected RSS must remain within usable memory.

The current throughput-retention default remains `0.97` for adapters already using
it. Adapters without a grid may select the safe ceiling and record that limitation.

## 9. Plan, run, resume, and revalidation

### 9.1 Plan creates one immutable selection

`plan` alone may perform representative probing, bounded throughput benchmarking,
automatic selection, and creation of authoritative `RUNTIME_SELECTION.json`.

An existing valid selection is not silently replanned. A new automatic decision uses
a new work directory, preserving failed evidence and avoiding mixed attempts.

### 9.2 Run never reselects

`run` must not call automatic selection, resource probing, or the throughput grid.
It must:

1. load and verify the immutable selection;
2. verify source, workload, scientific, adapter, and policy fingerprints;
3. verify run identity fixes the same worker count and selection digest;
4. reject any prior probe/scientific process still alive for the work directory;
5. re-read affinity binding, active cgroup, quota, cgroup usage, memory, and system
   busy cores;
6. project stored worker CPU/RAM reservations;
7. write attempt-local revalidation evidence; and
8. start with the exact planned count or fail closed.

Run may never silently change `112` to `80`, `20`, `1`, or another value.

### 9.3 Revalidation arithmetic

No workload worker from the work directory may be active before launch. Therefore the
three-sample maxima are external occupancy and are not reduced by stored demand.

```text
current_affinity_budget_cores =
    current_affinity_capacity_cores * cpu_fraction

current_quota_budget_cores =
    infinity when quota is unlimited
    otherwise current_quota_capacity_cores * cpu_fraction

selected_worker_cpu_cores =
    selected_workers * reserved_cpu_cores_per_worker

cpu_revalidation_ok =
    current_system_busy_cores_max
    + selected_worker_cpu_cores
        <= current_affinity_budget_cores
    and
    (
      quota is unlimited
      or
      current_cgroup_busy_cores_max
      + selected_worker_cpu_cores
          <= current_quota_budget_cores
    )

memory_revalidation_ok =
    selected_workers * reserved_memory_per_worker
        <= current_usable_memory
```

A changed affinity binding, cgroup path, quota reduction, fingerprint mismatch,
insufficient memory, unsafe CPU projection, or live-process conflict blocks with:

```text
RUNTIME_CAPACITY_CHANGED_REPLAN_REQUIRED
```

The original selection is never mutated or downshifted.

### 9.4 Revalidation artifact

Each run attempt writes:

```text
_runtime_resource_attempts/<attempt_id>/RUNTIME_REVALIDATION.json
```

It records:

- selection digest and workers;
- identity checks;
- affinity binding and current cgroup path;
- quota and three system/cgroup occupancy samples;
- stored worker CPU/RAM reservation;
- projected totals and both CPU constraints;
- live-process conflict audit;
- allow/block decision and reason;
- timestamp and sampling intervals.

It is provenance, not a second selection authority.

## 10. Selection identity, schema, and cache

### 10.1 Versioning

V2 introduces an explicit measured-CPU selector policy version. Raw-load-average
selections are incompatible cache misses.

At minimum selection records:

```text
document_schema_version
selector_policy_version
adapter id and implementation identity
source commit/worktree state
workload and scientific fingerprints
affinity binding and active-cgroup binding
selected_workers
selection_digest
```

### 10.2 Stable identity versus dynamic evidence

Stable identity includes selected workers, workload/scientific fingerprints, adapter
and policy versions, representative probe policy, CPU/RAM safety parameters,
affinity binding, and active-cgroup binding.

Dynamic evidence is excluded from the stable digest: load diagnostics, busy cores,
free memory, timestamps, throttling diagnostics, evidence paths, and probe/cache
route. Dynamic changes trigger revalidation. Stable changes require a new work
directory and selection.

### 10.3 No generic cache engine

V2 adds no cache service or abstraction. Existing adapter-owned measurement reuse may
continue only under exact workload, source, probe policy, selector policy, and
representative identity. `RUNTIME_SELECTION.json` is immutable authority, not a
mutable cache.

## 11. Implementation surface and cost

### 11.1 Production files

Preferred production changes remain within existing responsibilities:

```text
src/drpo/runtime_resource_autotune.py
src/drpo/runtime_resource_adapters.py
src/drpo/e7_ppo_w0_runtime_autotune.py
relevant E7 auto runner plan/run entrypoints
```

Thin high-c, squared-EXP night, and KL Stage A adapters inherit shared behavior and
need fingerprint/version plumbing only. Fixed launchers and scientific runners remain
unchanged.

Documentation/tests update:

```text
docs/runtime_resource_autotuning_v1.md or a versioned successor
docs/scopes/GOV-RUNTIME-RESOURCE-AUTOTUNE-01.md
docs/runtime_resource_autotune_evolution.md
existing runtime-resource and E7 wrapper tests
```

No handoff, registry, scientific config, or formal-channel modification is allowed.

### 11.2 Revised estimate after three correctness reviews

| Area | Estimate |
|---|---:|
| production code | 380–560 lines |
| deterministic tests | 550–850 lines |
| usage/scope updates | 80–160 lines |
| focused implementation | 4–5.5 engineer-days |
| CI and real-server shadow | 0.5–1.5 engineer-days |

The estimate increased because correct general behavior requires three details omitted
from the first estimate: current-cgroup path resolution, independent affinity/quota
arithmetic, and aligned same-cgroup occupancy measurement. These are not optional
architecture; omitting them creates concrete quota bugs.

This remains smaller than Draft PR `#50`'s project-neutral architecture. No portable
package, generic contract engine, scheduler backend, or Stage-2 integration is added.

## 12. Acceptance, risk, and rollback

### 12.1 Deterministic acceptance matrix

Required tests include:

1. current-cgroup path resolution for v2 and direct v1 compatibility;
2. root-escape, malformed-controller, and missing finite-quota usage rejection;
3. affinity 384 plus quota 376 records effective capacity 376;
4. fractional and unlimited quota behavior;
5. tight idle-host quota permits work up to its budget;
6. same-cgroup external work reduces tight-quota capacity;
7. generous quota plus saturated affinity is host constrained;
8. external host work outside the cgroup reduces affinity but not quota budget;
9. high load average with low measured execution does not collapse capacity;
10. low load average with high measured execution blocks unsafe concurrency;
11. `iowait` and guest accounting rules;
12. aligned system/cgroup/worker windows subtract probe demand once per domain;
13. non-positive deltas fail closed;
14. process-tree CPU includes threads and long-lived descendants;
15. CPU-, memory-, task-, configured-cap-, and growth-bound cases;
16. candidates never exceed safe ceiling;
17. fast but CPU/RAM-invalid candidate is rejected;
18. timeout/nonzero candidate is rejected;
19. plan writes immutable selection and digest;
20. run invokes no probe or throughput grid;
21. rising post-plan load average cannot change selected workers;
22. unsafe run revalidation blocks without downshift;
23. identity/fingerprint/policy mismatch blocks;
24. raw-load-average selection is invalidated;
25. blocked attempts preserve selection and write evidence;
26. live process conflict blocks;
27. all process groups are cleaned;
28. wrappers share the policy rather than copied formulas;
29. fixed launchers/scientific matrices remain unchanged;
30. three-sample revalidation uses the conservative maxima.

### 12.2 CI gates

Before shadow: Python compile, focused runtime/E7 tests, full pytest, Ruff, handoff
authority no-op, formal-channel validation, governance inventory/stage validation,
and exact scientific-field diff review.

### 12.3 Real CPU shadow

The exact reviewed commit is tested in a new work directory. The shadow must:

1. record affinity, active cgroup, quota, cgroup usage, throttling diagnostics, load
   averages, system busy, worker demand, RAM, safe ceiling, candidates, and selection;
2. exercise at least one candidate above one when capacity permits;
3. report candidate CPU/RAM validity separately from throughput validity;
4. show that benchmark-raised load average cannot alter frozen selection;
5. prove run consumes selection without a second grid;
6. verify three-sample CPU/RAM revalidation and unchanged digest;
7. verify no stale process or orphan;
8. suppress the full scientific sweep during selection-only shadow; and
9. complete separately approved small real-data liveness at the selected count before
   resuming the 150-branch Stage A run.

Tests, CI, one worker, or selection-only output do not independently establish full
runtime readiness.

### 12.4 Risks and controls

| Risk | Control |
|---|---|
| capacity overestimate | dual accounting domains, CPU fraction, worker reserve, RAM gate, candidate validation |
| tight-quota error | same-cgroup usage measured separately from host occupancy |
| short-window noise | aligned candidate windows and three-sample run maxima |
| representative mismatch | adapter hard fingerprint and representative identity |
| benchmark self-feedback | plan-only selection; run cannot reselect |
| stale environment | measured revalidation and fail-closed replan |
| legacy cache reuse | policy/source/fingerprint version checks |
| shared-core regression | wrapper coverage plus full tests |
| scientific drift | immutable scientific fingerprints and diff audit |

### 12.5 Rollback

1. Stop affected `*_auto.py` entrypoints.
2. Use unchanged fixed launchers or a separately verified fixed schedule.
3. Preserve failed work directories, selections, revalidations, summaries, and logs.
4. Revert the measured-CPU maintenance commit as one reviewed change.
5. Never reinterpret a failed resource shadow as scientific evidence.

## 13. Evolution ledger

### 13.1 `AUTOTUNE-2026-07-12-V1-MINIMAL`

**Status:** `active_opt_in` on the document base.  
**Claim:** `GOV-RUNTIME-RESOURCE-AUTOTUNE-01`.

Representative E7 RSS probe; load-average CPU limit; host/cgroup memory, task,
configured, fallback, and growth limits; E8 idle/free-VRAM device selection; one
process per GPU; machine-readable selection; fixed launchers preserved.

Known limitations: load average was treated as occupied CPU; E7 was not worker-demand
aware; E8 did not measure full lifecycle peaks.

### 13.2 `AUTOTUNE-2026-07-13-E7-PROBE-HORIZON`

**Status:** `implemented`; real-server re-acceptance required by scope.

A 20,000-step probe could finish before a 50,000-step evaluation interval. Replacement:
derive a probe horizon covering at least two frozen evaluation intervals while the
wall-clock sampler remains bounded. Formal execution is unchanged.

### 13.3 `AUTOTUNE-2026-07-13-GPU-LIVENESS-ONLY`

**Status:** `superseded`; development evidence only.

Eight H20 workers remained alive through model load without a real optimizer update
or maximum-shape evaluation. `slots_per_gpu=8` was not capacity evidence. Replacement:
explicit training and maximum-evaluation phases.

### 13.4 `AUTOTUNE-2026-07-14-GPU-PHASE-AWARE`

**Status:** `superseded` as a complete placement solution; its single-worker envelope
remains engineering evidence.

The phase contract completed, but load average `387.5` on 384 CPUs reduced an idle
eight-H20 pool to one GPU. Worker return code was null before controller cleanup.
Replacement: measured CPU plus clean zero exit/process-group disappearance.

### 13.5 `AUTOTUNE-2026-07-14-GPU-MEASURED-CPU-DRAFT`

**Status:** `ci_validated` on Draft PR `#53`; hardware shadow pending and not active on
`main`.

The draft uses `/proc/stat`, process-tree CPU demand, external occupancy, phase/exit
contracts, and cache invalidation. It motivates but does not implement E7 V2.

### 13.6 `AUTOTUNE-2026-07-14-E7-SELF-FEEDBACK-INCIDENT`

**Status:** confirmed engineering defect; server-report repository deposit pending at
this base.

```text
plan benchmarks counts -> selects 112 -> benchmark raises load average
-> run reselects -> cache rejects 112 -> load arithmetic yields one
-> run starts one worker
```

Root causes: invalid load semantics; duplicate selection ownership; mutable runtime
identity; dynamic load in cache acceptance. No scientific result was produced.

### 13.7 `AUTOTUNE-2026-07-14-E7-MEASURED-CPU-V2`

**Status:** `proposed`; Sections 4–12 are the implementation gate.

Benefits: measured capacity, affinity/quota awareness, worker demand, retained RAM and
throughput evidence, immutable plan, no silent downshift, revalidation evidence, and
one shared E7 implementation.

Rejected: scheduler service, cache engine, portable package, dynamic resizing,
migration, NUMA/affinity tuning, multi-node, batch/thread tuning, Stage-2 integration,
and automatic GPU/CPU policy unification.

### 13.8 `AUTOTUNE-2026-07-14-QUOTA-ARITHMETIC-REVIEW`

**Status:** design correction completed before implementation.

The first draft subtracted all host load from `min(affinity, quota)`. Review found this
incorrect for a tight quota on a large idle host. Replacement: independent affinity
and quota constraints. Current-cgroup path resolution was also made explicit.

### 13.9 `AUTOTUNE-2026-07-14-CGROUP-OCCUPANCY-REVIEW`

**Status:** design correction completed before implementation.

The second draft capped candidate worker demand by quota but did not subtract other
same-cgroup CPU usage. That could overcommit a tight shared quota even while host
capacity appeared ample. Replacement: aligned `cpu.stat`/`cpuacct.usage` accounting,
separate external system and external cgroup occupancy, and both constraints in plan,
candidate validation, and run revalidation. Pre-launch revalidation was strengthened
from one sample to conservative maxima over three one-second samples.

## 14. Design review record

### Review 1 — responsibility cohesion

Pass. The design remains resource preflight; runners retain execution, resume,
heartbeat, packaging, and scientific outputs.

### Review 2 — necessity and anti-overengineering

Pass after excluding standalone module, cache engine, cross-project package, dynamic
scaling, NUMA/affinity tuning, and scheduler/provider abstractions. Every remaining
component maps to a demonstrated defect or safety requirement.

### Review 3 — machine and cgroup correctness

Two real design flaws were found and corrected: host load must not be subtracted from
a tight quota, and same-cgroup external work must consume quota budget. Final design
uses current-cgroup resolution plus independent aligned system/cgroup accounting.
Pass after correction.

### Review 4 — CPU-time accounting

Pass after locking `iowait` as non-execution, avoiding guest double-counting,
including `steal`, aligning all windows, and subtracting probe demand exactly once per
domain.

### Review 5 — candidate validity

Pass after requiring throughput candidates to satisfy measured system CPU, cgroup
CPU, and RAM bounds. Throughput cannot override safety.

### Review 6 — lifecycle and self-feedback

Pass. Plan alone selects; run cannot probe/benchmark; digest is frozen; unsafe
capacity requires replan, not downshift.

### Review 7 — identity, cache, and provenance

Pass. Stable resource binding/fingerprints are separated from dynamic evidence. Old
policy selections are invalid.

### Review 8 — failure and cleanup

Pass. Failed/timed-out candidates are invalid; live conflicts block; process groups
are audited; selection is preserved.

### Review 9 — revalidation noise

Pass after replacing one pre-launch sample with conservative maxima over three
one-second samples. This remains a bounded preflight, not online monitoring.

### Review 10 — integration and scientific isolation

Pass. Only active subprocess count changes. Fixed launchers remain rollback. No
handoff, registry, scientific config, or formal-channel change is required.

## 15. Remaining uncertainties before implementation

The design is ready for implementation review, but these remain empirical:

- actual representative worker CPU demand;
- system/cgroup occupancy variance during candidate grids;
- whether `1.25` worker CPU reserve is sufficient across E7 workloads;
- selected worker count under V2;
- plan/run behavior after load average rises;
- Stage A wrapper compatibility after shared-core synchronization.

They are resolved by deterministic tests and exact-head CPU shadow, not by adding
unproven scheduler or dynamic-scaling features.
