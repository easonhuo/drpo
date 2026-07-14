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
| current-cgroup quota resolution | avoid quota overestimation | affinity may exceed the container's executable quota |
| `/proc/stat` busy-core sample | measure current CPU execution | load average includes history and uninterruptible waits |
| process-tree CPU demand | distinguish light and heavy workers | idle cores alone do not imply a worker count |
| CPU safety reserve | cover sampling and workload variance | short probes are estimates, not exact future demand |
| existing RAM/cgroup memory gate | prevent memory oversubscription | CPU capacity alone is insufficient |
| workload-specific throughput grid | avoid a safe but inefficient ceiling | already valuable for PPO-family E7 paths |
| candidate resource validation | reject a fast candidate that violates CPU/RAM bounds | throughput alone is not a safety gate |
| immutable plan selection | remove probe self-feedback and identity drift | run must not silently choose a different schedule |
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

The exact CPU IDs are also used to select per-CPU rows from `/proc/stat`. When
sched-affinity is unavailable, `os.cpu_count()` is allowed only as an explicitly
recorded compatibility fallback. Production Linux shadows must exercise the affinity
path.

### 5.2 Current-cgroup path and quota

The implementation resolves the current process cgroup rather than assuming that the
configured mount root is itself the active cgroup:

- cgroup v2: resolve `0::<relative_path>` from `/proc/self/cgroup`, then read
  `cpu.max` under the configured cgroup mount root;
- cgroup v1: use the `cpu` or `cpu,cpuacct` controller path and directly visible
  `cpu.cfs_quota_us` / `cpu.cfs_period_us` files;
- reject paths that escape the configured mount root;
- a namespaced environment that exposes the current cgroup at the mount root remains
  supported.

A finite cgroup quota is converted to equivalent CPU cores:

```text
quota_capacity_cores = quota_microseconds / period_microseconds
```

`max`, a negative v1 quota, or absent quota files in an environment with no detected
CPU controller mean unlimited quota. Malformed values, zero periods, contradictory
controller evidence, or a detected CPU cgroup whose active path cannot be resolved
make automatic selection unavailable rather than silently ignoring the quota.

Fractional quota capacity is preserved as a float and is never rounded up.

### 5.3 Reported effective capacity

For provenance:

```text
effective_cpu_capacity_cores =
    min(affinity_capacity_cores, finite_quota_capacity_cores)
```

When no finite quota exists, effective capacity equals affinity capacity.

This reported minimum is not by itself the worker budget because external host load
must be applied to the affinity constraint, not subtracted directly from a tight
cgroup quota. Section 8 defines the two independent constraints.

The existing default `cpu_fraction=0.85` remains the global execution headroom unless
an approved adapter policy explicitly supplies another value.

## 6. Actual CPU occupancy contract

### 6.1 Why load average is diagnostic only

Linux load average is an exponentially smoothed count of runnable and
uninterruptible tasks. It is not a count of CPU cores currently executing work. It
can remain high after a benchmark exits and can include I/O waits that do not consume
CPU execution capacity.

V2 records one-, five-, and fifteen-minute load average only as diagnostic provenance.
No load-average field participates in worker-capacity arithmetic or cache acceptance.

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

Only the standard fields through `steal` participate in the total; guest and
guest-nice are not added because Linux already includes guest execution in user/nice.

For a valid interval:

```text
system_busy_cores =
    affinity_capacity_cores * busy_tick_delta / total_tick_delta
```

The default sampling interval is one second. It is configurable for tests and
hardware shadows but is not a scientific parameter.

Missing per-CPU rows, non-positive total deltas, or a changing affinity set makes the
measured auto decision unavailable. The auto path fails closed; the unchanged fixed
launcher remains the rollback path.

### 6.3 Aligned measurement windows

Worker CPU demand and system busy cores must be measured over aligned monotonic
windows. A system sample from before or after a worker interval must not be mixed with
worker CPU demand from another interval.

The selection records interval start/end timestamps, elapsed seconds, CPU set, system
tick deltas, and process-tree CPU deltas.

## 7. Representative worker CPU demand

### 7.1 Measurement

The existing representative resource probe is extended to record cumulative user and
system CPU seconds for the probe process tree while it is alive:

```text
measured_cpu_cores_per_worker =
    process_tree_cpu_seconds / aligned_elapsed_seconds
```

Process-level CPU accounting includes its threads. Long-lived descendants are added
through process-tree sampling. The first V2 implementation does not add an E7 phase
framework or attempt exact recovery of descendants that start and exit entirely
between polls; the global reserve and concurrent candidate validation cover that
residual limitation.

The probe must:

- use the workload's existing representative branch and thread environment;
- remain in the dedicated non-scientific seed namespace;
- produce positive RSS and CPU-demand measurements;
- follow the existing bounded probe termination contract;
- leave no process-group descendants;
- preserve logs and small resource evidence while removing model payload.

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

These are runtime safety policy fields and are recorded in the resource fingerprint.
They do not alter scientific execution.

### 7.3 External occupancy during plan

The aligned system sample includes the probe itself, so plan estimates:

```text
external_busy_cores =
    max(0, system_busy_cores - measured_probe_cpu_cores)
```

Probe CPU is subtracted exactly once. The estimate cannot create capacity beyond the
affinity or quota budgets defined below.

## 8. Capacity and throughput selection

### 8.1 Two independent CPU constraints

External host load consumes affinity capacity, while cgroup quota independently caps
the workload. Subtracting all external host load directly from a small quota is
incorrect. V2 therefore computes:

```text
affinity_budget_cores = affinity_capacity_cores * cpu_fraction

quota_budget_cores =
    finite_quota_capacity_cores * cpu_fraction
    or infinity when quota is unlimited

affinity_worker_budget_cores =
    max(0, affinity_budget_cores - external_busy_cores)

worker_cpu_budget_cores =
    min(affinity_worker_budget_cores, quota_budget_cores)

cpu_worker_limit =
    floor(worker_cpu_budget_cores / reserved_cpu_cores_per_worker)
```

This correctly handles both important cases:

- a tight quota on a mostly idle large host still permits work up to the quota; and
- a generous quota on a saturated affinity set is constrained by actual host load.

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

Memory and CPU evidence must refer to the same representative workload identity.

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

The existing bounded growth limit remains a policy-level blast-radius cap. It is not
throughput evidence.

### 8.4 Throughput candidate search and resource validation

Only workloads that already use the bounded PPO-family throughput search retain it.
V2 does not create a generic throughput engine.

The candidate principle remains:

- benchmark a small set containing the verified fallback and points near 50%, 75%,
  and 100% of the safe ceiling;
- never launch a candidate above the safe ceiling;
- require every worker to exit successfully without timeout;
- measure aggregate completed optimizer updates per second;
- select the smallest resource-valid candidate reaching the configured fraction of
  peak aggregate throughput.

A candidate is resource-valid only when its aligned measurement satisfies both:

```text
candidate_worker_cpu_cores <= quota_budget_cores

candidate_external_busy_cores + candidate_worker_cpu_cores
    <= affinity_budget_cores
```

and its projected/observed host RSS remains within the usable memory budget.

The current throughput-retention default remains `0.97` for adapters that already use
it. Adapters without a throughput grid may select the safe ceiling directly and must
record that limitation.

## 9. Plan, run, resume, and revalidation

### 9.1 Plan creates one immutable selection

`plan` is the only operation that may perform:

- representative memory/CPU probing;
- bounded throughput candidate benchmarking;
- automatic worker-count selection; and
- creation of the authoritative `RUNTIME_SELECTION.json`.

An existing valid selection is not silently replanned. A new automatic decision uses
a new work directory. This preserves failed evidence and avoids mixed attempts.

### 9.2 Run never reselects

`run` must not call automatic selection, launch the representative resource probe, or
execute the throughput grid.

It must:

1. load and verify the immutable selection;
2. verify workload, source, adapter, policy, and scientific fingerprints;
3. verify that run identity fixes the same selected worker count and selection
   digest;
4. confirm that no prior probe worker or scientific worker from the same work
   directory is still alive;
5. discover current affinity, cgroup quota, host/cgroup memory, and actual busy cores;
6. project stored per-worker CPU and memory reservations;
7. write an attempt-local revalidation record; and
8. either start with the exact planned worker count or fail closed.

Run may never silently change `112` to `80`, `20`, `1`, or another value.

### 9.3 Revalidation arithmetic

Before launch, no workload worker from the work directory may be active. Therefore
the short pre-launch busy sample is external occupancy and is not reduced by the
stored worker demand.

```text
current_affinity_budget_cores =
    current_affinity_capacity_cores * cpu_fraction

current_quota_budget_cores =
    current_finite_quota_capacity_cores * cpu_fraction
    or infinity

selected_worker_cpu_cores =
    selected_workers * reserved_cpu_cores_per_worker

cpu_revalidation_ok =
    selected_worker_cpu_cores <= current_quota_budget_cores
    and
    current_system_busy_cores + selected_worker_cpu_cores
        <= current_affinity_budget_cores

memory_revalidation_ok =
    selected_workers * reserved_memory_per_worker
        <= current_usable_memory
```

A quota reduction, changed workload fingerprint, changed policy, insufficient memory,
unsafe CPU projection, or conflicting live process blocks launch with:

```text
RUNTIME_CAPACITY_CHANGED_REPLAN_REQUIRED
```

The original selection is never mutated or downshifted to make the attempt pass.

### 9.4 Revalidation artifact

Each run attempt writes:

```text
_runtime_resource_attempts/<attempt_id>/RUNTIME_REVALIDATION.json
```

It records:

- selection digest and selected workers;
- source/workload/scientific identity checks;
- current affinity and cgroup quota;
- current system busy cores;
- stored worker CPU/RAM reservation;
- projected CPU/RAM totals and both CPU constraints;
- live-process conflict audit;
- allow/block decision and structured reason;
- timestamp and sampling interval.

It is runtime provenance, not a second selection authority.

## 10. Selection identity, schema, and cache

### 10.1 Versioning

The implementation introduces an explicit measured-CPU selector policy version.
Selections created by the raw-load-average policy are incompatible cache misses.

At minimum the selection records:

```text
document_schema_version
selector_policy_version
adapter id and implementation identity
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
- declared resource binding.

Dynamic evidence is recorded but excluded from the stable digest:

- load-average diagnostics;
- busy cores and free memory;
- timestamps and evidence paths;
- cache or probe route.

Dynamic changes trigger revalidation. Selected-resource, workload/scientific,
adapter, or policy changes require a new selection in a new work directory.

### 10.3 No new generic cache engine

V2 adds no cache service or cache abstraction. Existing adapter-owned measurement
reuse may continue only under exact workload, source, probe policy, selector policy,
and representative identity. `RUNTIME_SELECTION.json` is immutable authority, not a
mutable measurement cache.

## 11. Implementation surface and cost

### 11.1 Production files

The preferred implementation modifies existing responsibilities:

```text
src/drpo/runtime_resource_autotune.py
src/drpo/runtime_resource_adapters.py
src/drpo/e7_ppo_w0_runtime_autotune.py
relevant E7 auto runner plan/run entrypoints
```

Thin high-c, squared-EXP night, and KL Stage A adapters inherit the shared behavior.
They need fingerprint/version plumbing, not copied arithmetic. Fixed launchers and
scientific runners remain unchanged.

Documentation and tests update:

```text
docs/runtime_resource_autotuning_v1.md or a versioned successor
docs/scopes/GOV-RUNTIME-RESOURCE-AUTOTUNE-01.md
docs/runtime_resource_autotune_evolution.md
existing focused runtime-resource and E7 wrapper tests
```

The implementation must not modify `docs/handoff.md`, `experiments/registry.yaml`,
formal scientific configuration, or the closed formal execution channel.

### 11.2 Revised estimate after design review

| Area | Estimate |
|---|---:|
| production code | 320–480 lines |
| deterministic tests | 450–700 lines |
| usage/scope updates | 80–160 lines |
| focused implementation | 3–4.5 engineer-days |
| CI and real-server shadow | 0.5–1.5 engineer-days |

The first estimate was increased after review found two necessary details that must
not be omitted: resolving the current cgroup path and validating affinity/quota as
independent constraints for both the single-worker estimate and concurrent
candidates.

This remains much smaller than Draft PR `#50`'s project-neutral policy architecture.
V2 reuses the existing DRPO core and does not add a portable package, generic contract
engine, scheduler backend, or Stage-2 integration.

## 12. Acceptance, risk, and rollback

### 12.1 Deterministic acceptance matrix

Required tests include:

1. current-cgroup path resolution for cgroup v2 and direct v1 compatibility;
2. root-escape and malformed-controller rejection;
3. affinity 384 and finite quota 376 records effective capacity 376;
4. fractional quota is preserved and unlimited quota uses affinity;
5. tight quota plus unrelated host load uses the quota budget rather than subtracting
   all host load from the quota;
6. generous quota plus saturated affinity is host-load constrained;
7. high load average with low measured busy cores does not collapse capacity;
8. low load average with high measured busy cores blocks unsafe concurrency;
9. `iowait` is not CPU execution and guest fields are not double-counted;
10. aligned worker/system windows subtract probe demand exactly once;
11. non-positive `/proc/stat` delta fails closed;
12. process-tree CPU measurement includes threads and long-lived descendants;
13. CPU-, memory-, task-, configured-cap-, and growth-bound cases;
14. throughput candidates never exceed the safe ceiling;
15. throughput-fast but resource-invalid candidate is rejected;
16. timed-out or failed candidate cannot be selected;
17. plan writes one immutable selection and stable digest;
18. run invokes no memory probe, CPU probe, or throughput benchmark;
19. plan worker count remains unchanged when load average rises afterward;
20. unsafe run revalidation blocks without silent downshift;
21. run identity, workload/scientific fingerprint, or policy mismatch blocks;
22. raw-load-average selections are invalidated;
23. blocked attempts preserve selection and write revalidation evidence;
24. conflicting live process groups block run;
25. all probe process groups are cleaned and no orphan remains;
26. high-c, squared-EXP night, PPO w(0), and Stage A wrappers share the policy;
27. fixed launchers and scientific branch matrices remain unchanged.

### 12.2 CI gates

Before server shadow:

- Python compilation and focused runtime-resource tests;
- affected E7 wrapper tests;
- full pytest and Ruff;
- handoff authority no-op verification;
- formal execution-channel validation;
- governance inventory and stage-status validation;
- exact diff review confirming no scientific-field change.

### 12.3 Real CPU shadow

The exact reviewed commit is tested in a new work directory on the target E7 server.
The shadow must:

1. record affinity, active cgroup path, quota, load averages, busy-core samples,
   worker CPU demand, RAM evidence, safe ceiling, every candidate, and selection;
2. exercise at least one candidate above one when capacity permits;
3. record candidate CPU/RAM validity separately from throughput validity;
4. prove the plan benchmark may raise load average without changing the frozen worker
   count;
5. invoke run in selection-consumption mode and prove no second throughput grid starts;
6. verify lightweight CPU/RAM revalidation and unchanged selection digest;
7. verify no stale process group or orphan;
8. avoid the full scientific sweep during the selection-only shadow; and
9. complete a separately approved small real-data liveness using the exact selected
   count before resuming the 150-branch Stage A run.

Tests, CI, a single-worker probe, or a selection-only report do not independently
establish full runtime readiness.

### 12.4 Risks and controls

| Risk | Control |
|---|---|
| capacity overestimate | dual CPU constraints, global fraction, worker reserve, RAM gate, candidate validation |
| capacity underestimate under tight quota | external host load applies to affinity, not subtracted directly from quota |
| short-window noise | aligned sample, safety factors, run revalidation |
| representative mismatch | adapter-owned hard fingerprint and representative identity |
| benchmark self-feedback | plan-only selection; run cannot reselect |
| stale environment | measured revalidation and fail-closed replan |
| legacy cache reuse | policy/source/fingerprint version checks |
| shared-core regression | wrapper coverage plus full repository tests |
| scientific drift | immutable scientific fingerprints and exact diff audit |

### 12.5 Rollback

1. Stop invoking affected `*_auto.py` entrypoints.
2. Use the unchanged fixed launcher or last separately verified fixed schedule.
3. Preserve failed work directories, selection/revalidation records, benchmark
   summaries, and logs.
4. Revert the measured-CPU maintenance commit as one reviewed change.
5. Never reinterpret a failed resource shadow as a scientific result.

## 13. Evolution ledger

### 13.1 `AUTOTUNE-2026-07-12-V1-MINIMAL`

**Status:** `active_opt_in` on the document base.  
**Claim:** `GOV-RUNTIME-RESOURCE-AUTOTUNE-01`.

**Design:** representative E7 RSS probe; load-average CPU limit; host/cgroup memory,
task, configured, fallback, and growth limits; E8 idle/free-VRAM device selection;
one process per GPU; machine-readable selection; fixed launchers preserved.

**Known limitations:** load average was treated as occupied CPU capacity; E7 was not
worker-CPU-demand aware; E8 did not measure full training/evaluation peaks.

### 13.2 `AUTOTUNE-2026-07-13-E7-PROBE-HORIZON`

**Status:** `implemented`; scope required real-server re-acceptance.

A 20,000-step probe could finish before a 50,000-step evaluation interval and leave
terminal evaluation history empty. The replacement derives a probe horizon covering
at least two frozen evaluation intervals while keeping the wall-clock sampler
bounded. Formal horizons and evaluation remain unchanged.

### 13.3 `AUTOTUNE-2026-07-13-GPU-LIVENESS-ONLY`

**Status:** `superseded`; development evidence only.

Eight H20 workers remained alive through model load without reaching a real optimizer
update or maximum-shape evaluation. `slots_per_gpu=8` was not workload-capacity
evidence. Replacement: explicit training and maximum-evaluation phases.

### 13.4 `AUTOTUNE-2026-07-14-GPU-PHASE-AWARE`

**Status:** `superseded` as a complete placement solution; its single-worker envelope
remains engineering evidence.

The worker completed `model_loaded`, `training_peak_completed`,
`evaluation_peak_completed`, and `probe_complete`. A second H20 shadow then showed
that load average `387.5` on 384 logical CPUs reduced an otherwise idle eight-H20
pool to one selected GPU. It also recorded a null worker return code before controller
cleanup. Replacement: measured CPU plus clean zero exit and process-group disappearance.

### 13.5 `AUTOTUNE-2026-07-14-GPU-MEASURED-CPU-DRAFT`

**Status:** `ci_validated` on Draft PR `#53`; hardware shadow pending and not active on
`main` at this base.

The draft uses `/proc/stat`, process-tree CPU demand, external occupancy, phase/exit
contracts, and old-cache invalidation. It motivates the E7 design but does not
implement E7 shared-core behavior.

### 13.6 `AUTOTUNE-2026-07-14-E7-SELF-FEEDBACK-INCIDENT`

**Status:** confirmed engineering defect; server-report repository deposit was pending
at this base.

```text
plan benchmarks worker counts
  -> selects 112
  -> benchmark raises load average above CPU count
  -> immediately following run calls auto selection again
  -> cache rejects 112
  -> load-average arithmetic yields one
  -> run replans and starts one worker
```

Root causes: invalid load-average semantics; plan and run both owned selection; plan
selection was not immutable identity; cache validation treated dynamic load as hard
capacity. No scientific result was produced.

### 13.7 `AUTOTUNE-2026-07-14-E7-MEASURED-CPU-V2`

**Status:** `proposed`; Sections 4–12 are the implementation gate.

Benefits: measured capacity, quota awareness, workload CPU demand, retained RAM and
throughput evidence, immutable plan, no silent downshift, revalidation provenance,
and one shared E7 implementation.

Rejected additions: scheduler service, cache engine, portable package, dynamic
resizing, migration, NUMA/affinity tuning, multi-node support, batch/thread tuning,
Stage-2 integration, and automatic GPU/CPU policy unification.

### 13.8 `AUTOTUNE-2026-07-14-QUOTA-ARITHMETIC-REVIEW`

**Status:** `proposed` design correction completed before implementation.

The first document draft computed `min(affinity, quota) * fraction - external_load`.
Review found that this incorrectly subtracts all host load from a tight quota even
when the host has ample idle affinity capacity. The corrected design keeps two
constraints:

```text
worker demand <= quota * fraction
external load + worker demand <= affinity * fraction
```

The same correction applies to initial capacity, concurrent candidate validation,
and run revalidation. Current-cgroup path resolution was also made explicit so quota
cannot be silently read from the wrong cgroup directory.

## 14. Design review record

### Review 1 — responsibility cohesion

Pass. Selection arithmetic, bounded probes, immutable plan, and revalidation remain
in the resource layer. Runners retain execution, resume, heartbeat, packaging, and
scientific outputs.

### Review 2 — necessity and anti-overengineering

Pass after excluding a standalone CPU module, cache engine, cross-project package,
dynamic scaling, NUMA/affinity tuning, and scheduler/provider abstractions. Every
remaining component maps to a demonstrated defect or safety requirement.

### Review 3 — measurement correctness

The first pass found a real flaw in quota arithmetic. The final design uses independent
quota and affinity constraints, aligned worker/system windows, no `iowait` charge,
no guest double-counting, one probe-demand subtraction, and current-cgroup resolution.
Pass after correction.

### Review 4 — candidate validity

Pass after requiring each throughput candidate to satisfy measured CPU and RAM bounds
in addition to completing quickly. Throughput cannot override capacity safety.

### Review 5 — lifecycle and self-feedback

Pass. Plan alone selects; run cannot probe or benchmark; selection digest is frozen;
unsafe environments require replan rather than downshift.

### Review 6 — identity, cache, and provenance

Pass. Selected resources, workload/scientific fingerprints, adapter, and policy are
stable identity. Busy cores, free memory, load diagnostics, timestamps, and evidence
routes are dynamic revalidation evidence. Old policy selections are invalid.

### Review 7 — failure and cleanup

Pass. Failed/timed-out candidates are invalid; process groups are audited; blocked
attempts preserve logs and selection; live conflicts block run.

### Review 8 — integration and scientific isolation

Pass. Only active subprocess count changes. Fixed launchers remain rollback paths.
No handoff, registry, scientific configuration, or formal-channel change is required.

## 15. Remaining uncertainties before implementation

The design is ready for implementation review, but these remain empirical:

- actual E7 representative worker CPU demand;
- short-window busy-core variance during the throughput grid;
- whether `1.25` worker CPU safety factor is sufficient across current E7 workloads;
- selected worker count under the measured model;
- plan/run revalidation after the benchmark raises load average;
- wrapper compatibility after Stage A is synchronized with the shared-core fix.

They are resolved by deterministic tests and exact-head CPU shadow, not by adding
unproven scheduler or dynamic-scaling features.
