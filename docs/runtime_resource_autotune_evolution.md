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
3. every finite cgroup quota domain that constrains the current process;
4. aligned CPU usage in those quota domains;
5. measured representative worker CPU demand;
6. existing host/cgroup memory evidence;
7. the existing bounded throughput candidate search where already used; and
8. immutable `plan` selection followed by lightweight `run` revalidation.

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
fingerprints. They must not copy capacity arithmetic.

### 4.3 Required components and necessity test

| Component | Direct purpose | Why it is required |
|---|---|---|
| affinity-aware CPU count | observe process-visible CPUs | `os.cpu_count()` may include unusable CPUs |
| cgroup hierarchy resolution | find every active quota domain | a parent quota can constrain a child with `cpu.max=max` |
| quota-domain CPU usage | account for siblings sharing an ancestor quota | current-child usage alone can overestimate capacity |
| `/proc/stat` busy-core sample | measure affinity-wide execution | load average includes history and waits |
| process-tree CPU demand | distinguish light and heavy workers | idle cores alone do not imply worker count |
| CPU safety reserve | cover sampling/workload variance | probes are estimates |
| existing RAM/cgroup memory gate | prevent memory oversubscription | CPU capacity is insufficient |
| workload-specific throughput grid | avoid a safe but inefficient ceiling | already valuable for PPO-family E7 paths |
| candidate resource validation | reject fast but unsafe candidates | throughput is not safety |
| immutable plan selection | remove probe self-feedback and identity drift | run must not choose another schedule |
| lightweight run revalidation | block unsafe later starts | immutable does not mean blindly reusable |
| schema/policy versioning | reject old selections | old semantics are incompatible |
| attempt-local revalidation record | preserve allow/block evidence | the plan artifact must remain immutable |

The final review justifies exactly one new production module:

```text
src/drpo/runtime_cpu_capacity.py
```

It owns Linux CPU/cgroup observation and pure capacity arithmetic. This avoids adding
hundreds of Linux-specific lines to the existing mixed CPU/RAM/GPU helper. It is not
a package, service, provider layer, scheduler, or plugin system.

## 5. CPU capacity and cgroup hierarchy contract

### 5.1 Affinity binding

The process-visible CPU set comes from `os.sched_getaffinity(0)`. The exact sorted CPU
IDs form an affinity binding fingerprint and select per-CPU rows from `/proc/stat`.
A changed set requires replan even when cardinality is unchanged because V2 does not
model NUMA/topology equivalence.

`os.cpu_count()` is allowed only as an explicitly recorded compatibility fallback.
Production Linux shadows must exercise affinity.

### 5.2 Current cgroup and ancestor quota domains

The implementation resolves `/proc/self/cgroup` against a configured cgroup mount
root and walks from the current cgroup to the controller mount root.

For cgroup v2 it reads, at every level:

```text
cpu.max
cpu.stat
```

For direct cgroup v1 compatibility it resolves the `cpu` / `cpu,cpuacct` controller
and reads, where present:

```text
cpu.cfs_quota_us
cpu.cfs_period_us
cpuacct.usage
```

Requirements:

- canonicalize and reject paths escaping the configured mount root;
- support cgroup namespaces exposing the current cgroup at mount root;
- preserve the ordered current-to-root hierarchy;
- treat each finite ancestor quota as an independent quota domain;
- record unlimited levels but do not create a budget constraint for them;
- fail closed on malformed quota, zero period, contradictory controller evidence, or
  missing aligned usage accounting for a finite domain.

A finite quota domain has:

```text
quota_capacity_cores = quota_microseconds / period_microseconds
```

Fractional capacity is preserved and never rounded up.

### 5.3 Quota-domain usage

Aligned usage is:

```text
v2: delta(cpu.stat usage_usec) / elapsed_microseconds
v1: delta(cpuacct.usage nanoseconds) / elapsed_nanoseconds
```

Ancestor usage includes current-cgroup work and sibling descendants sharing that
ancestor. This is why all finite domains must be evaluated rather than taking only
the numerically smallest quota.

V2 `nr_throttled` and `throttled_usec`, when present, are recorded per domain as
diagnostics. They are not a standalone hard gate in the first implementation;
measured budgets and candidate throughput remain authoritative.

### 5.4 Reported effective capacity

For compact provenance:

```text
effective_quota_capacity_cores =
    minimum finite quota capacity across domains
    or infinity when no finite domain exists

effective_cpu_capacity_cores =
    min(affinity_capacity_cores, effective_quota_capacity_cores)
```

This summary is not itself the worker budget because each quota domain can have a
different amount of external sibling usage. Section 8 evaluates every constraint.

The existing default `cpu_fraction=0.85` remains global headroom unless an approved
adapter policy supplies another value.

## 6. CPU occupancy contract

### 6.1 Load average is diagnostic only

Linux load average is an exponentially smoothed count of runnable and
uninterruptible tasks, not current execution. It can remain high after a benchmark
exits and include I/O waits.

V2 records one-, five-, and fifteen-minute load average only as diagnostics. No load
field participates in capacity arithmetic or cache acceptance.

### 6.2 Affinity-wide `/proc/stat` sampling

System execution occupancy is measured over an aligned monotonic interval using the
per-CPU rows for the affinity set.

Busy time:

```text
user + nice + system + irq + softirq + steal
```

Non-executing capacity:

```text
idle + iowait
```

Only fields through `steal` enter the total; guest fields are not added because guest
execution is already included in user/nice.

```text
system_busy_cores =
    affinity_capacity_cores * busy_tick_delta / total_tick_delta
```

Missing rows, non-positive deltas, or changing affinity makes auto measurement
unavailable.

### 6.3 Quota-domain sampling

The same monotonic interval samples usage in every finite quota domain:

```text
quota_domain_busy_cores[d] =
    quota_domain_cpu_time_delta[d] / elapsed_time
```

A finite quota without reliable aligned usage fails closed. The implementation must
not assume external processes live outside the quota hierarchy.

### 6.4 Sampling windows

Plan and candidate measurements use their aligned worker interval. Pre-launch run
revalidation uses three consecutive one-second samples and conservatively takes the
maximum affinity-wide busy value and, independently, the maximum busy value for each
quota domain.

Sample count and duration are test-configurable runtime policy, not scientific
parameters. Evidence records monotonic start/end, affinity set, `/proc/stat` deltas,
quota-domain deltas, throttling diagnostics, and process-tree CPU deltas.

## 7. Representative worker CPU demand

### 7.1 Measurement

The existing representative resource probe records cumulative process-tree user and
system CPU seconds:

```text
measured_cpu_cores_per_worker =
    process_tree_cpu_seconds / aligned_elapsed_seconds
```

Process-level accounting includes threads. Long-lived descendants are included by
process-tree sampling. V2 does not add an E7 phase framework or exact recovery of
descendants that start and exit entirely between polls; global reserve and concurrent
candidate validation cover that residual limitation.

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

These runtime safety fields enter the resource fingerprint and do not alter
scientific execution.

### 7.3 External occupancy during plan

The aligned affinity and quota-domain samples include the probe itself:

```text
external_system_busy_cores =
    max(0, system_busy_cores - measured_probe_cpu_cores)

external_quota_domain_busy_cores[d] =
    max(0, quota_domain_busy_cores[d] - measured_probe_cpu_cores)
```

Probe demand is subtracted exactly once in every accounting domain through which it
is charged.

## 8. Capacity and throughput selection

### 8.1 Independent CPU constraints

Affinity-wide external load and every finite quota domain independently constrain the
workload:

```text
affinity_budget_cores = affinity_capacity_cores * cpu_fraction

affinity_worker_budget_cores =
    max(0, affinity_budget_cores - external_system_busy_cores)

quota_worker_budget_cores[d] =
    max(
      0,
      quota_capacity_cores[d] * cpu_fraction
      - external_quota_domain_busy_cores[d]
    )

worker_cpu_budget_cores = min(
    affinity_worker_budget_cores,
    every quota_worker_budget_cores[d],
    infinity when no finite quota exists
)

cpu_worker_limit =
    floor(worker_cpu_budget_cores / reserved_cpu_cores_per_worker)
```

This handles tight child quota, tight shared parent quota, sibling quota use,
generous quota on saturated affinity, and unrelated host load outside the cgroup.
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
adds no generic throughput engine.

Candidate rules:

- benchmark the verified fallback and points near 50%, 75%, and 100% of the safe
  ceiling;
- never launch above the safe ceiling;
- require every worker to exit successfully without timeout;
- measure aggregate completed optimizer updates per second;
- measure aggregate worker CPU, affinity-wide CPU, every quota domain, and host RSS
  over the aligned candidate window;
- select the smallest resource-valid candidate reaching the configured fraction of
  peak aggregate throughput.

Resource validity requires:

```text
candidate_external_system_busy_cores
+ candidate_worker_cpu_cores
    <= affinity_capacity_cores * cpu_fraction

for every finite quota domain d:
candidate_external_quota_domain_busy_cores[d]
+ candidate_worker_cpu_cores
    <= quota_capacity_cores[d] * cpu_fraction
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
5. re-read affinity binding, cgroup hierarchy, quotas, quota-domain usage, memory, and
   affinity-wide busy cores;
6. project stored worker CPU/RAM reservations;
7. write attempt-local revalidation evidence; and
8. start with the exact planned count or fail closed.

Run may never silently change `112` to `80`, `20`, `1`, or another value.

### 9.3 Revalidation arithmetic

No workload worker from the work directory may be active before launch. Therefore the
three-sample maxima are external occupancy and are not reduced by stored demand.

```text
selected_worker_cpu_cores =
    selected_workers * reserved_cpu_cores_per_worker

cpu_revalidation_ok =
    current_system_busy_cores_max
    + selected_worker_cpu_cores
        <= current_affinity_capacity_cores * cpu_fraction
    and
    for every finite quota domain d:
      current_quota_domain_busy_cores_max[d]
      + selected_worker_cpu_cores
          <= current_quota_capacity_cores[d] * cpu_fraction

memory_revalidation_ok =
    selected_workers * reserved_memory_per_worker
        <= current_usable_memory
```

A changed affinity binding, changed cgroup hierarchy, quota reduction, fingerprint
mismatch, insufficient memory, unsafe CPU projection, or live-process conflict blocks
with:

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
- affinity binding and ordered cgroup quota domains;
- quota values, three affinity samples, and three samples per quota domain;
- stored worker CPU/RAM reservation;
- projected totals and every constraint;
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
affinity binding
ordered cgroup quota-domain binding
selected_workers
selection_digest
```

### 10.2 Stable identity versus dynamic evidence

Stable identity includes selected workers, workload/scientific fingerprints, adapter
and policy versions, representative probe policy, CPU/RAM safety parameters,
affinity binding, and ordered cgroup-domain binding.

Dynamic evidence is excluded from the stable digest: load diagnostics, busy cores,
free memory, timestamps, throttling diagnostics, evidence paths, and probe/cache
route. Dynamic changes trigger revalidation. Stable changes require new work directory
and selection.

### 10.3 No generic cache engine

V2 adds no cache service or abstraction. Existing adapter-owned measurement reuse may
continue only under exact workload, source, probe policy, selector policy, and
representative identity. `RUNTIME_SELECTION.json` is immutable authority, not a
mutable cache.

## 11. Implementation surface and cost

### 11.1 Production files

Exactly one new cohesive production module is justified:

```text
src/drpo/runtime_cpu_capacity.py
```

It owns:

- affinity and cgroup hierarchy resolution;
- `/proc/stat` and quota-domain sampling;
- process-tree CPU accounting helpers;
- immutable observation/result values;
- pure CPU budget arithmetic.

Existing integration files remain:

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
docs/runtime_resource_autotuning_v1.md or versioned successor
docs/scopes/GOV-RUNTIME-RESOURCE-AUTOTUNE-01.md
docs/runtime_resource_autotune_evolution.md
focused CPU-capacity, resource-core, adapter, and wrapper tests
```

No handoff, registry, scientific config, or formal-channel modification is allowed.

### 11.2 Revised estimate after hierarchy and module review

| Area | Estimate |
|---|---:|
| production code | 500–700 lines |
| deterministic tests | 750–1,100 lines |
| usage/scope updates | 80–180 lines |
| focused implementation | 5–7 engineer-days |
| CI and real-server shadow | 0.5–1.5 engineer-days |

The increase is evidence-driven: correct general behavior needs active hierarchy
resolution, multiple quota domains, aligned domain usage, independent constraints,
and plan/run lifecycle repair. The dedicated module is justified because placing all
of this in `runtime_resource_autotune.py` would create an unauditable mixed-resource
monolith.

This remains narrower than Draft PR `#50`: no portable package, generic contract
engine, scheduler backend, or Stage-2 integration.

## 12. Acceptance, risk, and rollback

### 12.1 Deterministic acceptance matrix

Required tests include:

1. affinity binding and changed-set rejection;
2. v2 current-cgroup and ancestor traversal;
3. direct v1 controller compatibility;
4. root-escape and malformed hierarchy rejection;
5. unlimited child plus finite parent quota;
6. multiple finite parent/child domains with different periods;
7. fractional quota preservation;
8. sibling work reducing shared-parent capacity;
9. child-only work reducing child and parent usage exactly once per domain;
10. generous quota plus saturated affinity;
11. unrelated host work reducing affinity but not a private quota domain;
12. high load average with low measured execution;
13. low load average with high measured execution;
14. `iowait`, guest, and steal accounting;
15. aligned system/domain/worker windows;
16. missing/zero deltas fail closed;
17. process-tree CPU includes threads and long-lived descendants;
18. CPU-, memory-, task-, configured-cap-, and growth-bound cases;
19. candidates never exceed safe ceiling;
20. fast but resource-invalid candidate rejection;
21. timeout/nonzero candidate rejection;
22. immutable selection/digest;
23. run invokes no probe or grid;
24. post-plan load average cannot change workers;
25. unsafe revalidation blocks without downshift;
26. changed affinity/cgroup hierarchy/fingerprint/policy blocks;
27. legacy selection invalidation;
28. blocked attempt preserves selection and writes evidence;
29. live process conflict and process-group cleanup;
30. wrapper sharing and no copied arithmetic;
31. fixed launchers/scientific matrices unchanged;
32. three-sample maxima per accounting domain.

### 12.2 CI gates

Before shadow: compile, focused CPU/runtime/E7 tests, full pytest, Ruff, handoff
authority no-op, formal-channel validation, governance inventory/stage validation,
and exact scientific-field diff review.

### 12.3 Real CPU shadow

The exact reviewed commit is tested in a new work directory. The shadow must:

1. record affinity, cgroup hierarchy, every quota/domain usage, throttling diagnostics,
   load averages, affinity busy, worker demand, RAM, safe ceiling, candidates, and
   selection;
2. exercise at least one candidate above one when capacity permits;
3. report candidate resource validity separately from throughput validity;
4. show benchmark-raised load average cannot alter frozen selection;
5. prove run consumes selection without a second grid;
6. verify three-sample revalidation in every accounting domain and unchanged digest;
7. verify no stale process or orphan;
8. suppress the full scientific sweep during selection-only shadow; and
9. complete separately approved small real-data liveness at the selected count before
   resuming the 150-branch Stage A run.

Tests, CI, one worker, or selection-only output do not independently establish full
runtime readiness.

### 12.4 Risks and controls

| Risk | Control |
|---|---|
| capacity overestimate | affinity plus every quota domain, CPU reserve, RAM gate, candidate validation |
| parent/sibling quota error | hierarchy traversal and ancestor usage accounting |
| short-window noise | aligned candidate windows and three-sample maxima |
| representative mismatch | adapter hard fingerprint and representative identity |
| benchmark self-feedback | plan-only selection; run cannot reselect |
| stale environment | measured revalidation and fail-closed replan |
| legacy cache reuse | policy/source/fingerprint version checks |
| shared-core regression | wrapper coverage plus full tests |
| module sprawl | exactly one cohesive Linux CPU module; no framework/package layers |
| scientific drift | immutable scientific fingerprints and diff audit |

### 12.5 Rollback

1. Stop affected `*_auto.py` entrypoints.
2. Use unchanged fixed launchers or separately verified fixed schedule.
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

Eight H20 workers remained alive through model load without optimizer update or
maximum-shape evaluation. `slots_per_gpu=8` was not capacity evidence. Replacement:
explicit training and maximum-evaluation phases.

### 13.4 `AUTOTUNE-2026-07-14-GPU-PHASE-AWARE`

**Status:** `superseded` as complete placement; single-worker envelope remains useful.

The phase contract completed, but load average `387.5` on 384 CPUs reduced an idle
eight-H20 pool to one GPU. Worker return code was null before controller cleanup.
Replacement: measured CPU plus clean exit/process-group disappearance.

### 13.5 `AUTOTUNE-2026-07-14-GPU-MEASURED-CPU-DRAFT`

**Status:** `ci_validated` on Draft PR `#53`; hardware shadow pending and not active on
`main`.

The draft uses `/proc/stat`, process-tree CPU demand, external occupancy, phase/exit
contracts, and cache invalidation. It motivates but does not implement E7 V2.

### 13.6 `AUTOTUNE-2026-07-14-E7-SELF-FEEDBACK-INCIDENT`

**Status:** confirmed engineering defect; server-report repository deposit pending.

```text
plan benchmarks -> selects 112 -> raises load average
-> run reselects -> cache rejects 112 -> load formula yields one
-> run starts one worker
```

Root causes: invalid load semantics; duplicate selection ownership; mutable runtime
identity; dynamic load in cache acceptance. No scientific result was produced.

### 13.7 `AUTOTUNE-2026-07-14-E7-MEASURED-CPU-V2`

**Status:** `proposed`; Sections 4–12 are implementation gate.

Benefits: measured affinity and quota-domain capacity, worker demand, retained RAM and
throughput evidence, immutable plan, no silent downshift, revalidation evidence, and
one shared E7 implementation.

Rejected: scheduler service, cache engine, portable package, dynamic resizing,
migration, NUMA/affinity tuning, multi-node, batch/thread tuning, Stage-2 integration,
and automatic GPU/CPU policy unification.

### 13.8 `AUTOTUNE-2026-07-14-QUOTA-ARITHMETIC-REVIEW`

**Status:** design correction completed before implementation.

The first draft subtracted all host load from `min(affinity, quota)`. Review found this
incorrect for a tight quota on a large idle host. Replacement: independent affinity
and quota constraints plus active cgroup path resolution.

### 13.9 `AUTOTUNE-2026-07-14-CGROUP-OCCUPANCY-REVIEW`

**Status:** design correction completed before implementation.

The second draft did not subtract other same-cgroup quota use. Replacement: aligned
cgroup usage, separate external affinity and cgroup occupancy, both constraints in
plan/candidates/run, and three-sample pre-launch maxima.

### 13.10 `AUTOTUNE-2026-07-14-CGROUP-HIERARCHY-REVIEW`

**Status:** design correction completed before implementation.

The third draft read only the current cgroup. Review found that a finite ancestor
quota and sibling descendants can be the real limiting domain. Replacement: walk all
ancestors, preserve every finite quota domain, measure usage in each domain, and
require every constraint to pass.

### 13.11 `AUTOTUNE-2026-07-14-MODULE-BOUNDARY-REVIEW`

**Status:** design correction completed before implementation.

The earlier plan prohibited a new module. After hierarchy and aligned-domain
requirements were made explicit, adding all Linux CPU logic to the existing mixed
resource helper would create a large monolith. Replacement: exactly one cohesive
`runtime_cpu_capacity.py`; no package, backend, provider, or scheduler abstraction.

## 14. Design review record

### Review 1 — responsibility cohesion

Pass. Resource observation/arithmetic remains preflight; runners retain execution,
resume, heartbeat, packaging, and scientific outputs.

### Review 2 — necessity and anti-overengineering

Pass. Every remaining component maps to a demonstrated defect or safety gap. Dynamic
scaling, NUMA tuning, providers, schedulers, generic cache, and portable policy core
remain excluded.

### Review 3 — quota arithmetic

Found and corrected invalid subtraction of all host load from a tight quota. Affinity
and quota are independent constraints.

### Review 4 — cgroup occupancy

Found and corrected omission of same-cgroup external usage. Quota-domain usage is
aligned and separately budgeted.

### Review 5 — hierarchy correctness

Found and corrected omission of ancestor quota domains and sibling descendants. Every
finite domain must pass.

### Review 6 — CPU-time accounting

Pass after locking `iowait` as non-execution, avoiding guest double-counting,
including steal, aligning windows, and subtracting probe demand once per domain.

### Review 7 — candidate validity

Pass after requiring throughput candidates to satisfy affinity, every quota domain,
and RAM bounds. Throughput cannot override safety.

### Review 8 — lifecycle and self-feedback

Pass. Plan alone selects; run cannot probe/benchmark; unsafe capacity requires replan,
not downshift.

### Review 9 — identity and provenance

Pass. Affinity/cgroup bindings and fingerprints are stable; busy/free/timestamp data
is dynamic revalidation evidence. Old selections are invalid.

### Review 10 — failure and cleanup

Pass. Failed/timed-out candidates are invalid; live conflicts block; process groups
are audited; selection is preserved.

### Review 11 — revalidation noise

Pass after conservative maxima over three one-second samples in every domain. This is
bounded preflight, not online monitoring.

### Review 12 — module cohesion

Pass after introducing exactly one CPU-capacity module. The module has one reason to
change and prevents the mixed resource core from becoming unauditable.

### Review 13 — scientific isolation

Pass. Only active subprocess count changes. Fixed launchers remain rollback. No
handoff, registry, scientific config, or formal-channel change is required.

## 15. Remaining uncertainties before implementation

The design is ready for implementation review, but these remain empirical:

- actual representative worker CPU demand;
- affinity and quota-domain variance during candidates;
- whether `1.25` worker reserve is sufficient across E7 workloads;
- selected worker count under V2;
- plan/run behavior after load average rises;
- Stage A wrapper compatibility after shared-core synchronization.

They are resolved by deterministic tests and exact-head CPU shadow, not by adding
unproven scheduler or dynamic-scaling features.
