# Runtime Resource Autotuning V2 — measured CPU E7 path

**Claim:** `GOV-RUNTIME-RESOURCE-AUTOTUNE-CPU-V2-01`  
**Status:** implemented on Draft PR `#65`; exact-head server shadow pending  
**Scientific impact:** none  
**Default-policy impact:** none; fixed launchers remain the rollback path

## Purpose

V2 replaces the E7 auto launchers' raw-load-average CPU arithmetic with measured
Linux CPU capacity. It selects only the number of independent subprocesses. It does
not change methods, seeds, batch size, thread settings, training horizon, evaluation,
or any other scientific coordinate.

The authoritative design and append-only history remain in
`docs/runtime_resource_autotune_evolution.md`.

## Supported E7 auto entrypoints

V2 is wired only into paths confirmed to share the existing E7 resource selector:

```text
scripts/run_e7_canonical_exp_horizon_joint_auto.py
scripts/run_e7_ppo_w0_grid_pilot_auto.py
scripts/run_e7_w0_highc_actor_auto.py
scripts/run_e7_squared_exp_night_auto.py
```

The canonical exp-horizon path uses the measured CPU/RAM safe ceiling. The PPO-family
w(0), high-c, and squared-EXP night paths additionally retain their bounded
throughput candidate grid. No new generic throughput engine was added.

## Measured CPU contract

V2 records the exact `sched_getaffinity(0)` CPU set and resolves the current process
cgroup from `/proc/self/cgroup`.

For every visible finite cgroup CPU quota from the current cgroup to the controller
mount root, it records:

```text
quota capacity
usage accounting path
aligned CPU usage during the sample
```

Supported accounting:

```text
cgroup v2: cpu.max + cpu.stat usage_usec
cgroup v1: cpu.cfs_quota_us + cpu.cfs_period_us + cpuacct.usage
```

A cgroup membership path that is not visible below the configured mount root fails
closed. A namespaced mount-root fallback is accepted only when `cgroup.procs` or
`tasks` explicitly lists the current process.

Affinity-wide execution is sampled from per-CPU `/proc/stat` rows. Busy execution is:

```text
user + nice + system + irq + softirq + steal
```

`idle` and `iowait` are non-executing capacity. Guest time is not added again.
One-, five-, and fifteen-minute load average remain diagnostic provenance only and do
not participate in worker-capacity or revalidation arithmetic.

## Representative worker demand

The existing representative branch probe now records, over one aligned monotonic
window:

```text
peak process-tree RSS
process-tree CPU seconds
average CPU cores used by the worker
system busy cores
CPU use in every finite quota domain
```

Default CPU reservation policy:

```text
per_worker_cpu_safety_factor = 1.25
minimum_cpu_cores_per_worker = 1.0
```

The reserved worker demand is:

```text
max(minimum_cpu_cores_per_worker,
    measured_cpu_cores_per_worker * per_worker_cpu_safety_factor)
```

The probe remains engineering-only, uses its existing non-scientific seed namespace,
and preserves logs and small evidence while removing generated model payload.

## Capacity arithmetic

Affinity and every finite quota domain are independent constraints.

For affinity:

```text
affinity_budget = affinity_cpu_count * cpu_fraction
external_affinity_use = max(0, measured_system_use - probe_use)
affinity_worker_budget = max(0, affinity_budget - external_affinity_use)
```

For each finite quota domain:

```text
domain_budget = quota_cores * cpu_fraction
external_domain_use = max(0, measured_domain_use - probe_use)
domain_worker_budget = max(0, domain_budget - external_domain_use)
```

The CPU worker budget is the minimum of the affinity and all quota-domain budgets.
The CPU limit is then divided by the reserved per-worker demand and rounded down.
The final safe ceiling also applies:

```text
host/cgroup memory limit
total task count
configured max-workers when present
bounded growth cap
```

A PPO-family throughput candidate is valid only when all workers finish without
timeout or controller cleanup and the aligned CPU, every quota domain, and aggregate
RSS remain within the measured budgets. A fast but resource-invalid candidate cannot
be selected.

## Plan and run lifecycle

### Plan

`plan` is the only operation permitted to:

```text
run the representative resource probe
run a bounded throughput candidate grid when supported
select workers
write RUNTIME_SELECTION.json
create the underlying RUN_IDENTITY.json
bind the selection digest and worker count into that identity
```

A work directory containing `RUNTIME_SELECTION.json` is not silently replanned. Use a
new work directory for a new automatic decision.

### Run

`run` first loads the frozen selection and validates `RUN_IDENTITY.json` before any
capacity sampling. It never calls automatic selection, the representative probe, or
the throughput grid.

It performs three consecutive one-second CPU/domain samples by default, taking the
maximum observed affinity pressure and the independent maximum for each quota domain.
It also validates current usable host/cgroup memory and rejects conflicting processes
whose command line refers to the same work directory.

Run either starts the exact planned worker count or fails with:

```text
RUNTIME_CAPACITY_CHANGED_REPLAN_REQUIRED
```

It must never silently change a planned value such as `112` to `80`, `20`, or `1`.

## Artifacts

### Immutable selection

```text
<work_dir>/RUNTIME_SELECTION.json
```

V2 E7 selections record:

```text
schema_version = 2
selector_policy_version = 2
adapter and implementation hashes
source commit/worktree state
resource fingerprint
exact affinity and cgroup binding
representative CPU/RSS evidence
safe limits and candidate results
selected_workers
selection_digest
load_average_is_diagnostic_only = true
scientific_matrix_changed = false
```

Selections produced by the old raw-load-average policy are incompatible and are not
reused.

### Attempt-local revalidation

```text
<work_dir>/_runtime_resource_attempts/<attempt_id>/RUNTIME_REVALIDATION.json
```

This records identity checks, current binding, three CPU/domain samples, conservative
pressure, CPU/RAM projections, live-process audit, and `ALLOW` or `BLOCK`. It is
runtime provenance and never replaces or mutates the selection.

## Default CLI controls

```text
--cpu-fraction 0.85
--memory-headroom-fraction 0.15
--per-worker-safety-factor 1.20
--per-worker-cpu-safety-factor 1.25
--minimum-cpu-cores-per-worker 1.0
--max-growth-factor 3.0
--revalidation-samples 3
--revalidation-sample-seconds 1.0
```

PPO-family paths also retain:

```text
--throughput-retention-fraction 0.97
```

These are runtime safety and efficiency policy fields. They are included in resource
identity and are not scientific variables.

## Example lifecycle

```bash
python scripts/run_e7_ppo_w0_grid_pilot_auto.py plan \
  --contract <contract.json> \
  --run-spec <run_spec.json> \
  --grid <grid.json> \
  --work-dir <new_work_dir>

python scripts/run_e7_ppo_w0_grid_pilot_auto.py run \
  --contract <contract.json> \
  --run-spec <run_spec.json> \
  --grid <grid.json> \
  --work-dir <same_work_dir> \
  --resume
```

The two commands must use identical resource-policy arguments. The run command
consumes the selection created by plan.

## Failure and rollback

V2 fails closed on malformed or unresolved cgroup evidence, changing affinity,
missing CPU rows, invalid counter deltas, missing quota usage, non-positive worker
demand, unsafe CPU/RAM projection, selection or source drift, missing run identity,
conflicting live processes, or probe/candidate cleanup failure.

Rollback:

1. stop using the affected `*_auto.py` entrypoint;
2. use the unchanged fixed launcher or a separately verified fixed schedule;
3. preserve the selection, revalidation, candidate summaries, logs, and failed work
   directory;
4. do not reinterpret resource evidence as a scientific result.

## Acceptance state

Deterministic tests and repository CI are necessary engineering gates. They do not
establish real-server readiness. Before merge or restoration of the stopped Stage A
workload, the exact reviewed commit still requires:

1. a selection-only CPU shadow in a new work directory;
2. actual execution of a candidate above one when capacity permits;
3. proof that plan-induced load-average elevation does not alter the frozen worker
   count;
4. proof that run launches no second probe or candidate grid;
5. successful three-sample revalidation with the unchanged selection digest;
6. no orphan process group; and
7. a separately approved small real-data liveness using the selected worker count.
