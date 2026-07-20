# Runtime Resource Autotuning V3 — squared-night low-first repair

**Claim:** `GOV-E7-RUNTIME-AUTOTUNE-ADAPTIVE-SEARCH-V3-01`  
**Parent:** `GOV-RUNTIME-RESOURCE-AUTOTUNE-CPU-V2-01`  
**Scope:** squared-night E7 adapter only  
**Scientific impact:** none

## Why V3 exists

V2 correctly replaced load-average arithmetic with measured affinity/cgroup CPU capacity and
made `plan` selection immutable. Its retained PPO-family throughput grid was nevertheless
unsafe on large machines because it started around 50 percent of the measured ceiling. A
large ceiling could therefore make the first candidate larger than a known startup-failure
region. V2 also reused the representative `probe_steps` horizon for every candidate, so a
long RSS/CPU observation accidentally multiplied into several long short-training grids.

V3 keeps the V2 CPU, cgroup, memory, identity, revalidation, process-group, and retained-peak
contracts. It replaces only the squared-night adapter's candidate and probe-horizon policy.

## Separate probe responsibilities

The representative resource probe owns:

- peak process-tree RSS;
- average process-tree CPU demand;
- affinity and quota-domain observations;
- the requested `probe_steps` and `probe_seconds` values.

The throughput grid owns only a short comparison of independent subprocess counts. Every
squared-night throughput candidate is capped at 5000 optimizer steps per branch. Therefore a
requested representative horizon of 100000 steps remains one 100000-step resource probe; it
no longer turns every concurrency candidate into another 100000-step run.

Both requested and effective candidate horizons are recorded in each
`BENCHMARK_SUMMARY.json`.

## Low-first candidate policy

Let `safe_cap` be the unchanged V2 minimum of CPU, memory, task, growth, and optional
configured ceilings. V3 constructs a sorted unique grid containing:

1. candidate `1`;
2. the eight equal-fraction boundaries of `safe_cap`;
3. the legacy fallback when it lies within the cap;
4. `safe_cap` itself.

No candidate exceeds `safe_cap`. Because candidate `1` is always first, a higher candidate can
fail without leaving the selector with zero completed evidence. The existing shared loop stops
on the first invalid candidate, and the unchanged retained-peak rule selects from the already
valid lower candidates.

`max_workers` remains supported, but only as an optional absolute ceiling. It is no longer
needed to pull the first candidate below the safe range.

Example for `safe_cap=130`, `fallback=60`:

```text
1, 17, 33, 49, 60, 65, 82, 98, 114, 130
```

If 49 is invalid, 1, 17, and 33 remain valid evidence and selection continues from those rows.
The plan does not fail merely because the first fraction of a large cap was unsafe.

## Policy and identity

Squared-night selections use selector policy version `3`. The resource fingerprint records:

- representative probe steps;
- effective throughput probe steps;
- candidate policy and divisions;
- fallback as a bounded candidate only;
- max workers as an optional absolute ceiling.

A V2 squared-night selection is not silently upgraded. Existing work directories remain
immutable and may resume only under their original source and policy. A V3 decision requires a
new work directory.

## One-click lifecycle

The squared-night one-click wrapper now behaves as follows:

```text
no RUNTIME_SELECTION + no RUN_IDENTITY -> plan, then run --resume
both files present                         -> run --resume only
exactly one file present                   -> fail closed
```

The E7-specific `E7_SQUARED_EXP_MAX_WORKERS` value has priority. When it is absent, the wrapper
inherits the unified RunSpec `DRPO_RUNTIME_MAX_WORKERS` ceiling.

## What V3 does not solve

V3 is not:

- online resizing;
- cross-run selection caching;
- NUMA or physical-core allocation;
- cgroup memory reservation;
- external-process preemption;
- protection against another workload exhausting a shared memory cgroup;
- permission to alter an existing immutable worker selection.

Long-running E7 and E8 jobs still require non-overlapping CPU affinity and operational control
of shared memory and storage pressure.

## Required validation

Before use on a full pilot:

1. focused deterministic tests and shell syntax;
2. full repository CI and governance no-op checks;
3. a selection-only server shadow in a new work directory;
4. confirmation that a long representative probe still produces 5000-step throughput rows;
5. confirmation that a failed higher candidate retains valid lower candidates;
6. confirmation that one-click consumes an existing V3 selection without another plan;
7. no scientific branch launch during shadow.
