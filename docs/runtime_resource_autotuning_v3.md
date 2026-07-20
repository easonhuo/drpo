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
region. V2 also reused the same requested horizon and timeout for the representative probe
and every candidate, so a large operator override multiplied into several long short-training
grids. Finally, an interrupted plan repeated every completed candidate because the shared V2
benchmark function deleted the candidate directory before rerunning it.

V3 keeps the V2 CPU, cgroup, memory, identity, revalidation, process-group, and retained-peak
contracts. It replaces only the squared-night adapter's candidate, bounded-probe, and
same-workdir checkpoint policy.

## Bounded probe policy

Squared-night preflight is capped at:

```text
optimizer steps per probe/candidate: 5000
wall-clock seconds per probe/candidate: 120
```

The limits apply to both the representative CPU/RSS probe and each throughput candidate. A
caller may request a smaller value. A larger request is recorded but cannot expand the
preflight beyond the V3 limits.

The representative probe still owns:

- peak process-tree RSS;
- average process-tree CPU demand;
- affinity and quota-domain observations.

The throughput grid still owns comparison of independent subprocess counts. Effective
step/time limits and the V3 hard limits enter the immutable resource fingerprint. Requested
and effective values are preserved as non-identity evidence in `RUNTIME_SELECTION.json` and
in every candidate `BENCHMARK_SUMMARY.json`. Therefore changing `100000/2500` to `5000/120`
after planning does not invalidate a selection when the effective policy is unchanged.

## Low-first candidate policy

Let `safe_cap` be the unchanged V2 minimum of CPU, memory, task, growth, and optional
configured ceilings. V3 constructs a sorted unique geometric sequence that:

1. always starts at candidate `1`;
2. grows by a factor of `1.75`;
3. includes the legacy fallback when it lies within the cap;
4. includes `safe_cap` exactly;
5. never contains a value above `safe_cap`.

Because candidate `1` is always first, a higher candidate can fail without leaving the
selector with zero completed evidence. The existing shared loop stops on the first invalid
candidate, and the unchanged retained-peak rule selects from the already valid lower
candidates.

`max_workers` remains supported, but only as an optional absolute ceiling. It is no longer
needed to pull the first candidate below the safe range.

Example for `safe_cap=130`, `fallback=60`:

```text
1, 2, 4, 7, 13, 23, 41, 60, 72, 126, 130
```

If 60 is invalid, candidates through 41 remain valid evidence and selection continues from
those rows. The plan does not fail merely because a fraction of a large machine ceiling was
unsafe.

## Interrupted-plan checkpoint policy

Every completed candidate summary records an exact checkpoint identity over:

- source commit and source dirty state;
- selector implementation digests and V3 adapter/policy identity;
- scientific input fingerprints;
- effective probe horizon and timeout;
- candidate concurrency and probe seed;
- CPU fraction, CPU safety factor, and exact CPU binding.

On a later `plan` attempt in the same work directory, V3 may reuse a candidate only when:

1. the summary exists and declares `valid=true`;
2. the exact checkpoint identity matches;
3. the current usable-memory budget still exceeds the candidate's observed aggregate peak RSS.

Invalid, incomplete, malformed, mismatched, or currently memory-infeasible summaries are not
reused. The currently interrupted candidate therefore reruns, but all earlier exact valid
candidates do not repeat. This is bounded plan continuation, not a cross-run cache service.
The representative resource probe still reruns so current machine pressure is remeasured.

## Policy and identity

Squared-night selections use a V3 adapter id and selector policy version `3`. The immutable
resource fingerprint records:

- effective probe steps and seconds;
- hard V3 probe limits;
- geometric candidate policy and growth factor;
- same-workdir valid-only checkpoint policy;
- fallback as a bounded candidate only;
- max workers as an optional absolute ceiling.

Requested probe values are operational provenance rather than identity when normalization
produces the same effective limits. They are recorded with `identity_affecting=false`.

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
4. confirmation that oversized requests are normalized to effective `5000/120`;
5. confirmation that a failed higher candidate retains valid lower candidates;
6. confirmation that an interrupted plan reuses exact valid lower candidates only;
7. confirmation that one-click consumes an existing V3 selection without another plan;
8. confirmation that changing only an oversized requested value does not change identity;
9. no scientific branch launch during shadow.
