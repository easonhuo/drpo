# Runtime-resource acceptance server correction 04

## Identity and authority

- Claim: `GOV-RUNTIME-RESOURCE-ACCEPTANCE-HARNESS-01`.
- User authorization: 2026-07-15, after confirming ResearchBench, AIDE, and the
  joblib/loky pool are permanent server workloads.
- Pre-correction harness head:
  `4c839e89dd02b75d34b9d1c64491d47e5dacebf7`.
- Scientific impact: none.
- Experiment status impact: none.

## Invalidated readiness assumption

The previous operator prompt required all ResearchBench, AIDE, and joblib/loky process
counts to reach zero before acceptance. That assumption is false on the target server:
the workloads are permanent. Repeating a process-count-zero gate would block forever.

Process existence alone is not the relevant safety predicate. The relevant predicate is
whether permanent external workloads can execute on the E7/E8 reserved CPUs.

## Corrected contract

The new route is opt-in and read-only with respect to system resource administration.
It requires the harness to already run inside a valid cgroup v2 cpuset partition whose
effective exclusive CPU set contains the union of the declared E7 and E8 pools.

The route:

1. automatically discovers the harness cgroup v2 path;
2. walks upward to a valid `root` or `isolated` cpuset partition;
3. verifies effective and exclusive CPU coverage;
4. inventories process cgroup paths and affinities;
5. permits permanent matched workloads only when they are outside the partition and
   have no reserved-CPU affinity overlap;
6. blocks any unrelated process found inside the acceptance partition;
7. writes structured partition evidence into Stage 0;
8. leaves all selector arithmetic, safety thresholds, worker counts, scientific inputs,
   seeds, steps, models, data, and evaluation unchanged.

## Non-goals

The repository does not create or modify cgroups, migrate unrelated tasks, kill
processes, renice workloads, alter their affinity, or choose permanent CPU allocations.
Server-administrator provisioning remains external and must be reviewed for the actual
systemd/container hierarchy.

## Required next execution

Do not use the old process-count-zero prompt. First run the new `--check-only` command
inside a pre-provisioned exclusive partition. Only when it returns
`exclusive_partition_proven=true` and `ready=true` may the partitioned one-click command
start the full engineering acceptance.

This remains engineering evidence only and cannot establish task performance, method
ranking, convergence, steady state, controlled mechanism identification, or OOD
generalization.
