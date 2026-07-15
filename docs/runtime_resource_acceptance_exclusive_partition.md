# Exclusive cgroup v2 partition acceptance

**Claim:** `GOV-RUNTIME-RESOURCE-ACCEPTANCE-HARNESS-01`  
**Scientific impact:** none  
**Default-policy impact:** none; this is an opt-in target-server acceptance route

## Why this route exists

The target server permanently runs ResearchBench, AIDE, and joblib/loky workers.
Process-count-zero is therefore not a valid readiness condition. Shared-host spare CPU
is also insufficient evidence because those permanent workloads may consume the same
CPUs during E7/E8 probes.

The partitioned route accepts permanent external workloads only when the kernel exposes
a valid cgroup v2 cpuset partition containing every declared E7/E8 CPU. A valid cpuset
partition gives the partition exclusive access to its exclusive CPU set; tasks outside
the partition cannot use those CPUs.

The harness does not create, modify, or delete cgroups. It does not migrate, kill,
renice, or rebind unrelated processes. Partition provisioning remains an explicit
server-administration action outside the acceptance run.

Authoritative kernel interface reference:

```text
https://docs.kernel.org/admin-guide/cgroup-v2.html
```

Relevant files are `cpuset.cpus.partition`, `cpuset.cpus.effective`, and
`cpuset.cpus.exclusive.effective`.

## Required server state

Before launch, a server administrator must provide a cgroup v2 partition root that:

1. reports `root` or `isolated` from `cpuset.cpus.partition`, without an invalid suffix;
2. includes the union of the profile's E7 and E8 CPU pools in both
   `cpuset.cpus.effective` and `cpuset.cpus.exclusive.effective`;
3. contains the foreground operator shell and harness process;
4. contains no unrelated process beyond the harness ancestor chain;
5. leaves ResearchBench, AIDE, and joblib/loky processes outside the partition;
6. exposes no outside process affinity overlap with the reserved E7/E8 CPUs.

Do not guess generic cgroup-creation commands on an unknown systemd/container layout.
Provisioning must be reviewed against the server's actual cgroup hierarchy.

## Read-only check

From inside the candidate partition:

```bash
python3 scripts/run_runtime_resource_acceptance_partitioned.py \
  --profile /absolute/path/runtime_resource_acceptance_server.json \
  --check-only
```

Exit code zero requires:

```text
exclusive_partition_proven = true
ready = true
```

The JSON report records:

- current cgroup v2 path;
- discovered partition root and state;
- effective and exclusive CPU sets;
- reserved E7/E8 CPU union;
- permanent external matches proven outside the partition;
- any unresolved affinity overlap;
- any contaminating process inside the partition.

The check is read-only.

## One-command acceptance

After the read-only check passes:

```bash
bash scripts/run_runtime_resource_acceptance_partitioned_one_click.sh \
  --profile /absolute/path/runtime_resource_acceptance_server.json
```

The shell command reruns the check immediately before acceptance. Stage 0 writes
`EXCLUSIVE_PARTITION_AUDIT.json` into the normal acceptance artifact.

The permanent external patterns checked in addition to the profile are:

```text
ResearchBench
collector_v2.py
AIDE
joblib.externals.loky
loky.process_executor
```

Their presence is allowed only when partition evidence proves that they are outside the
exclusive partition and their visible affinity has no reserved-CPU overlap.

## Status semantics

- no valid exclusive partition: `BLOCKED`;
- reserved CPUs not fully effective and exclusive: `BLOCKED`;
- unrelated process inside the acceptance partition: `BLOCKED`;
- outside permanent process with reserved-CPU affinity overlap: `BLOCKED`;
- permanent external processes outside a valid partition with no overlap: not a conflict;
- code, identity, cleanup, OOM, or numerical errors retain their existing `FAIL` rules.

A partition proof does not lower CPU/RAM/GPU headroom, worker-count immutability, or any
scientific gate. Measured capacity and runtime revalidation still execute unchanged.

## Rollback

Stop using the partitioned entrypoint and retain all generated evidence. The original
shared-host entrypoint remains unchanged. No scientific experiment state changes.
