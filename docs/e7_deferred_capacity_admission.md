# E7 Deferred Capacity Admission

**Engineering claim:** `EXT-H-E7-DEFERRED-CAPACITY-ADMISSION-01`  
**Scientific experiment:** `EXT-H-E7-PPO-W0-EXP-GRID-01`  
**Scientific impact:** none

## Evidence that triggered this correction

The shared-host acceptance package produced on harness commit
`5a146292ff65011559470fe999e038c119f3b083` ended as `BLOCKED`, not `FAIL`.

The terminal evidence showed:

- E7 planned `96` workers;
- E7 affinity pool capacity `192` logical CPUs with an `0.85` budget of `163.2` cores;
- launch-time conservative pool occupancy `189.374` cores;
- launch-time admitted workers `0`;
- E8 had no safe CPU capacity for one GPU worker;
- no OOM, NaN/Inf, orphan process, affinity escape, checkout mutation, or repository contamination.

The result is therefore an environment-capacity observation. It is not evidence of a
new implementation defect, and it must not be converted into a false PASS by forcing a
minimum worker count when measured safe capacity is zero.

## Operational correction

The acceptance harness remains a one-shot engineering diagnostic. It is no longer a
manual prerequisite that must be repeatedly rerun before starting the E7 development
pilot.

The E7 automatic launcher now separates three states:

1. `planned`: immutable `RUNTIME_SELECTION.json`, `EXECUTION_PLAN.json`, and
   `RUN_IDENTITY.json` define the reviewed scientific matrix and the planned worker
   ceiling;
2. `waiting_for_capacity`: launch-time admission is zero or below an explicit operator
   floor, so the foreground launcher records the attempt and sleeps before a fresh
   measurement;
3. `admitted`: current safe capacity is positive and reaches that floor, so the launcher
   uses the admitted worker count as the executor width and starts the unchanged branch
   matrix.

The launcher never invents capacity, never starts when admission is zero, never changes
running worker width, and never modifies scientific branches, datasets, seeds, methods,
training horizon, model architecture, optimizer, or evaluation rules.

## Default one-click behavior

The one-click and resume launchers:

- wait in the foreground rather than exiting on a temporary zero-capacity observation;
- poll every `300` seconds by default;
- wait without an automatic deadline by default;
- start at any positive measured safe capacity by default;
- allow operator overrides only for wait duration, poll interval, and minimum admitted
  worker count.

Environment controls:

```text
E7_PPO_W0_CAPACITY_WAIT_TIMEOUT_SECONDS   default: -1 (unbounded foreground wait)
E7_PPO_W0_CAPACITY_POLL_SECONDS           default: 300
E7_PPO_W0_MINIMUM_ADMITTED_WORKERS        default: 1
```

A negative wait timeout means no automatic deadline. Zero preserves one-shot behavior.
A positive value is a finite wait budget in seconds. Raising the minimum admitted worker
count changes only the wall-clock scheduling floor; it does not alter branch identities
or scientific coordinates.

## Evidence and supervision

The work directory records:

```text
RUNTIME_CAPACITY_WAIT.json
RUNTIME_CAPACITY_WAIT.jsonl
_runtime_resource_attempts/attempt-*/RUNTIME_REVALIDATION.json
_runtime_resource_attempts/attempt-*/RUNTIME_ADMISSION.json
```

Every poll refreshes the machine snapshot and CPU/RAM observation. Identity,
checkout, binding, configuration, or other non-capacity failures remain immediately
fatal. SIGINT/SIGTERM interrupts the foreground wait normally; it does not launch work
or leave a background scheduler.

## Scientific boundary

This correction changes scheduling only. The experiment remains a development
screening pilot. The `500k` endpoint is not convergence, held-out seeds remain untouched,
and task-performance degradation, support/variance boundary events, and NaN/Inf
numerical failure remain separately reported.
