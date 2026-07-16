# Runtime-resource acceptance server correction 07

**Claim:** `GOV-RUNTIME-RESOURCE-ACCEPTANCE-HARNESS-01`  
**Scientific impact:** none  
**Superseded server evidence:** shared-host acceptance from harness commit `5775f51e65cbae354554ec42c2e24e18ccad0d3a`

## Observed defect

The shared-host route correctly treated ResearchBench, AIDE, and joblib/loky as observation-only workloads and correctly measured residual capacity inside the declared E7 CPU pool. It proposed 123 workers during planning. Immediately before selected-count liveness, the measured pool-local capacity supported fewer workers, but the legacy immutable-selection rule rejected any worker-count change and classified the stage as `BLOCKED`.

That behavior does not implement the approved shared-host contract. The runtime worker count is an engineering concurrency limit, not a scientific coordinate. Before any E7 training branch starts, the final admitted worker count must be the minimum of:

- the planned worker count;
- the current pool-local CPU limit;
- the current memory limit;
- the configured hard cap and task limit already represented by the plan.

A positive lower count must be admitted and recorded instead of turning an otherwise runnable host into a capacity block. Zero safe workers remains `BLOCKED`.

## Corrected launch contract

The planned selection remains immutable provenance. Launch-time admission is a separate attempt-local artifact:

```text
RUNTIME_SELECTION.json       planned upper bound and selector evidence
RUNTIME_REVALIDATION.json    current CPU/RAM observation
RUNTIME_ADMISSION.json       final pre-launch admitted worker count
```

The admission artifact records:

- planned workers;
- admitted workers;
- whether a downshift occurred;
- CPU and memory worker limits;
- the exact revalidation evidence path;
- the unchanged selection digest;
- `scientific_matrix_changed=false`.

No running worker is resized. No scientific branch, seed, method, data, optimizer, training horizon, evaluation rule, threshold, or result identity is changed. This is a one-time decision before the bounded liveness launch.

## Reporting corrections

1. E7 stage evidence must report planned and admitted workers separately. The compatibility field `selected_workers` means the actually launched admitted count.
2. E8 evidence must report selected devices, slots per GPU, selected total runtime slots, and maximum actually probed concurrency separately. A selection of eight one-slot devices with only one concurrently probed worker remains `INCONCLUSIVE`, not a one-slot selection.
3. OOM classification must use explicit structured OOM evidence or exact OOM phrases. Substrings such as `headroom` must never trigger an OOM classification.
4. The repository-owned acceptance archive already contains a standard `sha256sum -c` manifest. Server delivery must return that archive directly and must not wrap it in a second package with a non-standard three-column manifest.

## Terminal semantics

- positive admitted count and successful bounded liveness: `PASS` when more than one worker is actually exercised, otherwise `INCONCLUSIVE`;
- zero safe admitted workers: `BLOCKED`;
- identity, affinity, implementation, cleanup, OOM, or NaN/Inf failure: `FAIL`.

The previous server package remains preserved as failed engineering evidence. It is not a scientific result and is not reinterpreted as a successful acceptance.
