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

No running worker is resized. No scientific branch, seed, method, data, optimizer, training horizon, evaluation rule, threshold, or branch identity is changed. This is a one-time decision before launch.

The same distinction is enforced in both places that matter:

1. the acceptance liveness path launches the actually admitted count;
2. the real PPO w(0) E7 auto-runner retains the reviewed planned count in `EXECUTION_PLAN.json` and `RUN_IDENTITY.json`, while the canonical branch executor uses the lower attempt-local admitted width.

Therefore a capacity downshift changes only the number of simultaneously active subprocesses. It does not remove branches, alter branch identities, or change resume compatibility. `RUN_SUMMARY.json` records both `planned_max_workers` and `runtime_admitted_workers`.

## Reporting corrections

1. E7 stage evidence must report planned and admitted workers separately. The compatibility field `selected_workers` means the actually launched admitted count.
2. E8 evidence must report selected devices, slots per GPU, selected total runtime slots, and maximum actually probed concurrency separately. A selection of eight one-slot devices with only one concurrently probed worker remains `INCONCLUSIVE`, not a one-slot selection.
3. OOM classification must use explicit structured OOM evidence or exact OOM phrases. Substrings such as `headroom` must never trigger an OOM classification.
4. The repository-owned acceptance archive already contains a standard `sha256sum -c` manifest. Server delivery must return that archive directly and must not wrap it in a second package with a non-standard three-column manifest.

## Validation requirement

Before another server command is issued, the final PR head must pass:

- tiered test-plan shadow;
- Python compilation;
- shell syntax;
- handoff authority;
- formal execution-channel validation;
- governance inventory and stage status;
- full pytest;
- Ruff.

The exact validated head and workflow run belong in the PR review record rather than this versioned document, avoiding a self-referential commit identity.

Focused regression coverage includes:

- the observed 123-to-113 pre-launch CPU downshift;
- immutable `RUNTIME_SELECTION.json` bytes;
- zero safe capacity remaining blocked;
- non-capacity failures never being downshifted;
- planned E7 identity retained while the executor width uses the admitted count;
- E8 eight-slot selection reported separately from one-worker probe coverage;
- `headroom` not being misclassified as OOM.

## Terminal semantics

- positive admitted count and successful bounded liveness: `PASS` when more than one worker is actually exercised, otherwise `INCONCLUSIVE`;
- zero safe admitted workers: `BLOCKED`;
- identity, affinity, implementation, cleanup, OOM, or NaN/Inf failure: `FAIL`.

The previous server package remains preserved as failed engineering evidence. It is not a scientific result and is not reinterpreted as a successful acceptance.
