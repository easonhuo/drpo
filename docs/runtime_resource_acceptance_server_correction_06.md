# Runtime-resource acceptance server correction 06

## Identity

- Claim: `GOV-RUNTIME-RESOURCE-ACCEPTANCE-HARNESS-01`.
- Pre-correction head: `ff40efd23945c74889b6b4ca405e010686463bc3`.
- Scientific impact: none.
- Experiment-status impact: none.

## Review finding

Correction 05 established the correct default model for the target server: bounded
shared-host execution with pool-local measured capacity, not exclusive CPU ownership.
During preparation of the final server-operator prompt, one remaining preflight boundary
was found to be too broad.

The first shared-host wrapper converted every configured process-pattern match into
observation-only evidence. The profile also contains patterns for real DRPO E7/E8 runs.
Consequently, another active DRPO acceptance or scientific run could have been treated the
same way as permanent ResearchBench/AIDE/loky workloads. That would permit duplicate DRPO
work to contend inside the same declared ceilings.

## Corrected process policy

Process handling is now split into two classes.

### Permanent external workloads: observation only

The following target-server workloads may remain alive and do not block merely because
of their names:

- `ResearchBench`;
- `collector_v2.py`;
- `AIDE`;
- `joblib.externals.loky`;
- `loky.process_executor`.

They are discovered from the full Stage-0 process inventory even when they are not listed
in `conflict_process_patterns`. Their actual CPU use remains accounted for by pool-local
capacity measurement.

### Competing DRPO workloads: blocking conflicts

Configured DRPO patterns such as `run_e7_`, `countdown_e8`, `EXT-H-E7`, and `EXT-C-E8`
remain blocking conflicts. The harness does not kill or modify them. It stops before
starting a second overlapping DRPO acceptance or experiment.

When both classes are present, Stage 0 records the permanent external processes as
observation-only evidence and remains `BLOCKED` because of the competing DRPO process.

## Unchanged contract

This correction does not change:

- E7/E8 CPU pools or GPU IDs;
- worker, device, slot, thread, headroom, or safety-factor limits;
- selector arithmetic or launch revalidation;
- scientific inputs, variables, seeds, steps, data, models, or evaluation;
- cgroup state or unrelated process affinity;
- status semantics outside this process-class split.

The default operator route remains:

```bash
bash scripts/run_runtime_resource_acceptance_one_click.sh \
  --profile /absolute/path/runtime_resource_acceptance_server.json
```

The exclusive-partition route remains optional history and is not required on the target
server.
