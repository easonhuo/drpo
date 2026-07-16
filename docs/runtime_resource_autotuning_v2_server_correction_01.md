# Measured-CPU V2 server correction 01

**Claim:** `GOV-RUNTIME-RESOURCE-AUTOTUNE-CPU-V2-01`

A target-server acceptance run of the stacked runtime-resource harness exposed a
PPO w(0) plan/run lifecycle mismatch. Measured CPU/RAM selection and bounded
throughput candidates completed successfully, but the plan wrapper failed after
the delegated plan wrote `EXECUTION_PLAN.json`: the wrapper incorrectly assumed
that the delegated plan also wrote `RUN_IDENTITY.json`.

The delegated canonical runner creates run identity during its `run` command,
not its `plan` command. Measured-CPU V2 nevertheless requires the selected
worker count and immutable selection digest to be bound before revalidation or
run.

The correction is therefore narrow:

1. after a successful measured plan, load the just-written
   `EXECUTION_PLAN.json` when `RUN_IDENTITY.json` is absent;
2. compute the run identity using the same stable-plan rule as the delegated run
   path: exclude only `created_utc`, then canonical-JSON SHA-256;
3. write `RUN_IDENTITY.json` atomically;
4. bind exact `selected_workers` and `selection_digest`;
5. preserve the existing run-time identity and revalidation gates.

This does not change selector arithmetic, candidate grids, affinity/cgroup
measurement, CPU/RAM reservations, load-average semantics, seeds, steps,
scientific matrices, data, configs, thresholds, or default activation.

The original target-server run remains a failed engineering acceptance record.
This correction requires exact-head CI and a fresh server acceptance before the
PR can be approved or merged.