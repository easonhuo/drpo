# EXT-H-E7-PPO-W0-EXP-GRID-01 execution notes

## Code-first sequence

The implementation was pushed before full authority materialization at the user's explicit request. The sequence is:

1. GitHub CI and code review on the protected dev-branch implementation;
2. immediate real-data liveness on the server;
3. after liveness passes, launch the complete 186-branch development screening pilot so compute overlaps the slower registry/handoff process;
4. finish authoritative registration, ready-RunSpec promotion, terminal review, and merge governance in parallel.

A dev-branch run remains `pilot / formal_evidence_allowed=false`. It may provide provisional screening data, but it may not be cited as formal evidence or upgraded to a stable ranking until provenance and terminal audit are accepted.

## Current registration and execution state

- The schema-v3 authority update `EXT-H-E7-PPO-W0-EXP-GRID-REGISTRATION-2026-07-13` has materialized the handoff block and added the complete registry entity on this Draft PR branch.
- Registry status is `pilot`, result status is `running`, and no terminal result is currently registered.
- The user reported that the server run has started from implementation commit `d8bb6141092969a7daccb42b87c4f2da6e8371c6`.
- This running state does not certify liveness acceptance, branch completion, convergence, stability, or a method ranking.
- Held-out seeds `204--207` remain reserved and must not be consumed by this development screening run.

## Server commands

After checking out `dev/e7-ppo-w0-grid-pilot`, run the non-scientific liveness gate first:

```bash
bash scripts/run_e7_ppo_w0_grid_liveness_one_click.sh
```

This performs only representative 500-update resource probes and materializes the deterministic 186-branch plan. It does not launch scientific branches.

After `RUNTIME_SELECTION.json` and `EXECUTION_PLAN.json` are reviewed and the liveness gate passes, launch the complete development pilot:

```bash
bash scripts/run_e7_ppo_w0_grid_pilot_auto_one_click.sh
```

For an interrupted run whose `RUNTIME_SELECTION.json` and `RUN_IDENTITY.json` are intact, use:

```bash
bash scripts/run_e7_ppo_w0_grid_pilot_resume_one_click.sh
```

Default external inputs remain:

```text
/root/d4rl2/configs/e7_canonical_contract_9task.json
/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json
```

Operators may override paths through `E7_CANONICAL_CONTRACT` and `E7_CANONICAL_RUN_SPEC`. The scientific grid defaults to `configs/e7_ppo_w0_exp_grid_pilot_v1.json` and must not be modified during execution.

## Runtime provenance

The launcher writes `RUNTIME_SELECTION.json` before the scientific run identity. It records the machine snapshot, representative peak-RSS probe, bounded throughput candidates, failures, peak throughput, 97% retained-peak threshold, and selected workers. A resumed work directory must reuse the exact worker count fixed by `RUN_IDENTITY.json`.

## Outputs

```text
outputs/e7/ppo_w0_exp_grid_pilot_001/
  RUNTIME_SELECTION.json
  EXECUTION_PLAN.json
  RUN_IDENTITY.json
  RUN_SUMMARY.json
  branches/*
  aggregate/per_branch_summary.csv
  aggregate/grid_summary.csv
  aggregate/aggregate_summary.json
  aggregate/terminal_audit.json
```

Probe payloads live below `_runtime_resource_probe/`; generated trainer/checkpoint payload is removed after each probe, while small benchmark summaries and logs remain as runtime evidence.

## Failure semantics

- a resource candidate that times out or fails is runtime evidence only, not scientific evidence;
- no safe successful candidate blocks launch;
- a scientific branch failure leaves `FAILED.json`, blocks successful aggregation, and writes a failing `aggregate/terminal_audit.json` before re-raising;
- task degradation is not relabeled as NaN/Inf failure;
- absence of a registered support/variance boundary instrument is reported as unavailable;
- 500k is a screening endpoint, not convergence.
