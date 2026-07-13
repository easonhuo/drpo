# EXT-H-E7-PPO-W0-EXP-GRID-01 execution notes

## Code-first sequence

The implementation was pushed before the full authority materialization at the user's explicit request so the server path can be prepared without waiting for the complete governance pass. This does not authorize scientific launch from the dev branch.

## Current command

After the RunSpec is pinned and promoted to `runspecs/ready/`, the intended one-click command is:

```bash
bash scripts/run_e7_ppo_w0_grid_pilot_auto_one_click.sh
```

Default external inputs remain:

```text
/root/d4rl2/configs/e7_canonical_contract_9task.json
/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json
```

Operators may override paths through `E7_CANONICAL_CONTRACT` and `E7_CANONICAL_RUN_SPEC`. The scientific grid file defaults to `configs/e7_ppo_w0_exp_grid_pilot_v1.json` and must not be modified during execution.

## Runtime provenance

The launcher writes `RUNTIME_SELECTION.json` before the scientific run identity. It records the machine snapshot, memory probe, bounded throughput candidates, failures, peak throughput, 97% retained-peak threshold, and selected workers. A resumed work directory must reuse the exact worker count fixed by `RUN_IDENTITY.json`.

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

- a resource candidate that times out or fails is not scientific evidence;
- no safe successful candidate blocks launch;
- a scientific branch failure leaves `FAILED.json` and blocks automatic aggregation;
- task degradation is not relabeled as NaN/Inf failure;
- absence of a registered support/variance boundary instrument is reported as unavailable;
- 500k is a screening endpoint, not convergence.
