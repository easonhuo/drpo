# E8 3B DRPO `(c,tau)` Scale-Transfer Operator Guide

Experiment ID: `EXT-C-E8-DRPO-CTAU-SCALE-TRANSFER-3B-01`

Run ID: `E8_DRPO_CTAU_SCALE_TRANSFER_3B_20260723_01`

Status: implementation prepared; full pilot not yet run.

## Canonical files

- profile/runtime: `src/drpo/countdown_e8_drpo_ctau_scale_transfer_3b.py`;
- base config: `configs/countdown_e8_base_rl_replay_3b.yaml`;
- grid config: `configs/countdown_e8_drpo_ctau_scale_transfer_3b.yaml`;
- scope: `docs/scopes/EXT-C-E8-DRPO-CTAU-SCALE-TRANSFER-3B-01.md`;
- protocol: `docs/experiments/E8_DRPO_CTAU_SCALE_TRANSFER_3B_PROTOCOL.md`;
- RunSpec: `runspecs/ready/E8_DRPO_CTAU_SCALE_TRANSFER_3B_20260723_01.yaml`.

## Canonical launch

Execute only through the governed E8 lane after the RunSpec is registered and
reviewed:

```bash
python scripts/agent/run_lane.py \
  --lane e8 \
  --run-id E8_DRPO_CTAU_SCALE_TRANSFER_3B_20260723_01 \
  --once \
  --json
```

Do not directly launch the module for the scientific pilot. Direct module use is
limited to static plan inspection and local tests because it bypasses RunSpec
claim, packaging, and durable delivery.

## Required server inputs

The RunSpec uses these default server paths:

```text
model: /root/models/Qwen2.5-3B-Instruct
bank:  /root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl
val:   /root/experiment_output/e8_oracle_bank_v2/data/val.jsonl
```

A different absolute model location is allowed only when its `config.json`
passes the exact Qwen2.5-3B-Instruct identity gate. The bank and validation files
must match the registered frozen identities; replacing their content is not a
path override.

## Preflight evidence

Before treating training as started, require:

- clean source checkout at or descended from the RunSpec implementation SHA;
- `RUNTIME_SELECTION.json` selects exactly GPU `0,1,2,3`;
- `RUNTIME_SELECTION.json` identifies the 3B transfer adapter;
- `SWEEP_PLAN.json` lists exactly four `(c,tau)` points and eight cells;
- every plan row records label, `c`, `tau`, and seed offset;
- resource contract is four GPUs, one slot per GPU, four concurrent cells, two waves;
- `SMOKE_GATE.json` is `PASS` and `scientific_evidence=false`;
- model identity is Qwen2.5-3B-Instruct;
- separate test split is unused.

## Completion audit

Require:

- `SWEEP_COMPLETE.json`;
- eight cell summaries and metric CSVs;
- `RUNTIME_SLOTS.json` with `runtime_slots_per_gpu=1`, `gpu_count=4`, and
  `total_runtime_slots=4`;
- aggregate per-cell CSV and terminal audit;
- late-window and terminal reporting for A-D on both seeds;
- best-checkpoint metrics marked supplementary only;
- task performance, valid-expression/structure diagnostics, and NaN/Inf reported separately;
- automatic delivery status for `easonhuo/drpo-results@ingest/e8`;
- no model, adapter, checkpoint, optimizer, or dataset-like file in the remote review package.

The two waves must use the same code commit, model identity, bank, validation
split, base config, grid config, and scientific matrix. Identity-checked resume
may skip a completed matching cell but must reject stale or mismatched output.

## Delivery failure handling

A delivery failure after successful computation must not trigger retraining.
Repair only the results-repository credential or network path, then run:

```bash
python scripts/agent/upload_runspec_result.py \
  --run-id E8_DRPO_CTAU_SCALE_TRANSFER_3B_20260723_01 \
  --json
```

Do not edit the four points, seeds, model identity, horizon, bank, evaluation
window, one-slot allocation, work directory, or delivery destination on the
server.
