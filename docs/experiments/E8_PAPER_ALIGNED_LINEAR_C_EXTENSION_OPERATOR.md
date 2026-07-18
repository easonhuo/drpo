# E8 Paper-Aligned Linear-C Extension — Operator Launch

Experiment ID:
`EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LINEAR-C-EXTENSION-0.5B-01`

Status: code-first development pilot; not yet run.

## Canonical files

- config:
  `configs/countdown_e8_oracle_offline_v2_linear_c_extension_0p5b.yaml`;
- scope:
  `docs/scopes/EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LINEAR-C-EXTENSION-0.5B-01.md`;
- protocol:
  `docs/experiments/E8_PAPER_ALIGNED_LINEAR_C_EXTENSION_PROTOCOL.md`;
- RunSpec:
  `runspecs/ready/E8_PAPER_ALIGNED_LINEAR_C_EXTENSION_20260717_01.yaml`.

## Direct launch

The existing launcher is reused with an explicit grid-config override:

```bash
E8_ALPHA1_HIGHC_WORK_DIR=outputs/e8/paper_aligned_linear_c_extension_001 \
E8_ALPHA1_HIGHC_GRID_CONFIG=configs/countdown_e8_oracle_offline_v2_linear_c_extension_0p5b.yaml \
  bash scripts/run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto_one_click.sh
```

The full sweep contains exactly 16 cells and should occupy one wave across GPU
`0-7` with two cells per GPU. No Positive-only cell is launched.

## Required preflight evidence

Before treating training as started, require:

- `RUNTIME_SELECTION.json` lists GPU `0-7`;
- `RUNTIME_SLOTS.json` reports two slots per GPU and 16 total slots;
- `SWEEP_PLAN.json` reports 8 parameter points and 16 cells;
- every planned cell has `alpha=1.0` and one of the eight frozen `c` values;
- no planned cell has method `positive_only` or `global`;
- `SMOKE_GATE.json` is `PASS` and `scientific_evidence=false`.

## RunSpec lane

Execute through the E8 lane with:

```bash
python scripts/agent/run_lane.py \
  --lane e8 \
  --run-id E8_PAPER_ALIGNED_LINEAR_C_EXTENSION_20260717_01 \
  --once \
  --json
```

The RunSpec enables automatic text-first delivery to
`easonhuo/drpo-results`, branch `ingest/e8`. Delivery failure must not trigger a
training rerun; use the canonical manual uploader after repairing credentials.

## Completion audit

Require:

- `SWEEP_COMPLETE.json`;
- 16 cell summaries and metric CSVs;
- `aggregate/per_cell_summary.csv`;
- `aggregate/terminal_audit.json`;
- zero undeclared test access;
- separate task-performance, valid-structure/support-proxy and NaN/Inf reports;
- successful or explicitly downgraded result-repository delivery status.

Do not relaunch into the predecessor work directory. Do not edit the c list,
seeds, horizon, alpha, bank or evaluation rule on the server.
