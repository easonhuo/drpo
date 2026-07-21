# E8 AsymRE Delta-V Scan — Operator Launch

Experiment ID:
`EXT-C-E8-ORACLE-OFFLINE-V2-ASYMRE-DELTAV-SCAN-0.5B-01`

Run ID:
`E8_ASYMRE_DELTAV_SCAN_20260721_01`

Status: code-first development pilot; not yet run.

## Canonical files

- config: `configs/countdown_e8_oracle_offline_v2_asymre_deltav_scan_0p5b.yaml`;
- scope: `docs/scopes/EXT-C-E8-ORACLE-OFFLINE-V2-ASYMRE-DELTAV-SCAN-0.5B-01.md`;
- protocol: `docs/experiments/E8_ASYMRE_DELTAV_SCAN_PROTOCOL.md`;
- RunSpec: `runspecs/ready/E8_ASYMRE_DELTAV_SCAN_20260721_01.yaml`.

## Canonical launch

Execute through the E8 lane:

```bash
python scripts/agent/run_lane.py \
  --lane e8 \
  --run-id E8_ASYMRE_DELTAV_SCAN_20260721_01 \
  --once \
  --json
```

Do not bypass the RunSpec with a direct shell launch for the real pilot. The RunSpec
binds provenance, recovery, packaging, and automatic result delivery.

## Required preflight evidence

Before treating training as started, require:

- `RUNTIME_SELECTION.json` lists GPU `0-7`;
- `RUNTIME_SLOTS.json` reports two slots per GPU and 16 total slots;
- `SWEEP_PLAN.json` reports 8 `delta_v` points and 16 cells;
- every planned cell has method `asymre` and one frozen `delta_v` value;
- `SMOKE_GATE.json` is `PASS` and `scientific_evidence=false`;
- the split role is recorded as structurally disjoint held-out evaluation;
- no value network, remoteness taper, or unregistered coefficient appears.

## Completion audit

Require:

- `SWEEP_COMPLETE.json`;
- 16 cell summaries and metric CSVs;
- aggregate per-cell summary and terminal audit;
- late-window and terminal reporting for all eight points;
- best-checkpoint metrics marked supplementary only;
- task performance, valid-expression/structure diagnostics, and NaN/Inf separated;
- automatic delivery status for `easonhuo/drpo-results@ingest/e8`;
- no model, adapter, checkpoint, optimizer, or dataset-like file in the remote review package.

## Delivery failure handling

A delivery failure does not invalidate successful training and must not trigger a rerun.
Repair the repository-scoped credential or network path, then invoke only:

```bash
python scripts/agent/upload_runspec_result.py \
  --run-id E8_ASYMRE_DELTAV_SCAN_20260721_01 \
  --json
```

Identical retries must return `ALREADY_DELIVERED`; conflicting content must fail closed.

Do not edit the `delta_v` list, seeds, horizon, bank, evaluation window, work directory,
or delivery destination on the server.
