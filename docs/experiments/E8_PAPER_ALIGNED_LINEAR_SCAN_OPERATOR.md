# E8 Paper-Aligned Linear Scan — Operator Launch

Experiment ID:
`EXT-C-E8-ORACLE-OFFLINE-V2-PAPER-ALIGNED-LINEAR-SCAN-0.5B-01`

Status: code-first development pilot; not yet run.

## Canonical source

- implementation commit: `ed3f132d63a378ef272126c0cd1e6fddd6b0d5c5`;
- RunSpec: `runspecs/ready/E8_PAPER_ALIGNED_LINEAR_SCAN_20260716_01.yaml`;
- protocol: `docs/experiments/E8_PAPER_ALIGNED_LINEAR_SCAN_PROTOCOL.md`.

Do not use the removed `run_countdown_e8_paper_aligned_lambda_*` launchers from
the superseded 18-cell plan.

## Direct launch

From a clean checkout containing the implementation commit:

```bash
E8_ALPHA1_HIGHC_WORK_DIR=outputs/e8/paper_aligned_linear_scan_001 \
  bash scripts/run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto_one_click.sh
```

The launcher performs resource selection, plan, two-step smoke, and the resumable
full sweep. It requires all eight configured GPUs. Runtime remains two cells per
GPU, giving 16 concurrent cells and two waves for the 32-cell matrix.

The default model, bank and validation paths may be overridden only through the
existing `E8_ALPHA1_HIGHC_*` environment variables. Changing paths requires a new
identity and an empty/new work directory. Scientific variables, seeds, formula,
training budget and evaluation rules must not be overridden.

## RunSpec lane

After the E8 lane is configured, the canonical executor command remains:

```bash
python scripts/agent/run_lane.py --once
```

The RunSpec is single-use by `run_id`. Do not manually copy it back into
`runspecs/ready/` after the lane moves it.

## Required launch evidence

Before treating the sweep as started, verify:

- `RUNTIME_SELECTION.json` lists exactly GPU 0 through GPU 7;
- `RUNTIME_SLOTS.json` reports `runtime_slots_per_gpu=2` and
  `total_runtime_slots=16`;
- `SWEEP_PLAN.json` reports 16 parameter points and 32 cells;
- `SMOKE_GATE.json` is `PASS` and `scientific_evidence=false`.

At completion, require `SWEEP_COMPLETE.json`, all 32 summaries and
`aggregate/terminal_audit.json`. A smoke run, partial sweep, fixed 1200-step
endpoint or best checkpoint alone is not a formal result.
