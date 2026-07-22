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

## Governed RunSpec launch

The canonical non-interactive E8 entry is the scoped RunSpec wrapper:

```bash
/root/.config/drpo-results/run_lane_scoped.sh \
  --lane e8 \
  --run-id E8_PAPER_ALIGNED_LINEAR_SCAN_20260716_01 \
  --once \
  --json
```

The RunSpec executor performs resource selection, plan, two-step smoke, the
resumable full sweep, artifact packaging, and automatic results-repository
delivery. For E7/E8, a missing `delivery` block defaults to:

```yaml
enabled: true
auto: true
mode: results_repo
repository: easonhuo/drpo-results
branch: ingest/e8
export_profile: manifest_text_v1
```

Successful completion must report `delivery_status=PASS` or the idempotent
`ALREADY_DELIVERED`, together with `results_commit`, `result_path`, and
`manifest_sha256`.

If training completes but delivery fails, do not rerun training. Retry only:

```bash
/root/.config/drpo-results/upload_result_scoped.sh \
  --run-id E8_PAPER_ALIGNED_LINEAR_SCAN_20260716_01 \
  --json
```

The RunSpec is single-use by `run_id`. Do not manually copy it back into
`runspecs/ready/` after the lane moves it.

## Underlying one-click launcher

The RunSpec entrypoint remains:

```bash
E8_ALPHA1_HIGHC_WORK_DIR=outputs/e8/paper_aligned_linear_scan_001 \
  bash scripts/run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto_one_click.sh
```

This lower-level launcher owns training only. Running it directly bypasses
RunSpec state transitions, artifact packaging, automatic upload, and deferred
registration closure. It is not the governed production entry and must not be
used to claim delivered completion.

The launcher requires all eight configured GPUs. Runtime remains two cells per
GPU, giving 16 concurrent cells and two waves for the 32-cell matrix.

The default model, bank and validation paths may be overridden only through the
existing `E8_ALPHA1_HIGHC_*` environment variables. Changing paths requires a new
identity and an empty/new work directory. Scientific variables, seeds, formula,
training budget and evaluation rules must not be overridden.

## Required launch evidence

Before treating the sweep as started, verify:

- `RUNTIME_SELECTION.json` lists exactly GPU 0 through GPU 7;
- `RUNTIME_SLOTS.json` reports `runtime_slots_per_gpu=2` and
  `total_runtime_slots=16`;
- `SWEEP_PLAN.json` reports 16 parameter points and 32 cells;
- `SMOKE_GATE.json` is `PASS` and `scientific_evidence=false`.

At completion, require `SWEEP_COMPLETE.json`, all 32 summaries,
`aggregate/terminal_audit.json`, and the immutable results-repository locator.
A smoke run, partial sweep, fixed 1200-step endpoint or best checkpoint alone is
not a formal result.
