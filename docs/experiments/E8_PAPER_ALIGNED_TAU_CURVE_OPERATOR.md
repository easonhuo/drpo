# E8 paper-aligned tau curve operator contract

## Existing entrypoint

Reuse the existing one-click shell script. Do not add another launcher:

```bash
E8_ALPHA1_HIGHC_GRID_CONFIG=configs/countdown_e8_oracle_offline_v2_paper_aligned_tau_curve_0p5b.yaml \
E8_ALPHA1_HIGHC_WORK_DIR=outputs/e8/paper_aligned_tau_curve_001 \
bash scripts/run_countdown_e8_oracle_offline_v2_alpha1_highc_scan_auto_one_click.sh
```

The governed RunSpec is the preferred execution route after its source commit
has passed review and registration requirements.

## Required server inputs

The inherited defaults remain:

- model: `/root/models/Qwen2.5-0.5B-Instruct`;
- bank: `/root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl`;
- validation: `/root/experiment_output/e8_oracle_bank_v2/data/val.jsonl`;
- base config: `configs/countdown_e8_base_rl_replay_0p5b.yaml`.

Override them only through the already supported environment variables. Do not
change the frozen grid configuration.

## Expected execution shape

- 32 cells;
- GPU IDs `0-7`;
- two runtime slots per GPU;
- 16 concurrent cells in two waves;
- one mandatory representative liveness cell before the full sweep;
- identity-checked resume for completed cells.

## Required outputs

The work directory must contain:

- `SMOKE_GATE.json`;
- `SWEEP_PLAN.json`;
- `RUNTIME_SELECTION.json` and `RUNTIME_SLOTS.json`;
- per-cell summaries, metrics, diagnostics, and logs;
- aggregate CSV/JSON;
- `aggregate/terminal_audit.json`;
- `SWEEP_COMPLETE.json`.

The compact text evidence must be delivered to `easonhuo/drpo-results` on
`ingest/e8`. Local LoRA checkpoints remain server-local and are excluded from
the text delivery package.

## Stop conditions

Stop and preserve evidence when any of the following occurs:

- liveness failure;
- grid identity mismatch;
- stale result identity;
- missing expected cell;
- NaN/Inf loss, gradient, or parameter event;
- infrastructure failure that prevents terminal audit or durable delivery.

Do not reinterpret a partial sweep as a trend result and do not launch a
second-seed follow-up automatically.
