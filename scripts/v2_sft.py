#!/usr/bin/env python3
"""v2 SFT with per-epoch checkpoints -> matrix B/C inits (GLM Dev Agent, one-off).

The A/B/C matrix's B init = ultra-low SFT (epoch_1) and C init = full-SFT
(best_adapter). The old matrix reused adapters SFT'd on the pilot_run_7 puzzle
set; for a faithful v2 re-run both inits must be SFT'd on the v2 corpus (same
puzzle set the v2 bank is built from) so init-level and bank/split are
coherent.

Uses the canonical `arena.cmd_sft` with `save_every_epoch=True` (opt-in, already
in the arena) and the capacity_diag reference-SFT hyperparameters (epochs=6,
min_epochs=3, lr=2e-4, grad_accum=32, micro_batch=2, seed=2026070700). SFT
train_data = the v2 bank file (rows have `prompt` + `oracle` = oracle_positive);
val_data = v2 val.jsonl. This is oracle SFT (positives only) — same SFT
procedure as the old matrix's B/C, only the puzzle set is v2.

Outputs: /root/experiment_output/e8_v2_sft/{epoch_1_adapter..epoch_6_adapter,
best_adapter, sft_metrics.csv}. B init = epoch_1_adapter, C init = best_adapter.
Diagnostic only — single-seed; not a method-ranking claim.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

DRPO_SRC = "/root/drpo/src/drpo"
sys.path.insert(0, DRPO_SRC)
import countdown_qwen_arena_onefile as arena  # noqa: E402

MODEL = "/root/models/Qwen2.5-0.5B-Instruct"
V2_BANK = "/root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl"
V2_VAL = "/root/experiment_output/e8_oracle_bank_v2/data/val.jsonl"
OUT = Path("/root/experiment_output/e8_v2_sft")


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")
    OUT.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        model_path=MODEL,
        train_data=V2_BANK,        # rows have prompt + oracle (=oracle_positive)
        val_data=V2_VAL,
        output_dir=str(OUT),
        seed=2026070700,
        max_length=256,
        max_new_tokens=80,
        epochs=6,
        min_epochs=3,
        early_stop_patience=2,
        parameterization="lora",
        micro_batch=2,
        grad_accum=32,
        lr=2.0e-4,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        num_workers=0,
        eval_examples=500,
        eval_batch=8,
        pass_k=8,
        eval_seed=2026070790,
        selection_metric="pass_at_k",
        selection_delta=0.0,
        log_every=20,
        load_in_4bit=False,
        dtype="bf16",
        result_status="pilot",
        save_every_epoch=True,
    )
    print(f"[v2_sft] cmd_sft save_every_epoch -> {OUT}", flush=True)
    t0 = time.time()
    arena.cmd_sft(args)
    print(f"[v2_sft] done in {time.time()-t0:.0f}s", flush=True)
    # report per-epoch greedy/pass@8 from sft_metrics.csv for B-init selection
    metrics_csv = OUT / "sft_metrics.csv"
    if metrics_csv.exists():
        print("[v2_sft] sft_metrics.csv:", flush=True)
        print(metrics_csv.read_text(), flush=True)
    print(f"[v2_sft] B init = {OUT}/epoch_1_adapter", flush=True)
    print(f"[v2_sft] C init = {OUT}/best_adapter", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())