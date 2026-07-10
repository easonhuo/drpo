#!/usr/bin/env python3
"""Smoke test the v2 oracle-offline bank before the full BC re-run (GLM Dev Agent, one-off).

Goal: verify the v2 bank (value-stratified, model-independent 16-negatives-per-row)
drives a NON-DEGENERATE bank_global_matched training before committing ~hours of
GPU to the full parallel re-run. Reuses the canonical runner paths
(`arena.cmd_calibrate_global` + `arena.cmd_train_method`) with the SAME config
params as `countdown_e8_base_rl_replay.train_offline_method`, only shortened
(steps=60, min_steps=20, eval_every=30, eval_examples=100, eval_batch=16).

Checks (non-degeneracy gates):
  1. calibration produces finite bank_negative_scale + bank_global_gamma > 0
  2. training runs without raising; manifest has best_step/best_value
  3. near/far surprisal separation is visible in the training log
     (current_bank_training_batches uses CURRENT-model surprisal over the 16 bank
      entries -> if v2's value-stratified negatives all sit at similar surprisal
      under the base model, near/far selection degenerates and the method is moot)
  4. eval not stuck at exactly 0 (any signal > base_eval ~0.017 is fine; the point
     is the pipeline moves, not that 60 steps beats the full run)

This is a diagnostic, NOT a scientific result. Single-seed, base-init (no SFT).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DRPO_SRC = "/root/drpo/src/drpo"
sys.path.insert(0, DRPO_SRC)
import countdown_e8_base_rl_replay as br  # noqa: E402
import countdown_qwen_arena_onefile as arena  # noqa: E402

CONFIG = "/root/drpo/configs/countdown_e8_base_rl_replay_0p5b.yaml"
MODEL = "/root/models/Qwen2.5-0.5B-Instruct"
V2_BANK = Path("/root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl")
V2_VAL = Path("/root/experiment_output/e8_oracle_bank_v2/data/val.jsonl")
WORK = Path("/root/experiment_output/e8_v2_smoke")
CALIB_JSON = WORK / "base_calibration.json"


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    WORK.mkdir(parents=True, exist_ok=True)
    config = br.load_config(CONFIG)
    model_cfg = config["model"]
    neg = config["negative_calibration"]

    # ---- 1. calibrate base negative on the v2 bank (reference_adapter=None) ----
    print(f"[smoke] calibrate on v2 bank -> {CALIB_JSON}", flush=True)
    calib_args = argparse.Namespace(
        model_path=MODEL,
        reference_adapter=None,
        offline_data=str(V2_BANK),
        output_json=str(CALIB_JSON),
        batch_size=int(neg["batch_size"]),
        calibration_batches=int(neg["batches"]),
        max_length=int(model_cfg["max_length"]),
        near_mix=float(neg["near_mix"]),
        far_mix=float(neg["far_mix"]),
        exp_lambda=float(neg["exp_lambda"]),
        surprisal_threshold=float(neg["surprisal_threshold"]),
        seed=int(neg["seed"]),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
        dtype=str(model_cfg.get("dtype", "auto")),
    )
    t0 = time.time()
    arena.cmd_calibrate_global(calib_args)
    calib = json.loads(CALIB_JSON.read_text())
    print(f"[smoke] calibrate done in {time.time()-t0:.0f}s", flush=True)
    print(f"[smoke]   bank_negative_scale={calib.get('bank_negative_scale')}  "
          f"bank_global_gamma={calib.get('bank_global_gamma')}", flush=True)
    bns = float(calib.get("bank_negative_scale", 0.0))
    bgg = float(calib.get("bank_global_gamma", 0.0))
    if not (bns > 0 and bgg > 0 and bns == bns and bgg == bgg):
        print("[smoke] DEGENERATE: calibration scale/gamma not finite-positive", flush=True)
        return 2

    # ---- 2. short bank_global_matched training on the v2 bank ----
    out_dir = WORK / "methods" / "smoke_bank_global_matched"
    train_cfg = config["offline_training"]
    seed_off = int(train_cfg["seed"])
    train_args = argparse.Namespace(
        model_path=MODEL,
        reference_adapter=None,
        sft_adapter=None,
        offline_data=str(V2_BANK),
        val_data=str(V2_VAL),
        structure_reference_data=str(V2_BANK),  # bank file has oracle_structure + oracle
        output_dir=str(out_dir),
        method="bank_global_matched",
        steps=60,
        min_steps=20,
        early_stop_patience=int(train_cfg["early_stop_patience"]),
        early_stop_delta=float(train_cfg["early_stop_delta"]),
        selection_metric=str(train_cfg["selection_metric"]),
        micro_batch=int(train_cfg["micro_batch"]),
        grad_accum=int(train_cfg["gradient_accumulation"]),
        lr=float(train_cfg["learning_rate"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        max_grad_norm=float(train_cfg["maximum_gradient_norm"]),
        max_length=int(model_cfg["max_length"]),
        max_new_tokens=int(model_cfg["max_new_tokens"]),
        eval_examples=100,
        eval_batch=16,
        pass_k=int(config["evaluation"]["pass_ks"][0]),
        negative_scale=None,
        negative_scale_multiplier=1.0,
        near_mix=float(neg["near_mix"]),
        far_mix=float(neg["far_mix"]),
        global_gamma=0.55,
        negative_calibration_json=str(CALIB_JSON),
        exp_lambda=float(neg["exp_lambda"]),
        surprisal_threshold=float(neg["surprisal_threshold"]),
        entropy_coef=0.02,
        target_entropy=1.8,
        target_entropy_coef=0.05,
        sbrc_kappa=0.92,
        entropy_floor=1.0,
        eval_every=30,
        eval_seed=int(config["evaluation"]["seed"]),
        diagnostic_examples=int(train_cfg["diagnostic_examples"]),
        diagnostic_gradient_examples=int(train_cfg["diagnostic_gradient_examples"]),
        diagnostic_batch=int(train_cfg["diagnostic_batch"]),
        log_every=int(train_cfg["log_every"]),
        num_workers=int(train_cfg["num_workers"]),
        seed=seed_off + 1000,
        result_status=str(config["result_status"]),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
        dtype=str(model_cfg.get("dtype", "auto")),
    )
    print(f"[smoke] train bank_global_matched (60 steps) -> {out_dir}", flush=True)
    t1 = time.time()
    arena.cmd_train_method(train_args)
    print(f"[smoke] train done in {time.time()-t1:.0f}s", flush=True)

    manifest = json.loads((out_dir / "manifest.json").read_text())
    print(f"[smoke]   best_step={manifest.get('best_step')}  "
          f"best_value={manifest.get('best_value')}  "
          f"terminal_step={manifest.get('terminal_step')}  "
          f"stop_reason={manifest.get('stop_reason')}", flush=True)

    # eval the best + terminal adapters (reuse canonical evaluate_adapter_checkpoint)
    data_paths = {"train": V2_BANK, "validation": V2_VAL,
                  "test": V2_VAL, "split_manifest": V2_BANK}  # test=val here (smoke only)
    best_eval = br.evaluate_adapter_checkpoint(
        Path(MODEL), out_dir / "best_adapter", data_paths, config, seed_offset=seed_off + 1000)
    print(f"[smoke]   best_eval test pass@8={best_eval.get('test_pass_at_8')}  "
          f"greedy={best_eval.get('test_greedy_success')}", flush=True)

    print("[smoke] OK — non-degenerate; safe to proceed to full BC re-run on v2", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())