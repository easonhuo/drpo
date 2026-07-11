#!/usr/bin/env python3
"""Smoke-test the v2 oracle-offline bank before any full training (GLM Dev Agent, one-off).

Goal: verify the v2 bank (value-stratified, model-independent 16-negatives-per-row)
drives a NON-DEGENERATE ``bank_global_matched`` training before committing ~hours
of GPU to a full parallel re-run. Reuses the canonical runner paths
(``arena.cmd_calibrate_global`` + ``arena.cmd_train_method``) with the SAME config
params as ``countdown_e8_base_rl_replay.train_offline_method``, only shortened
(steps=60, min_steps=20, eval_every=30, eval_examples=100, eval_batch=16).

NON-DEGENERACY GATES (real asserts, fail-closed — no unconditional "OK"):

  G1  calibration produced finite bank_negative_scale > 0
  G2  calibration produced finite bank_global_gamma > 0
  G3  training ran without raising and wrote manifest.json with best_step>=0
  G4  best_adapter directory + adapter_config.json exist on disk
  G5  terminal or last_finite adapter exists (training produced a checkpoint beyond init)
  G6  dynamic_diagnostics.jsonl exists and is non-empty (current-policy reselection ran)
  G7  bank_current_far_surprisal_mean is finite and > bank_current_near_surprisal_mean
      (near=argmin, far=argmax CURRENT-model surprisal; if far<=near the v2 value
       stratification has collapsed under the base model and the bank method is moot)
  G8  best-adapter eval produced finite test greedy_success / pass_at_8 / valid_rate
  G9  no NaN/Inf anywhere in calibration scale/gamma, manifest best_value, or eval metrics
  G10 calibration scale is NOT silently reused across inits (record reference_adapter=None;
       this is the base-init calibration only — the v2 init-matrix routes low_sft
       calibration to a separate reference_adapter=epoch_1)

This is a diagnostic, NOT a scientific result. Single-seed, base-init (no SFT).

A machine-readable ``SMOKE_GATE.json`` is written with every gate's PASS/FAIL, the
source commit + corpus/calibration hashes, the smoke config, and
``result_status=smoke_not_scientific_result``. Any gate FAIL exits non-zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
GATE_JSON = WORK / "SMOKE_GATE.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", "/root/drpo", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _is_finite(value: object) -> bool:
    if value is None:
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _gate(record: dict, name: str, condition: bool, detail: str) -> bool:
    record["gates"][name] = {"status": "PASS" if condition else "FAIL", "detail": detail}
    return bool(condition)


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    WORK.mkdir(parents=True, exist_ok=True)
    config = br.load_config(CONFIG)
    model_cfg = config["model"]
    neg = config["negative_calibration"]

    record: dict = {
        "smoke_id": "v2_oracle_offline_bank_smoke",
        "result_status": "smoke_not_scientific_result",
        "source_commit": _git_commit(),
        "config_path": CONFIG,
        "model_path": MODEL,
        "v2_bank_sha256": _sha256(V2_BANK),
        "v2_val_sha256": _sha256(V2_VAL),
        "calibration_json": str(CALIB_JSON),
        "smoke_config": {
            "steps": 60, "min_steps": 20, "eval_every": 30,
            "eval_examples": 100, "eval_batch": 16,
            "method": "bank_global_matched", "init": "base", "reference_adapter": None,
        },
        "gates": {},
        "calibration": {},
        "manifest": {},
        "eval": {},
    }
    all_pass = True

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
    bns = calib.get("bank_negative_scale")
    bgg = calib.get("bank_global_gamma")
    print(f"[smoke]   bank_negative_scale={bns}  bank_global_gamma={bgg}", flush=True)
    record["calibration"] = {
        "bank_negative_scale": bns,
        "bank_global_gamma": bgg,
        "negative_scale": calib.get("negative_scale"),
        "global_gamma": calib.get("global_gamma"),
        "reference_adapter": calib.get("reference_adapter"),
        "batches": calib.get("batches"),
        "frozen_before_method_training": calib.get("frozen_before_method_training"),
    }
    all_pass &= _gate(record, "G1_calibration_scale_finite_positive",
                     _is_finite(bns) and float(bns) > 0.0, f"bank_negative_scale={bns}")
    all_pass &= _gate(record, "G2_calibration_gamma_finite_positive",
                     _is_finite(bgg) and float(bgg) > 0.0, f"bank_global_gamma={bgg}")

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

    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        all_pass &= _gate(record, "G3_manifest_written", False, "manifest.json missing")
        all_pass &= _gate(record, "G4_best_adapter_present", False, "no manifest")
        all_pass &= _gate(record, "G5_terminal_or_last_finite_adapter", False, "no manifest")
        all_pass &= _gate(record, "G6_dynamic_diagnostics_written", False, "no manifest")
        all_pass &= _gate(record, "G7_far_surprisal_gt_near", False, "no manifest")
        all_pass &= _gate(record, "G8_eval_metrics_finite", False, "no manifest")
        all_pass &= _gate(record, "G9_no_nan_inf", False, "no manifest")
        all_pass &= _gate(record, "G10_base_init_calibration_isolated", False, "no manifest")
        record["overall"] = "FAIL"
        GATE_JSON.write_text(json.dumps(record, indent=2))
        print("[smoke] FAIL — see SMOKE_GATE.json", flush=True)
        return 2

    manifest = json.loads(manifest_path.read_text())
    record["manifest"] = {
        "best_step": manifest.get("best_step"),
        "best_value": manifest.get("best_value"),
        "terminal_step": manifest.get("terminal_step"),
        "stop_reason": manifest.get("stop_reason"),
        "numerical_failure": manifest.get("numerical_failure"),
        "negative_scale": manifest.get("negative_scale"),
        "global_matched_gamma": manifest.get("global_matched_gamma"),
        "dynamic_diagnostics_path": manifest.get("dynamic_diagnostics_path"),
    }
    print(f"[smoke]   best_step={manifest.get('best_step')}  "
          f"best_value={manifest.get('best_value')}  "
          f"terminal_step={manifest.get('terminal_step')}  "
          f"stop_reason={manifest.get('stop_reason')}", flush=True)

    best_step = manifest.get("best_step")
    all_pass &= _gate(record, "G3_manifest_written",
                     _is_finite(best_step) and float(best_step) >= 0.0,
                     f"best_step={best_step}")

    best_dir = out_dir / "best_adapter"
    best_cfg = best_dir / "adapter_config.json"
    all_pass &= _gate(record, "G4_best_adapter_present",
                     best_dir.is_dir() and best_cfg.exists(),
                     f"best_adapter dir + adapter_config.json at {best_dir}")

    terminal_dir = out_dir / "terminal_adapter"
    last_finite_dir = out_dir / "last_finite_adapter"
    has_post_init_ckpt = (
        (terminal_dir.is_dir() and (terminal_dir / "adapter_config.json").exists())
        or (last_finite_dir.is_dir() and (last_finite_dir / "adapter_config.json").exists())
    )
    all_pass &= _gate(record, "G5_terminal_or_last_finite_adapter",
                     has_post_init_ckpt,
                     f"terminal_adapter={terminal_dir.is_dir()} "
                     f"last_finite_adapter={last_finite_dir.is_dir()}")

    diag_path = out_dir / "dynamic_diagnostics.jsonl"
    diag_lines = [ln for ln in diag_path.read_text().splitlines() if ln.strip()] if diag_path.exists() else []
    all_pass &= _gate(record, "G6_dynamic_diagnostics_written",
                     diag_path.exists() and len(diag_lines) >= 1,
                     f"dynamic_diagnostics.jsonl lines={len(diag_lines)}")

    # G7: current-model far surprisal > near surprisal (the v2 non-degeneracy core)
    far_mean = near_mean = None
    for ln in diag_lines:
        row = json.loads(ln)
        if _is_finite(row.get("bank_current_far_surprisal_mean")):
            far_mean = float(row["bank_current_far_surprisal_mean"])
        if _is_finite(row.get("bank_current_near_surprisal_mean")):
            near_mean = float(row["bank_current_near_surprisal_mean"])
        if far_mean is not None and near_mean is not None:
            break
    cond_g7 = (
        _is_finite(far_mean) and _is_finite(near_mean)
        and far_mean > near_mean
    )
    all_pass &= _gate(record, "G7_far_surprisal_gt_near", cond_g7,
                      f"bank_current_far_surprisal_mean={far_mean} "
                      f"bank_current_near_surprisal_mean={near_mean}")

    # ---- 3. eval the best adapter (reuse canonical evaluate_adapter_checkpoint) ----
    data_paths = {"train": V2_BANK, "validation": V2_VAL,
                  "test": V2_VAL, "split_manifest": V2_BANK}  # test=val here (smoke only)
    best_eval = br.evaluate_adapter_checkpoint(
        Path(MODEL), best_dir, data_paths, config, seed_offset=seed_off + 1000)
    tgs = best_eval.get("test_greedy_success")
    tp8 = best_eval.get("test_pass_at_8")
    tvr = best_eval.get("test_valid_rate")
    print(f"[smoke]   best_eval test pass@8={tp8}  greedy={tgs}  valid={tvr}", flush=True)
    record["eval"] = {
        "test_greedy_success": tgs,
        "test_pass_at_8": tp8,
        "test_valid_rate": tvr,
        # evaluate_model uses prefix="validation" -> key is "validation_*" (not "val_*")
        "val_greedy_success": best_eval.get("validation_greedy_success"),
        "val_pass_at_8": best_eval.get("validation_pass_at_8"),
    }
    vgs = best_eval.get("validation_greedy_success")
    vp8 = best_eval.get("validation_pass_at_8")
    cond_g8 = all(_is_finite(v) for v in (tgs, tp8, tvr, vgs, vp8))
    all_pass &= _gate(record, "G8_eval_metrics_finite", cond_g8,
                      f"test greedy={tgs} pass@8={tp8} valid={tvr} | "
                      f"val greedy={vgs} pass@8={vp8}")

    # G9: no NaN/Inf anywhere in the numeric trail
    nan_scan = [bns, bgg, manifest.get("best_value"), tgs, tp8, tvr, vgs, vp8]
    found_nonfinite = [str(v) for v in nan_scan if not _is_finite(v)]
    all_pass &= _gate(record, "G9_no_nan_inf", not found_nonfinite,
                      f"non_finite_values={found_nonfinite}")

    # G10: this calibration is base-init only (reference_adapter=None); the v2
    # init-matrix must route low_sft calibration to a SEPARATE reference_adapter.
    # Asserting reference_adapter is None here documents the isolation boundary.
    cond_g10 = calib.get("reference_adapter") is None
    all_pass &= _gate(record, "G10_base_init_calibration_isolated", cond_g10,
                      f"reference_adapter={calib.get('reference_adapter')} "
                      f"(low_sft must use a separate reference_adapter=epoch_1)")

    record["overall"] = "PASS" if all_pass else "FAIL"
    GATE_JSON.write_text(json.dumps(record, indent=2))
    if all_pass:
        print("[smoke] PASS — non-degenerate; safe to proceed to full v2 init-matrix "
              "(NOT a scientific result). See SMOKE_GATE.json", flush=True)
        return 0
    print("[smoke] FAIL — one or more non-degeneracy gates failed; see SMOKE_GATE.json",
          flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
