#!/usr/bin/env python3
"""A/B/C offline-bank matrix — one-off staging orchestration (GLM Dev Agent).

Approved design (user, 2026-07-09): init-level × training-condition matrix on
Qwen2.5-0.5B-Instruct, reusing the FROZEN E8 offline-bank products from
pilot_run_7 (V4.4-OFFLINE-BANK):

  bank        = pilot_run_7/data/offline_6000.jsonl  (positive + negative_bank[16])
  calibration = pilot_run_7/negative_budget_calibration.json
                (bank_negative_scale=0.0890, bank_global_gamma=0.3360 — frozen)
  splits      = pilot_run_7/data/{train,val,test}.jsonl

Init levels:
  A = Qwen base, no SFT        -> fresh untrained LoRA (cmd_init_adapter)
  B = ultra-low SFT (greedy~4%) -> e8_lowsft_rft/sft/epoch_1_adapter
  C = full-SFT reference        -> pilot_run_7/sft_adapter/best_adapter

Cells (single-seed 1234; training replicates pilot_run_7's exact method config,
only reference_adapter / method / seed vary):
  A0 eval-only | A1 positive_only | A2 bank_global_matched
  B0 eval-only | B1 positive_only | B2 bank_global_matched
  C0 eval-only | C1 positive_only (negative control)   [no C2 per spec]

All 9 cells get a UNIFORM final test eval (pass_k=8, max_new_tokens=80,
batch_size=32, seed=7234) on test.jsonl so A0/B0/C0 are like-for-like with the
trained cells (we re-eval A0/C0 rather than reuse pilot_run_7's pre-computed
base/reference test metrics, which do not record their eval seed).

Parallelism: 8 pipelines fanned across 8 GPUs (one process per GPU via
CUDA_VISIBLE_DEVICES). init_A runs first (blocking; A0/A1/A2 depend on it).
Each training pipeline chains train_method -> evaluate(best) -> evaluate(terminal)
on the same GPU. Single-seed pilot; no method-ranking claim; no 3B/7B
generalization. Execution-only — no scientific variable beyond the 4 user
decisions (B=epoch_1, A2/B2=bank_global_matched, seed 1234, abandon E8-TAPER/D-U1).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# ---- fixed config ----
MODEL = "/root/models/Qwen2.5-0.5B-Instruct"
RUNNER_PY = "/root/drpo/src/drpo/countdown_qwen_arena_onefile.py"
PY = sys.executable
WORK = Path("/root/experiment_output/abc_offline_matrix")
LOGS = Path("/root/countdown/logs")
BANK = "/root/experiment_output/pilot_run_7/data/offline_6000.jsonl"
CALIB = "/root/experiment_output/pilot_run_7/negative_budget_calibration.json"
TRAIN = "/root/experiment_output/pilot_run_7/data/train.jsonl"
VAL = "/root/experiment_output/pilot_run_7/data/val.jsonl"
TEST = "/root/experiment_output/pilot_run_7/data/test.jsonl"
ADAPTER_B = "/root/experiment_output/e8_lowsft_rft/sft/epoch_1_adapter"
ADAPTER_C = "/root/experiment_output/pilot_run_7/sft_adapter/best_adapter"
ADAPTER_A = str(WORK / "A_init_adapter")
SEED = 1234
EVAL_SEED = 7234
GPUS = ["0", "1", "2", "3", "4", "5", "6", "7"]

# pilot_run_7 method training config (replicated verbatim)
TRAIN_FLAGS = [
    "--steps", "1128", "--min_steps", "376",
    "--early_stop_patience", "2", "--early_stop_delta", "0.002",
    "--selection_metric", "greedy_success",
    "--micro_batch", "4", "--grad_accum", "8", "--lr", "5e-5",
    "--warmup_ratio", "0.03", "--max_grad_norm", "1.0",
    "--max_length", "256", "--max_new_tokens", "80",
    "--eval_examples", "500", "--eval_batch", "32", "--pass_k", "8",
    "--eval_every", "188", "--eval_seed", str(EVAL_SEED),
    "--near_mix", "0.5", "--far_mix", "0.5", "--global_gamma", "0.55",
    "--negative_scale_multiplier", "1.0",
    "--entropy_coef", "0.02", "--target_entropy", "1.8",
    "--target_entropy_coef", "0.05", "--sbrc_kappa", "0.92",
    "--entropy_floor", "1.0", "--exp_lambda", "0.7",
    "--surprisal_threshold", "2.0",
    "--diagnostic_examples", "32", "--diagnostic_gradient_examples", "8",
    "--diagnostic_batch", "8", "--log_every", "10", "--num_workers", "2",
    "--seed", str(SEED), "--result_status", "pilot",
    "--structure_reference_data", TRAIN, "--val_data", VAL,
    "--offline_data", BANK, "--dtype", "bf16",
]
EVAL_FLAGS = ["--max_new_tokens", "80", "--pass_k", "8",
              "--batch_size", "32", "--seed", str(EVAL_SEED),
              "--structure_reference_data", TRAIN, "--dtype", "bf16"]

# ---- pipeline definitions ----
# Each pipeline: list of (label, argv, log_suffix) run sequentially on one GPU.
# init_A is run separately (blocking) before these.

def train_cmd(cell, method, ref):
    out = str(WORK / cell)
    argv = [PY, RUNNER_PY, "train_method", "--model_path", MODEL,
            "--reference_adapter", ref, "--output_dir", out,
            "--method", method] + TRAIN_FLAGS
    if method == "bank_global_matched":
        argv += ["--negative_calibration_json", CALIB]
    return argv

def eval_cmd(cell, adapter, suffix):
    out = str(WORK / cell / f"test_metrics_{suffix}.json")
    argv = [PY, RUNNER_PY, "evaluate", "--model_path", MODEL,
            "--adapter", adapter, "--data", TEST, "--output_json", out] + EVAL_FLAGS
    return argv

# pipelines keyed by gpu index
PIPELINES = {
    "0": [("A0_eval", eval_cmd("A0", ADAPTER_A, "init"), "A0_eval")],
    "1": [("A1_train", train_cmd("A1", "positive_only", ADAPTER_A), "A1_train"),
          ("A1_eval_best", eval_cmd("A1", str(WORK / "A1" / "best_adapter"), "best"), "A1_eval_best"),
          ("A1_eval_term", eval_cmd("A1", str(WORK / "A1" / "terminal_adapter"), "terminal"), "A1_eval_term")],
    "2": [("A2_train", train_cmd("A2", "bank_global_matched", ADAPTER_A), "A2_train"),
          ("A2_eval_best", eval_cmd("A2", str(WORK / "A2" / "best_adapter"), "best"), "A2_eval_best"),
          ("A2_eval_term", eval_cmd("A2", str(WORK / "A2" / "terminal_adapter"), "terminal"), "A2_eval_term")],
    "3": [("B0_eval", eval_cmd("B0", ADAPTER_B, "init"), "B0_eval")],
    "4": [("B1_train", train_cmd("B1", "positive_only", ADAPTER_B), "B1_train"),
          ("B1_eval_best", eval_cmd("B1", str(WORK / "B1" / "best_adapter"), "best"), "B1_eval_best"),
          ("B1_eval_term", eval_cmd("B1", str(WORK / "B1" / "terminal_adapter"), "terminal"), "B1_eval_term")],
    "5": [("B2_train", train_cmd("B2", "bank_global_matched", ADAPTER_B), "B2_train"),
          ("B2_eval_best", eval_cmd("B2", str(WORK / "B2" / "best_adapter"), "best"), "B2_eval_best"),
          ("B2_eval_term", eval_cmd("B2", str(WORK / "B2" / "terminal_adapter"), "terminal"), "B2_eval_term")],
    "6": [("C0_eval", eval_cmd("C0", ADAPTER_C, "init"), "C0_eval")],
    "7": [("C1_train", train_cmd("C1", "positive_only", ADAPTER_C), "C1_train"),
          ("C1_eval_best", eval_cmd("C1", str(WORK / "C1" / "best_adapter"), "best"), "C1_eval_best"),
          ("C1_eval_term", eval_cmd("C1", str(WORK / "C1" / "terminal_adapter"), "terminal"), "C1_eval_term")],
}

results: dict[str, dict] = {}
results_lock = threading.Lock()
progress: list[str] = []
prog_lock = threading.Lock()

def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with prog_lock:
        progress.append(line)

def run_subprocess(argv: list[str], log_path: Path, gpu: str, label: str) -> int:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as fh:
        fh.write(f"=== {label} on GPU {gpu} ===\n")
        fh.write(" ".join(argv) + "\n")
        fh.flush()
        proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.readline(), ""):
            fh.write(chunk)
            fh.flush()
        rc = proc.wait()
        fh.write(f"=== returncode={rc} ===\n")
    return rc

def run_pipeline(gpu: str, steps: list[tuple[str, list[str], str]]) -> None:
    for label, argv, suffix in steps:
        log(f"GPU{gpu} START {label}")
        rc = run_subprocess(argv, LOGS / f"abc_{suffix}.log", gpu, label)
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        log(f"GPU{gpu} END   {label} -> {status}")
        with results_lock:
            results[label] = {"gpu": gpu, "returncode": rc, "status": status}
        if rc != 0:
            # stop this pipeline on failure, but let others continue
            return

def run_init_a(gpu: str) -> bool:
    log("Phase 0: init_adapter A (fresh untrained LoRA on base Qwen)")
    argv = [PY, RUNNER_PY, "init_adapter", "--model_path", MODEL,
            "--output_dir", ADAPTER_A, "--seed", str(SEED), "--dtype", "bf16"]
    rc = run_subprocess(argv, LOGS / "abc_A_init.log", gpu, "A_init")
    log(f"Phase 0 done: A_init returncode={rc}")
    return rc == 0

def collect_results() -> dict:
    """Read each cell's test_metrics_*.json + manifest into a matrix dict."""
    matrix = {}
    for cell in ["A0", "A1", "A2", "B0", "B1", "B2", "C0", "C1"]:
        entry: dict = {"cell": cell}
        for suffix in ["init", "best", "terminal"]:
            p = WORK / cell / f"test_metrics_{suffix}.json"
            if p.exists():
                try:
                    d = json.loads(p.read_text())
                    entry[f"test_{suffix}"] = {
                        "greedy_success": d.get("greedy_success"),
                        "pass_at_k": d.get("pass_at_k"),
                        "valid_rate": d.get("valid_rate"),
                    }
                except Exception as e:
                    entry[f"test_{suffix}"] = {"error": str(e)}
        mf = WORK / cell / "manifest.json"
        if mf.exists():
            try:
                m = json.loads(mf.read_text())
                entry["train"] = {
                    "best_step": m.get("best_step"),
                    "terminal_step": m.get("terminal_step"),
                    "best_value": m.get("best_value"),
                    "method": m.get("method"),
                    "reference_adapter": m.get("reference_adapter"),
                }
            except Exception as e:
                entry["train"] = {"error": str(e)}
        matrix[cell] = entry
    return matrix

def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    log(f"=== A/B/C offline-bank matrix start; WORK={WORK} ===")
    log(f"bank={BANK}")
    log(f"calib={CALIB}  A={ADAPTER_A}  B={ADAPTER_B}  C={ADAPTER_C}")

    # Phase 0: init_A (must precede A0/A1/A2)
    if not run_init_a(GPUS[0]):
        log("FATAL: init_A failed; aborting")
        return 1

    # Phase 1: 8 pipelines in parallel
    log("Phase 1: launching 8 pipelines across 8 GPUs")
    threads = []
    for gpu in GPUS:
        t = threading.Thread(target=run_pipeline, args=(gpu, PIPELINES[gpu]),
                             name=f"gpu{gpu}")
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    log("Phase 1: all pipelines joined")

    # Aggregate
    matrix = collect_results()
    (WORK / "matrix_results.json").write_text(
        json.dumps({"matrix": matrix, "step_results": results,
                    "config": {"seed": SEED, "eval_seed": EVAL_SEED,
                               "bank": BANK, "calibration": CALIB,
                               "A_init": ADAPTER_A, "B_init": ADAPTER_B,
                               "C_init": ADAPTER_C}},
                   indent=2, ensure_ascii=False))
    # RESULTS.md
    lines = ["# A/B/C offline-bank matrix — results\n",
             f"seed={SEED} eval_seed={EVAL_SEED}  bank=offline_6000.jsonl\n",
             "| cell | init | condition | test_greedy | test_pass@8 | valid |",
             "|---|---|---|---|---|---|"]
    init_label = {"A": "base", "B": "ultra-low(ep1)", "C": "full-SFT"}
    cond = {"0": "eval-only", "1": "positive-only", "2": "pos+neg(bank_global_matched)"}
    for cell in ["A0", "A1", "A2", "B0", "B1", "B2", "C0", "C1"]:
        lvl, kind = cell[0], cell[1]
        m = matrix[cell]
        # pick the representative test metric: best for trained, init for eval-only
        key = "test_best" if "test_best" in m else ("test_init" if "test_init" in m else None)
        t = m.get(key, {})
        g = t.get("greedy_success"); p = t.get("pass_at_k"); v = t.get("valid_rate")
        def fmt(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else "-"
        lines.append(f"| {cell} | {init_label[lvl]} | {cond[kind]} | {fmt(g)} | {fmt(p)} | {fmt(v)} |")
    (WORK / "RESULTS.md").write_text("\n".join(lines) + "\n")
    log(f"=== matrix done; wrote {WORK}/matrix_results.json + RESULTS.md ===")
    # final status summary
    fails = [k for k, v in results.items() if v["returncode"] != 0]
    log(f"failures: {fails if fails else 'none'}")
    return 1 if fails else 0

if __name__ == "__main__":
    raise SystemExit(main())