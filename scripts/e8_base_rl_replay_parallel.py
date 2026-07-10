#!/usr/bin/env python3
"""Parallel accelerator for EXT-C-E8-BASE-RL-REPLAY-0.5B-01 (GLM Dev Agent, one-off).

The registered runner `countdown_e8_base_rl_replay.py:cmd_run` trains its 6
remaining method-cells SEQUENTIALLY on one GPU. The user asked to parallelize
the queued cells. This orchestrator fans the not-yet-completed cells across
free GPUs, each in a GPU-pinned subprocess that imports the runner module and
calls the SAME function cmd_run would call (train_offline_method /
train_online_or_replay), so every subprocess writes a byte-faithful
`summary.json` in the canonical location/format.

After all cells have summary.json, re-launch the guarded run
(scripts/run_countdown_e8_base_rl_replay.py --allow-dirty): with the
skip-if-summary.json idempotency added to the runner, cmd_run skips every
completed method and runs only final aggregation -> guard packages the
registered zip.

Cells (9 total). Already done on disk (summary.json present, skipped here):
  base_eval, base_oracle_offline_positive_only,
  base_oracle_offline_pos_neg_recalibrated_x0p25

Run here (6 cells):
  recover_x0p5  : x0p5 training finished (best+terminal adapters + manifest on
                  disk) but terminal eval was interrupted -> recover by eval'ing
                  the existing best/terminal adapters and assembling summary
                  (mirrors train_offline_method's tail, lines 407-431).
  train_x1p0    : full train_offline_method, multiplier 1.0
  train_x2p0    : full train_offline_method, multiplier 2.0
  online        : train_online_or_replay base_onpolicy_positive_only
  replay_pos    : train_online_or_replay base_online_replay_positive_only
  replay_posneg : train_online_or_replay base_online_replay_pos_neg

Execution-only: same config, same data, same seeds, same functions as cmd_run.
No scientific variable changed. Single-seed pilot.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

DRPO_SRC = "/root/drpo/src/drpo"
CONFIG = "/root/drpo/configs/countdown_e8_base_rl_replay_0p5b.yaml"
MODEL = "/root/models/Qwen2.5-0.5B-Instruct"
WORK = Path("/root/experiment_output/e8_base_rl_replay")
LOGS = Path("/root/countdown/logs")
SELF = Path(__file__).resolve()

# GPU assignment for the 6 jobs (avoid cards with zombie memory if present).
# Filled at launch from free GPUs; default order prefers 0,1,2,3,6,7.
DEFAULT_GPUS = ["0", "1", "2", "3", "6", "7"]

JOBS = [
    "recover_x0p5",
    "train_x1p0",
    "train_x2p0",
    "online",
    "replay_pos",
    "replay_posneg",
]

progress: list[str] = []
prog_lock = threading.Lock()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with prog_lock:
        progress.append(line)


def worker(job: str) -> int:
    """Run one method cell in-process (CUDA_VISIBLE_DEVICES already set by caller)."""
    if DRPO_SRC not in sys.path:
        sys.path.insert(0, DRPO_SRC)
    import countdown_e8_base_rl_replay as br  # noqa: E402

    config = br.load_config(CONFIG)
    model_path = Path(MODEL)
    work = WORK
    data_paths = br.generate_or_load_data(work, config)
    offline_data = work / "offline_bank" / "offline_6000.jsonl"
    calib = work / "negative_calibration" / "base_negative_budget_calibration.json"
    if not offline_data.exists():
        raise SystemExit(f"offline bank missing: {offline_data}")
    if not calib.exists():
        raise SystemExit(f"calibration missing: {calib}")
    seed_off = int(config["offline_training"]["seed"])
    neg_method = str(config["negative_calibration"]["method"])

    def assemble_offline_summary(out_dir: Path, output_name: str, method: str,
                                  seed: int, multiplier: float,
                                  calibration_json: Path) -> dict:
        manifest = json.loads((out_dir / "manifest.json").read_text())
        best_eval = br.evaluate_adapter_checkpoint(
            model_path, out_dir / "best_adapter", data_paths, config, seed_offset=seed
        )
        terminal_eval = None
        if (out_dir / "terminal_adapter" / "adapter_config.json").exists():
            terminal_eval = br.evaluate_adapter_checkpoint(
                model_path, out_dir / "terminal_adapter", data_paths, config,
                seed_offset=seed + 17,
            )
        summary = {
            "method": output_name,
            "arena_method": method,
            "seed": seed,
            "best_step": manifest.get("best_step"),
            "best_value": manifest.get("best_value"),
            "terminal_step": manifest.get("terminal_step"),
            "stop_reason": manifest.get("stop_reason"),
            "negative_scale_multiplier": multiplier,
            "negative_calibration_json": str(calibration_json) if calibration_json else None,
            "best_evaluation": best_eval,
            "terminal_evaluation": terminal_eval,
            "summary_path": str(out_dir / "manifest.json"),
        }
        br._atomic_json(out_dir / "summary.json", summary)
        return summary

    if job == "recover_x0p5":
        out_dir = work / "methods" / "base_oracle_offline_pos_neg_recalibrated_x0p5"
        seed = seed_off + 500
        s = assemble_offline_summary(out_dir, "base_oracle_offline_pos_neg_recalibrated_x0p5",
                                     neg_method, seed, 0.5, calib)
        print(f"RECOVER_DONE x0p5 best_pass@8={s['best_evaluation'].get('test_pass_at_k')} "
              f"terminal_pass@8={(s.get('terminal_evaluation') or {}).get('test_pass_at_k')}")
        return 0

    if job == "train_x1p0":
        s = br.train_offline_method(
            model_path, work, data_paths, offline_data, config,
            method=neg_method,
            output_name="base_oracle_offline_pos_neg_recalibrated_x1p0",
            seed=seed_off + 1000, calibration_json=calib,
            negative_scale_multiplier=1.0,
        )
        print(f"TRAIN_DONE x1p0 best_pass@8={s['best_evaluation'].get('test_pass_at_k')}")
        return 0

    if job == "train_x2p0":
        s = br.train_offline_method(
            model_path, work, data_paths, offline_data, config,
            method=neg_method,
            output_name="base_oracle_offline_pos_neg_recalibrated_x2p0",
            seed=seed_off + 2000, calibration_json=calib,
            negative_scale_multiplier=2.0,
        )
        print(f"TRAIN_DONE x2p0 best_pass@8={s['best_evaluation'].get('test_pass_at_k')}")
        return 0

    if job == "online":
        seed = int(config["onpolicy_training"]["seeds"][0])
        s = br.train_online_or_replay(
            model_path, work, data_paths, config,
            method="base_onpolicy_positive_only", seed=seed,
            replay=False, use_negatives=False,
        )
        st = s.get("status"); bv = s.get("best_validation_value")
        print(f"ONLINE_DONE status={st} best_val={bv}")
        return 0

    if job == "replay_pos":
        seed = int(config["replay_training"]["seeds"][0])
        s = br.train_online_or_replay(
            model_path, work, data_paths, config,
            method="base_online_replay_positive_only", seed=seed,
            replay=True, use_negatives=False,
        )
        st = s.get("status"); bv = s.get("best_validation_value")
        print(f"REPLAY_POS_DONE status={st} best_val={bv}")
        return 0

    if job == "replay_posneg":
        seed = int(config["replay_training"]["seeds"][0]) + 100
        s = br.train_online_or_replay(
            model_path, work, data_paths, config,
            method="base_online_replay_pos_neg", seed=seed,
            replay=True, use_negatives=True,
        )
        st = s.get("status"); bv = s.get("best_validation_value")
        print(f"REPLAY_POSNEG_DONE status={st} best_val={bv}")
        return 0

    raise SystemExit(f"unknown job: {job}")


def run_worker_subprocess(job: str, gpu: str) -> tuple[int, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    log_path = LOGS / f"base_rl_replay_parallel_{job}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, str(SELF), "--worker", job]
    with log_path.open("w") as fh:
        fh.write(f"=== {job} on GPU {gpu} ===\n" + " ".join(argv) + "\n")
        fh.flush()
        proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.readline(), ""):
            fh.write(chunk)
            fh.flush()
        rc = proc.wait()
        fh.write(f"=== returncode={rc} ===\n")
    return rc, str(log_path)


results: dict[str, dict] = {}
results_lock = threading.Lock()


def run_job(job: str, gpu: str) -> None:
    log(f"GPU{gpu} START {job}")
    rc, log_path = run_worker_subprocess(job, gpu)
    status = "OK" if rc == 0 else f"FAIL(rc={rc})"
    log(f"GPU{gpu} END   {job} -> {status}  ({log_path})")
    with results_lock:
        results[job] = {"gpu": gpu, "returncode": rc, "status": status, "log": log_path}


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--worker":
        return worker(args[1])
    # orchestrator
    LOGS.mkdir(parents=True, exist_ok=True)
    gpus = DEFAULT_GPUS[:len(JOBS)]
    log(f"=== E8 base-rl-replay parallel acceleration start; {len(JOBS)} jobs on GPUs {gpus} ===")
    log(f"WORK={WORK}  config={CONFIG}")
    threads = []
    for job, gpu in zip(JOBS, gpus):
        t = threading.Thread(target=run_job, args=(job, gpu), name=job)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    log("=== all jobs joined ===")
    fails = [j for j, v in results.items() if v["returncode"] != 0]
    # write status json
    (WORK / "parallel_accel_status.json").write_text(
        json.dumps({"jobs": results, "failures": fails}, indent=2, ensure_ascii=False))
    log(f"failures: {fails if fails else 'none'}")
    if fails:
        log("NOTE: failed cells will be retried by the subsequent guarded aggregation run "
            "if their summary.json is absent; cells with summary.json are skipped.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())