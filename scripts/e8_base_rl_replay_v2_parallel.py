#!/usr/bin/env python3
"""Parallel orchestrator: EXT-C-E8-BASE-RL-REPLAY on the v2 oracle-bank (one-off).

Re-runs the 9 base-init cells of EXT-C-E8-BASE-RL-REPLAY-0.5B-01 on the v2
oracle-offline bank (model-independent, value-stratified). Mirrors the
registered runner's `cmd_run` cell list, seeds, and output_names EXACTLY; only
the dataset changes (v2 bank + v2 splits + v2 base calibration) — the approved
scientific variable. Execution-only.

Cells (9), all base-init (reference_adapter=None, no SFT warmstart):
  base_eval                                            eval-only
  base_oracle_offline_positive_only        (B)         positive_only,  seed=seed_off
  base_oracle_offline_pos_neg_recalibrated_x0p25 (C')  bank_global_matched, mult 0.25, seed+250
  base_oracle_offline_pos_neg_recalibrated_x0p5  (C')  bank_global_matched, mult 0.5,  seed+500
  base_oracle_offline_pos_neg_recalibrated_x1p0  (C')  bank_global_matched, mult 1.0,  seed+1000
  base_oracle_offline_pos_neg_recalibrated_x2p0  (C')  bank_global_matched, mult 2.0,  seed+2000
  base_onpolicy_positive_only              (D)         online, seed=2026070805
  base_online_replay_positive_only         (F1)        replay pos, seed=2026070806
  base_online_replay_pos_neg               (F2)        replay pos+neg, seed=2026070906

Each cell runs in a GPU-pinned subprocess that imports the runner module and
calls the SAME function cmd_run would call (train_offline_method /
train_online_or_replay / evaluate_base), so every subprocess writes a
byte-faithful `summary.json` in the canonical location/format. A 6-GPU pool
(GPUs 0-5) serves the 9 jobs; GPUs 6-7 are reserved for the concurrent v2 SFT
and matrix init. Single-seed pilot; not a method-ranking claim.
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
WORK = Path("/root/experiment_output/e8_base_rl_replay_v2")
LOGS = Path("/root/countdown/logs")
SELF = Path(__file__).resolve()

V2_BANK = Path("/root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl")
V2_TRAIN = Path("/root/experiment_output/e8_oracle_bank_v2/data/oracle_offline_bank_v2_train.jsonl")
V2_VAL = Path("/root/experiment_output/e8_oracle_bank_v2/data/val.jsonl")
V2_TEST = Path("/root/experiment_output/e8_oracle_bank_v2/data/test.jsonl")
V2_MANIFEST = Path("/root/experiment_output/e8_oracle_bank_v2/data/split_manifest.json")
CALIB = Path("/root/experiment_output/e8_v2_bank_calib/base_calibration.json")

POOL_GPUS = ["0", "1", "2", "3", "4", "5"]   # 6-GPU pool; 6/7 reserved for SFT+init

progress: list[str] = []
prog_lock = threading.Lock()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with prog_lock:
        progress.append(line)


def worker(job: str) -> int:
    """Run one cell in-process. CUDA_VISIBLE_DEVICES already set by caller."""
    if DRPO_SRC not in sys.path:
        sys.path.insert(0, DRPO_SRC)
    import countdown_e8_base_rl_replay as br  # noqa: E402

    config = br.load_config(CONFIG)
    model_path = Path(MODEL)
    # v2 data_paths (NOT generate_or_load_data -> that would build old splits).
    data_paths = {
        "train": V2_TRAIN,
        "validation": V2_VAL,
        "test": V2_TEST,
        "split_manifest": V2_MANIFEST,
    }
    seed_off = int(config["offline_training"]["seed"])
    neg_method = str(config["negative_calibration"]["method"])

    if job == "base_eval":
        s = br.evaluate_base(model_path, WORK, data_paths, config)
        print(f"BASE_EVAL_DONE test_pass@8={s.get('test_pass_at_8')} greedy={s.get('test_greedy_success')}")
        return 0

    if job == "B":
        s = br.train_offline_method(
            model_path, WORK, data_paths, V2_BANK, config,
            method="positive_only",
            output_name="base_oracle_offline_positive_only",
            seed=seed_off, calibration_json=None, negative_scale_multiplier=1.0,
        )
        print(f"B_DONE best_pass@8={s['best_evaluation'].get('test_pass_at_8')}")
        return 0

    if job == "x0p25":
        s = br.train_offline_method(
            model_path, WORK, data_paths, V2_BANK, config,
            method=neg_method, output_name="base_oracle_offline_pos_neg_recalibrated_x0p25",
            seed=seed_off + 250, calibration_json=CALIB, negative_scale_multiplier=0.25,
        )
        print(f"X0P25_DONE best_pass@8={s['best_evaluation'].get('test_pass_at_8')}")
        return 0

    if job == "x0p5":
        s = br.train_offline_method(
            model_path, WORK, data_paths, V2_BANK, config,
            method=neg_method, output_name="base_oracle_offline_pos_neg_recalibrated_x0p5",
            seed=seed_off + 500, calibration_json=CALIB, negative_scale_multiplier=0.5,
        )
        print(f"X0P5_DONE best_pass@8={s['best_evaluation'].get('test_pass_at_8')}")
        return 0

    if job == "x1p0":
        s = br.train_offline_method(
            model_path, WORK, data_paths, V2_BANK, config,
            method=neg_method, output_name="base_oracle_offline_pos_neg_recalibrated_x1p0",
            seed=seed_off + 1000, calibration_json=CALIB, negative_scale_multiplier=1.0,
        )
        print(f"X1P0_DONE best_pass@8={s['best_evaluation'].get('test_pass_at_8')}")
        return 0

    if job == "x2p0":
        s = br.train_offline_method(
            model_path, WORK, data_paths, V2_BANK, config,
            method=neg_method, output_name="base_oracle_offline_pos_neg_recalibrated_x2p0",
            seed=seed_off + 2000, calibration_json=CALIB, negative_scale_multiplier=2.0,
        )
        print(f"X2P0_DONE best_pass@8={s['best_evaluation'].get('test_pass_at_8')}")
        return 0

    if job == "online":
        seed = int(config["onpolicy_training"]["seeds"][0])
        s = br.train_online_or_replay(
            model_path, WORK, data_paths, config,
            method="base_onpolicy_positive_only", seed=seed,
            replay=False, use_negatives=False,
        )
        print(f"ONLINE_DONE status={s.get('status')} best_val={s.get('best_validation_value')}")
        return 0

    if job == "replay_pos":
        seed = int(config["replay_training"]["seeds"][0])
        s = br.train_online_or_replay(
            model_path, WORK, data_paths, config,
            method="base_online_replay_positive_only", seed=seed,
            replay=True, use_negatives=False,
        )
        print(f"REPLAY_POS_DONE status={s.get('status')} best_val={s.get('best_validation_value')}")
        return 0

    if job == "replay_posneg":
        seed = int(config["replay_training"]["seeds"][0]) + 100
        s = br.train_online_or_replay(
            model_path, WORK, data_paths, config,
            method="base_online_replay_pos_neg", seed=seed,
            replay=True, use_negatives=True,
        )
        print(f"REPLAY_POSNEG_DONE status={s.get('status')} best_val={s.get('best_validation_value')}")
        return 0

    raise SystemExit(f"unknown job: {job}")


def run_worker_subprocess(job: str, gpu: str) -> tuple[int, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    log_path = LOGS / f"base_rl_replay_v2_{job}.log"
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


# job order: quick base_eval first, then offline cells, then slow online/replay
JOBS = ["base_eval", "B", "x0p25", "x0p5", "x1p0", "x2p0", "online", "replay_pos", "replay_posneg"]

results: dict[str, dict] = {}
results_lock = threading.Lock()
gpu_pool = threading.Semaphore(len(POOL_GPUS))
gpu_q: list[str] = list(POOL_GPUS)
gpu_q_lock = threading.Lock()


def take_gpu() -> str:
    with gpu_q_lock:
        return gpu_q.pop(0)


def give_gpu(g: str) -> None:
    with gpu_q_lock:
        gpu_q.append(g)


def run_job(job: str) -> None:
    gpu_pool.acquire()
    gpu = take_gpu()
    log(f"GPU{gpu} START {job}")
    rc, log_path = run_worker_subprocess(job, gpu)
    status = "OK" if rc == 0 else f"FAIL(rc={rc})"
    log(f"GPU{gpu} END   {job} -> {status}  ({log_path})")
    with results_lock:
        results[job] = {"gpu": gpu, "returncode": rc, "status": status, "log": log_path}
    give_gpu(gpu)
    gpu_pool.release()


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--worker":
        return worker(args[1])
    WORK.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    # sanity: v2 bank + calib must exist
    for p in [V2_BANK, V2_TRAIN, V2_VAL, V2_TEST, CALIB]:
        if not Path(p).exists():
            raise SystemExit(f"missing required v2 artifact: {p}")
    log(f"=== E8 base-rl-replay v2 start; {len(JOBS)} jobs, pool GPUs {POOL_GPUS} ===")
    log(f"WORK={WORK}  bank={V2_BANK}  calib={CALIB}")
    threads = [threading.Thread(target=run_job, args=(job,), name=job) for job in JOBS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("=== all jobs joined ===")
    fails = [j for j, v in results.items() if v["returncode"] != 0]
    (WORK / "parallel_accel_status.json").write_text(
        json.dumps({"jobs": results, "failures": fails}, indent=2, ensure_ascii=False))
    log(f"failures: {fails if fails else 'none'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())