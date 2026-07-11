#!/usr/bin/env python3
"""Guarded launcher for EXT-C-E8-ORACLE-OFFLINE-V2-INIT-MATRIX-0.5B-01.

Wraps the v2 canonical runner (`src/drpo/countdown_e8_oracle_offline_v2_matrix.py`)
under `scripts/run_experiment_guard_hardened.py` so the experiment gets:
clean-worktree provenance, commit/branch/source-file recording, heartbeat,
exit status, terminal audit, and artifact packaging.

This launcher only sets up provenance + guard; it does NOT change any scientific
variable. All scientific params come from the shared config (the base-rl-replay
config's offline_training/evaluation/negative_calibration blocks) and the frozen
v2 corpus/bank passed via --bank/--val/--test.

Usage:
  python scripts/run_countdown_e8_oracle_offline_v2_matrix.py \
      --model_path /root/models/Qwen2.5-0.5B-Instruct \
      --work_dir /path/to/empty/work_dir \
      --bank /root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl \
      --val  /root/experiment_output/e8_oracle_bank_v2/data/val.jsonl \
      --test /root/experiment_output/e8_oracle_bank_v2/data/test.jsonl \
      --sft_dir /path/to/v2_sft_output \
      --gpu 0 --gpus 0,1,2,3,4,5 --allow-dirty
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENT_ID = "EXT-C-E8-ORACLE-OFFLINE-V2-INIT-MATRIX-0.5B-01"
CONFIG = "configs/countdown_e8_base_rl_replay_0p5b.yaml"
RUNNER = "src/drpo/countdown_e8_oracle_offline_v2_matrix.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the v2 init-matrix under the hardened guard")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--bank", required=True, help="v2 offline_bank_v2.jsonl")
    parser.add_argument("--val", required=True, help="v2 val.jsonl")
    parser.add_argument("--test", required=True, help="v2 test.jsonl")
    parser.add_argument("--sft_dir", default=None, help="v2 SFT dir (epoch_1+best adapters)")
    parser.add_argument("--gpu", default="0", help="GPU for calibration")
    parser.add_argument("--gpus", default="0,1,2,3,4,5", help="GPU pool for training cells")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--logs_dir", default="logs")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    guard = repo / "scripts" / "run_experiment_guard_hardened.py"
    runner = repo / RUNNER
    config = (repo / args.config).resolve()
    model = Path(args.model_path).resolve()
    work = Path(args.work_dir).resolve()
    artifact = work.parent / f"{EXPERIMENT_ID}_pilot.zip"
    command = [
        sys.executable,
        str(guard),
        "--run-class",
        "pilot",
        "--experiment-id",
        EXPERIMENT_ID,
        "--output-root",
        str(work),
        "--artifact-output",
        str(artifact),
        "--source-file",
        RUNNER,
        "--source-file",
        "src/drpo/countdown_e8_base_rl_replay.py",
        "--source-file",
        "src/drpo/countdown_qwen_arena_onefile.py",
        "--source-file",
        args.config,
        "--source-file",
        "docs/handoff.md",
        "--source-file",
        "experiments/registry.yaml",
        "--progress-glob",
        "v2_matrix_status.json",
        "--progress-glob",
        "methods/*/summary.json",
    ]
    if args.allow_dirty:
        command.append("--allow-dirty")
    command.extend([
        "--",
        sys.executable,
        str(runner),
        "run",
        "--model_path",
        str(model),
        "--work_dir",
        str(work),
        "--bank",
        str(Path(args.bank).resolve()),
        "--val",
        str(Path(args.val).resolve()),
        "--test",
        str(Path(args.test).resolve()),
        "--config",
        str(config),
        "--gpu",
        args.gpu,
        "--gpus",
        args.gpus,
        "--logs_dir",
        str(Path(args.logs_dir).resolve()),
    ])
    if args.sft_dir:
        command.extend(["--sft_dir", str(Path(args.sft_dir).resolve())])
    print(f"Running {EXPERIMENT_ID} under the hardened guard.")
    result = subprocess.run(command, cwd=repo)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
