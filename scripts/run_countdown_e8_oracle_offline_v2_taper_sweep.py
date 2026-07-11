#!/usr/bin/env python3
"""One-click guarded launcher for the eight-GPU E8 V2 taper sweep."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENT_ID = "EXT-C-E8-ORACLE-OFFLINE-V2-TAPER-SWEEP-0.5B-01"
RUNNER = "src/drpo/countdown_e8_oracle_offline_v2_taper_sweep.py"
SWEEP_CONFIG = "configs/countdown_e8_oracle_offline_v2_taper_sweep_0p5b.yaml"
BASE_CONFIG = "configs/countdown_e8_base_rl_replay_0p5b.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the E8 V2 taper sweep on eight GPUs")
    parser.add_argument(
        "--model_path", default="/root/models/Qwen2.5-0.5B-Instruct"
    )
    parser.add_argument(
        "--work_dir", default="/root/experiment_output/e8_v2_taper_sweep"
    )
    parser.add_argument(
        "--bank",
        default="/root/experiment_output/e8_oracle_bank_v2/data/offline_bank_v2.jsonl",
    )
    parser.add_argument(
        "--val", default="/root/experiment_output/e8_oracle_bank_v2/data/val.jsonl"
    )
    parser.add_argument(
        "--test", default="/root/experiment_output/e8_oracle_bank_v2/data/test.jsonl"
    )
    parser.add_argument(
        "--global_calibration",
        default="/root/experiment_output/e8_v2_matrix/calibration/base/calibration.json",
    )
    parser.add_argument("--base_config", default=BASE_CONFIG)
    parser.add_argument("--sweep_config", default=SWEEP_CONFIG)
    parser.add_argument("--calibration_gpu", default="0")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    work_dir = Path(args.work_dir).resolve()
    artifact = work_dir.parent / f"{EXPERIMENT_ID}_pilot.zip"
    command = [
        sys.executable,
        str(repo / "scripts" / "run_experiment_guard_hardened.py"),
        "--run-class",
        "pilot",
        "--experiment-id",
        EXPERIMENT_ID,
        "--output-root",
        str(work_dir),
        "--artifact-output",
        str(artifact),
        "--source-file",
        RUNNER,
        "--source-file",
        args.sweep_config,
        "--source-file",
        args.base_config,
        "--source-file",
        "src/drpo/countdown_qwen_arena_onefile.py",
        "--source-file",
        "src/drpo/countdown_e8_base_rl_replay.py",
        "--source-file",
        "experiments/registry.yaml",
        "--source-file",
        "docs/handoff.md",
        "--progress-glob",
        "SWEEP_STATUS.json",
        "--progress-glob",
        "methods/*/summary.json",
    ]
    if args.allow_dirty:
        command.append("--allow-dirty")
    command.extend(
        [
            "--",
            sys.executable,
            str(repo / RUNNER),
            "run",
            "--model_path",
            str(Path(args.model_path).resolve()),
            "--work_dir",
            str(work_dir),
            "--bank",
            str(Path(args.bank).resolve()),
            "--val",
            str(Path(args.val).resolve()),
            "--test",
            str(Path(args.test).resolve()),
            "--global_calibration",
            str(Path(args.global_calibration).resolve()),
            "--base_config",
            str((repo / args.base_config).resolve()),
            "--sweep_config",
            str((repo / args.sweep_config).resolve()),
            "--calibration_gpu",
            args.calibration_gpu,
            "--gpus",
            args.gpus,
        ]
    )
    print(
        f"Running {EXPERIMENT_ID}: 72 Linear/Quadratic/Exp cells on GPUs {args.gpus}"
    )
    return int(subprocess.run(command, cwd=repo, check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
