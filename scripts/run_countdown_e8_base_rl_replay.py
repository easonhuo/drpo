#!/usr/bin/env python3
"""Guarded launcher for EXT-C-E8-BASE-RL-REPLAY-0.5B-01."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENT_ID = "EXT-C-E8-BASE-RL-REPLAY-0.5B-01"
CONFIG = "configs/countdown_e8_base_rl_replay_0p5b.yaml"
RUNNER = "src/drpo/countdown_e8_base_rl_replay.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Countdown E8 base RL/replay pilot under the hardened guard")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip_base_eval", action="store_true")
    parser.add_argument("--skip_offline", action="store_true")
    parser.add_argument("--skip_online", action="store_true")
    parser.add_argument("--skip_replay", action="store_true")
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
        "src/drpo/countdown_e8_onpolicy.py",
        "--source-file",
        "src/drpo/countdown_qwen_arena_onefile.py",
        "--source-file",
        args.config,
        "--source-file",
        "docs/handoff.md",
        "--source-file",
        "experiments/registry.yaml",
        "--progress-glob",
        "*.json",
        "--progress-glob",
        "methods/*/*/summary.json",
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
        "--config",
        str(config),
        "--gpu",
        args.gpu,
    ])
    for flag in ("skip_base_eval", "skip_offline", "skip_online", "skip_replay"):
        if getattr(args, flag):
            command.append("--" + flag)
    print("Running EXT-C-E8 base-start RL/replay pilot: oracle-offline, online on-policy, and online replay branches.")
    result = subprocess.run(command, cwd=repo)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
