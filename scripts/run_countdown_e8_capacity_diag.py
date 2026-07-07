#!/usr/bin/env python3
"""Guarded launcher for EXT-C-E8-ONPOLICY-CAPACITY-DIAG-0.5B-01."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENT_ID = "EXT-C-E8-ONPOLICY-CAPACITY-DIAG-0.5B-01"
CONFIG = "configs/countdown_e8_onpolicy_capacity_diag_0p5b.yaml"
RUNNER = "src/drpo/countdown_e8_capacity_diag.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Countdown E8 capacity diagnostic under the hardened guard")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--gpu_ids", default="0,1,2,3")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--sft_adapter_path", default=None)
    parser.add_argument("--serial", action="store_true", help="Debug fallback; formal pilot default is parallel branch workers.")
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
    artifact = work / f"{EXPERIMENT_ID}.zip"
    command = [
        sys.executable,
        str(guard),
        "--run-class",
        "pilot",
        "--experiment-id",
        EXPERIMENT_ID,
        "--work-dir",
        str(work),
        "--artifact",
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
        "logs/*.log",
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
        "--gpu_ids",
        args.gpu_ids,
    ])
    if args.serial:
        command.append("--serial")
    if args.sft_adapter_path:
        command.extend(["--sft_adapter_path", str(Path(args.sft_adapter_path).resolve())])
    print(f"Running {EXPERIMENT_ID} with branch/seed workers; single seed attempts remain on-policy sequential.")
    result = subprocess.run(command, cwd=repo)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
