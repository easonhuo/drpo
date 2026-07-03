#!/usr/bin/env python3
"""One-command launcher for EXT-C-E8-TAPER-0.5B-01.

Example:
    python3 scripts/run_countdown_e8_taper.py \
      --model_path /ABS/PATH/Qwen2.5-0.5B-Instruct \
      --work_dir /ABS/PATH/e8-taper-run \
      --gpus auto
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENT_ID = "EXT-C-E8-TAPER-0.5B-01"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the registered Countdown continuous-surprisal taper pilot"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--work_dir", required=True, help="New persistent run directory")
    parser.add_argument("--gpus", default="auto", help="Visible GPU ids or auto")
    parser.add_argument("--artifact_output", default=None)
    parser.add_argument("--allow_dirty", action="store_true", help="Pilot-only dirty launch with captured diff")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    script = Path(__file__).resolve()
    repo = Path(_git(script.parent, "rev-parse", "--show-toplevel")).resolve()
    head = _git(repo, "rev-parse", "HEAD")
    if len(head) != 40:
        raise RuntimeError(f"Expected full Git SHA, got {head!r}")
    model = Path(args.model_path).resolve()
    work = Path(args.work_dir).resolve()
    if not model.is_dir():
        raise SystemExit(f"Model directory does not exist: {model}")
    artifact = (
        Path(args.artifact_output).resolve()
        if args.artifact_output
        else work.parent / f"{work.name}_{EXPERIMENT_ID}_RAW_COMPLETE.zip"
    )
    if artifact.exists():
        raise SystemExit(f"Artifact already exists; choose a new output: {artifact}")

    guard = repo / "scripts" / "run_experiment_guard_hardened.py"
    runner = repo / "src" / "drpo" / "countdown_e8_taper.py"
    config = repo / "configs" / "countdown_e8_taper_0p5b.yaml"
    command = [
        sys.executable,
        str(guard),
        "--experiment-id",
        EXPERIMENT_ID,
        "--repo-root",
        str(repo),
        "--output-root",
        str(work),
        "--artifact-output",
        str(artifact),
        "--run-class",
        "pilot",
        "--expected-commit",
        head,
        "--large-file-persistence",
        "persistent_local",
        "--required-output",
        "RUN_COMPLETE.json",
        "--required-output",
        "terminal_audit.json",
        "--required-output",
        "scientific_run_manifest.json",
        "--required-output",
        "arena_summary.csv",
        "--required-output",
        "taper_calibration.json",
        "--required-output",
        "replay_pool_manifest.json",
        "--required-output",
        "surprisal_bin_diagnostics.csv",
        "--required-output",
        "run_config.json",
        "--source-file",
        "scripts/run_countdown_e8_taper.py",
        "--source-file",
        "src/drpo/countdown_e8_taper.py",
        "--source-file",
        "src/drpo/countdown_qwen_arena_onefile.py",
        "--source-file",
        "configs/countdown_e8_taper_0p5b.yaml",
        "--source-file",
        "docs/handoff.md",
        "--source-file",
        "experiments/registry.yaml",
        "--progress-glob",
        "logs/*.log",
        "--progress-glob",
        "methods/*/*/metrics.csv",
        "--progress-glob",
        "methods/*/*/training_log.csv",
    ]
    if args.allow_dirty:
        command.append("--allow-dirty")
    command.extend(
        [
            "--",
            sys.executable,
            str(runner),
            "run",
            "--model_path",
            str(model),
            "--work_dir",
            str(work),
            "--gpus",
            args.gpus,
            "--config",
            str(config),
        ]
    )
    print("Countdown E8 TAPER is fully specified; no interactive choices are required.")
    print(f"Git commit: {head}")
    print(f"Run directory: {work}")
    print(f"Artifact: {artifact}")
    result = subprocess.run(command, cwd=repo)
    if result.returncode == 0:
        print(f"Pilot packaged successfully: {artifact}")
    else:
        print(
            f"Pilot failed with exit {result.returncode}; the hardened guard preserved evidence in {work}.",
            file=sys.stderr,
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
