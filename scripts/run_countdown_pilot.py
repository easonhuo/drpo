#!/usr/bin/env python3
"""One-decision Countdown V4.3 dynamic-remoteness pilot launcher.

The operator supplies only the local Qwen2.5-0.5B-Instruct path and a new
work directory. The launcher binds the current Git commit, starts the hardened
foreground guard, lets the arena choose the base/SFT path, schedules safe
multi-GPU stages, audits terminal checkpoints, and creates the durable result
artifact automatically.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENT_ID = "EXT-C-E8-V4.3"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the registered Countdown V4.3 pilot under the hardened guard"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--work_dir", required=True, help="New or empty persistent run directory")
    parser.add_argument("--gpus", default="auto", help="Visible GPU ids or auto (default)")
    parser.add_argument("--artifact_output", default=None)
    parser.add_argument("--allow_dirty", action="store_true", help="Engineering pilot only; captures launch diff")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    script = Path(__file__).resolve()
    repo = Path(_git(script.parent, "rev-parse", "--show-toplevel")).resolve()
    head = _git(repo, "rev-parse", "HEAD")
    if len(head) != 40:
        raise RuntimeError(f"Expected a full Git SHA, got {head!r}")

    model = Path(args.model_path).resolve()
    work = Path(args.work_dir).resolve()
    if not model.is_dir():
        raise SystemExit(f"Model directory does not exist: {model}")
    artifact = (
        Path(args.artifact_output).resolve()
        if args.artifact_output
        else work.parent / f"{work.name}_{EXPERIMENT_ID}_pilot.zip"
    )
    if artifact.exists():
        raise SystemExit(f"Artifact output already exists; use a new work_dir: {artifact}")

    guard = repo / "scripts" / "run_experiment_guard_hardened.py"
    runner = repo / "src" / "drpo" / "countdown_qwen_arena_onefile.py"
    command = [
        sys.executable,
        str(guard),
        "--experiment-id", EXPERIMENT_ID,
        "--repo-root", str(repo),
        "--output-root", str(work),
        "--artifact-output", str(artifact),
        "--run-class", "pilot",
        "--expected-commit", head,
        "--large-file-persistence", "persistent_local",
        "--required-output", "RUN_COMPLETE.json",
        "--required-output", "terminal_audit.json",
        "--required-output", "arena_summary.csv",
        "--source-file", "scripts/run_countdown_pilot.py",
        "--source-file", "src/drpo/countdown_qwen_arena_onefile.py",
        "--source-file", "docs/handoff.md",
        "--source-file", "experiments/registry.yaml",
        "--progress-glob", "logs/*.log",
        "--progress-glob", "methods/*/metrics.csv",
    ]
    if args.allow_dirty:
        command.append("--allow-dirty")
    command.extend([
        "--",
        sys.executable,
        str(runner),
        "run",
        "--model_path", str(model),
        "--work_dir", str(work),
        "--gpus", args.gpus,
        "--preset", "0.5b",
        "--memory_mode", "bf16",
        "--seed", "1234",
    ])

    print("Countdown pilot is fully specified; no interactive decisions are required.")
    print(f"Git commit: {head}")
    print(f"Run directory: {work}")
    print(f"Result artifact: {artifact}")
    print(f"GPU selection: {args.gpus}")
    result = subprocess.run(command, cwd=repo)
    if result.returncode == 0:
        print(f"Pilot packaged successfully: {artifact}")
    else:
        print(
            f"Pilot did not complete (exit {result.returncode}). The guard preserved failure evidence "
            f"in {work} and attempted a recovery package at {artifact}.",
            file=sys.stderr,
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
