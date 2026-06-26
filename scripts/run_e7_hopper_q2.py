#!/usr/bin/env python3
"""Launch EXT-H-E7-Q2 through the canonical hardened execution channel."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENT_ID = "EXT-H-E7-Q2"
DEFAULT_CONFIG = "configs/e7_hopper_q2_medium_replay_v2.yaml"


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the registered Hopper E7-Q2 pipeline under the hardened guard"
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument(
        "--work-dir", required=True, help="New or empty persistent run directory"
    )
    parser.add_argument("--run-class", choices=("pilot", "formal"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-output")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Pilot only; the hardened guard captures a bounded launch snapshot",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    script = Path(__file__).resolve()
    repo = Path(git_text(script.parent, "rev-parse", "--show-toplevel")).resolve()
    head = git_text(repo, "rev-parse", "HEAD")
    if len(head) != 40:
        raise RuntimeError(f"Expected a full Git SHA, got {head!r}")
    if args.run_class == "formal" and args.allow_dirty:
        raise SystemExit("--allow-dirty is forbidden for formal runs")

    dataset = Path(args.dataset_path).expanduser().resolve()
    work = Path(args.work_dir).expanduser().resolve()
    config = (
        (repo / args.config).resolve()
        if not Path(args.config).is_absolute()
        else Path(args.config).resolve()
    )
    if not dataset.is_file():
        raise SystemExit(f"Dataset does not exist: {dataset}")
    if not config.is_file():
        raise SystemExit(f"Config does not exist: {config}")
    try:
        config_relative = config.relative_to(repo)
    except ValueError as exc:
        raise SystemExit(
            "The E7-Q2 config must be a repository-relative tracked file so the "
            "hardened source snapshot is complete"
        ) from exc
    try:
        git_text(repo, "ls-files", "--error-unmatch", str(config_relative))
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Config is not tracked by Git: {config_relative}") from exc
    artifact = (
        Path(args.artifact_output).expanduser().resolve()
        if args.artifact_output
        else work.parent / f"{work.name}_{EXPERIMENT_ID}_{args.run_class}.zip"
    )
    if artifact.exists():
        raise SystemExit(f"Artifact output already exists: {artifact}")

    guard = repo / "scripts" / "run_experiment_guard_hardened.py"
    runner = repo / "src" / "drpo" / "e7_hopper_q2.py"
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
        args.run_class,
        "--expected-commit",
        head,
        "--large-file-persistence",
        "persistent_local",
        "--required-output",
        "RUN_COMPLETE.json",
        "--required-output",
        "terminal_audit.json",
        "--required-output",
        "aggregate_summary.json",
        "--required-output",
        "per_seed_summary.csv",
        "--source-file",
        "scripts/run_e7_hopper_q2.py",
        "--source-file",
        "src/drpo/e7_hopper_q2.py",
        "--source-file",
        str(config_relative),
        "--source-file",
        "docs/handoff.md",
        "--source-file",
        "experiments/registry.yaml",
        "--progress-glob",
        "scientific_heartbeat.json",
        "--progress-glob",
        "events.jsonl",
        "--progress-glob",
        "seeds/*/methods/*/curves.csv",
    ]
    if args.run_class == "formal":
        command.append("--require-origin-main-match")
    if args.allow_dirty:
        command.append("--allow-dirty")
    command.extend(
        [
            "--",
            sys.executable,
            str(runner),
            "run",
            "--mode",
            args.run_class,
            "--dataset-path",
            str(dataset),
            "--work-dir",
            str(work),
            "--config",
            str(config),
            "--repo-root",
            str(repo),
            "--device",
            args.device,
        ]
    )
    if args.allow_dirty:
        command.append("--allow-dirty")

    print("EXT-H-E7-Q2 is fully specified; no interactive decisions are required.")
    print(f"Git commit: {head}")
    print(f"Run class: {args.run_class}")
    print(f"Dataset: {dataset}")
    print(f"Run directory: {work}")
    print(f"Result artifact: {artifact}")
    result = subprocess.run(command, cwd=repo)
    if result.returncode == 0:
        print(f"Hardened artifact created: {artifact}")
    else:
        print(
            f"E7-Q2 exited with code {result.returncode}; the hardened guard preserved "
            f"failure evidence in {work} and attempted recovery packaging at {artifact}.",
            file=sys.stderr,
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
