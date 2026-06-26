#!/usr/bin/env python3
"""Launch D-U1 E6 semantic long-run through the canonical hardened channel."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENT_ID = "D-U1-E6-SEMANTIC-LONGRUN-01"
DEFAULT_CONFIG = "configs/du1_e6_semantic_longrun.yaml"


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen E6 formal long-run under the hardened guard"
    )
    parser.add_argument("--work-dir", required=True, help="New or empty persistent run directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    script = Path(__file__).resolve()
    repo = Path(git_text(script.parent, "rev-parse", "--show-toplevel")).resolve()
    head = git_text(repo, "rev-parse", "HEAD")
    if len(head) != 40:
        raise RuntimeError(f"Expected a full Git SHA, got {head!r}")
    if git_text(repo, "status", "--porcelain"):
        raise SystemExit("Formal E6 requires a clean worktree")

    work = Path(args.work_dir).expanduser().resolve()
    config = (
        (repo / args.config).resolve()
        if not Path(args.config).is_absolute()
        else Path(args.config).resolve()
    )
    if not config.is_file():
        raise SystemExit(f"Config does not exist: {config}")
    try:
        config_relative = config.relative_to(repo)
    except ValueError as exc:
        raise SystemExit("Formal E6 config must be inside the repository") from exc
    try:
        git_text(repo, "ls-files", "--error-unmatch", str(config_relative))
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Config is not tracked by Git: {config_relative}") from exc

    artifact = (
        Path(args.artifact_output).expanduser().resolve()
        if args.artifact_output
        else work.parent / f"{work.name}_{EXPERIMENT_ID}_RAW_COMPLETE.zip"
    )
    if artifact.exists():
        raise SystemExit(f"Artifact output already exists: {artifact}")

    guard = repo / "scripts" / "run_experiment_guard_hardened.py"
    runner = repo / "src" / "drpo" / "du1_e6_semantic_longrun.py"
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
        "formal",
        "--expected-commit",
        head,
        "--require-origin-main-match",
        "--heartbeat-seconds",
        "60",
        "--stale-seconds",
        "900",
        "--fail-on-stale",
        "--large-file-persistence",
        "persistent_local",
        "--required-output",
        "RUN_COMPLETE.json",
        "--required-output",
        "scientific_run_manifest.json",
        "--required-output",
        "terminal_audit.json",
        "--required-output",
        "aggregate_summary.json",
        "--required-output",
        "per_run_summary.csv",
        "--required-output",
        "formal_protocol_freeze.json",
        "--required-output",
        "run_manifest.json",
        "--source-file",
        "scripts/run_du1_e6_semantic_longrun.py",
        "--source-file",
        "src/drpo/du1_e6_semantic_longrun.py",
        "--source-file",
        "src/drpo/du1_e6_semantic.py",
        "--source-file",
        str(config_relative),
        "--source-file",
        "docs/handoff.md",
        "--source-file",
        "experiments/registry.yaml",
        "--progress-glob",
        "trajectories.jsonl",
        "--progress-glob",
        "runs/seed_*/*.summary.json",
        "--progress-glob",
        "checkpoints/*/CHECKPOINT_COMPLETE.json",
        "--",
        sys.executable,
        str(runner),
        "--config",
        str(config),
        "--output-root",
        str(work),
        "--device",
        args.device,
    ]

    print("D-U1-E6-SEMANTIC-LONGRUN-01 is frozen and fully specified.")
    print(f"Git commit: {head}")
    print(f"Run directory: {work}")
    print(f"Result artifact: {artifact}")
    result = subprocess.run(command, cwd=repo)
    if result.returncode == 0:
        print(f"Hardened artifact created: {artifact}")
    else:
        print(
            f"E6 long-run exited with code {result.returncode}; the hardened guard "
            f"preserved failure evidence in {work} and attempted packaging at {artifact}.",
            file=sys.stderr,
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
