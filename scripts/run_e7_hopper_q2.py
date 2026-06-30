#!/usr/bin/env python3
"""One-command launcher for EXT-H-E7-Q2.

Typical formal use after the dataset has been placed in a standard location::

    python3 scripts/run_e7_hopper_q2.py

The launcher resolves the registered dataset, creates a timestamped persistent
work directory, and delegates the scientific run plus recovery packaging to the
canonical hardened guard.  ``--plan-only`` performs all local resolution and
prints the exact command without starting training.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_ID = "EXT-H-E7-Q2"
DEFAULT_CONFIG = "configs/e7_hopper_q2_medium_replay_v2.yaml"
DATASET_BASENAME = "hopper_medium_replay-v2.hdf5"
DATASET_ENV = "DRPO_HOPPER_MEDIUM_REPLAY"


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the registered Hopper E7-Q2 pipeline through the hardened guard; "
            "formal is the default and requires no interactive decisions"
        )
    )
    parser.add_argument(
        "--run-class", choices=("pilot", "formal"), default="formal"
    )
    parser.add_argument(
        "--dataset-path",
        help=(
            f"HDF5 path. If omitted, use ${DATASET_ENV} and registered standard locations."
        ),
    )
    parser.add_argument(
        "--work-dir",
        help="Persistent run directory. If omitted, create runs/e7_q2/<UTC stamp>_<class>.",
    )
    parser.add_argument(
        "--output-root", default="runs/e7_q2", help="Parent used by automatic work-dir"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-output")
    parser.add_argument(
        "--critic-artifact",
        help="Optional exact v4.3 schema-3 canonical critic artifact to verify and reuse",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Pilot only; formal continues to require a clean current main checkout",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Resolve paths and print the hardened command without executing it",
    )
    return parser


def resolve_dataset(repo: Path, explicit: str | None) -> tuple[Path, str]:
    candidates: list[tuple[Path, str]] = []
    if explicit:
        candidates.append((Path(explicit).expanduser(), "--dataset-path"))
    env_value = os.environ.get(DATASET_ENV)
    if env_value:
        candidates.append((Path(env_value).expanduser(), f"${DATASET_ENV}"))
    candidates.extend(
        [
            (
                Path("/root/d4rl/d4rl_datasets/locomotion") / DATASET_BASENAME,
                "registered server location",
            ),
            (
                Path("/root/d4rl/datasets") / DATASET_BASENAME,
                "alternate server location",
            ),
            (repo / "data" / DATASET_BASENAME, "repository data directory"),
            (repo.parent / "d4rl_datasets" / DATASET_BASENAME, "sibling data directory"),
        ]
    )
    checked: list[str] = []
    for candidate, source in candidates:
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if resolved.is_file():
            return resolved, source
    raise SystemExit(
        "Could not resolve the registered Hopper dataset. Set "
        f"{DATASET_ENV}, pass --dataset-path, or place {DATASET_BASENAME} in a "
        "registered location. Checked:\n  - " + "\n  - ".join(checked)
    )


def resolve_repo(script: Path) -> Path:
    try:
        return Path(git_text(script.parent, "rev-parse", "--show-toplevel")).resolve()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit("This launcher must run from a Git checkout of drpo") from exc


def resolve_config(repo: Path, raw: str) -> tuple[Path, Path]:
    config = Path(raw)
    config = config.resolve() if config.is_absolute() else (repo / config).resolve()
    if not config.is_file():
        raise SystemExit(f"Config does not exist: {config}")
    try:
        relative = config.relative_to(repo)
    except ValueError as exc:
        raise SystemExit("The E7 config must be inside the repository") from exc
    try:
        git_text(repo, "ls-files", "--error-unmatch", str(relative))
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Config is not tracked by Git: {relative}") from exc
    return config, relative


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    script = Path(__file__).resolve()
    repo = resolve_repo(script)
    head = git_text(repo, "rev-parse", "HEAD")
    if len(head) != 40:
        raise RuntimeError(f"Expected a full Git SHA, got {head!r}")
    if args.run_class == "formal" and args.allow_dirty:
        raise SystemExit("--allow-dirty is forbidden for formal runs")

    dataset, dataset_source = resolve_dataset(repo, args.dataset_path)
    config, config_relative = resolve_config(repo, args.config)
    if args.work_dir:
        work = Path(args.work_dir).expanduser().resolve()
    else:
        output_root = Path(args.output_root).expanduser()
        output_root = (
            output_root.resolve()
            if output_root.is_absolute()
            else (repo / output_root).resolve()
        )
        work = output_root / f"{utc_stamp()}_{args.run_class}"
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f"Work directory must be new or empty: {work}")

    artifact = (
        Path(args.artifact_output).expanduser().resolve()
        if args.artifact_output
        else work.parent / f"{work.name}_{EXPERIMENT_ID}_{args.run_class}.zip"
    )
    if artifact.exists():
        raise SystemExit(f"Artifact output already exists: {artifact}")

    guard = repo / "scripts" / "run_experiment_guard_hardened.py"
    runner = repo / "src" / "drpo" / "e7_hopper_q2.py"
    for required in (guard, runner):
        if not required.is_file():
            raise SystemExit(f"Required repository file is missing: {required}")

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
        "--required-output",
        "ROLLOUT_PREFLIGHT.json",
        "--required-output",
        "CANONICAL_CRITIC_REFERENCE.json",
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
        "--progress-glob",
        "canonical_critic/training/critic_metrics.csv",
        "--progress-glob",
        "rollout_preflight/rollout_preflight.json",
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
    if args.critic_artifact:
        critic_artifact = Path(args.critic_artifact).expanduser().resolve()
        if not critic_artifact.is_dir():
            raise SystemExit(
                f"Canonical critic artifact does not exist: {critic_artifact}"
            )
        command.extend(["--critic-artifact", str(critic_artifact)])
    if args.allow_dirty:
        command.append("--allow-dirty")

    plan = {
        "experiment_id": EXPERIMENT_ID,
        "git_commit": head,
        "run_class": args.run_class,
        "dataset": str(dataset),
        "dataset_resolution": dataset_source,
        "config": str(config),
        "work_dir": str(work),
        "artifact": str(artifact),
        "critic_artifact": args.critic_artifact or "train_once_inside_run",
    }
    return command, plan


def main() -> int:
    args = build_parser().parse_args()
    command, plan = build_command(args)
    print(json.dumps(plan, indent=2, sort_keys=True))
    print("Command:")
    print("  " + shlex.join(command))
    if args.plan_only:
        print("Plan only: no training process was started.")
        return 0
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1])
    if result.returncode == 0:
        print(f"Hardened result artifact created: {plan['artifact']}")
    else:
        print(
            f"E7-Q2 exited with code {result.returncode}; the hardened guard preserved "
            f"failure evidence in {plan['work_dir']} and attempted recovery packaging.",
            file=sys.stderr,
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
