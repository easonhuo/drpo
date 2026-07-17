"""Resumable two-stage coordinator and CLI for EXT-H-E7-SQEXP-GAE-01."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from drpo.e7_sqexp_gae_actor_runtime import aggregate_results, train_actor_branch
from drpo.e7_sqexp_gae_contract import (
    DEVELOPMENT_SEEDS, EXPECTED_BRANCHES, EXPERIMENT_ID, RUNNER_VERSION,
    SCIENTIFIC_STATUS, FrozenProtocol, atomic_json, build_actor_branches,
    build_critic_jobs, canonical_hash, load_protocol, load_source_run_spec, utc_now,
)
from drpo.e7_sqexp_gae_preparation import train_frozen_critic_and_prepare

def execution_plan(protocol: FrozenProtocol) -> dict[str, Any]:
    critics = build_critic_jobs(protocol)
    branches = build_actor_branches(protocol)
    return {
        "experiment_id": EXPERIMENT_ID,
        "runner_version": RUNNER_VERSION,
        "scientific_status": SCIENTIFIC_STATUS,
        "critic_jobs": [dataclasses.asdict(job) | {"job_id": job.id} for job in critics],
        "actor_branches": [dataclasses.asdict(branch) | {"branch_id": branch.id} for branch in branches],
        "critic_job_count": len(critics),
        "actor_branch_count": len(branches),
        "shared_critic_scope": "one_per_dataset_seed_shared_by_16_actor_branches",
        "held_out_seeds": list(protocol.held_out_seeds),
        "held_out_seeds_scheduled": False,
        "formal_run_allowed": False,
    }


def run_identity(protocol: FrozenProtocol, run_spec_sha256: str) -> str:
    return canonical_hash(
        {
            "experiment_id": EXPERIMENT_ID,
            "runner_version": RUNNER_VERSION,
            "protocol_sha256": protocol.config_sha256,
            "source_run_spec_sha256": run_spec_sha256,
            "matrix": execution_plan(protocol),
        }
    )


def _parallel(
    commands: Sequence[tuple[str, list[str], Path]],
    max_workers: int,
    *,
    cpus_per_worker: int,
    stage: str,
    heartbeat_path: Path,
) -> list[dict[str, Any]]:
    if not commands:
        return []
    from drpo import e7_bench

    jobs = [
        {"job_id": job_id, "command": command, "log_path": log_path}
        for job_id, command, log_path in commands
    ]
    return e7_bench.run_parallel_stage(
        jobs,
        max_workers=max_workers,
        cpus_per_worker=cpus_per_worker,
        stage=stage,
        heartbeat_path=heartbeat_path,
    )


def _resume_complete_or_raise(path: Path, run_identity_sha256: str) -> bool:
    marker = path / "WORKER_COMPLETE.json"
    if not marker.is_file():
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f"incomplete worker output requires a new/clean path: {path}")
        return False
    payload = json.loads(marker.read_text())
    if payload.get("run_identity_sha256") != run_identity_sha256:
        raise RuntimeError(f"stale worker identity at {path}")
    return True


def run_coordinator(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.config)
    run_spec_path = args.run_spec or protocol.source_run_spec
    datasets, run_spec_sha = load_source_run_spec(run_spec_path)
    by_id = {dataset.id: dataset for dataset in datasets}
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    identity = run_identity(protocol, run_spec_sha)
    identity_path = work_dir / "RUN_IDENTITY.json"
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text())
        if existing.get("run_identity_sha256") != identity:
            raise RuntimeError("work directory belongs to a different frozen protocol identity")
        if not args.resume:
            raise RuntimeError("existing run requires --resume")
    elif any(work_dir.iterdir()):
        raise RuntimeError("non-empty work directory lacks RUN_IDENTITY.json")
    else:
        atomic_json(
            identity_path,
            {
                "run_identity_sha256": identity,
                "experiment_id": EXPERIMENT_ID,
                "protocol_sha256": protocol.config_sha256,
                "source_run_spec_sha256": run_spec_sha,
                "created_utc": utc_now(),
            },
        )
    atomic_json(work_dir / "EXECUTION_PLAN.json", execution_plan(protocol))
    python = sys.executable
    runner = str(Path(__file__).resolve())
    critic_commands: list[tuple[str, list[str], Path]] = []
    for job in build_critic_jobs(protocol):
        output = work_dir / "critics" / job.dataset_id / f"seed_{job.seed}"
        if args.resume and _resume_complete_or_raise(output, identity):
            continue
        command = [
            python,
            runner,
            "critic-worker",
            "--config",
            str(args.config),
            "--run-spec",
            str(run_spec_path),
            "--dataset-id",
            job.dataset_id,
            "--seed",
            str(job.seed),
            "--output-dir",
            str(output),
            "--device",
            args.device,
        ]
        critic_commands.append((job.id, command, work_dir / "logs" / f"{job.id}.log"))
    _parallel(
        critic_commands,
        args.critic_workers,
        cpus_per_worker=args.critic_cpus_per_worker,
        stage="parallel_shared_critic_and_advantage_preparation",
        heartbeat_path=work_dir / "critic_stage_heartbeat.json",
    )
    branch_commands: list[tuple[str, list[str], Path]] = []
    for branch in build_actor_branches(protocol):
        output = work_dir / "branches" / branch.dataset_id / f"seed_{branch.seed}" / branch.id
        if args.resume and _resume_complete_or_raise(output, identity):
            continue
        critic_dir = work_dir / "critics" / branch.dataset_id / f"seed_{branch.seed}"
        if not (critic_dir / "WORKER_COMPLETE.json").is_file():
            raise RuntimeError(f"shared critic is incomplete: {critic_dir}")
        command = [
            python,
            runner,
            "actor-worker",
            "--config",
            str(args.config),
            "--run-spec",
            str(run_spec_path),
            "--branch-id",
            branch.id,
            "--critic-dir",
            str(critic_dir),
            "--output-dir",
            str(output),
            "--device",
            args.device,
        ]
        branch_commands.append((branch.id, command, work_dir / "logs" / f"{branch.id}.log"))
    _parallel(
        branch_commands,
        args.actor_workers,
        cpus_per_worker=args.actor_cpus_per_worker,
        stage="parallel_192_equal_horizon_actor_branches",
        heartbeat_path=work_dir / "actor_stage_heartbeat.json",
    )
    aggregate_results(work_dir, protocol)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/e7_sqexp_gae_v1.yaml")
        command.add_argument("--run-spec")
        command.add_argument("--work-dir", required=name == "run")
        command.add_argument("--device", default="auto")
        command.add_argument("--critic-workers", type=int, default=12)
        command.add_argument("--actor-workers", type=int, default=64)
        command.add_argument("--critic-cpus-per-worker", type=int, default=4)
        command.add_argument("--actor-cpus-per-worker", type=int, default=1)
        command.add_argument("--resume", action="store_true")
    critic = sub.add_parser("critic-worker", help=argparse.SUPPRESS)
    critic.add_argument("--config", required=True)
    critic.add_argument("--run-spec", required=True)
    critic.add_argument("--dataset-id", required=True)
    critic.add_argument("--seed", type=int, required=True)
    critic.add_argument("--output-dir", required=True)
    critic.add_argument("--device", default="auto")
    actor = sub.add_parser("actor-worker", help=argparse.SUPPRESS)
    actor.add_argument("--config", required=True)
    actor.add_argument("--run-spec", required=True)
    actor.add_argument("--branch-id", required=True)
    actor.add_argument("--critic-dir", required=True)
    actor.add_argument("--output-dir", required=True)
    actor.add_argument("--device", default="auto")
    aggregate = sub.add_parser("aggregate", help=argparse.SUPPRESS)
    aggregate.add_argument("--config", required=True)
    aggregate.add_argument("--work-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_protocol(args.config)
    if args.command == "plan":
        print(json.dumps(execution_plan(protocol), indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        return run_coordinator(args)
    if args.command == "aggregate":
        aggregate_results(Path(args.work_dir).resolve(), protocol)
        return 0
    datasets, run_spec_sha = load_source_run_spec(args.run_spec)
    by_id = {dataset.id: dataset for dataset in datasets}
    if args.command == "critic-worker":
        if args.dataset_id not in by_id or args.seed not in DEVELOPMENT_SEEDS:
            raise ValueError("critic worker is outside the frozen dataset/seed matrix")
        manifest = train_frozen_critic_and_prepare(
            dataset=by_id[args.dataset_id],
            seed=args.seed,
            protocol=protocol,
            source_run_spec_sha256=run_spec_sha,
            output_dir=Path(args.output_dir).resolve(),
            device_name=args.device,
        )
        manifest["source_run_spec_sha256"] = run_spec_sha
        manifest["run_identity_sha256"] = run_identity(protocol, run_spec_sha)
        atomic_json(Path(args.output_dir) / "prepared_advantage_manifest.json", manifest)
        atomic_json(Path(args.output_dir) / "WORKER_COMPLETE.json", manifest)
        return 0
    branches = {branch.id: branch for branch in build_actor_branches(protocol)}
    branch = branches.get(args.branch_id)
    if branch is None:
        raise ValueError("actor branch is outside the frozen 192-cell matrix")
    summary = train_actor_branch(
        branch=branch,
        dataset=by_id[branch.dataset_id],
        protocol=protocol,
        critic_dir=Path(args.critic_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        device_name=args.device,
        source_run_spec_sha256=run_spec_sha,
    )
    summary["run_identity_sha256"] = run_identity(protocol, run_spec_sha)
    atomic_json(Path(args.output_dir) / "summary.json", summary)
    atomic_json(Path(args.output_dir) / "WORKER_COMPLETE.json", summary)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
