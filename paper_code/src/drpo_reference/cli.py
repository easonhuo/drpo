"""Command-line entry point for the DRPO reference code."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from drpo_reference.categorical.du1_public import run_du1
from drpo_reference.categorical.structured_generation import (
    STRUCTURED_GENERATION_METHODS,
    TASK_NAMES,
)
from drpo_reference.continuous.cu1_suite import (
    STAGES,
    run_cu1_stage,
)
from drpo_reference.experiments import (
    D4RL_METHODS,
    run_d4rl,
)
from drpo_reference.experiments.structured_generation import run_structured_generation


def _seed_list(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def _task_list(value: str) -> tuple[str, ...]:
    return tuple(value.split(","))


def _method_list(value: str) -> tuple[str, ...]:
    return tuple(value.split(","))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drpo-reference",
        description="Run the DRPO reference experiments.",
    )
    experiments = parser.add_subparsers(
        dest="experiment",
        required=True,
    )
    cu1 = experiments.add_parser(
        "cu1",
        help="C-U1 same-distribution held-out-context experiments",
    )
    cu1.add_argument(
        "--stage",
        choices=STAGES,
        required=True,
        help="paper evidence stage to run",
    )
    cu1.add_argument("--output", type=Path, required=True)
    cu1.add_argument(
        "--seeds",
        type=_seed_list,
        help="optional comma-separated seed subset",
    )
    cu1.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device such as cpu, cuda, cuda:0, or auto",
    )
    du1 = experiments.add_parser(
        "du1",
        help="D-U1 revision-4 utility×rarity experiment",
    )
    du1.add_argument("--output", type=Path, required=True)
    du1.add_argument(
        "--seeds",
        type=_seed_list,
        help="optional comma-separated seed subset",
    )
    du1.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device such as cpu, cuda, cuda:0, or auto",
    )
    d4rl = experiments.add_parser(
        "d4rl",
        help="D4RL-9 training and rollout runner",
    )
    d4rl.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="directory containing canonical D4RL-v2 HDF5 files",
    )
    d4rl.add_argument("--output", type=Path, required=True)
    d4rl.add_argument(
        "--tasks",
        type=_task_list,
        help="optional comma-separated task IDs; defaults to all nine",
    )
    d4rl.add_argument(
        "--methods",
        type=_method_list,
        help=(
            "optional comma-separated methods; omitted means ExpRank "
            "only. Available: " + ", ".join(D4RL_METHODS)
        ),
    )
    d4rl.add_argument(
        "--seeds",
        type=_seed_list,
        required=True,
        help="comma-separated seeds",
    )
    d4rl.add_argument(
        "--steps",
        type=int,
        required=True,
        help="training updates",
    )
    d4rl.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="training batch size",
    )
    d4rl.add_argument(
        "--device",
        default="auto",
        help="PyTorch device such as cpu, cuda, cuda:0, or auto",
    )
    d4rl.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help=("real Gymnasium/MuJoCo episodes per seed; zero disables rollout evaluation"),
    )
    d4rl.add_argument(
        "--eval-max-steps",
        type=int,
        default=1000,
        help="maximum environment steps per evaluation episode",
    )
    structured = experiments.add_parser(
        "structured-generation",
        help="unified nine-task Structured Generation Qwen/LoRA runner",
    )
    structured.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON runtime configuration",
    )
    structured.add_argument("--output", type=Path, required=True)
    structured.add_argument(
        "--tasks",
        type=_task_list,
        help=("optional comma-separated task subset; available: " + ", ".join(TASK_NAMES)),
    )
    structured.add_argument(
        "--methods",
        type=_method_list,
        help=(
            "optional comma-separated method subset; available: "
            + ", ".join(STRUCTURED_GENERATION_METHODS)
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.experiment == "cu1":
        run_cu1_stage(
            stage=args.stage,
            output_root=args.output,
            seeds=args.seeds,
            device=args.device,
        )
        return 0
    if args.experiment == "du1":
        run_du1(
            output_root=args.output,
            seeds=args.seeds,
            device=args.device,
        )
        return 0
    if args.experiment == "d4rl":
        run_d4rl(
            dataset_root=args.dataset_root,
            output_root=args.output,
            task_ids=args.tasks,
            seeds=args.seeds,
            steps=args.steps,
            batch_size=args.batch_size,
            device=args.device,
            eval_episodes=args.eval_episodes,
            eval_max_steps=args.eval_max_steps,
            methods=args.methods,
        )
        return 0
    if args.experiment == "structured-generation":
        run_structured_generation(
            config_path=args.config,
            output_root=args.output,
            tasks=args.tasks,
            methods=args.methods,
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
