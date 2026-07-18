"""Command-line entry point for the paper-facing DRPO reference code."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from drpo_reference.categorical.du1_public import run_du1
from drpo_reference.continuous.cu1_suite import (
    STAGES,
    run_cu1_all,
    run_cu1_stage,
)
from drpo_reference.experiments.hopper import run_hopper


def _seed_list(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from exc
    if not seeds:
        raise argparse.ArgumentTypeError(
            "at least one seed is required"
        )
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(
            "seed list contains duplicates"
        )
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drpo-reference",
        description="Reproduce the paper-facing DRPO experiments.",
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
        choices=(*STAGES, "all"),
        required=True,
        help="paper evidence stage to run",
    )
    cu1.add_argument("--output", type=Path, required=True)
    cu1.add_argument(
        "--seeds",
        type=_seed_list,
        help=(
            "optional comma-separated subset; "
            "subsets are never formal evidence"
        ),
    )
    cu1.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device such as cpu, cuda, cuda:0, or auto",
    )
    cu1.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "run a tiny integration path; "
            "never eligible for scientific evidence"
        ),
    )

    du1 = experiments.add_parser(
        "du1",
        help="D-U1 revision-4 utility×rarity experiment",
    )
    du1.add_argument("--output", type=Path, required=True)
    du1.add_argument(
        "--seeds",
        type=_seed_list,
        help=(
            "optional comma-separated subset; "
            "subsets are never formal evidence"
        ),
    )
    du1.add_argument(
        "--device",
        default="cpu",
        help=(
            "formal revision-4 runs require cpu; "
            "smoke may use another PyTorch device"
        ),
    )
    du1.add_argument(
        "--workers",
        type=int,
        help="parallel seed workers; defaults to 8 formal and 1 smoke",
    )
    du1.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "run one tiny six-method matrix; "
            "never eligible for scientific evidence"
        ),
    )

    hopper = experiments.add_parser(
        "hopper",
        help="Hopper E7-Q2 external mechanism-validation pipeline",
    )
    hopper.add_argument("--dataset", type=Path, required=True)
    hopper.add_argument("--output", type=Path, required=True)
    hopper.add_argument(
        "--seeds",
        type=_seed_list,
        help=(
            "registered-order formal subset; "
            "subsets are never formal evidence"
        ),
    )
    hopper.add_argument(
        "--device",
        default="auto",
        help="PyTorch device such as cpu, cuda, cuda:0, or auto",
    )
    hopper.add_argument(
        "--critic-artifact",
        type=Path,
        help="optional exact canonical critic artifact to verify and reuse",
    )
    hopper.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "run the tiny integration protocol; "
            "never eligible for scientific evidence"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.experiment == "cu1":
        if args.stage == "all":
            if args.seeds is not None:
                raise SystemExit(
                    "--seeds is not valid with --stage all"
                )
            run_cu1_all(
                output_root=args.output,
                smoke=args.smoke,
                device=args.device,
            )
        else:
            run_cu1_stage(
                stage=args.stage,
                output_root=args.output,
                seeds=args.seeds,
                smoke=args.smoke,
                device=args.device,
            )
        return 0
    if args.experiment == "du1":
        run_du1(
            output_root=args.output,
            seeds=args.seeds,
            smoke=args.smoke,
            device=args.device,
            workers=args.workers,
        )
        return 0
    if args.experiment == "hopper":
        run_hopper(
            dataset_path=args.dataset,
            output_root=args.output,
            seeds=args.seeds,
            smoke=args.smoke,
            device=args.device,
            critic_artifact=args.critic_artifact,
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
