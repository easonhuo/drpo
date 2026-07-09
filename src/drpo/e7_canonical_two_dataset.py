"""Two-dataset canonical-agent pilot wrapper for E7.

This wrapper exists to make the GLM/dev-agent handoff less error-prone.  It
uses the already registered canonical-source adapter but fixes the intended
first validation scope to exactly two D4RL cells:

* hopper-medium-replay-v2;
* hopper-medium-expert-v2.

It does not implement a new D4RL algorithm.  It fingerprints the user's old
D4RL source tree, writes a concrete run spec for the historical
``train_sna2c_variant.py`` + ``SNA2C_IQLV_ExpRankAgent`` backbone, then delegates
planning/running to ``drpo.e7_canonical_sweep``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from drpo.e7_canonical_injection import sha256_file, write_fingerprint_contract

EXPERIMENT_ID = "EXT-H-E7-BENCH-01"
DEFAULT_GRID = "configs/e7_canonical_two_dataset_taper_grid_v1.json"
CONTRACT_VERSION_NOTE = "canonical-agent-two-dataset-v1"

DATASET_IDS = ("hopper-medium-replay-v2", "hopper-medium-expert-v2")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _sha_or_compute(path: str, supplied: str | None) -> str:
    if supplied:
        value = supplied.lower()
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"invalid sha256 for {path}: {supplied!r}")
        return value
    return sha256_file(path)


def build_trainer_template(args: argparse.Namespace) -> list[str]:
    template = [
        "--dataset",
        "{dataset_id}",
        "--hdf5",
        "{dataset_path}",
        "--variant",
        "iqlv_exp_rank",
        "--alpha",
        str(args.alpha),
        "--tau",
        str(args.tau),
        "--temp",
        str(args.temp),
        "--steps",
        str(args.steps),
        "--batch",
        str(args.batch),
        "--lr",
        str(args.lr),
        "--eval_interval",
        str(args.eval_interval),
        "--eval_episodes",
        str(args.eval_episodes),
        "--seed",
        "{seed}",
        "--out_dir",
        "{output_dir}",
        "--ckpt_dir",
        "{output_dir}/ckpts",
        "--ckpt_interval",
        str(args.ckpt_interval),
        "--last_pct",
        str(args.last_pct),
    ]
    if args.device is not None:
        template.extend(["--device", str(args.device)])
    if args.eval_max_steps is not None:
        template.extend(["--eval_max_steps", str(args.eval_max_steps)])
    return template


def write_run_spec(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    replay_path = Path(args.hopper_medium_replay_hdf5).expanduser().resolve()
    expert_path = Path(args.hopper_medium_expert_hdf5).expanduser().resolve()
    if not replay_path.is_file():
        raise FileNotFoundError(f"missing hopper-medium-replay-v2 dataset: {replay_path}")
    if not expert_path.is_file():
        raise FileNotFoundError(f"missing hopper-medium-expert-v2 dataset: {expert_path}")
    seeds = [int(item) for item in args.seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"seeds must be unique: {seeds}")
    payload = {
        "run_kind": "pilot",
        "experiment_id": EXPERIMENT_ID,
        "wrapper": CONTRACT_VERSION_NOTE,
        "scope": "two_dataset_canonical_agent_validation_only",
        "datasets": [
            {
                "id": DATASET_IDS[0],
                "path": str(replay_path),
                "sha256": _sha_or_compute(str(replay_path), args.hopper_medium_replay_sha256),
            },
            {
                "id": DATASET_IDS[1],
                "path": str(expert_path),
                "sha256": _sha_or_compute(str(expert_path), args.hopper_medium_expert_sha256),
            },
        ],
        "seeds": seeds,
        "trainer_argv_template": build_trainer_template(args),
        "passthrough_variants": [
            {
                "id": "original_exp_rank_mr",
                "template_values": {},
                "description": (
                    "Unchanged SNA2C_IQLV_ExpRankAgent passthrough from the "
                    "fingerprinted old source tree."
                ),
            }
        ],
        "environment": {
            "OMP_NUM_THREADS": str(args.omp_threads),
            "MKL_NUM_THREADS": str(args.omp_threads),
            "OPENBLAS_NUM_THREADS": str(args.omp_threads),
        },
        "interpretation_boundary": (
            "Pilot-only canonical-backbone parity/taper validation.  Do not use "
            "as a formal D4RL-9 method ranking."
        ),
    }
    atomic_write_json(output, payload)
    return payload


def prepare_common(args: argparse.Namespace) -> tuple[Path, Path]:
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    contract_path = Path(args.contract_output or (work_dir / "canonical_contract.json"))
    run_spec_path = Path(args.run_spec_output or (work_dir / "run_spec.json"))
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    run_spec_path.parent.mkdir(parents=True, exist_ok=True)

    write_fingerprint_contract(
        canonical_root=args.canonical_root,
        agents_relpath=args.agents_relpath,
        trainer_relpath=args.trainer_relpath,
        module_name=args.module_name,
        target_class=args.target_class,
        expected_canonical_alpha=args.alpha,
        output=contract_path,
        return_mode=args.return_mode,
    )
    write_run_spec(args, run_spec_path)
    return contract_path.resolve(), run_spec_path.resolve()


def sweep_command(args: argparse.Namespace, command: str, contract: Path, run_spec: Path) -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    grid_path = Path(args.grid).expanduser()
    if not grid_path.is_absolute():
        grid_path = repo_root / grid_path
    return [
        sys.executable,
        str(repo_root / "scripts" / "run_e7_canonical_sweep.py"),
        command,
        "--contract",
        str(contract),
        "--run-spec",
        str(run_spec),
        "--grid",
        str(grid_path.resolve()),
        "--work-dir",
        str(Path(args.work_dir).expanduser().resolve()),
        "--max-workers",
        str(args.max_workers),
    ]


def run_subprocess(command: Sequence[str]) -> int:
    print(" ".join(str(item) for item in command), flush=True)
    completed = subprocess.run(list(command), check=False)
    return int(completed.returncode)


def cmd_prepare(args: argparse.Namespace) -> int:
    contract, run_spec = prepare_common(args)
    print(json.dumps({"contract": str(contract), "run_spec": str(run_spec)}, indent=2))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    contract, run_spec = prepare_common(args)
    return run_subprocess(sweep_command(args, "plan", contract, run_spec))


def cmd_run(args: argparse.Namespace) -> int:
    contract, run_spec = prepare_common(args)
    command = sweep_command(args, "run", contract, run_spec)
    if args.resume:
        command.append("--resume")
    return run_subprocess(command)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--agents-relpath", default="agents.py")
    parser.add_argument("--trainer-relpath", default="train_sna2c_variant.py")
    parser.add_argument("--module-name", default="agents")
    parser.add_argument("--target-class", default="SNA2C_IQLV_ExpRankAgent")
    parser.add_argument("--return-mode", choices=["zero_float", "metrics_dict"], default="zero_float")
    parser.add_argument("--hopper-medium-replay-hdf5", required=True)
    parser.add_argument("--hopper-medium-replay-sha256")
    parser.add_argument("--hopper-medium-expert-hdf5", required=True)
    parser.add_argument("--hopper-medium-expert-sha256")
    parser.add_argument("--seeds", type=int, nargs="+", default=[200, 201])
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--contract-output")
    parser.add_argument("--run-spec-output")
    parser.add_argument("--grid", default=DEFAULT_GRID)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.11)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--temp", type=float, default=5.0)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-max-steps", type=int, default=None)
    parser.add_argument("--ckpt-interval", type=int, default=50_000)
    parser.add_argument("--last-pct", type=float, default=0.10)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--omp-threads", type=int, default=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func, help_text in (
        ("prepare", cmd_prepare, "write canonical_contract.json and run_spec.json only"),
        ("plan", cmd_plan, "prepare files and validate/materialize the execution plan"),
        ("run", cmd_run, "prepare files and run the two-dataset pilot"),
    ):
        child = sub.add_parser(name, help=help_text)
        add_common_args(child)
        child.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive")
    if args.omp_threads < 1:
        raise ValueError("--omp-threads must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
