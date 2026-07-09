"""Self-contained two-dataset canonical-agent pilot wrapper for E7.

This wrapper fixes the first E7 canonical-agent validation scope to two D4RL
Hopper cells:

* hopper-medium-replay-v2;
* hopper-medium-expert-v2.

It does not implement a new D4RL algorithm.  It fingerprints a vendored copy of
the user's old D4RL source tree, writes a concrete run spec for the historical
``train_sna2c_variant.py`` + ``SNA2C_IQLV_ExpRankAgent`` backbone, and then
delegates planning/running to ``drpo.e7_canonical_sweep``.

The runner is intentionally staged:

* ``reproduce``: original ExpRank_MR passthrough only, 1M steps by default;
* ``taper-pilot``: small taper/control grid, 300k steps by default;
* ``smoke``: original passthrough only, 20k steps by default for liveness;
* ``full-grid``: the broad exploratory grid, 1M steps by default.

Only ``reproduce`` is intended to check whether the old backbone can recover its
historical high-score scale.  Shorter profiles are liveness or pilot evidence,
not formal method results.
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
CONTRACT_VERSION_NOTE = "canonical-agent-two-dataset-v2-self-contained"

DATASET_IDS = ("hopper-medium-replay-v2", "hopper-medium-expert-v2")

PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "smoke": {
        "grid": "configs/e7_canonical_two_dataset_grid_reproduce_v2.json",
        "steps": 20_000,
        "eval_interval": 10_000,
        "ckpt_interval": 10_000,
        "eval_episodes": 1,
        "max_workers": 1,
        "run_kind": "smoke",
    },
    "reproduce": {
        "grid": "configs/e7_canonical_two_dataset_grid_reproduce_v2.json",
        "steps": 1_000_000,
        "eval_interval": 50_000,
        "ckpt_interval": 50_000,
        "eval_episodes": 10,
        "max_workers": 8,
        "run_kind": "pilot",
    },
    "taper-pilot": {
        "grid": "configs/e7_canonical_two_dataset_grid_small_taper_v2.json",
        "steps": 300_000,
        "eval_interval": 25_000,
        "ckpt_interval": 25_000,
        "eval_episodes": 5,
        "max_workers": 16,
        "run_kind": "pilot",
    },
    "full-grid": {
        "grid": "configs/e7_canonical_two_dataset_taper_grid_v1.json",
        "steps": 1_000_000,
        "eval_interval": 50_000,
        "ckpt_interval": 50_000,
        "eval_episodes": 10,
        "max_workers": 16,
        "run_kind": "pilot",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_canonical_root() -> Path:
    return repo_root() / "src" / "drpo" / "e7_canonical_vendor" / "d4rl"


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


def _resolve_dataset_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    replay = args.hopper_medium_replay_hdf5
    expert = args.hopper_medium_expert_hdf5
    data_dir = args.data_dir or os.environ.get("D4RL_DATA_DIR")
    if (replay is None or expert is None) and data_dir is None:
        raise ValueError(
            "provide either --data-dir/D4RL_DATA_DIR or both explicit HDF5 paths"
        )
    if replay is None:
        replay = str(Path(data_dir) / "hopper-medium-replay-v2.hdf5")  # type: ignore[arg-type]
    if expert is None:
        expert = str(Path(data_dir) / "hopper-medium-expert-v2.hdf5")  # type: ignore[arg-type]
    replay_path = Path(replay).expanduser().resolve()
    expert_path = Path(expert).expanduser().resolve()
    if not replay_path.is_file():
        raise FileNotFoundError(f"missing hopper-medium-replay-v2 dataset: {replay_path}")
    if not expert_path.is_file():
        raise FileNotFoundError(f"missing hopper-medium-expert-v2 dataset: {expert_path}")
    return replay_path, expert_path


def _profile_value(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return PROFILE_DEFAULTS[args.profile][name]


def resolve_grid_path(args: argparse.Namespace) -> Path:
    raw = Path(args.grid or PROFILE_DEFAULTS[args.profile]["grid"]).expanduser()
    if raw.is_absolute():
        return raw
    return repo_root() / raw


def build_trainer_template(args: argparse.Namespace) -> list[str]:
    steps = int(_profile_value(args, "steps"))
    eval_interval = int(_profile_value(args, "eval_interval"))
    eval_episodes = int(_profile_value(args, "eval_episodes"))
    ckpt_interval = int(_profile_value(args, "ckpt_interval"))
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
        str(steps),
        "--batch",
        str(args.batch),
        "--lr",
        str(args.lr),
        "--eval_interval",
        str(eval_interval),
        "--eval_episodes",
        str(eval_episodes),
        "--seed",
        "{seed}",
        "--out_dir",
        "{output_dir}",
        "--ckpt_dir",
        "{output_dir}/ckpts",
        "--ckpt_interval",
        str(ckpt_interval),
        "--last_pct",
        str(args.last_pct),
    ]
    if args.device is not None:
        template.extend(["--device", str(args.device)])
    if args.eval_max_steps is not None:
        template.extend(["--eval_max_steps", str(args.eval_max_steps)])
    return template


def write_run_spec(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    replay_path, expert_path = _resolve_dataset_paths(args)
    seeds = [int(item) for item in args.seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"seeds must be unique: {seeds}")
    profile_defaults = PROFILE_DEFAULTS[args.profile]
    payload = {
        "run_kind": profile_defaults["run_kind"],
        "experiment_id": EXPERIMENT_ID,
        "wrapper": CONTRACT_VERSION_NOTE,
        "profile": args.profile,
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
                    "fingerprinted vendored old source tree."
                ),
            }
        ],
        "environment": {
            "OMP_NUM_THREADS": str(args.omp_threads),
            "MKL_NUM_THREADS": str(args.omp_threads),
            "OPENBLAS_NUM_THREADS": str(args.omp_threads),
        },
        "interpretation_boundary": (
            "Two-dataset canonical-backbone parity/taper validation.  Smoke and "
            "short pilot profiles are not formal D4RL results.  Only the 1M-step "
            "original passthrough profile can be compared to historical ExpRank_MR "
            "scale, and even that remains a two-dataset pilot until expanded and "
            "terminal-audited."
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

    canonical_root = Path(args.canonical_root).expanduser().resolve()
    if not canonical_root.is_dir():
        raise FileNotFoundError(f"canonical root does not exist: {canonical_root}")
    write_fingerprint_contract(
        canonical_root=canonical_root,
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
    grid_path = resolve_grid_path(args)
    max_workers = int(args.max_workers if args.max_workers is not None else PROFILE_DEFAULTS[args.profile]["max_workers"])
    return [
        sys.executable,
        str(repo_root() / "scripts" / "run_e7_canonical_sweep.py"),
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
        str(max_workers),
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
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_DEFAULTS),
        default="reproduce",
        help=(
            "Execution profile. reproduce=1M original ExpRank_MR only; "
            "taper-pilot=300k small grid; smoke=20k liveness; full-grid=1M broad grid."
        ),
    )
    parser.add_argument("--canonical-root", default=str(default_canonical_root()))
    parser.add_argument("--agents-relpath", default="agents.py")
    parser.add_argument("--trainer-relpath", default="train_sna2c_variant.py")
    parser.add_argument("--module-name", default="agents")
    parser.add_argument("--target-class", default="SNA2C_IQLV_ExpRankAgent")
    parser.add_argument("--return-mode", choices=["zero_float", "metrics_dict"], default="zero_float")
    parser.add_argument("--data-dir")
    parser.add_argument("--hopper-medium-replay-hdf5")
    parser.add_argument("--hopper-medium-replay-sha256")
    parser.add_argument("--hopper-medium-expert-hdf5")
    parser.add_argument("--hopper-medium-expert-sha256")
    parser.add_argument("--seeds", type=int, nargs="+", default=[200, 201, 202, 203])
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--contract-output")
    parser.add_argument("--run-spec-output")
    parser.add_argument("--grid")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.11)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--temp", type=float, default=5.0)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--eval-max-steps", type=int, default=None)
    parser.add_argument("--ckpt-interval", type=int)
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
    return int(args.func(args))
