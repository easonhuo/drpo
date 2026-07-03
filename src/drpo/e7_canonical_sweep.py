"""Parallel, resumable pilot launcher for canonical D4RL negative-weight grids."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from drpo.e7_canonical_injection import (
    CanonicalContract,
    NegativeControl,
    sha256_file,
    write_fingerprint_contract,
)

EXPERIMENT_ID = "EXT-H-E7-BENCH-01"
SCIENTIFIC_STATUS = "lineage_recovery_weight_sweep_pilot_only"
RUNNER_VERSION = "1.0.0-canonical-source-adapter"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_sha256(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    id: str
    path: str
    sha256: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DatasetSpec":
        spec = cls(
            id=str(raw["id"]),
            path=str(raw["path"]),
            sha256=str(raw["sha256"]).lower(),
        )
        if len(spec.sha256) != 64:
            raise ValueError(f"dataset {spec.id} has invalid sha256")
        return spec

    def verify(self) -> dict[str, Any]:
        source = Path(self.path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"dataset {self.id} is missing: {source}")
        actual = sha256_file(source)
        if actual != self.sha256:
            raise RuntimeError(
                f"dataset {self.id} SHA-256 mismatch: expected {self.sha256}, "
                f"got {actual}"
            )
        return {
            "id": self.id,
            "path": str(source),
            "sha256": actual,
            "size_bytes": source.stat().st_size,
        }


@dataclasses.dataclass(frozen=True)
class Branch:
    branch_id: str
    branch_kind: str
    dataset: DatasetSpec
    seed: int
    template_values: dict[str, str]
    negative_control: NegativeControl | None

    def identity_payload(
        self,
        contract: CanonicalContract,
        grid_sha256: str,
        run_spec_sha256: str,
    ) -> dict[str, Any]:
        return {
            "experiment_id": EXPERIMENT_ID,
            "scientific_status": SCIENTIFIC_STATUS,
            "runner_version": RUNNER_VERSION,
            "contract": dataclasses.asdict(contract),
            "grid_sha256": grid_sha256,
            "run_spec_sha256": run_spec_sha256,
            "branch_id": self.branch_id,
            "branch_kind": self.branch_kind,
            "dataset_id": self.dataset.id,
            "dataset_sha256": self.dataset.sha256,
            "seed": self.seed,
            "template_values": self.template_values,
            "negative_control": (
                None
                if self.negative_control is None
                else dataclasses.asdict(self.negative_control)
            ),
        }


def _format_value(value: str, context: Mapping[str, Any]) -> str:
    try:
        return value.format_map(context)
    except KeyError as exc:
        raise ValueError(f"unknown trainer template placeholder: {exc.args[0]}") from exc


def load_grid(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = json.loads(source.read_text())
    if raw.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"grid experiment_id must be {EXPERIMENT_ID}")
    if raw.get("run_kind") != "pilot":
        raise ValueError("canonical weight sweep is pilot-only; formal launch is blocked")
    methods = {str(name).lower() for name in raw.get("negative_scale_grid", {})}
    if any("quartic" in name for name in methods):
        raise ValueError("quartic taper is intentionally excluded from this sweep")
    return raw, sha256_file(source)


def load_run_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = json.loads(source.read_text())
    if raw.get("run_kind") != "pilot":
        raise ValueError("run_spec.run_kind must be 'pilot'")
    required = {"datasets", "seeds", "trainer_argv_template"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"run spec is missing fields: {missing}")
    if not raw["datasets"] or not raw["seeds"]:
        raise ValueError("run spec requires at least one dataset and one seed")
    if len(set(int(seed) for seed in raw["seeds"])) != len(raw["seeds"]):
        raise ValueError("run spec seeds must be unique")
    return raw, sha256_file(source)


def _scale_label(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def expand_injected_controls(grid: Mapping[str, Any]) -> list[NegativeControl]:
    canonical_alpha = float(grid["canonical_alpha"])
    common = {
        "canonical_alpha": canonical_alpha,
        "reference_distance": float(grid["reference_distance"]),
        "reciprocal_linear_coefficient": float(
            grid["coefficients"]["reciprocal_linear"]
        ),
        "reciprocal_quadratic_coefficient": float(
            grid["coefficients"]["reciprocal_quadratic"]
        ),
        "exponential_coefficient": float(grid["coefficients"]["exponential"]),
    }
    controls = [
        NegativeControl(method="positive_only", negative_scale=0.0, **common),
        NegativeControl(method="canonical_signed", negative_scale=1.0, **common),
    ]
    for method, scales in grid["negative_scale_grid"].items():
        for scale in scales:
            controls.append(
                NegativeControl(
                    method=str(method),
                    negative_scale=float(scale),
                    **common,
                )
            )
    identities = [
        (control.method, control.negative_scale)
        for control in controls
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("negative control grid contains duplicate branches")
    return controls


def build_branches(
    contract: CanonicalContract,
    run_spec: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> list[Branch]:
    datasets = [DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
    seeds = [int(value) for value in run_spec["seeds"]]
    injected_values = {
        str(key): str(value)
        for key, value in run_spec.get("injected_template_values", {}).items()
    }
    branches: list[Branch] = []
    for dataset in datasets:
        for seed in seeds:
            for control in expand_injected_controls(grid):
                if control.canonical_alpha != contract.expected_canonical_alpha:
                    raise ValueError(
                        "grid canonical_alpha does not match canonical contract"
                    )
                branch_id = (
                    f"{dataset.id}__seed{seed}__{control.method}__"
                    f"scale{_scale_label(control.negative_scale)}"
                )
                branches.append(
                    Branch(
                        branch_id=branch_id,
                        branch_kind="injected",
                        dataset=dataset,
                        seed=seed,
                        template_values=dict(injected_values),
                        negative_control=control,
                    )
                )
            for raw_variant in run_spec.get("passthrough_variants", []):
                variant_id = str(raw_variant["id"])
                values = {
                    str(key): str(value)
                    for key, value in raw_variant.get("template_values", {}).items()
                }
                branch_id = f"{dataset.id}__seed{seed}__baseline__{variant_id}"
                branches.append(
                    Branch(
                        branch_id=branch_id,
                        branch_kind="passthrough",
                        dataset=dataset,
                        seed=seed,
                        template_values=values,
                        negative_control=None,
                    )
                )
    ids = [branch.branch_id for branch in branches]
    if len(ids) != len(set(ids)):
        raise ValueError("branch IDs are not unique")
    return branches


def branch_command(
    *,
    contract_path: Path,
    contract: CanonicalContract,
    branch: Branch,
    branch_dir: Path,
    trainer_argv_template: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    context: dict[str, Any] = {
        "canonical_root": str(contract.source_root),
        "dataset_id": branch.dataset.id,
        "dataset_path": str(Path(branch.dataset.path).expanduser().resolve()),
        "dataset_sha256": branch.dataset.sha256,
        "seed": branch.seed,
        "output_dir": str(branch_dir / "trainer_output"),
        "branch_id": branch.branch_id,
        **branch.template_values,
    }
    trainer_args = [_format_value(str(item), context) for item in trainer_argv_template]
    branch_config = {
        "branch_id": branch.branch_id,
        "branch_kind": branch.branch_kind,
        "dataset_id": branch.dataset.id,
        "dataset_sha256": branch.dataset.sha256,
        "seed": branch.seed,
        "template_values": branch.template_values,
        "negative_control": (
            None
            if branch.negative_control is None
            else dataclasses.asdict(branch.negative_control)
        ),
    }
    branch_config_path = branch_dir / "branch_config.json"
    atomic_write_json(branch_config_path, branch_config)
    command = [
        sys.executable,
        "-m",
        "drpo.e7_canonical_bootstrap",
        "--contract",
        str(contract_path),
        "--branch-config",
        str(branch_config_path),
        "--branch-manifest",
        str(branch_dir / "branch_manifest.json"),
        "--",
        *trainer_args,
    ]
    return command, branch_config


def _stream_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with log_path.open("w") as handle:
        for line in process.stdout:
            handle.write(line)
    return int(process.wait())


def execute_branch(
    *,
    contract_path: Path,
    contract: CanonicalContract,
    branch: Branch,
    work_dir: Path,
    grid_sha256: str,
    run_spec_sha256: str,
    trainer_argv_template: Sequence[str],
    base_environment: Mapping[str, str],
    resume: bool,
) -> dict[str, Any]:
    branch_dir = work_dir / "branches" / branch.branch_id
    branch_dir.mkdir(parents=True, exist_ok=True)
    identity_payload = branch.identity_payload(
        contract, grid_sha256, run_spec_sha256
    )
    identity = canonical_json_sha256(identity_payload)
    identity_path = branch_dir / "BRANCH_IDENTITY.json"
    done_path = branch_dir / "COMPLETED.json"
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text())
        if existing.get("identity_sha256") != identity:
            raise RuntimeError(
                f"branch directory identity mismatch: {branch.branch_id}"
            )
        if done_path.is_file() and resume:
            done = json.loads(done_path.read_text())
            return {"branch_id": branch.branch_id, "status": "skipped", **done}
        if not resume:
            raise RuntimeError(
                f"branch directory already exists; pass --resume: {branch.branch_id}"
            )
    else:
        if any(branch_dir.iterdir()):
            raise RuntimeError(
                f"non-empty branch directory has no identity: {branch_dir}"
            )
        atomic_write_json(
            identity_path,
            {
                **identity_payload,
                "identity_sha256": identity,
                "created_utc": utc_now(),
            },
        )

    command, branch_config = branch_command(
        contract_path=contract_path,
        contract=contract,
        branch=branch,
        branch_dir=branch_dir,
        trainer_argv_template=trainer_argv_template,
    )
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in base_environment.items()})
    environment["PYTHONUNBUFFERED"] = "1"
    environment["DRPO_E7_BRANCH_ID"] = branch.branch_id
    start = utc_now()
    atomic_write_json(
        branch_dir / "LAUNCH.json",
        {
            "started_utc": start,
            "command": command,
            "command_shell": shlex.join(command),
            "branch_config": branch_config,
        },
    )
    code = _stream_process(
        command,
        cwd=contract.source_root,
        environment=environment,
        log_path=branch_dir / "stdout_stderr.log",
    )
    result = {
        "branch_id": branch.branch_id,
        "return_code": code,
        "started_utc": start,
        "finished_utc": utc_now(),
    }
    if code == 0:
        atomic_write_json(done_path, result)
        return {"status": "completed", **result}
    atomic_write_json(branch_dir / "FAILED.json", result)
    return {"status": "failed", **result}


def write_plan(
    *,
    contract: CanonicalContract,
    branches: Sequence[Branch],
    grid_sha256: str,
    run_spec_sha256: str,
    work_dir: Path,
    max_workers: int,
) -> dict[str, Any]:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": SCIENTIFIC_STATUS,
        "runner_version": RUNNER_VERSION,
        "formal_launch_allowed": False,
        "formal_blocking_reason": (
            "The registered formal nine-task protocol remains blocked.  This "
            "grid is a lineage-recovery pilot and cannot establish a formal "
            "method ranking."
        ),
        "canonical_contract": dataclasses.asdict(contract),
        "grid_sha256": grid_sha256,
        "run_spec_sha256": run_spec_sha256,
        "max_workers": max_workers,
        "branch_count": len(branches),
        "branches": [
            {
                "branch_id": branch.branch_id,
                "branch_kind": branch.branch_kind,
                "dataset_id": branch.dataset.id,
                "seed": branch.seed,
                "negative_control": (
                    None
                    if branch.negative_control is None
                    else dataclasses.asdict(branch.negative_control)
                ),
                "template_values": branch.template_values,
            }
            for branch in branches
        ],
        "created_utc": utc_now(),
    }
    atomic_write_json(work_dir / "EXECUTION_PLAN.json", payload)
    return payload


def cmd_fingerprint(args: argparse.Namespace) -> int:
    contract = write_fingerprint_contract(
        canonical_root=args.canonical_root,
        agents_relpath=args.agents_relpath,
        trainer_relpath=args.trainer_relpath,
        module_name=args.module_name,
        target_class=args.target_class,
        expected_canonical_alpha=args.expected_canonical_alpha,
        output=args.output,
        return_mode=args.return_mode,
    )
    print(json.dumps(dataclasses.asdict(contract), indent=2))
    return 0


def _prepare(args: argparse.Namespace) -> tuple[
    Path,
    CanonicalContract,
    dict[str, Any],
    dict[str, Any],
    str,
    str,
    list[Branch],
    Path,
]:
    contract_path = Path(args.contract).expanduser().resolve()
    contract = CanonicalContract.load(contract_path)
    contract.verify_runtime()
    run_spec, run_spec_sha256 = load_run_spec(args.run_spec)
    grid, grid_sha256 = load_grid(args.grid)
    branches = build_branches(contract, run_spec, grid)
    for dataset in {branch.dataset for branch in branches}:
        dataset.verify()
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    return (
        contract_path,
        contract,
        run_spec,
        grid,
        grid_sha256,
        run_spec_sha256,
        branches,
        work_dir,
    )


def cmd_plan(args: argparse.Namespace) -> int:
    (
        _,
        contract,
        _,
        _,
        grid_sha256,
        run_spec_sha256,
        branches,
        work_dir,
    ) = _prepare(args)
    plan = write_plan(
        contract=contract,
        branches=branches,
        grid_sha256=grid_sha256,
        run_spec_sha256=run_spec_sha256,
        work_dir=work_dir,
        max_workers=args.max_workers,
    )
    print(json.dumps(plan, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    (
        contract_path,
        contract,
        run_spec,
        _,
        grid_sha256,
        run_spec_sha256,
        branches,
        work_dir,
    ) = _prepare(args)
    plan = write_plan(
        contract=contract,
        branches=branches,
        grid_sha256=grid_sha256,
        run_spec_sha256=run_spec_sha256,
        work_dir=work_dir,
        max_workers=args.max_workers,
    )
    stable_plan = {key: value for key, value in plan.items() if key != "created_utc"}
    run_identity = canonical_json_sha256(stable_plan)
    run_identity_path = work_dir / "RUN_IDENTITY.json"
    if run_identity_path.is_file():
        existing = json.loads(run_identity_path.read_text())
        if existing.get("run_identity_sha256") != run_identity:
            raise RuntimeError(
                "work directory belongs to another canonical sweep; use a new path"
            )
        if not args.resume:
            raise RuntimeError("work directory exists; pass --resume or use a new path")
    else:
        atomic_write_json(
            run_identity_path,
            {"run_identity_sha256": run_identity, "plan": plan},
        )

    environment = {
        str(key): str(value)
        for key, value in run_spec.get("environment", {}).items()
    }
    trainer_template = [str(item) for item in run_spec["trainer_argv_template"]]
    results: list[dict[str, Any]] = []
    print_lock = threading.Lock()

    def run_one(branch: Branch) -> dict[str, Any]:
        result = execute_branch(
            contract_path=contract_path,
            contract=contract,
            branch=branch,
            work_dir=work_dir,
            grid_sha256=grid_sha256,
            run_spec_sha256=run_spec_sha256,
            trainer_argv_template=trainer_template,
            base_environment=environment,
            resume=args.resume,
        )
        with print_lock:
            print(json.dumps(result, sort_keys=True), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_one, branch) for branch in branches]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda row: row["branch_id"])
    summary = {
        "finished_utc": utc_now(),
        "branch_count": len(results),
        "completed": sum(row["status"] in {"completed", "skipped"} for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "results": results,
    }
    atomic_write_json(work_dir / "RUN_SUMMARY.json", summary)
    if summary["failed"]:
        raise RuntimeError(f"{summary['failed']} canonical sweep branches failed")
    return 0


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--contract", required=True)
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=40)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fingerprint = subparsers.add_parser(
        "fingerprint",
        help="fingerprint the user's original canonical D4RL source tree",
    )
    fingerprint.add_argument("--canonical-root", required=True)
    fingerprint.add_argument("--agents-relpath", required=True)
    fingerprint.add_argument("--trainer-relpath", required=True)
    fingerprint.add_argument("--module-name", default="agents")
    fingerprint.add_argument("--target-class", required=True)
    fingerprint.add_argument("--expected-canonical-alpha", type=float, default=0.11)
    fingerprint.add_argument(
        "--return-mode",
        choices=["zero_float", "metrics_dict"],
        default="zero_float",
    )
    fingerprint.add_argument("--output", required=True)
    fingerprint.set_defaults(func=cmd_fingerprint)

    plan = subparsers.add_parser("plan", help="validate and materialize all jobs")
    add_common_run_args(plan)
    plan.set_defaults(func=cmd_plan)

    run = subparsers.add_parser("run", help="run all branches in parallel")
    add_common_run_args(run)
    run.add_argument("--resume", action="store_true")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "max_workers") and args.max_workers < 1:
        raise ValueError("max_workers must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
