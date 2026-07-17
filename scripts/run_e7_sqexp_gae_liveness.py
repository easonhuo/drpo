#!/usr/bin/env python3
"""Run one matched TD/GAE real-data liveness pair for EXT-H-E7-SQEXP-GAE-01."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import h5py

from drpo import e7_canonical_sweep as base
from drpo import e7_sqexp_gae as pilot

REPRESENTATIVE_DATASET = pilot.EXPECTED_DATASETS[0]
REPRESENTATIVE_SEED = pilot.EXPECTED_SEEDS[0]
REPRESENTATIVE_COEFFICIENT = 128.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default=os.environ.get(
            "E7_CANONICAL_CONTRACT",
            "/root/d4rl2/configs/e7_canonical_contract_9task.json",
        ),
    )
    parser.add_argument(
        "--run-spec",
        default=os.environ.get(
            "E7_CANONICAL_RUN_SPEC",
            "/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json",
        ),
    )
    parser.add_argument("--grid", default="configs/e7_sqexp_gae_v2.json")
    parser.add_argument(
        "--work-dir",
        default=os.environ.get(
            "E7_SQEXP_GAE_LIVENESS_WORK_DIR",
            "outputs/e7/sqexp_gae_joint_liveness_001",
        ),
    )
    return parser


def _replace_flag(argv: list[str], flag: str, expected: str, replacement: str) -> None:
    positions = [index for index, token in enumerate(argv) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise RuntimeError(f"probe template must contain exactly one {flag}")
    index = positions[0] + 1
    if argv[index] != expected:
        raise RuntimeError(f"canonical {flag} changed: {argv[index]} != {expected}")
    argv[index] = replacement


def _probe_template(run_spec: dict[str, Any], probe_steps: int) -> list[str]:
    argv = [str(value) for value in run_spec["trainer_argv_template"]]
    _replace_flag(argv, "--eval_interval", "50000", str(probe_steps))
    _replace_flag(argv, "--eval_episodes", "10", "1")
    return argv


def _representative(branches: Sequence[base.Branch], estimator: str) -> base.Branch:
    matches = [
        branch
        for branch in branches
        if branch.dataset.id == REPRESENTATIVE_DATASET
        and branch.seed == REPRESENTATIVE_SEED
        and branch.template_values["advantage_estimator"] == estimator
        and branch.template_values["weight_method"] == "squared_exponential"
        and math.isclose(
            float(branch.template_values["exp_coefficient"]),
            REPRESENTATIVE_COEFFICIENT,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one representative {estimator} branch, found {len(matches)}")
    return matches[0]


def _probe_branch(branch: base.Branch, probe_steps: int) -> base.Branch:
    values = dict(branch.template_values)
    values["steps"] = str(probe_steps)
    return dataclasses.replace(
        branch,
        branch_id=branch.branch_id.replace("steps1m", f"liveness_steps{probe_steps}"),
        template_values=values,
    )


def _transition_count(dataset_path: str | Path) -> int:
    with h5py.File(Path(dataset_path).expanduser().resolve(), "r") as handle:
        count = int(handle["rewards"].shape[0])
    if count <= 0:
        raise RuntimeError("representative dataset is empty")
    return count


def _require_clean(repo: Path) -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip()
    if status:
        raise RuntimeError("refusing to run liveness from a dirty checkout")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    _require_clean(repo)
    work_dir = (repo / args.work_dir).resolve()
    if work_dir.exists() and any(work_dir.iterdir()):
        raise RuntimeError(f"liveness work directory is not empty: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    contract_path = Path(args.contract).expanduser().resolve()
    contract = base.CanonicalContract.load(contract_path)
    contract.verify_runtime()
    run_spec, run_spec_sha256 = pilot._load_run_spec(args.run_spec)  # noqa: SLF001
    grid, grid_sha256 = pilot._load_grid(args.grid)  # noqa: SLF001
    branches = pilot._build_branches(contract, run_spec, grid)  # noqa: SLF001
    representative = _representative(branches, "td")
    representative.dataset.verify()
    refresh_interval = math.ceil(
        _transition_count(representative.dataset.path) / pilot.CANONICAL_BATCH_SIZE
    )
    probe_steps = refresh_interval + 1
    trainer_template = _probe_template(run_spec, probe_steps)
    environment = {str(key): str(value) for key, value in run_spec.get("environment", {}).items()}
    environment["DRPO_RUNTIME_RESOURCE_PROBE"] = "1"

    previous = (
        base.EXPERIMENT_ID,
        base.SCIENTIFIC_STATUS,
        base.RUNNER_VERSION,
        base.branch_command,
    )
    base.EXPERIMENT_ID = pilot.EXPERIMENT_ID
    base.SCIENTIFIC_STATUS = pilot.SCIENTIFIC_STATUS
    base.RUNNER_VERSION = pilot.RUNNER_VERSION
    base.branch_command = pilot._branch_command  # noqa: SLF001
    records: list[dict[str, Any]] = []
    try:
        for estimator in pilot.ESTIMATORS:
            branch = _probe_branch(_representative(branches, estimator), probe_steps)
            started = time.monotonic()
            result = base.execute_branch(
                contract_path=contract_path,
                contract=contract,
                branch=branch,
                work_dir=work_dir,
                grid_sha256=grid_sha256,
                run_spec_sha256=run_spec_sha256,
                trainer_argv_template=trainer_template,
                base_environment=environment,
                resume=False,
            )
            manifest = json.loads(
                (work_dir / "branches" / branch.branch_id / "branch_manifest.json").read_text()
            )
            snapshot = manifest["trajectory_snapshot"]
            if result["status"] != "completed":
                raise RuntimeError(f"{estimator} liveness branch did not complete")
            if int(snapshot["snapshot_count"]) < 2 or snapshot["critic_evolution_observed"] is not True:
                raise RuntimeError(f"{estimator} liveness did not observe joint-critic evolution")
            if int(snapshot["snapshot_refresh_interval"]) != refresh_interval:
                raise RuntimeError(f"{estimator} snapshot cadence changed")
            records.append(
                {
                    "estimator": estimator,
                    "branch_id": branch.branch_id,
                    "elapsed_seconds": time.monotonic() - started,
                    "snapshot_count": int(snapshot["snapshot_count"]),
                    "critic_evolution_observed": True,
                }
            )
        summary = {
            "status": "PASS",
            "experiment_id": pilot.EXPERIMENT_ID,
            "scientific_result_available": False,
            "formal_evidence_allowed": False,
            "full_matrix_branch_count": len(branches),
            "liveness_branch_count": len(records),
            "dataset": REPRESENTATIVE_DATASET,
            "seed": REPRESENTATIVE_SEED,
            "exp_coefficient": REPRESENTATIVE_COEFFICIENT,
            "probe_steps": probe_steps,
            "snapshot_refresh_interval": refresh_interval,
            "critic_updated_during_actor_training": True,
            "prepared_advantage_artifact_used": False,
            "held_out_seeds_touched": False,
            "task_performance_collapse_event": "not_adjudicated_liveness_only",
            "support_or_variance_boundary_event": "not_adjudicated_liveness_only",
            "nan_inf_numerical_failure": False,
            "branches": records,
        }
        base.atomic_write_json(work_dir / "LIVENESS_SUMMARY.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        (
            base.EXPERIMENT_ID,
            base.SCIENTIFIC_STATUS,
            base.RUNNER_VERSION,
            base.branch_command,
        ) = previous


if __name__ == "__main__":
    raise SystemExit(main())
