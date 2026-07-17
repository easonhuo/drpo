#!/usr/bin/env python3
"""Run the matched real-data TD/GAE liveness gate for EXT-H-E7-SQEXP-GAE-01."""
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

DATASET, SEED, COEFFICIENT = pilot.EXPECTED_DATASETS[0], pilot.EXPECTED_SEEDS[0], 128.0
_DEFAULT_CONTRACT = "/root/d4rl2/configs/e7_canonical_contract_9task.json"
_DEFAULT_RUN_SPEC = "/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json"
_DEFAULT_WORK_DIR = "outputs/e7/sqexp_gae_joint_liveness_001"


def _probe_template(run_spec: dict[str, Any], steps: int) -> list[str]:
    argv = [str(value) for value in run_spec["trainer_argv_template"]]
    for flag, expected, replacement in (
        ("--eval_interval", "50000", str(steps)),
        ("--eval_episodes", "10", "1"),
    ):
        positions = [index for index, token in enumerate(argv) if token == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(argv):
            raise RuntimeError(f"probe template must contain exactly one {flag}")
        index = positions[0] + 1
        if argv[index] != expected:
            raise RuntimeError(f"canonical {flag} changed: {argv[index]} != {expected}")
        argv[index] = replacement
    return argv


def _representative(branches: Sequence[base.Branch], estimator: str) -> base.Branch:
    matches = [
        branch
        for branch in branches
        if branch.dataset.id == DATASET
        and branch.seed == SEED
        and branch.template_values["advantage_estimator"] == estimator
        and branch.template_values["weight_method"] == "squared_exponential"
        and float(branch.template_values["exp_coefficient"]) == COEFFICIENT
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one representative {estimator} branch, found {len(matches)}")
    return matches[0]


def _probe_branch(branch: base.Branch, steps: int) -> base.Branch:
    return dataclasses.replace(
        branch,
        branch_id=branch.branch_id.replace("steps1m", f"liveness_steps{steps}"),
        template_values={**branch.template_values, "steps": str(steps)},
    )


def _validate_matched_critic(records: list[dict[str, Any]]) -> None:
    if len(records) != len(pilot.ESTIMATORS):
        raise RuntimeError("paired liveness did not produce both estimators")
    if len({tuple(record["snapshot_hashes"]) for record in records}) != 1:
        raise RuntimeError("TD and GAE critic snapshot trajectories diverged")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=os.getenv("E7_CANONICAL_CONTRACT", _DEFAULT_CONTRACT))
    parser.add_argument("--run-spec", default=os.getenv("E7_CANONICAL_RUN_SPEC", _DEFAULT_RUN_SPEC))
    parser.add_argument("--grid", default="configs/e7_sqexp_gae_v2.json")
    parser.add_argument("--work-dir", default=os.getenv("E7_SQEXP_GAE_LIVENESS_WORK_DIR", _DEFAULT_WORK_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise RuntimeError("refusing to run liveness from a dirty checkout")
    work = (repo / args.work_dir).resolve()
    if work.exists() and any(work.iterdir()):
        raise RuntimeError(f"liveness work directory is not empty: {work}")
    work.mkdir(parents=True, exist_ok=True)

    contract_path = Path(args.contract).expanduser().resolve()
    contract = base.CanonicalContract.load(contract_path)
    contract.verify_runtime()
    run_spec, run_sha = pilot._load_run_spec(args.run_spec)  # noqa: SLF001
    grid, grid_sha = pilot._load_grid(args.grid)  # noqa: SLF001
    branches = pilot._build_branches(contract, run_spec, grid)  # noqa: SLF001
    anchor = _representative(branches, "td")
    anchor.dataset.verify()
    with h5py.File(Path(anchor.dataset.path).expanduser().resolve(), "r") as handle:
        transition_count = int(handle["rewards"].shape[0])
    if transition_count <= 0:
        raise RuntimeError("representative dataset is empty")
    interval = math.ceil(transition_count / pilot.CANONICAL_BATCH_SIZE)
    steps, trainer_template = interval + 1, _probe_template(run_spec, interval + 1)
    environment = {str(key): str(value) for key, value in run_spec.get("environment", {}).items()}
    environment["DRPO_RUNTIME_RESOURCE_PROBE"] = "1"

    previous = base.EXPERIMENT_ID, base.SCIENTIFIC_STATUS, base.RUNNER_VERSION, base.branch_command
    base.EXPERIMENT_ID, base.SCIENTIFIC_STATUS, base.RUNNER_VERSION = (
        pilot.EXPERIMENT_ID,
        pilot.SCIENTIFIC_STATUS,
        pilot.RUNNER_VERSION,
    )
    base.branch_command = pilot._branch_command  # noqa: SLF001
    records: list[dict[str, Any]] = []
    try:
        for estimator in pilot.ESTIMATORS:
            branch = _probe_branch(_representative(branches, estimator), steps)
            started = time.monotonic()
            result = base.execute_branch(
                contract_path=contract_path,
                contract=contract,
                branch=branch,
                work_dir=work,
                grid_sha256=grid_sha,
                run_spec_sha256=run_sha,
                trainer_argv_template=trainer_template,
                base_environment=environment,
                resume=False,
            )
            manifest_path = work / "branches" / branch.branch_id / "branch_manifest.json"
            snapshot = json.loads(manifest_path.read_text())["trajectory_snapshot"]
            if result["status"] != "completed":
                raise RuntimeError(f"{estimator} liveness branch did not complete")
            if snapshot["critic_evolution_observed"] is not True or int(snapshot["snapshot_count"]) < 2:
                raise RuntimeError(f"{estimator} liveness did not observe critic evolution")
            if int(snapshot["snapshot_refresh_interval"]) != interval:
                raise RuntimeError(f"{estimator} snapshot cadence changed")
            records.append(
                {
                    "estimator": estimator,
                    "branch_id": branch.branch_id,
                    "elapsed_seconds": time.monotonic() - started,
                    "snapshot_count": int(snapshot["snapshot_count"]),
                    "snapshot_hashes": list(snapshot["snapshot_hashes"]),
                    "critic_evolution_observed": True,
                }
            )
        _validate_matched_critic(records)
        summary = {
            "status": "PASS",
            "experiment_id": pilot.EXPERIMENT_ID,
            "scientific_result_available": False,
            "formal_evidence_allowed": False,
            "full_matrix_branch_count": len(branches),
            "liveness_branch_count": len(records),
            "dataset": DATASET,
            "seed": SEED,
            "exp_coefficient": COEFFICIENT,
            "probe_steps": steps,
            "snapshot_refresh_interval": interval,
            "matched_critic_snapshot_trajectories": True,
            "critic_updated_during_actor_training": True,
            "prepared_advantage_artifact_used": False,
            "held_out_seeds_touched": False,
            "task_performance_collapse_event": "not_adjudicated_liveness_only",
            "support_or_variance_boundary_event": "not_adjudicated_liveness_only",
            "nan_inf_numerical_failure": False,
            "branches": records,
        }
        base.atomic_write_json(work / "LIVENESS_SUMMARY.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        base.EXPERIMENT_ID, base.SCIENTIFIC_STATUS, base.RUNNER_VERSION, base.branch_command = previous


if __name__ == "__main__":
    raise SystemExit(main())
