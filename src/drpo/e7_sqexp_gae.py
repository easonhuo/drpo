"""Orchestrate preparation and the 192-branch EXT-H-E7-SQEXP-GAE-01 pilot."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from drpo import e7_canonical_sweep as base
from drpo import e7_sqexp_gae_contract as protocol
from drpo.e7_canonical_injection import CanonicalContract
from drpo.e7_offline_gae import atomic_write_json, sha256_file
from drpo.e7_sqexp_gae_aggregate import aggregate as aggregate_results
from drpo.e7_sqexp_gae_prepare import main as prepare_main


EXPERIMENT_ID = protocol.EXPERIMENT_ID
SCIENTIFIC_STATUS = protocol.SCIENTIFIC_STATUS
RUNNER_VERSION = protocol.RUNNER_VERSION
EXPECTED_SEEDS = protocol.EXPECTED_SEEDS
HELD_OUT_SEEDS = protocol.HELD_OUT_SEEDS
EXPECTED_DATASETS = protocol.EXPECTED_DATASETS
EXPECTED_PAIRS = 12
EXPECTED_BRANCHES = protocol.EXPECTED_BRANCHES
EXPECTED_PREPARER_VERSION = "1.0.1-e7-sqexp-gae"
GAE_CROSSCHECK_ATOL = 1e-6
load_grid = protocol.load_grid

_ORIGINAL_WRITE_PLAN = base.write_plan


def _activate_protocol() -> None:
    """Compatibility hook retained for focused tests; the contract is already frozen."""


def _prepared_dir(work_dir: Path, dataset_id: str, seed: int) -> Path:
    return work_dir / "prepared" / dataset_id / f"seed{seed}"


def _verify_prepared(
    directory: Path,
    *,
    dataset_id: str,
    dataset_sha256: str,
    seed: int,
) -> dict[str, Any]:
    manifest_path = directory / "ADVANTAGE_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"prepare TD/GAE artifacts before plan/run; missing {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"prepared artifact is incomplete: {directory}")
    if manifest.get("preparer_version") != EXPECTED_PREPARER_VERSION:
        raise RuntimeError(
            "prepared artifact predates the corrected precision gate: "
            f"{directory}"
        )
    if manifest.get("dataset_id") != dataset_id:
        raise RuntimeError(f"prepared artifact dataset mismatch: {directory}")
    if int(manifest.get("seed", -1)) != seed:
        raise RuntimeError(f"prepared artifact seed mismatch: {directory}")
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise RuntimeError(f"prepared artifact dataset hash mismatch: {directory}")
    critic = manifest.get("critic", {})
    advantages = manifest.get("advantages", {})
    critic_path = Path(critic.get("path", "")).expanduser().resolve()
    advantages_path = Path(advantages.get("path", "")).expanduser().resolve()
    if not critic_path.is_file() or sha256_file(critic_path) != critic.get("sha256"):
        raise RuntimeError(f"prepared critic verification failed: {directory}")
    if not advantages_path.is_file() or sha256_file(advantages_path) != advantages.get(
        "sha256"
    ):
        raise RuntimeError(f"prepared advantage verification failed: {directory}")
    with np.load(advantages_path, allow_pickle=False) as arrays:
        if "td" not in arrays or "gae" not in arrays:
            raise RuntimeError(f"prepared advantage keys are incomplete: {directory}")
        if arrays["td"].dtype != np.float32 or arrays["gae"].dtype != np.float32:
            raise RuntimeError(
                f"prepared actor advantages must remain float32: {directory}"
            )
    audit = manifest.get("trajectory_audit", {})
    if audit.get("status") != "PASS":
        raise RuntimeError(f"ordered trajectory audit is not PASS: {directory}")
    diagnostics = advantages.get("diagnostics", {})
    if float(diagnostics.get("lambda_zero_max_abs_error", 1.0)) > GAE_CROSSCHECK_ATOL:
        raise RuntimeError(f"lambda=0 regression failed: {directory}")
    float64_error = float(
        diagnostics.get("numpy_torch_float64_max_abs_error", 1.0)
    )
    compatibility_error = float(diagnostics.get("numpy_torch_max_abs_error", 1.0))
    if float64_error > GAE_CROSSCHECK_ATOL:
        raise RuntimeError(f"NumPy/Torch float64 GAE cross-check failed: {directory}")
    if not math.isclose(
        compatibility_error,
        float64_error,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError(f"GAE compatibility diagnostic mismatch: {directory}")
    if diagnostics.get("stored_gae_dtype") != "float32":
        raise RuntimeError(f"stored GAE dtype verification failed: {directory}")
    if diagnostics.get("stored_gae_matches_float64_reference_cast") is not True:
        raise RuntimeError(f"stored GAE reference-cast verification failed: {directory}")
    if float(diagnostics.get("stored_gae_vs_float64_cast_max_abs_error", 1.0)) != 0.0:
        raise RuntimeError(f"stored GAE reference-cast mismatch: {directory}")
    quantization_error = float(
        diagnostics.get("gae_float32_storage_quantization_max_abs_error", math.nan)
    )
    if not math.isfinite(quantization_error) or quantization_error < 0.0:
        raise RuntimeError(f"invalid GAE storage quantization diagnostic: {directory}")
    return manifest


def _write_plan_with_prepared_gate(**kwargs: Any) -> dict[str, Any]:
    work_dir = Path(kwargs["work_dir"]).expanduser().resolve()
    seen: set[tuple[str, int]] = set()
    for branch in kwargs["branches"]:
        key = (branch.dataset.id, int(branch.seed))
        if key in seen:
            continue
        seen.add(key)
        _verify_prepared(
            _prepared_dir(work_dir, branch.dataset.id, int(branch.seed)),
            dataset_id=branch.dataset.id,
            dataset_sha256=branch.dataset.sha256,
            seed=int(branch.seed),
        )
    if len(seen) != EXPECTED_PAIRS:
        raise RuntimeError(f"expected {EXPECTED_PAIRS} prepared pairs, got {len(seen)}")
    return _ORIGINAL_WRITE_PLAN(**kwargs)


def cmd_prepare(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Prepare shared frozen-critic TD/GAE")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(list(argv))

    contract = CanonicalContract.load(args.contract)
    grid, _ = protocol.load_grid(args.grid)
    run_spec, _ = protocol.load_run_spec(args.run_spec)
    branches = protocol.build_branches(contract, run_spec, grid)
    pairs: dict[tuple[str, int], Any] = {}
    for branch in branches:
        pairs[(branch.dataset.id, int(branch.seed))] = branch.dataset
    if len(pairs) != EXPECTED_PAIRS:
        raise RuntimeError(f"expected {EXPECTED_PAIRS} dataset/seed pairs, got {len(pairs)}")

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    shared = grid["shared_critic"]
    results: list[dict[str, Any]] = []
    for (dataset_id, seed), dataset in sorted(pairs.items()):
        dataset.verify()
        output_dir = _prepared_dir(work_dir, dataset_id, seed)
        command = [
            "--contract",
            str(Path(args.contract).expanduser().resolve()),
            "--dataset-id",
            dataset_id,
            "--dataset-path",
            dataset.path,
            "--dataset-sha256",
            dataset.sha256,
            "--seed",
            str(seed),
            "--output-dir",
            str(output_dir),
            "--critic-steps",
            str(shared["steps"]),
            "--batch-size",
            str(shared["batch_size"]),
            "--learning-rate",
            str(shared["learning_rate"]),
            "--gamma",
            str(shared["gamma"]),
            "--expectile",
            str(shared["expectile"]),
            "--gae-lambda",
            str(shared["gae_lambda"]),
            "--device",
            str(shared["device"]),
        ]
        if args.resume:
            command.append("--resume")
        prepare_main(command)
        manifest = _verify_prepared(
            output_dir,
            dataset_id=dataset_id,
            dataset_sha256=dataset.sha256,
            seed=seed,
        )
        results.append(
            {
                "dataset_id": dataset_id,
                "seed": seed,
                "status": manifest["status"],
                "trajectory_audit": manifest["trajectory_audit"],
            }
        )
        print(json.dumps(results[-1], sort_keys=True), flush=True)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "prepared_pairs": len(results),
        "expected_pairs": EXPECTED_PAIRS,
        "branches_started": 0,
        "held_out_seeds_touched": False,
        "formal_result": False,
        "results": results,
    }
    atomic_write_json(work_dir / "PREPARE_SUMMARY.json", summary)
    return 0


def _work_dir_from_run(argv: list[str]) -> str | None:
    if not argv or argv[0] != "run":
        return None
    if "--work-dir" not in argv:
        raise ValueError("run command is missing --work-dir")
    index = argv.index("--work-dir")
    if index + 1 >= len(argv):
        raise ValueError("run command is missing the --work-dir value")
    return argv[index + 1]


def main(argv: list[str] | None = None) -> int:
    delegated = list(sys.argv[1:] if argv is None else argv)
    if delegated and delegated[0] == "prepare":
        return cmd_prepare(delegated[1:])

    previous = (
        base.EXPERIMENT_ID,
        base.SCIENTIFIC_STATUS,
        base.RUNNER_VERSION,
        base.load_grid,
        base.load_run_spec,
        base.build_branches,
        base.branch_command,
        base.write_plan,
    )
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.SCIENTIFIC_STATUS = SCIENTIFIC_STATUS
    base.RUNNER_VERSION = RUNNER_VERSION
    base.load_grid = protocol.load_grid
    base.load_run_spec = protocol.load_run_spec
    base.build_branches = protocol.build_branches
    base.branch_command = protocol.branch_command
    base.write_plan = _write_plan_with_prepared_gate
    work_dir = _work_dir_from_run(delegated)
    try:
        try:
            result = int(base.main(delegated))
        except BaseException:
            if work_dir is not None:
                try:
                    aggregate_results(work_dir)
                except BaseException:
                    pass
            raise
        if work_dir is not None:
            aggregate_results(work_dir)
        return result
    finally:
        (
            base.EXPERIMENT_ID,
            base.SCIENTIFIC_STATUS,
            base.RUNNER_VERSION,
            base.load_grid,
            base.load_run_spec,
            base.build_branches,
            base.branch_command,
            base.write_plan,
        ) = previous


if __name__ == "__main__":
    raise SystemExit(main())
