"""Orchestrate preparation and the 192-branch EXT-H-E7-SQEXP-GAE-01 pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from drpo import e7_canonical_sweep as base
from drpo import e7_sqexp_gae_contract as protocol
from drpo.e7_canonical_injection import CanonicalContract
from drpo.e7_offline_gae import atomic_write_json, sha256_file
from drpo.e7_sqexp_gae_aggregate import aggregate as aggregate_results
from drpo.e7_sqexp_gae_prepare import main as prepare_main


EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-01"
SCIENTIFIC_STATUS = "frozen_critic_trajectory_gae_development_pilot_only"
RUNNER_VERSION = "1.0.0-e7-sqexp-gae"
EXPECTED_SEEDS = (200, 201, 202, 203)
HELD_OUT_SEEDS = (204, 205, 206, 207)
EXPECTED_PAIRS = 12
EXPECTED_BRANCHES = 192
EXPECTED_DATASETS = (
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
)
EXPECTED_ACTOR_MODES = ("a2c", "ppo_clip_k4")
EXPECTED_ADVANTAGE_MODES = ("one_step_td", "gae_lambda_0p95")
EXPECTED_COEFFICIENTS = (64.0, 128.0, 256.0)

_ORIGINAL_WRITE_PLAN = base.write_plan


def _activate_protocol() -> None:
    """Apply the reviewed successor seed contract to the low-level builder."""

    protocol.EXPECTED_SEEDS = EXPECTED_SEEDS
    protocol.EXPECTED_BRANCHES = EXPECTED_BRANCHES


def _require_exact_mapping(
    raw: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if dict(raw) != dict(expected):
        raise ValueError(f"{label} changed from the frozen GAE pilot contract")


def load_grid(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load one canonical schema and normalize it for the low-level branch builder."""

    source = Path(path)
    raw = json.loads(source.read_text())
    compatibility_fields = {
        "seeds",
        "advantage_estimators",
        "shared_critic",
        "expected_branches",
    }
    duplicated = sorted(compatibility_fields & set(raw))
    if duplicated:
        raise ValueError(
            "GAE grid must use the canonical schema only; duplicated compatibility "
            f"fields are forbidden: {duplicated}"
        )
    if raw.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("GAE grid experiment_id mismatch")
    if raw.get("run_kind") != "pilot" or raw.get("status") != "not_run":
        raise ValueError("GAE grid must remain an unrun development pilot")
    if raw.get("scientific_status") != SCIENTIFIC_STATUS:
        raise ValueError("GAE scientific_status changed")
    if tuple(raw.get("datasets", ())) != EXPECTED_DATASETS:
        raise ValueError("GAE dataset matrix changed")
    if tuple(int(value) for value in raw.get("development_seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("GAE development seeds changed")
    if tuple(int(value) for value in raw.get("held_out_seeds", ())) != HELD_OUT_SEEDS:
        raise ValueError("GAE held-out seeds changed")
    if tuple(raw.get("actor_update_modes", ())) != EXPECTED_ACTOR_MODES:
        raise ValueError("GAE actor update modes changed")
    if tuple(raw.get("advantage_modes", ())) != EXPECTED_ADVANTAGE_MODES:
        raise ValueError("GAE advantage modes changed")
    if int(raw.get("steps", -1)) != 1_000_000:
        raise ValueError("GAE actor horizon changed")
    if int(raw.get("evaluation_interval", -1)) != 50_000:
        raise ValueError("GAE evaluation interval changed")
    if int(raw.get("evaluation_episodes", -1)) != 10:
        raise ValueError("GAE evaluation episode count changed")

    shared = raw.get("shared_frozen_critic")
    if not isinstance(shared, Mapping):
        raise ValueError("shared_frozen_critic must be a mapping")
    _require_exact_mapping(
        shared,
        {
            "steps": 100_000,
            "batch": 256,
            "gamma": 0.99,
            "tau": 0.5,
            "lr": 3e-4,
            "temperature": 5.0,
            "device": "cpu",
            "shared_per_dataset_seed": True,
            "updated_during_actor_training": False,
        },
        label="shared frozen critic",
    )
    trajectory = raw.get("trajectory_advantage")
    if not isinstance(trajectory, Mapping):
        raise ValueError("trajectory_advantage must be a mapping")
    _require_exact_mapping(
        trajectory,
        {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ordered_behavior_trajectory": True,
            "terminal_bootstrap": False,
            "timeout_bootstrap": True,
            "terminal_stops_recursion": True,
            "timeout_stops_recursion": True,
            "tail_bootstrap_and_stop_recursion": True,
            "lambda_zero_must_equal_one_step": True,
            "normalization": "none",
            "clipping": "none",
        },
        label="trajectory advantage",
    )
    weight = raw.get("weight_control")
    if not isinstance(weight, Mapping):
        raise ValueError("weight_control must be a mapping")
    if float(weight.get("weight_at_zero", -1.0)) != 1.0:
        raise ValueError("squared EXP w(0) changed")
    if weight.get("positive_only_anchor") is not True:
        raise ValueError("Positive-only anchor was removed")
    if float(weight.get("reference_distance", -1.0)) != 2.0:
        raise ValueError("reference distance changed")
    if weight.get("formula") != "w(d)=w(0)*exp(-c*(d/2)^2)":
        raise ValueError("squared-remoteness formula changed")
    coefficients = tuple(float(value) for value in weight.get("exp_coefficients", ()))
    if coefficients != EXPECTED_COEFFICIENTS:
        raise ValueError("coefficient shortlist changed")
    if int(raw.get("expected_controls_per_actor_advantage_cell", -1)) != 4:
        raise ValueError("control count changed")
    if int(raw.get("expected_total_branches", -1)) != EXPECTED_BRANCHES:
        raise ValueError("expected branch count changed")
    if raw.get("screening_only") is not True or raw.get("formal_evidence_allowed") is not False:
        raise ValueError("pilot evidence boundary changed")

    normalized = dict(raw)
    normalized.update(
        {
            "seeds": list(EXPECTED_SEEDS),
            "advantage_estimators": ["td", "gae"],
            "shared_critic": {
                "steps": int(shared["steps"]),
                "batch_size": int(shared["batch"]),
                "learning_rate": float(shared["lr"]),
                "expectile": float(shared["tau"]),
                "gamma": float(shared["gamma"]),
                "gae_lambda": float(trajectory["gae_lambda"]),
                "device": str(shared["device"]),
            },
            "expected_branches": EXPECTED_BRANCHES,
        }
    )
    return normalized, sha256_file(source)


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
    audit = manifest.get("trajectory_audit", {})
    if audit.get("status") != "PASS":
        raise RuntimeError(f"ordered trajectory audit is not PASS: {directory}")
    diagnostics = advantages.get("diagnostics", {})
    if float(diagnostics.get("lambda_zero_max_abs_error", 1.0)) > 1e-6:
        raise RuntimeError(f"lambda=0 regression failed: {directory}")
    if float(diagnostics.get("numpy_torch_max_abs_error", 1.0)) > 1e-6:
        raise RuntimeError(f"NumPy/Torch GAE cross-check failed: {directory}")
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

    _activate_protocol()
    contract = CanonicalContract.load(args.contract)
    grid, _ = load_grid(args.grid)
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

    _activate_protocol()
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
    base.load_grid = load_grid
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
