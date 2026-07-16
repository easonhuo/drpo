"""Code-first runner for Stage A squared-EXP KL threshold tuning."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from drpo import e7_canonical_sweep as base
from drpo import e7_squared_exp_night as predecessor
from drpo.e7_canonical_injection import sha256_file
from drpo.e7_squared_exp_kernel import FORMULA
from drpo.e7_squared_exp_kl_tune_stage_a_aggregate import aggregate as aggregate_results


EXPERIMENT_ID = "EXT-H-E7-SQUARED-EXP-KL-TUNE-01"
STAGE_ID = "stage_a_kl_threshold_and_reference_lifecycle_screen"
SCIENTIFIC_STATUS = (
    "squared_remoteness_kl_threshold_and_reference_lifecycle_"
    "development_screening_only"
)
RUNNER_VERSION = "1.0.0-e7-squared-exp-kl-tune-stage-a"

EXPECTED_DATASETS = predecessor.EXPECTED_DATASETS
EXPECTED_SEEDS = (200, 201)
HELD_OUT_SEEDS = (204, 205, 206, 207)
EXPECTED_COEFFICIENTS = (4.0, 8.0, 16.0, 32.0)
EXPECTED_LIFECYCLES = (
    "ppo_clip_k4",
    "ppo_clip_k16",
    "ppo_clip_kl_k16_t0p003",
    "ppo_clip_kl_k16_t0p01",
    "ppo_clip_kl_k16_t0p03",
)
ADAPTIVE_LIFECYCLES = (
    "ppo_clip_kl_k16_t0p003",
    "ppo_clip_kl_k16_t0p01",
    "ppo_clip_kl_k16_t0p03",
)
EXPECTED_STEPS = 1_000_000
EXPECTED_CONTROLS_PER_LIFECYCLE = 5
EXPECTED_TOTAL_BRANCHES = 150
REFERENCE_DISTANCE = 2.0
INTERNAL_CANONICAL_ALPHA = 0.11
DIAGNOSTICS_INTERVAL = 1000
SAMPLED_VALUES_PER_UPDATE = 16

LIFECYCLE_SPECS: dict[str, dict[str, Any]] = {
    "ppo_clip_k4": {
        "clip_epsilon": 0.2,
        "updates_per_old_policy": 4,
        "analytic_kl_early_refresh": False,
        "target_kl": None,
    },
    "ppo_clip_k16": {
        "clip_epsilon": 0.2,
        "updates_per_old_policy": 16,
        "analytic_kl_early_refresh": False,
        "target_kl": None,
    },
    "ppo_clip_kl_k16_t0p003": {
        "clip_epsilon": 0.2,
        "updates_per_old_policy": 16,
        "analytic_kl_early_refresh": True,
        "target_kl": 0.003,
    },
    "ppo_clip_kl_k16_t0p01": {
        "clip_epsilon": 0.2,
        "updates_per_old_policy": 16,
        "analytic_kl_early_refresh": True,
        "target_kl": 0.01,
    },
    "ppo_clip_kl_k16_t0p03": {
        "clip_epsilon": 0.2,
        "updates_per_old_policy": 16,
        "analytic_kl_early_refresh": True,
        "target_kl": 0.03,
    },
}


def _label(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def _flag_value(argv: list[str], flag: str) -> str:
    return predecessor._flag_value(argv, flag)  # noqa: SLF001


def lifecycle_spec(lifecycle_id: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(LIFECYCLE_SPECS[lifecycle_id])
    except KeyError as exc:
        raise ValueError(f"unsupported reference lifecycle: {lifecycle_id}") from exc


def load_grid(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = json.loads(source.read_text())
    if raw.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"grid experiment_id must be {EXPERIMENT_ID}")
    if raw.get("stage_id") != STAGE_ID:
        raise ValueError(f"grid stage_id must be {STAGE_ID}")
    if raw.get("run_kind") != "pilot" or raw.get("scientific_status") != SCIENTIFIC_STATUS:
        raise ValueError("Stage A grid must remain the frozen development pilot")
    if tuple(raw.get("datasets", ())) != EXPECTED_DATASETS:
        raise ValueError("Stage A datasets changed")
    if tuple(int(value) for value in raw.get("development_seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("Stage A development seeds changed")
    if tuple(int(value) for value in raw.get("held_out_seeds", ())) != HELD_OUT_SEEDS:
        raise ValueError("held-out seed reservation changed")
    if int(raw.get("steps", -1)) != EXPECTED_STEPS:
        raise ValueError("Stage A branches must run exactly 1,000,000 updates")
    if int(raw.get("evaluation_interval", -1)) != 50_000:
        raise ValueError("evaluation interval must remain 50,000")
    if int(raw.get("evaluation_episodes", -1)) != 10:
        raise ValueError("evaluation episodes must remain 10")

    weight = raw.get("weight_control", {})
    if str(weight.get("formula")) != FORMULA:
        raise ValueError("weight formula must remain squared remoteness")
    if not math.isclose(float(weight.get("weight_at_zero")), 1.0, abs_tol=1e-12):
        raise ValueError("weight_at_zero must remain 1")
    if weight.get("positive_only_anchor") is not True:
        raise ValueError("Positive-only anchor must remain enabled")
    if not math.isclose(
        float(weight.get("reference_distance")),
        REFERENCE_DISTANCE,
        abs_tol=1e-12,
    ):
        raise ValueError("reference_distance must remain 2")
    coefficients = tuple(float(value) for value in weight.get("exp_coefficients", ()))
    if coefficients != EXPECTED_COEFFICIENTS:
        raise ValueError("Stage A squared-EXP coefficient set changed")

    lifecycles = raw.get("reference_lifecycles")
    if not isinstance(lifecycles, list):
        raise ValueError("reference_lifecycles must be a list")
    lifecycle_ids = tuple(str(item.get("id")) for item in lifecycles)
    if lifecycle_ids != EXPECTED_LIFECYCLES:
        raise ValueError("Stage A reference lifecycle order changed")
    for item in lifecycles:
        lifecycle_id = str(item["id"])
        expected = lifecycle_spec(lifecycle_id)
        if not math.isclose(float(item.get("clip_epsilon")), 0.2, abs_tol=1e-12):
            raise ValueError(f"{lifecycle_id} clip epsilon changed")
        if int(item.get("max_updates_per_old_policy", -1)) != int(
            expected["updates_per_old_policy"]
        ):
            raise ValueError(f"{lifecycle_id} reference window changed")
        if item.get("analytic_kl_early_refresh") is not expected[
            "analytic_kl_early_refresh"
        ]:
            raise ValueError(f"{lifecycle_id} KL refresh mode changed")
        target = item.get("target_kl")
        expected_target = expected["target_kl"]
        if expected_target is None:
            if target is not None:
                raise ValueError(f"{lifecycle_id} must not define target_kl")
        elif not math.isclose(float(target), float(expected_target), abs_tol=1e-12):
            raise ValueError(f"{lifecycle_id} target_kl changed")

    if int(raw.get("expected_controls_per_lifecycle", -1)) != (
        EXPECTED_CONTROLS_PER_LIFECYCLE
    ):
        raise ValueError("control count changed")
    if int(raw.get("expected_reference_lifecycles", -1)) != len(
        EXPECTED_LIFECYCLES
    ):
        raise ValueError("reference lifecycle count changed")
    if int(raw.get("expected_runnable_branches", -1)) != EXPECTED_TOTAL_BRANCHES:
        raise ValueError("runnable branch count changed")
    if raw.get("formal_evidence_allowed") is not False:
        raise ValueError("development screening cannot allow formal evidence")
    if raw.get("registration_blocks_launch") is not False:
        raise ValueError("code-first Stage A launch must not be registration-blocked")
    return raw, sha256_file(source)


def load_run_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    run_spec, digest = predecessor.load_run_spec(path)
    if tuple(int(value) for value in run_spec["seeds"]) != EXPECTED_SEEDS:
        raise ValueError("source run spec seeds changed")
    return run_spec, digest


def control_points(grid: Mapping[str, Any]) -> list[tuple[float, float | None]]:
    points: list[tuple[float, float | None]] = [(0.0, None)]
    points.extend(
        (1.0, float(coefficient))
        for coefficient in grid["weight_control"]["exp_coefficients"]
    )
    if len(points) != EXPECTED_CONTROLS_PER_LIFECYCLE:
        raise ValueError("Stage A must contain five controls per lifecycle")
    if len(points) != len(set(points)):
        raise ValueError("Stage A control points are not unique")
    return points


def build_branches(
    contract: base.CanonicalContract,
    run_spec: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> list[base.Branch]:
    if not math.isclose(
        contract.expected_canonical_alpha,
        INTERNAL_CANONICAL_ALPHA,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("canonical source alpha changed from 0.11")
    datasets = [base.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
    if tuple(item.id for item in datasets) != EXPECTED_DATASETS:
        raise ValueError("expanded dataset subset changed")
    seeds = [int(value) for value in run_spec["seeds"]]
    if tuple(seeds) != EXPECTED_SEEDS:
        raise ValueError("expanded development seeds changed")

    common = {
        "steps": str(EXPECTED_STEPS),
        "diagnostics_interval": str(DIAGNOSTICS_INTERVAL),
        "sampled_values_per_update": str(SAMPLED_VALUES_PER_UPDATE),
        "stage": "stage_a",
    }
    branches: list[base.Branch] = []
    for lifecycle_id in EXPECTED_LIFECYCLES:
        lifecycle = lifecycle_spec(lifecycle_id)
        for weight_at_zero, coefficient in control_points(grid):
            if coefficient is None:
                method = "positive_only"
                coefficient_value = 0.0
                control_label = "positive_only__w0_0"
            else:
                method = "squared_exponential"
                coefficient_value = coefficient
                control_label = f"sqexp__w0_1__c_{_label(coefficient)}"
            for dataset in datasets:
                for seed in seeds:
                    branch_id = (
                        f"{dataset.id}__seed{seed}__{control_label}__"
                        f"{lifecycle_id}__steps1m"
                    )
                    target_kl = lifecycle["target_kl"]
                    branches.append(
                        base.Branch(
                            branch_id=branch_id,
                            branch_kind="injected",
                            dataset=dataset,
                            seed=seed,
                            template_values={
                                **common,
                                "actor_update_mode": lifecycle_id,
                                "weight_method": method,
                                "weight_at_zero": f"{weight_at_zero:.17g}",
                                "exp_coefficient": f"{coefficient_value:.17g}",
                                "reference_distance": f"{REFERENCE_DISTANCE:.17g}",
                                "updates_per_old_policy": str(
                                    lifecycle["updates_per_old_policy"]
                                ),
                                "analytic_kl_early_refresh": (
                                    "true"
                                    if lifecycle["analytic_kl_early_refresh"]
                                    else "false"
                                ),
                                "target_kl": (
                                    ""
                                    if target_kl is None
                                    else f"{float(target_kl):.17g}"
                                ),
                            },
                            negative_control=None,
                        )
                    )
    ids = [branch.branch_id for branch in branches]
    if len(ids) != len(set(ids)):
        raise ValueError("Stage A branch IDs are not unique")
    if len(branches) != EXPECTED_TOTAL_BRANCHES:
        raise ValueError(
            f"expected {EXPECTED_TOTAL_BRANCHES} branches, built {len(branches)}"
        )
    return branches


def branch_command(
    *,
    contract_path: Path,
    contract: base.CanonicalContract,
    branch: base.Branch,
    branch_dir: Path,
    trainer_argv_template: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    values = branch.template_values
    weight_at_zero = float(values["weight_at_zero"])
    coefficient = float(values["exp_coefficient"])
    method = str(values["weight_method"])
    lifecycle_id = str(values["actor_update_mode"])
    lifecycle = lifecycle_spec(lifecycle_id)
    if method == "positive_only" and (weight_at_zero != 0.0 or coefficient != 0.0):
        raise ValueError("Positive-only branch requires w(0)=0,c=0")
    if method == "squared_exponential" and weight_at_zero != 1.0:
        raise ValueError("squared EXP requires w(0)=1")

    context: dict[str, Any] = {
        "canonical_root": str(contract.source_root),
        "dataset_id": branch.dataset.id,
        "dataset_path": str(Path(branch.dataset.path).expanduser().resolve()),
        "dataset_sha256": branch.dataset.sha256,
        "seed": branch.seed,
        "output_dir": str(branch_dir / "trainer_output"),
        "branch_id": branch.branch_id,
        "variant": "iqlv_exp_rank",
        **values,
    }
    trainer_args = [
        base._format_value(str(item), context)  # noqa: SLF001
        for item in trainer_argv_template
    ]
    weight_control = {
        "method": method,
        "weight_at_zero": weight_at_zero,
        "exp_coefficient": coefficient,
        "reference_distance": REFERENCE_DISTANCE,
        "formula": FORMULA,
    }
    reference_lifecycle = {
        "id": lifecycle_id,
        "clip_epsilon": float(lifecycle["clip_epsilon"]),
        "max_updates_per_old_policy": int(lifecycle["updates_per_old_policy"]),
        "analytic_kl_early_refresh": bool(
            lifecycle["analytic_kl_early_refresh"]
        ),
        "target_kl": lifecycle["target_kl"],
    }
    branch_config = {
        "experiment_id": EXPERIMENT_ID,
        "stage_id": STAGE_ID,
        "branch_id": branch.branch_id,
        "branch_kind": branch.branch_kind,
        "dataset_id": branch.dataset.id,
        "dataset_sha256": branch.dataset.sha256,
        "seed": branch.seed,
        "template_values": values,
        "weight_control": weight_control,
        "reference_lifecycle": reference_lifecycle,
    }
    branch_config_path = branch_dir / "branch_config.json"
    base.atomic_write_json(branch_config_path, branch_config)
    command = [
        sys.executable,
        "-m",
        "drpo.e7_squared_exp_kl_tune_stage_a_bootstrap",
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


def main(argv: list[str] | None = None) -> int:
    previous = (
        base.EXPERIMENT_ID,
        base.SCIENTIFIC_STATUS,
        base.RUNNER_VERSION,
        base.load_grid,
        base.load_run_spec,
        base.build_branches,
        base.branch_command,
    )
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.SCIENTIFIC_STATUS = SCIENTIFIC_STATUS
    base.RUNNER_VERSION = RUNNER_VERSION
    base.load_grid = load_grid
    base.load_run_spec = load_run_spec
    base.build_branches = build_branches
    base.branch_command = branch_command
    try:
        delegated = list(sys.argv[1:] if argv is None else argv)
        result = base.main(delegated)
        if delegated and delegated[0] == "run":
            if "--work-dir" not in delegated:
                raise ValueError("run command is missing --work-dir")
            index = delegated.index("--work-dir")
            if index + 1 >= len(delegated):
                raise ValueError("run command has no --work-dir value")
            aggregate_results(delegated[index + 1])
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
        ) = previous


if __name__ == "__main__":
    raise SystemExit(main())
