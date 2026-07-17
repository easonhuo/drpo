"""Code-first runner for the E7 squared-remoteness night suite."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from drpo import e7_canonical_sweep as base
from drpo import e7_w0_highc_actor as predecessor
from drpo.e7_canonical_injection import sha256_file
from drpo.e7_squared_exp_kernel import FORMULA
from drpo.e7_squared_exp_night_aggregate import aggregate as aggregate_results


EXPERIMENT_ID = "EXT-H-E7-SQUARED-EXP-NIGHT-01"
SCIENTIFIC_STATUS = (
    "squared_remoteness_and_ppo_reference_lifecycle_development_screening_only"
)
RUNNER_VERSION = "1.0.0-e7-squared-exp-night-1m"
GAE_EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-01"
GAE_SCIENTIFIC_STATUS = "canonical_joint_critic_trajectory_snapshot_gae_pilot_only"
GAE_RUNNER_VERSION = "5.0.0-existing-pipeline-gae"

EXPECTED_DATASETS = (
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
)
EXPECTED_SEEDS = (200, 201)
GAE_EXPECTED_SEEDS = (200, 201, 202, 203)
HELD_OUT_SEEDS = (204, 205, 206, 207)
EXPECTED_COEFFICIENTS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
GAE_COEFFICIENTS = (64.0, 128.0, 256.0)
EXPECTED_ACTOR_MODES = ("a2c", "ppo_clip_k4", "ppo_clip_kl_k16")
EXPECTED_STEPS = 1_000_000
EXPECTED_CONTROLS_PER_MODE = 7
EXPECTED_STAGE_A_BRANCHES = 84
EXPECTED_STAGE_B_BRANCHES = 42
EXPECTED_TOTAL_BRANCHES = 126
GAE_EXPECTED_BRANCHES = 96
REFERENCE_DISTANCE = 2.0
INTERNAL_CANONICAL_ALPHA = 0.11
DIAGNOSTICS_INTERVAL = 1000
SAMPLED_VALUES_PER_UPDATE = 16
GAE_LAMBDA = 0.95
GAE_CANONICAL_BATCH_SIZE = 256
GAE_LIVENESS_DATASET = "hopper-medium-expert-v2"
GAE_LIVENESS_SEED = 200
GAE_LIVENESS_COEFFICIENT = 128.0

_ACTIVE_EXPERIMENT_ID = EXPERIMENT_ID
_LIVENESS_STEPS: int | None = None


def _label(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def _flag_value(argv: list[str], flag: str) -> str:
    positions = [index for index, token in enumerate(argv) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"trainer_argv_template must contain exactly one {flag}")
    return argv[positions[0] + 1]


def configure_execution(
    grid_path: str | Path, *, liveness_pair: bool = False, liveness_steps: int | None = None
) -> None:
    global _ACTIVE_EXPERIMENT_ID, _LIVENESS_STEPS
    experiment_id = str(json.loads(Path(grid_path).read_text()).get("experiment_id"))
    if experiment_id not in {EXPERIMENT_ID, GAE_EXPERIMENT_ID}:
        raise ValueError(f"unsupported squared-night experiment_id={experiment_id!r}")
    if liveness_pair != (liveness_steps is not None):
        raise ValueError("liveness_pair and liveness_steps must be set together")
    if liveness_pair and (experiment_id != GAE_EXPERIMENT_ID or int(liveness_steps) < 2):
        raise ValueError("GAE liveness requires the GAE grid and at least two updates")
    _ACTIVE_EXPERIMENT_ID = experiment_id
    _LIVENESS_STEPS = None if liveness_steps is None else int(liveness_steps)


def active_experiment_id() -> str:
    return _ACTIVE_EXPERIMENT_ID


def active_scientific_status() -> str:
    return GAE_SCIENTIFIC_STATUS if _is_gae() else SCIENTIFIC_STATUS


def active_expected_branch_count() -> int:
    return 2 if _LIVENESS_STEPS else (GAE_EXPECTED_BRANCHES if _is_gae() else EXPECTED_TOTAL_BRANCHES)


def active_runtime_profile() -> dict[str, Any]:
    if _is_gae():
        return {
            "adapter_id": "e7_squared_exp_night_gae_cpu_v2",
            "dataset": GAE_LIVENESS_DATASET,
            "seed": GAE_LIVENESS_SEED,
            "actor_update_mode": "a2c",
            "advantage_estimator": "gae",
            "weight_at_zero": 1.0,
            "exp_coefficient": GAE_LIVENESS_COEFFICIENT,
            "gae_lambda": GAE_LAMBDA,
        }
    return {
        "adapter_id": "e7_squared_exp_night_cpu_v2",
        "dataset": "walker2d-medium-v2",
        "seed": EXPECTED_SEEDS[0],
        "actor_update_mode": "ppo_clip_kl_k16",
        "advantage_estimator": "one_step_td",
        "weight_at_zero": 1.0,
        "exp_coefficient": 4.0,
        "clip_epsilon": 0.2,
        "max_updates_per_old_policy": 16,
        "target_kl": 0.01,
    }


def _is_gae() -> bool:
    return _ACTIVE_EXPERIMENT_ID == GAE_EXPERIMENT_ID


def _check(raw: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    changed = [key for key, value in expected.items() if raw.get(key) != value]
    if changed:
        raise ValueError(f"{label} changed: {changed}")


def load_grid(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = json.loads(source.read_text())
    _check(
        raw,
        {
            "experiment_id": _ACTIVE_EXPERIMENT_ID,
            "run_kind": "pilot",
            "datasets": list(EXPECTED_DATASETS),
            "held_out_seeds": list(HELD_OUT_SEEDS),
            "steps": EXPECTED_STEPS,
            "evaluation_interval": 50_000,
            "evaluation_episodes": 10,
            "formal_evidence_allowed": False,
        },
        "squared-night grid",
    )
    weight = raw.get("weight_control", {})
    if weight.get("formula") != FORMULA or weight.get("positive_only_anchor") is not True:
        raise ValueError("squared-remoteness weight contract changed")
    if not math.isclose(float(weight.get("reference_distance")), REFERENCE_DISTANCE):
        raise ValueError("reference_distance changed")
    if _is_gae():
        _check(
            raw,
            {
                "status": "not_run",
                "scientific_status": GAE_SCIENTIFIC_STATUS,
                "predecessor_experiment_id": EXPERIMENT_ID,
                "development_seeds": list(GAE_EXPECTED_SEEDS),
                "actor_update_modes": ["a2c"],
                "advantage_modes": ["one_step_td", "gae_lambda_0p95"],
                "expected_total_branches": GAE_EXPECTED_BRANCHES,
                "screening_only": True,
            },
            "GAE grid",
        )
        snapshot = raw.get("trajectory_snapshot", {})
        _check(
            snapshot,
            {
                "gae_lambda": GAE_LAMBDA,
                "canonical_batch_size": GAE_CANONICAL_BATCH_SIZE,
                "td_and_gae_share_snapshot": True,
                "critic_updated_every_step": True,
                "prepared_advantage_artifact": False,
                "terminal_bootstrap": False,
                "timeout_bootstrap": True,
                "terminal_timeout_recursion_stop": True,
                "dataset_tail_recursion_stop": True,
            },
            "GAE snapshot contract",
        )
        coefficients = GAE_COEFFICIENTS
    else:
        _check(
            raw,
            {
                "scientific_status": SCIENTIFIC_STATUS,
                "development_seeds": list(EXPECTED_SEEDS),
                "expected_stage_a_branches": EXPECTED_STAGE_A_BRANCHES,
                "expected_stage_b_branches": EXPECTED_STAGE_B_BRANCHES,
                "expected_runnable_branches": EXPECTED_TOTAL_BRANCHES,
            },
            "historical grid",
        )
        by_id = {str(stage.get("id")): stage for stage in raw.get("stages", [])}
        stage_c = by_id.get("stage_c_gae", {})
        if stage_c.get("enabled") is not False or stage_c.get("status") != "blocked_pending_verified_trajectory_contract":
            raise ValueError("historical Stage C contract changed")
        coefficients = EXPECTED_COEFFICIENTS
    if tuple(float(value) for value in weight.get("exp_coefficients", ())) != coefficients:
        raise ValueError("squared EXP coefficient set changed")
    return raw, sha256_file(source)


def load_run_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    raw, digest = predecessor._BASE_LOAD_RUN_SPEC(path)  # noqa: SLF001
    run_spec = copy.deepcopy(raw)
    if run_spec.get("experiment_id") != "EXT-H-E7-BENCH-01":
        raise ValueError("source run_spec experiment_id changed")
    source_ids = tuple(str(item["id"]) for item in run_spec["datasets"])
    if source_ids != predecessor.EXPECTED_SOURCE_DATASETS:
        raise ValueError("source run_spec nine-dataset order changed")
    by_id = {str(item["id"]): item for item in run_spec["datasets"]}
    run_spec["datasets"] = [copy.deepcopy(by_id[name]) for name in EXPECTED_DATASETS]
    if tuple(int(value) for value in run_spec["seeds"]) != EXPECTED_SEEDS:
        raise ValueError("source run_spec seeds changed")
    run_spec["seeds"] = list(GAE_EXPECTED_SEEDS if _is_gae() else EXPECTED_SEEDS)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if str(run_spec.get("environment", {}).get(name)) != "1":
            raise ValueError(f"run_spec {name} must remain 1")
    argv = [str(item) for item in run_spec["trainer_argv_template"]]
    for flag, expected in {
        "--alpha": "0.11",
        "--batch": "256",
        "--lr": "0.0003",
        "--eval_interval": "50000",
        "--eval_episodes": "10",
        "--steps": "1000000",
    }.items():
        if _flag_value(argv, flag) != expected:
            raise ValueError(f"source run_spec {flag} changed")
    if _is_gae() and "--ret_weight_mode" in argv and _flag_value(argv, "--ret_weight_mode") != "none":
        raise ValueError("GAE transition IDs require ret_weight_mode=none")
    argv[argv.index("--steps") + 1] = "{steps}"
    run_spec["trainer_argv_template"] = argv
    run_spec["passthrough_variants"] = []
    return run_spec, digest


def control_points(grid: Mapping[str, Any]) -> list[tuple[float, float | None]]:
    points = [(0.0, None), *[(1.0, float(c)) for c in grid["weight_control"]["exp_coefficients"]]]
    expected = 4 if _is_gae() else EXPECTED_CONTROLS_PER_MODE
    if len(points) != expected or len(points) != len(set(points)):
        raise ValueError("squared-night controls are not unique or complete")
    return points


def _control(weight_at_zero: float, coefficient: float | None) -> tuple[str, float, str]:
    if coefficient is None:
        return "positive_only", 0.0, "positive_only__w0_0"
    return "squared_exponential", coefficient, f"sqexp__w0_1__c_{_label(coefficient)}"


def _gae_branches(run_spec: Mapping[str, Any], grid: Mapping[str, Any]) -> list[base.Branch]:
    datasets = [base.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
    branches = []
    for estimator in ("td", "gae"):
        for w0, coefficient in control_points(grid):
            method, c, label = _control(w0, coefficient)
            for dataset in datasets:
                for seed in GAE_EXPECTED_SEEDS:
                    branches.append(
                        base.Branch(
                            branch_id=f"{dataset.id}__seed{seed}__{estimator}__{label}__a2c__steps1m",
                            branch_kind="injected",
                            dataset=dataset,
                            seed=seed,
                            template_values={
                                "steps": str(EXPECTED_STEPS),
                                "stage": "stage_c_joint_gae",
                                "actor_update_mode": "a2c",
                                "advantage_estimator": estimator,
                                "weight_method": method,
                                "weight_at_zero": f"{w0:.17g}",
                                "exp_coefficient": f"{c:.17g}",
                                "reference_distance": f"{REFERENCE_DISTANCE:.17g}",
                                "diagnostics_interval": str(DIAGNOSTICS_INTERVAL),
                                "sampled_values_per_update": str(SAMPLED_VALUES_PER_UPDATE),
                                "execution_mode": "full",
                            },
                            negative_control=None,
                        )
                    )
    if _LIVENESS_STEPS:
        branches = [
            dataclasses.replace(
                branch,
                branch_id=f"{branch.branch_id}__liveness_steps{_LIVENESS_STEPS}",
                template_values={**branch.template_values, "steps": str(_LIVENESS_STEPS), "execution_mode": "liveness"},
            )
            for branch in branches
            if branch.dataset.id == GAE_LIVENESS_DATASET
            and branch.seed == GAE_LIVENESS_SEED
            and float(branch.template_values["exp_coefficient"]) == GAE_LIVENESS_COEFFICIENT
        ]
    return branches


def build_branches(
    contract: base.CanonicalContract,
    run_spec: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> list[base.Branch]:
    if not math.isclose(contract.expected_canonical_alpha, INTERNAL_CANONICAL_ALPHA):
        raise ValueError("canonical source alpha changed from 0.11")
    if _is_gae():
        branches = _gae_branches(run_spec, grid)
    else:
        datasets = [base.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
        common = {
            "steps": str(EXPECTED_STEPS),
            "diagnostics_interval": str(DIAGNOSTICS_INTERVAL),
            "sampled_values_per_update": str(SAMPLED_VALUES_PER_UPDATE),
        }
        branches = []
        for actor_mode in EXPECTED_ACTOR_MODES:
            stage = "stage_b" if actor_mode == "ppo_clip_kl_k16" else "stage_a"
            for w0, coefficient in control_points(grid):
                method, c, label = _control(w0, coefficient)
                for dataset in datasets:
                    for seed in EXPECTED_SEEDS:
                        branches.append(
                            base.Branch(
                                branch_id=f"{dataset.id}__seed{seed}__{label}__{actor_mode}__steps1m",
                                branch_kind="injected",
                                dataset=dataset,
                                seed=seed,
                                template_values={
                                    **common,
                                    "stage": stage,
                                    "actor_update_mode": actor_mode,
                                    "weight_method": method,
                                    "weight_at_zero": f"{w0:.17g}",
                                    "exp_coefficient": f"{c:.17g}",
                                    "reference_distance": f"{REFERENCE_DISTANCE:.17g}",
                                },
                                negative_control=None,
                            )
                        )
    ids = [branch.branch_id for branch in branches]
    if len(ids) != active_expected_branch_count() or len(ids) != len(set(ids)):
        raise ValueError("squared-night branch matrix changed")
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
    w0 = float(values["weight_at_zero"])
    coefficient = float(values["exp_coefficient"])
    method = str(values["weight_method"])
    actor_mode = str(values["actor_update_mode"])
    if actor_mode not in (("a2c",) if _is_gae() else EXPECTED_ACTOR_MODES):
        raise ValueError("branch actor mode changed")
    if method == "positive_only" and (w0 != 0.0 or coefficient != 0.0):
        raise ValueError("Positive-only branch requires w(0)=0,c=0")
    if method == "squared_exponential" and w0 != 1.0:
        raise ValueError("squared EXP requires w(0)=1")
    context = {
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
    trainer_args = [base._format_value(str(item), context) for item in trainer_argv_template]  # noqa: SLF001
    if values.get("execution_mode") == "liveness":
        for flag, value in (("--eval_interval", values["steps"]), ("--eval_episodes", "1")):
            trainer_args[trainer_args.index(flag) + 1] = value
    branch_config = {
        "experiment_id": _ACTIVE_EXPERIMENT_ID,
        "branch_id": branch.branch_id,
        "branch_kind": branch.branch_kind,
        "dataset_id": branch.dataset.id,
        "dataset_sha256": branch.dataset.sha256,
        "seed": branch.seed,
        "template_values": values,
        "weight_control": {
            "method": method,
            "weight_at_zero": w0,
            "exp_coefficient": coefficient,
            "reference_distance": REFERENCE_DISTANCE,
            "formula": FORMULA,
        },
    }
    if _is_gae():
        branch_config.update(canonical_root=str(contract.source_root), dataset_path=context["dataset_path"])
    branch_config_path = branch_dir / "branch_config.json"
    base.atomic_write_json(branch_config_path, branch_config)
    return [
        sys.executable,
        "-m",
        "drpo.e7_squared_exp_night_bootstrap",
        "--contract",
        str(contract_path),
        "--branch-config",
        str(branch_config_path),
        "--branch-manifest",
        str(branch_dir / "branch_manifest.json"),
        "--",
        *trainer_args,
    ], branch_config


def main(argv: list[str] | None = None) -> int:
    global _ACTIVE_EXPERIMENT_ID, _LIVENESS_STEPS
    delegated = list(sys.argv[1:] if argv is None else argv)
    grid_index = delegated.index("--grid")
    previous_profile = (_ACTIVE_EXPERIMENT_ID, _LIVENESS_STEPS)
    env_steps = os.environ.get("DRPO_E7_GAE_LIVENESS_STEPS")
    configure_execution(
        delegated[grid_index + 1],
        liveness_pair=env_steps is not None,
        liveness_steps=None if env_steps is None else int(env_steps),
    )
    previous = (
        base.EXPERIMENT_ID,
        base.SCIENTIFIC_STATUS,
        base.RUNNER_VERSION,
        base.load_grid,
        base.load_run_spec,
        base.build_branches,
        base.branch_command,
    )
    base.EXPERIMENT_ID = _ACTIVE_EXPERIMENT_ID
    base.SCIENTIFIC_STATUS = active_scientific_status()
    base.RUNNER_VERSION = GAE_RUNNER_VERSION if _is_gae() else RUNNER_VERSION
    base.load_grid, base.load_run_spec = load_grid, load_run_spec
    base.build_branches, base.branch_command = build_branches, branch_command
    try:
        result = int(base.main(delegated))
        if delegated[0] == "run":
            aggregate_results(delegated[delegated.index("--work-dir") + 1])
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
        _ACTIVE_EXPERIMENT_ID, _LIVENESS_STEPS = previous_profile


if __name__ == "__main__":
    raise SystemExit(main())
