"""Code-first runner for the E7 squared-remoteness night suite and GAE successor."""

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
EXPECTED_ACTOR_MODES = ("a2c", "ppo_clip_k4", "ppo_clip_kl_k16")
EXPECTED_STEPS = 1_000_000
EXPECTED_CONTROLS_PER_MODE = 7
EXPECTED_STAGE_A_BRANCHES = 84
EXPECTED_STAGE_B_BRANCHES = 42
EXPECTED_TOTAL_BRANCHES = 126
REFERENCE_DISTANCE = 2.0
INTERNAL_CANONICAL_ALPHA = 0.11
DIAGNOSTICS_INTERVAL = 1000
SAMPLED_VALUES_PER_UPDATE = 16

GAE_ESTIMATORS = ("td", "gae")
GAE_COEFFICIENTS = (64.0, 128.0, 256.0)
GAE_LAMBDA = 0.95
GAE_CANONICAL_BATCH_SIZE = 256
GAE_EXPECTED_BRANCHES = 96
GAE_LIVENESS_DATASET = "hopper-medium-expert-v2"
GAE_LIVENESS_SEED = 200
GAE_LIVENESS_COEFFICIENT = 128.0

_ACTIVE_EXPERIMENT_ID = EXPERIMENT_ID
_LIVENESS_PAIR = False
_LIVENESS_STEPS: int | None = None


def _label(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def _flag_value(argv: list[str], flag: str) -> str:
    positions = [index for index, token in enumerate(argv) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"trainer_argv_template must contain exactly one {flag}")
    return argv[positions[0] + 1]


def _replace_flag(argv: list[str], flag: str, value: str) -> None:
    positions = [index for index, token in enumerate(argv) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"trainer args must contain exactly one {flag}")
    argv[positions[0] + 1] = value


def configure_execution(
    grid_path: str | Path,
    *,
    liveness_pair: bool = False,
    liveness_steps: int | None = None,
) -> None:
    """Select the historical night profile or its registered GAE successor."""

    global _ACTIVE_EXPERIMENT_ID, _LIVENESS_PAIR, _LIVENESS_STEPS
    experiment_id = str(json.loads(Path(grid_path).read_text()).get("experiment_id"))
    if experiment_id not in {EXPERIMENT_ID, GAE_EXPERIMENT_ID}:
        raise ValueError(f"unsupported squared-night experiment_id={experiment_id!r}")
    if liveness_pair and experiment_id != GAE_EXPERIMENT_ID:
        raise ValueError("matched liveness is defined only for the GAE successor")
    if liveness_pair and (liveness_steps is None or liveness_steps < 2):
        raise ValueError("GAE liveness requires at least two optimizer updates")
    if not liveness_pair and liveness_steps is not None:
        raise ValueError("liveness_steps requires liveness_pair=True")
    _ACTIVE_EXPERIMENT_ID = experiment_id
    _LIVENESS_PAIR = bool(liveness_pair)
    _LIVENESS_STEPS = None if liveness_steps is None else int(liveness_steps)


def active_experiment_id() -> str:
    return _ACTIVE_EXPERIMENT_ID


def active_scientific_status() -> str:
    return (
        GAE_SCIENTIFIC_STATUS
        if _ACTIVE_EXPERIMENT_ID == GAE_EXPERIMENT_ID
        else SCIENTIFIC_STATUS
    )


def active_runner_version() -> str:
    return (
        GAE_RUNNER_VERSION
        if _ACTIVE_EXPERIMENT_ID == GAE_EXPERIMENT_ID
        else RUNNER_VERSION
    )


def active_expected_branch_count() -> int:
    if _LIVENESS_PAIR:
        return 2
    if _ACTIVE_EXPERIMENT_ID == GAE_EXPERIMENT_ID:
        return GAE_EXPECTED_BRANCHES
    return EXPECTED_TOTAL_BRANCHES


def active_runtime_profile() -> dict[str, Any]:
    if _ACTIVE_EXPERIMENT_ID == GAE_EXPERIMENT_ID:
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


def _validate_common_grid(raw: Mapping[str, Any]) -> None:
    if tuple(raw.get("datasets", ())) != EXPECTED_DATASETS:
        raise ValueError("night-suite datasets changed")
    if tuple(int(value) for value in raw.get("held_out_seeds", ())) != HELD_OUT_SEEDS:
        raise ValueError("held-out seed reservation changed")
    if int(raw.get("steps", -1)) != EXPECTED_STEPS:
        raise ValueError("steps must remain 1,000,000")
    if int(raw.get("evaluation_interval", -1)) != 50_000:
        raise ValueError("evaluation_interval must remain 50,000")
    if int(raw.get("evaluation_episodes", -1)) != 10:
        raise ValueError("evaluation_episodes must remain 10")
    if raw.get("formal_evidence_allowed") is not False:
        raise ValueError("development screening cannot allow formal evidence")


def _validate_legacy_grid(raw: Mapping[str, Any]) -> None:
    if raw.get("run_kind") != "pilot" or raw.get("scientific_status") != SCIENTIFIC_STATUS:
        raise ValueError("night grid must remain the frozen development pilot")
    if tuple(int(value) for value in raw.get("development_seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("development seeds changed")
    weight = raw.get("weight_control", {})
    if str(weight.get("formula")) != FORMULA:
        raise ValueError("weight formula must be squared remoteness")
    if not math.isclose(float(weight.get("weight_at_zero")), 1.0, abs_tol=1e-12):
        raise ValueError("weight_at_zero must remain 1")
    if weight.get("positive_only_anchor") is not True:
        raise ValueError("Positive-only anchor must remain enabled")
    if not math.isclose(
        float(weight.get("reference_distance")), REFERENCE_DISTANCE, abs_tol=1e-12
    ):
        raise ValueError("reference_distance must remain 2")
    if tuple(float(value) for value in weight.get("exp_coefficients", ())) != EXPECTED_COEFFICIENTS:
        raise ValueError("squared EXP coefficient set changed")

    stages = raw.get("stages")
    if not isinstance(stages, list) or len(stages) != 3:
        raise ValueError("night suite requires exactly three stage records")
    by_id = {str(stage.get("id")): stage for stage in stages}
    stage_a = by_id.get("stage_a_squared_kernel", {})
    stage_b = by_id.get("stage_b_ppo_kl_early_refresh", {})
    stage_c = by_id.get("stage_c_gae", {})
    if stage_a.get("enabled") is not True or tuple(stage_a.get("actor_update_modes", ())) != (
        "a2c",
        "ppo_clip_k4",
    ):
        raise ValueError("Stage A actor matrix changed")
    if stage_b.get("enabled") is not True or tuple(stage_b.get("actor_update_modes", ())) != (
        "ppo_clip_kl_k16",
    ):
        raise ValueError("Stage B actor matrix changed")
    stage_a_ppo = stage_a.get("ppo", {})
    stage_b_ppo = stage_b.get("ppo", {})
    if float(stage_a_ppo.get("clip_epsilon")) != 0.2 or int(
        stage_a_ppo.get("updates_per_old_policy", -1)
    ) != 4:
        raise ValueError("Stage A PPO settings changed")
    if float(stage_b_ppo.get("clip_epsilon")) != 0.2 or int(
        stage_b_ppo.get("max_updates_per_old_policy", -1)
    ) != 16:
        raise ValueError("Stage B PPO settings changed")
    if stage_b_ppo.get("analytic_kl_early_refresh") is not True or not math.isclose(
        float(stage_b_ppo.get("target_kl")), 0.01, abs_tol=1e-12
    ):
        raise ValueError("Stage B KL contract changed")
    if stage_c.get("enabled") is not False or float(stage_c.get("gae_lambda")) != GAE_LAMBDA:
        raise ValueError("historical Stage C must remain blocked at lambda=0.95")
    if stage_c.get("status") != "blocked_pending_verified_trajectory_contract":
        raise ValueError("historical Stage C block reason changed")
    if int(raw.get("expected_stage_a_branches", -1)) != EXPECTED_STAGE_A_BRANCHES:
        raise ValueError("Stage A branch count changed")
    if int(raw.get("expected_stage_b_branches", -1)) != EXPECTED_STAGE_B_BRANCHES:
        raise ValueError("Stage B branch count changed")
    if int(raw.get("expected_runnable_branches", -1)) != EXPECTED_TOTAL_BRANCHES:
        raise ValueError("runnable branch count changed")


def _validate_gae_grid(raw: Mapping[str, Any]) -> None:
    required = {
        "run_kind": "pilot",
        "status": "not_run",
        "scientific_status": GAE_SCIENTIFIC_STATUS,
        "predecessor_experiment_id": EXPERIMENT_ID,
        "development_seeds": list(GAE_EXPECTED_SEEDS),
        "actor_update_modes": ["a2c"],
        "advantage_modes": ["one_step_td", "gae_lambda_0p95"],
        "expected_total_branches": GAE_EXPECTED_BRANCHES,
        "screening_only": True,
    }
    changed = [key for key, expected in required.items() if raw.get(key) != expected]
    weight = raw.get("weight_control", {})
    snapshot = raw.get("trajectory_snapshot", {})
    if changed:
        raise ValueError(f"joint-critic GAE grid changed: {changed}")
    if str(weight.get("formula")) != FORMULA:
        raise ValueError("GAE grid must retain the squared-remoteness formula")
    if tuple(float(value) for value in weight.get("exp_coefficients", ())) != GAE_COEFFICIENTS:
        raise ValueError("GAE coefficient set changed")
    if weight.get("positive_only_anchor") is not True:
        raise ValueError("GAE Positive-only anchor changed")
    if float(snapshot.get("gae_lambda", -1.0)) != GAE_LAMBDA:
        raise ValueError("GAE lambda changed")
    if int(snapshot.get("canonical_batch_size", -1)) != GAE_CANONICAL_BATCH_SIZE:
        raise ValueError("GAE canonical batch size changed")
    if snapshot.get("td_and_gae_share_snapshot") is not True:
        raise ValueError("TD and GAE must share each critic snapshot")
    if snapshot.get("critic_updated_every_step") is not True:
        raise ValueError("critic must remain active on every optimizer step")
    if snapshot.get("prepared_advantage_artifact") is not False:
        raise ValueError("prepared advantage artifacts are forbidden")
    if snapshot.get("terminal_bootstrap") is not False:
        raise ValueError("terminal bootstrap must remain disabled")
    if snapshot.get("timeout_bootstrap") is not True:
        raise ValueError("timeout bootstrap must remain enabled")
    if snapshot.get("terminal_timeout_recursion_stop") is not True:
        raise ValueError("terminal/timeout recursion stop changed")
    if snapshot.get("dataset_tail_recursion_stop") is not True:
        raise ValueError("dataset-tail recursion stop changed")


def load_grid(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = json.loads(source.read_text())
    experiment_id = raw.get("experiment_id")
    if experiment_id not in {EXPERIMENT_ID, GAE_EXPERIMENT_ID}:
        raise ValueError("unsupported squared-night grid experiment_id")
    _validate_common_grid(raw)
    if experiment_id == EXPERIMENT_ID:
        _validate_legacy_grid(raw)
    else:
        _validate_gae_grid(raw)
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
    run_spec["seeds"] = list(
        GAE_EXPECTED_SEEDS
        if _ACTIVE_EXPERIMENT_ID == GAE_EXPERIMENT_ID
        else EXPECTED_SEEDS
    )
    environment = run_spec.get("environment", {})
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if str(environment.get(name)) != "1":
            raise ValueError(f"run_spec {name} must remain 1")
    argv = [str(item) for item in run_spec["trainer_argv_template"]]
    expected_flags = {
        "--alpha": "0.11",
        "--batch": "256",
        "--lr": "0.0003",
        "--eval_interval": "50000",
        "--eval_episodes": "10",
        "--steps": "1000000",
    }
    for flag, expected in expected_flags.items():
        actual = _flag_value(argv, flag)
        if actual != expected:
            raise ValueError(f"source run_spec {flag} changed: {actual} != {expected}")
    if _ACTIVE_EXPERIMENT_ID == GAE_EXPERIMENT_ID:
        ret_weight = None
        positions = [index for index, token in enumerate(argv) if token == "--ret_weight_mode"]
        if positions:
            if len(positions) != 1 or positions[0] + 1 >= len(argv):
                raise ValueError("run_spec has malformed --ret_weight_mode")
            ret_weight = argv[positions[0] + 1]
        if ret_weight not in {None, "none"}:
            raise ValueError("GAE transition IDs require uniform ret_weight_mode=none")
    argv[argv.index("--steps") + 1] = "{steps}"
    run_spec["trainer_argv_template"] = argv
    run_spec["passthrough_variants"] = []
    return run_spec, digest


def control_points(grid: Mapping[str, Any]) -> list[tuple[float, float | None]]:
    points: list[tuple[float, float | None]] = [(0.0, None)]
    points.extend(
        (1.0, float(coefficient))
        for coefficient in grid["weight_control"]["exp_coefficients"]
    )
    expected = 4 if grid.get("experiment_id") == GAE_EXPERIMENT_ID else EXPECTED_CONTROLS_PER_MODE
    if len(points) != expected or len(points) != len(set(points)):
        raise ValueError("squared-night controls are not unique or complete")
    return points


def _control_fields(weight_at_zero: float, coefficient: float | None) -> tuple[str, float, str]:
    if coefficient is None:
        return "positive_only", 0.0, "positive_only__w0_0"
    return (
        "squared_exponential",
        float(coefficient),
        f"sqexp__w0_1__c_{_label(float(coefficient))}",
    )


def _legacy_branches(
    contract: base.CanonicalContract,
    run_spec: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> list[base.Branch]:
    datasets = [base.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
    seeds = [int(value) for value in run_spec["seeds"]]
    common = {
        "steps": str(EXPECTED_STEPS),
        "diagnostics_interval": str(DIAGNOSTICS_INTERVAL),
        "sampled_values_per_update": str(SAMPLED_VALUES_PER_UPDATE),
    }
    branches: list[base.Branch] = []
    for actor_mode in EXPECTED_ACTOR_MODES:
        stage = "stage_b" if actor_mode == "ppo_clip_kl_k16" else "stage_a"
        for weight_at_zero, coefficient in control_points(grid):
            method, coefficient_value, control_label = _control_fields(
                weight_at_zero, coefficient
            )
            for dataset in datasets:
                for seed in seeds:
                    branches.append(
                        base.Branch(
                            branch_id=(
                                f"{dataset.id}__seed{seed}__{control_label}__"
                                f"{actor_mode}__steps1m"
                            ),
                            branch_kind="injected",
                            dataset=dataset,
                            seed=seed,
                            template_values={
                                **common,
                                "stage": stage,
                                "actor_update_mode": actor_mode,
                                "weight_method": method,
                                "weight_at_zero": f"{weight_at_zero:.17g}",
                                "exp_coefficient": f"{coefficient_value:.17g}",
                                "reference_distance": f"{REFERENCE_DISTANCE:.17g}",
                            },
                            negative_control=None,
                        )
                    )
    return branches


def _gae_branches(
    run_spec: Mapping[str, Any], grid: Mapping[str, Any]
) -> list[base.Branch]:
    datasets = [base.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
    controls = control_points(grid)
    branches: list[base.Branch] = []
    for estimator in GAE_ESTIMATORS:
        for weight_at_zero, coefficient in controls:
            method, coefficient_value, _ = _control_fields(weight_at_zero, coefficient)
            control_label = (
                "positive_only"
                if coefficient is None
                else f"sqexp_c{coefficient_value:g}"
            )
            for dataset in datasets:
                for seed in GAE_EXPECTED_SEEDS:
                    branches.append(
                        base.Branch(
                            branch_id=(
                                f"{dataset.id}__seed{seed}__{estimator}__"
                                f"{control_label}__a2c__steps1m"
                            ),
                            branch_kind="injected",
                            dataset=dataset,
                            seed=seed,
                            template_values={
                                "steps": str(EXPECTED_STEPS),
                                "stage": "stage_c_joint_gae",
                                "actor_update_mode": "a2c",
                                "advantage_estimator": estimator,
                                "weight_method": method,
                                "weight_at_zero": f"{weight_at_zero:.17g}",
                                "exp_coefficient": f"{coefficient_value:.17g}",
                                "reference_distance": f"{REFERENCE_DISTANCE:.17g}",
                                "diagnostics_interval": str(DIAGNOSTICS_INTERVAL),
                                "sampled_values_per_update": str(SAMPLED_VALUES_PER_UPDATE),
                                "execution_mode": "full",
                            },
                            negative_control=None,
                        )
                    )
    if _LIVENESS_PAIR:
        assert _LIVENESS_STEPS is not None
        selected = [
            branch
            for branch in branches
            if branch.dataset.id == GAE_LIVENESS_DATASET
            and branch.seed == GAE_LIVENESS_SEED
            and branch.template_values["weight_method"] == "squared_exponential"
            and math.isclose(
                float(branch.template_values["exp_coefficient"]),
                GAE_LIVENESS_COEFFICIENT,
                abs_tol=1e-12,
            )
        ]
        branches = [
            dataclasses.replace(
                branch,
                branch_id=f"{branch.branch_id}__liveness_steps{_LIVENESS_STEPS}",
                template_values={
                    **branch.template_values,
                    "steps": str(_LIVENESS_STEPS),
                    "execution_mode": "liveness",
                },
            )
            for branch in selected
        ]
    return branches


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
    datasets = tuple(str(item["id"]) for item in run_spec["datasets"])
    if datasets != EXPECTED_DATASETS:
        raise ValueError("expanded dataset subset changed")
    if grid.get("experiment_id") == GAE_EXPERIMENT_ID:
        branches = _gae_branches(run_spec, grid)
    else:
        if tuple(int(value) for value in run_spec["seeds"]) != EXPECTED_SEEDS:
            raise ValueError("expanded development seeds changed")
        branches = _legacy_branches(contract, run_spec, grid)
    ids = [branch.branch_id for branch in branches]
    if len(ids) != len(set(ids)):
        raise ValueError("night-suite branch IDs are not unique")
    expected = active_expected_branch_count()
    if len(branches) != expected:
        raise ValueError(f"expected {expected} branches, built {len(branches)}")
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
    actor_mode = str(values["actor_update_mode"])
    allowed_modes = {"a2c"} if _ACTIVE_EXPERIMENT_ID == GAE_EXPERIMENT_ID else set(EXPECTED_ACTOR_MODES)
    if actor_mode not in allowed_modes:
        raise ValueError("branch actor mode changed")
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
    if values.get("execution_mode") == "liveness":
        _replace_flag(trainer_args, "--eval_interval", str(values["steps"]))
        _replace_flag(trainer_args, "--eval_episodes", "1")
    weight_control = {
        "method": method,
        "weight_at_zero": weight_at_zero,
        "exp_coefficient": coefficient,
        "reference_distance": REFERENCE_DISTANCE,
        "formula": FORMULA,
    }
    branch_config: dict[str, Any] = {
        "experiment_id": _ACTIVE_EXPERIMENT_ID,
        "branch_id": branch.branch_id,
        "branch_kind": branch.branch_kind,
        "dataset_id": branch.dataset.id,
        "dataset_sha256": branch.dataset.sha256,
        "seed": branch.seed,
        "template_values": values,
        "weight_control": weight_control,
    }
    if _ACTIVE_EXPERIMENT_ID == GAE_EXPERIMENT_ID:
        branch_config.update(
            canonical_root=str(contract.source_root),
            dataset_path=str(Path(branch.dataset.path).expanduser().resolve()),
        )
    branch_config_path = branch_dir / "branch_config.json"
    base.atomic_write_json(branch_config_path, branch_config)
    command = [
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
    ]
    return command, branch_config


def _grid_argument(argv: Sequence[str]) -> str:
    values = list(argv)
    positions = [index for index, token in enumerate(values) if token == "--grid"]
    if len(positions) != 1 or positions[0] + 1 >= len(values):
        raise ValueError("squared-night command requires exactly one --grid")
    return values[positions[0] + 1]


def main(argv: list[str] | None = None) -> int:
    global _ACTIVE_EXPERIMENT_ID, _LIVENESS_PAIR, _LIVENESS_STEPS
    delegated = list(sys.argv[1:] if argv is None else argv)
    previous_profile = (_ACTIVE_EXPERIMENT_ID, _LIVENESS_PAIR, _LIVENESS_STEPS)
    env_liveness = os.environ.get("DRPO_E7_GAE_LIVENESS_PAIR") == "1"
    env_steps = os.environ.get("DRPO_E7_GAE_LIVENESS_STEPS")
    configure_execution(
        _grid_argument(delegated),
        liveness_pair=env_liveness,
        liveness_steps=int(env_steps) if env_steps is not None else None,
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
    base.EXPERIMENT_ID = active_experiment_id()
    base.SCIENTIFIC_STATUS = active_scientific_status()
    base.RUNNER_VERSION = active_runner_version()
    base.load_grid = load_grid
    base.load_run_spec = load_run_spec
    base.build_branches = build_branches
    base.branch_command = branch_command
    try:
        result = base.main(delegated)
        if delegated and delegated[0] == "run":
            if "--work-dir" not in delegated:
                raise ValueError("run command is missing --work-dir")
            index = delegated.index("--work-dir")
            if index + 1 >= len(delegated):
                raise ValueError("run command has no --work-dir value")
            aggregate_results(delegated[index + 1])
        return int(result)
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
        _ACTIVE_EXPERIMENT_ID, _LIVENESS_PAIR, _LIVENESS_STEPS = previous_profile


if __name__ == "__main__":
    raise SystemExit(main())
