"""Frozen matrix and branch contract for EXT-H-E7-SQEXP-GAE-01."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from drpo import e7_canonical_sweep as base
from drpo import e7_w0_highc_actor as predecessor
from drpo.e7_canonical_injection import CanonicalContract, sha256_file
from drpo.e7_squared_exp_kernel import FORMULA


EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-01"
SCIENTIFIC_STATUS = "shared_frozen_critic_behavior_trajectory_gae_pilot_not_run"
RUNNER_VERSION = "1.0.0-e7-sqexp-gae"
EXPECTED_SOURCE_DATASETS = predecessor.EXPECTED_SOURCE_DATASETS
EXPECTED_SOURCE_SEEDS = predecessor.EXPECTED_SEEDS
EXPECTED_DATASETS = (
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
)
EXPECTED_SEEDS = (204, 205, 206, 207)
EXPECTED_COEFFICIENTS = (64.0, 128.0, 256.0)
EXPECTED_ACTOR_MODES = ("a2c", "ppo_clip_k4")
EXPECTED_ADVANTAGE_ESTIMATORS = ("td", "gae")
EXPECTED_STEPS = 1_000_000
EXPECTED_BRANCHES = 192
REFERENCE_DISTANCE = 2.0
INTERNAL_CANONICAL_ALPHA = 0.11
DIAGNOSTICS_INTERVAL = 1000
SAMPLED_VALUES_PER_UPDATE = 16


def _label(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def _flag_value(argv: list[str], flag: str) -> str:
    positions = [index for index, token in enumerate(argv) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"trainer template must contain exactly one {flag}")
    return argv[positions[0] + 1]


def _require_float_flag(
    argv: list[str],
    flag: str,
    expected: float,
    *,
    abs_tol: float = 1e-12,
) -> None:
    actual_text = _flag_value(argv, flag)
    try:
        actual = float(actual_text)
    except ValueError as exc:
        raise ValueError(
            f"source trainer {flag} is not numeric: {actual_text!r}"
        ) from exc
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=abs_tol):
        raise ValueError(f"source trainer {flag} changed: {actual} != {expected}")


def _require_int_flag(argv: list[str], flag: str, expected: int) -> None:
    actual_text = _flag_value(argv, flag)
    try:
        actual = int(actual_text)
    except ValueError as exc:
        raise ValueError(
            f"source trainer {flag} is not an integer: {actual_text!r}"
        ) from exc
    if actual != expected:
        raise ValueError(f"source trainer {flag} changed: {actual} != {expected}")


def load_grid(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = json.loads(source.read_text())
    if raw.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"grid experiment_id must be {EXPERIMENT_ID}")
    if raw.get("run_kind") != "pilot" or raw.get("status") != "not_run":
        raise ValueError("GAE grid must remain a not-run pilot")
    if tuple(raw.get("datasets", ())) != EXPECTED_DATASETS:
        raise ValueError("GAE datasets changed")
    if tuple(int(value) for value in raw.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("GAE confirmatory seeds changed")
    if tuple(raw.get("advantage_estimators", ())) != EXPECTED_ADVANTAGE_ESTIMATORS:
        raise ValueError("advantage estimator matrix changed")
    if tuple(raw.get("actor_update_modes", ())) != EXPECTED_ACTOR_MODES:
        raise ValueError("actor update matrix changed")
    control = raw.get("weight_control", {})
    if control.get("formula") != FORMULA:
        raise ValueError("squared-remoteness formula changed")
    if not math.isclose(float(control.get("weight_at_zero")), 1.0, abs_tol=1e-12):
        raise ValueError("weight_at_zero must remain 1")
    if control.get("positive_only_anchor") is not True:
        raise ValueError("Positive-only anchor must remain enabled")
    if not math.isclose(
        float(control.get("reference_distance")),
        REFERENCE_DISTANCE,
        abs_tol=1e-12,
    ):
        raise ValueError("reference_distance must remain 2")
    coefficients = tuple(float(value) for value in control.get("exp_coefficients", ()))
    if coefficients != EXPECTED_COEFFICIENTS:
        raise ValueError("GAE coefficient shortlist changed")
    if int(raw.get("steps", -1)) != EXPECTED_STEPS:
        raise ValueError("actor steps must remain 1,000,000")
    if int(raw.get("expected_branches", -1)) != EXPECTED_BRANCHES:
        raise ValueError("expected branch count changed")
    prepared = raw.get("shared_critic", {})
    expected_prepared = {
        "steps": 100_000,
        "batch_size": 256,
        "learning_rate": 0.0003,
        "expectile": 0.5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "device": "cpu",
    }
    if prepared != expected_prepared:
        raise ValueError("shared critic/GAE preparation settings changed")
    if raw.get("formal_evidence_allowed") is not False:
        raise ValueError("implementation pilot cannot allow formal evidence")
    return raw, sha256_file(source)


def load_run_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    raw, digest = predecessor._BASE_LOAD_RUN_SPEC(path)  # noqa: SLF001
    run_spec = copy.deepcopy(raw)
    if run_spec.get("experiment_id") != "EXT-H-E7-BENCH-01":
        raise ValueError("source run spec experiment_id changed")
    source_dataset_ids = tuple(str(item["id"]) for item in run_spec["datasets"])
    if source_dataset_ids != EXPECTED_SOURCE_DATASETS:
        raise ValueError("source nine-task dataset order changed")
    source_seeds = tuple(int(value) for value in run_spec["seeds"])
    if source_seeds != EXPECTED_SOURCE_SEEDS:
        raise ValueError("source development seeds changed")
    by_id = {str(item["id"]): item for item in run_spec["datasets"]}
    run_spec["datasets"] = [copy.deepcopy(by_id[name]) for name in EXPECTED_DATASETS]
    run_spec["seeds"] = list(EXPECTED_SEEDS)
    environment = run_spec.get("environment", {})
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if str(environment.get(name)) != "1":
            raise ValueError(f"run spec {name} must remain 1")
    argv = [str(item) for item in run_spec["trainer_argv_template"]]
    if _flag_value(argv, "--variant") != "iqlv_exp_rank":
        raise ValueError("source trainer variant changed from iqlv_exp_rank")
    _require_float_flag(argv, "--alpha", 0.11)
    _require_float_flag(argv, "--tau", 0.5)
    _require_float_flag(argv, "--temp", 5.0)
    _require_int_flag(argv, "--batch", 256)
    _require_float_flag(argv, "--lr", 0.0003)
    _require_int_flag(argv, "--eval_interval", 50_000)
    _require_int_flag(argv, "--eval_episodes", 10)
    _require_int_flag(argv, "--steps", 1_000_000)
    argv[argv.index("--seed") + 1] = "{seed}"
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
    if len(points) != 4 or len(points) != len(set(points)):
        raise ValueError("GAE pilot requires Positive-only plus three c values")
    return points


def build_branches(
    contract: CanonicalContract,
    run_spec: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> list[base.Branch]:
    if not math.isclose(
        contract.expected_canonical_alpha,
        INTERNAL_CANONICAL_ALPHA,
        abs_tol=1e-12,
    ):
        raise ValueError("canonical alpha changed from 0.11")
    datasets = [base.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
    if tuple(item.id for item in datasets) != EXPECTED_DATASETS:
        raise ValueError("expanded dataset subset changed")
    seeds = [int(value) for value in run_spec["seeds"]]
    if tuple(seeds) != EXPECTED_SEEDS:
        raise ValueError("expanded seed set changed")
    common = {
        "steps": str(EXPECTED_STEPS),
        "diagnostics_interval": str(DIAGNOSTICS_INTERVAL),
        "sampled_values_per_update": str(SAMPLED_VALUES_PER_UPDATE),
    }
    branches: list[base.Branch] = []
    for estimator in EXPECTED_ADVANTAGE_ESTIMATORS:
        for actor_mode in EXPECTED_ACTOR_MODES:
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
                            f"{dataset.id}__seed{seed}__{estimator}__"
                            f"{control_label}__{actor_mode}__steps1m"
                        )
                        branches.append(
                            base.Branch(
                                branch_id=branch_id,
                                branch_kind="injected",
                                dataset=dataset,
                                seed=seed,
                                template_values={
                                    **common,
                                    "advantage_estimator": estimator,
                                    "actor_update_mode": actor_mode,
                                    "weight_method": method,
                                    "weight_at_zero": f"{weight_at_zero:.17g}",
                                    "exp_coefficient": f"{coefficient_value:.17g}",
                                    "reference_distance": f"{REFERENCE_DISTANCE:.17g}",
                                },
                                negative_control=None,
                            )
                        )
    ids = [branch.branch_id for branch in branches]
    if len(ids) != EXPECTED_BRANCHES or len(ids) != len(set(ids)):
        raise ValueError(
            f"expected {EXPECTED_BRANCHES} unique branches, got {len(ids)}"
        )
    return branches


def branch_command(
    *,
    contract_path: Path,
    contract: CanonicalContract,
    branch: base.Branch,
    branch_dir: Path,
    trainer_argv_template: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    values = branch.template_values
    work_dir = branch_dir.parent.parent
    prepared_manifest = (
        work_dir
        / "prepared"
        / branch.dataset.id
        / f"seed{branch.seed}"
        / "ADVANTAGE_MANIFEST.json"
    )
    context: dict[str, Any] = {
        "canonical_root": str(contract.source_root),
        "dataset_id": branch.dataset.id,
        "dataset_path": str(Path(branch.dataset.path).expanduser().resolve()),
        "dataset_sha256": branch.dataset.sha256,
        "seed": branch.seed,
        "output_dir": str(branch_dir / "trainer_output"),
        "branch_id": branch.branch_id,
        **values,
    }
    trainer_args = [
        base._format_value(str(item), context)  # noqa: SLF001
        for item in trainer_argv_template
    ]
    method = str(values["weight_method"])
    weight_at_zero = float(values["weight_at_zero"])
    coefficient = float(values["exp_coefficient"])
    branch_config = {
        "experiment_id": EXPERIMENT_ID,
        "branch_id": branch.branch_id,
        "branch_kind": "injected",
        "dataset_id": branch.dataset.id,
        "dataset_sha256": branch.dataset.sha256,
        "seed": branch.seed,
        "template_values": values,
        "advantage_manifest": str(prepared_manifest),
        "weight_control": {
            "method": method,
            "weight_at_zero": weight_at_zero,
            "exp_coefficient": coefficient,
            "reference_distance": REFERENCE_DISTANCE,
            "formula": FORMULA,
        },
    }
    branch_config_path = branch_dir / "branch_config.json"
    base.atomic_write_json(branch_config_path, branch_config)
    command = [
        sys.executable,
        "-m",
        "drpo.e7_sqexp_gae_bootstrap",
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
