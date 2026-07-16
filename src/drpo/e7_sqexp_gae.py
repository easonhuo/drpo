"""Thin 192-branch wrapper over the existing canonical E7 sweep."""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from drpo import e7_canonical_sweep as base
from drpo import e7_w0_highc_actor as predecessor
from drpo.e7_canonical_injection import sha256_file
from drpo.e7_squared_exp_kernel import FORMULA

EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-01"
SCIENTIFIC_STATUS = "frozen_critic_trajectory_gae_development_pilot_only"
RUNNER_VERSION = "2.0.0-minimal-canonical-wrapper"
EXPECTED_DATASETS = (
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
)
EXPECTED_SEEDS = (200, 201, 202, 203)
HELD_OUT_SEEDS = (204, 205, 206, 207)
ACTOR_MODES = ("a2c", "ppo_clip_k4")
ESTIMATORS = ("td", "gae")
COEFFICIENTS = (64.0, 128.0, 256.0)
EXPECTED_BRANCHES = 192
STEPS = 1_000_000
REFERENCE_DISTANCE = 2.0
_ORIGINAL_WRITE_PLAN = base.write_plan


def _label(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def _flag(argv: list[str], name: str) -> str:
    positions = [index for index, token in enumerate(argv) if token == name]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"trainer template must contain exactly one {name}")
    return argv[positions[0] + 1]


def load_grid(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = json.loads(source.read_text())
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "run_kind": "pilot",
        "status": "not_run",
        "scientific_status": SCIENTIFIC_STATUS,
        "datasets": list(EXPECTED_DATASETS),
        "development_seeds": list(EXPECTED_SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "actor_update_modes": list(ACTOR_MODES),
        "advantage_estimators": list(ESTIMATORS),
        "steps": STEPS,
        "evaluation_interval": 50_000,
        "evaluation_episodes": 10,
        "expected_total_branches": EXPECTED_BRANCHES,
        "formal_evidence_allowed": False,
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f"GAE grid field changed: {key}")
    critic = raw.get("shared_frozen_critic", {})
    if critic != {
        "steps": 100_000,
        "batch": 256,
        "gamma": 0.99,
        "tau": 0.5,
        "lr": 3e-4,
        "temperature": 5.0,
        "shared_per_dataset_seed": True,
        "updated_during_actor_training": False,
    }:
        raise ValueError("shared frozen critic contract changed")
    trajectory = raw.get("trajectory_advantage", {})
    if trajectory != {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "terminal_bootstrap": False,
        "timeout_bootstrap": True,
        "terminal_stops_recursion": True,
        "timeout_stops_recursion": True,
        "tail_stops_recursion": True,
        "normalization": "none",
        "clipping": "none",
    }:
        raise ValueError("trajectory GAE contract changed")
    control = raw.get("weight_control", {})
    if (
        control.get("formula") != FORMULA
        or float(control.get("weight_at_zero", -1)) != 1.0
        or control.get("positive_only_anchor") is not True
        or float(control.get("reference_distance", -1)) != REFERENCE_DISTANCE
        or tuple(float(value) for value in control.get("exp_coefficients", ())) != COEFFICIENTS
    ):
        raise ValueError("squared-EXP control contract changed")
    return raw, sha256_file(source)


def load_run_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    raw, digest = predecessor._BASE_LOAD_RUN_SPEC(path)  # noqa: SLF001
    run_spec = copy.deepcopy(raw)
    if run_spec.get("experiment_id") != "EXT-H-E7-BENCH-01":
        raise ValueError("source run spec experiment_id changed")
    source_ids = tuple(str(item["id"]) for item in run_spec["datasets"])
    if source_ids != predecessor.EXPECTED_SOURCE_DATASETS:
        raise ValueError("source nine-task dataset order changed")
    source_seeds = tuple(int(value) for value in run_spec["seeds"])
    if source_seeds != predecessor.EXPECTED_SEEDS:
        raise ValueError("source development seeds changed")
    by_id = {str(item["id"]): item for item in run_spec["datasets"]}
    run_spec["datasets"] = [copy.deepcopy(by_id[name]) for name in EXPECTED_DATASETS]
    run_spec["seeds"] = list(EXPECTED_SEEDS)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if str(run_spec.get("environment", {}).get(name)) != "1":
            raise ValueError(f"source run spec {name} changed")
    argv = [str(item) for item in run_spec["trainer_argv_template"]]
    for flag, expected in {
        "--alpha": 0.11,
        "--tau": 0.5,
        "--temp": 5.0,
        "--batch": 256,
        "--lr": 3e-4,
        "--eval_interval": 50_000,
        "--eval_episodes": 10,
        "--steps": STEPS,
    }.items():
        if not math.isclose(float(_flag(argv, flag)), float(expected), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"source trainer {flag} changed")
    argv[argv.index("--steps") + 1] = "{steps}"
    run_spec["trainer_argv_template"] = argv
    run_spec["passthrough_variants"] = []
    return run_spec, digest


def build_branches(
    contract: base.CanonicalContract,
    run_spec: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> list[base.Branch]:
    if not math.isclose(contract.expected_canonical_alpha, 0.11, abs_tol=1e-12):
        raise ValueError("canonical source alpha changed")
    datasets = [base.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
    controls = [("positive_only", 0.0, 0.0)] + [
        ("squared_exponential", 1.0, coefficient) for coefficient in COEFFICIENTS
    ]
    branches: list[base.Branch] = []
    for estimator in ESTIMATORS:
        for actor_mode in ACTOR_MODES:
            for method, weight_at_zero, coefficient in controls:
                label = "positive_only" if method == "positive_only" else f"sqexp_c{_label(coefficient)}"
                for dataset in datasets:
                    for seed in EXPECTED_SEEDS:
                        branches.append(
                            base.Branch(
                                branch_id=(
                                    f"{dataset.id}__seed{seed}__{estimator}__{label}__"
                                    f"{actor_mode}__steps1m"
                                ),
                                branch_kind="injected",
                                dataset=dataset,
                                seed=seed,
                                template_values={
                                    "steps": str(STEPS),
                                    "actor_update_mode": actor_mode,
                                    "advantage_estimator": estimator,
                                    "weight_method": method,
                                    "weight_at_zero": f"{weight_at_zero:.17g}",
                                    "exp_coefficient": f"{coefficient:.17g}",
                                    "reference_distance": f"{REFERENCE_DISTANCE:.17g}",
                                    "diagnostics_interval": "1000",
                                    "sampled_values_per_update": "16",
                                },
                                negative_control=None,
                            )
                        )
    ids = [branch.branch_id for branch in branches]
    if len(ids) != EXPECTED_BRANCHES or len(ids) != len(set(ids)):
        raise RuntimeError(f"expected {EXPECTED_BRANCHES} unique branches")
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
    work_dir = branch_dir.parent.parent
    manifest = work_dir / "prepared" / branch.dataset.id / f"seed{branch.seed}" / "ADVANTAGE_MANIFEST.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing preserved shared-critic artifact: {manifest}")
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
    branch_config = {
        "experiment_id": EXPERIMENT_ID,
        "branch_id": branch.branch_id,
        "branch_kind": branch.branch_kind,
        "dataset_id": branch.dataset.id,
        "dataset_path": context["dataset_path"],
        "dataset_sha256": branch.dataset.sha256,
        "seed": branch.seed,
        "template_values": values,
        "advantage_manifest": str(manifest),
        "weight_control": {
            "method": values["weight_method"],
            "weight_at_zero": float(values["weight_at_zero"]),
            "exp_coefficient": float(values["exp_coefficient"]),
            "reference_distance": REFERENCE_DISTANCE,
            "formula": FORMULA,
        },
    }
    config_path = branch_dir / "branch_config.json"
    base.atomic_write_json(config_path, branch_config)
    return (
        [
            sys.executable,
            "-m",
            "drpo.e7_sqexp_gae_minimal",
            "--contract",
            str(contract_path),
            "--branch-config",
            str(config_path),
            "--branch-manifest",
            str(branch_dir / "branch_manifest.json"),
            "--",
            *trainer_args,
        ],
        branch_config,
    )


def write_plan(**kwargs: Any) -> dict[str, Any]:
    work_dir = Path(kwargs["work_dir"]).expanduser().resolve()
    pairs = {(branch.dataset.id, branch.seed) for branch in kwargs["branches"]}
    missing = [
        str(work_dir / "prepared" / dataset / f"seed{seed}" / "ADVANTAGE_MANIFEST.json")
        for dataset, seed in sorted(pairs)
        if not (work_dir / "prepared" / dataset / f"seed{seed}" / "ADVANTAGE_MANIFEST.json").is_file()
    ]
    if missing:
        raise FileNotFoundError("missing preserved prepared artifacts: " + ", ".join(missing))
    return _ORIGINAL_WRITE_PLAN(**kwargs)


def main(argv: list[str] | None = None) -> int:
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
    base.load_grid, base.load_run_spec = load_grid, load_run_spec
    base.build_branches, base.branch_command, base.write_plan = build_branches, branch_command, write_plan
    try:
        return int(base.main(list(sys.argv[1:] if argv is None else argv)))
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
