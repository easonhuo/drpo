"""Canonical E7 PPO-stability pilot adapter.

The adapter reuses the generic canonical sweep runner and the unchanged source
contract. It freezes a 96-branch matrix and changes only the injected actor
update mode between historical A2C and PPO clipping.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from drpo import e7_canonical_sweep as base
from drpo.e7_canonical_injection import NegativeControl, sha256_file
from drpo.e7_ppo_stability_aggregate import aggregate as aggregate_results

EXPERIMENT_ID = "EXT-H-E7-PPO-STABILITY-01"
SCIENTIFIC_STATUS = "ppo_actor_stability_pilot_only"
RUNNER_VERSION = "1.0.0-canonical-ppo-stability"
EXPECTED_SOURCE_DATASETS = (
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-expert-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
)
EXPECTED_DATASETS = (
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
)
SOURCE_RUN_SPEC_SEEDS = (200, 201)
EXPECTED_SEEDS = (200, 201, 202, 203)
EXPECTED_COEFFICIENTS = (0.5, 1.0, 1.5)
EXPECTED_ACTOR_UPDATES = ("a2c", "ppo_clip")
EXPECTED_TOTAL_BRANCHES = 96
EXPECTED_STEPS = 1_000_000
EXPECTED_CLIP_EPSILON = 0.2
EXPECTED_UPDATES_PER_OLD_POLICY = 4
EXPECTED_DIAGNOSTICS_INTERVAL = 1000

_BASE_LOAD_RUN_SPEC = base.load_run_spec


def _label(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def _flag_value(argv: list[str], flag: str) -> str:
    positions = [index for index, token in enumerate(argv) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"trainer_argv_template must contain exactly one {flag}")
    return argv[positions[0] + 1]


def load_ppo_grid(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = json.loads(source.read_text())
    if raw.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"grid experiment_id must be {EXPERIMENT_ID}")
    if raw.get("run_kind") != "pilot":
        raise ValueError("PPO stability grid must remain a pilot")
    if raw.get("scientific_status") != SCIENTIFIC_STATUS:
        raise ValueError("unexpected scientific_status")
    if tuple(raw.get("datasets", ())) != EXPECTED_DATASETS:
        raise ValueError("PPO stability datasets changed")
    if tuple(int(value) for value in raw.get("development_seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("PPO stability development seeds changed")
    if tuple(int(value) for value in raw.get("held_out_seeds", ())) != (
        204,
        205,
        206,
        207,
    ):
        raise ValueError("held-out seed reservation changed")
    if tuple(float(value) for value in raw.get("exp_coefficients", ())) != EXPECTED_COEFFICIENTS:
        raise ValueError("EXP coefficient set changed")
    if tuple(raw.get("actor_update_modes", ())) != EXPECTED_ACTOR_UPDATES:
        raise ValueError("actor update modes changed")
    if int(raw.get("steps", -1)) != EXPECTED_STEPS:
        raise ValueError("steps must remain 1,000,000")
    if int(raw.get("expected_total_branches", -1)) != EXPECTED_TOTAL_BRANCHES:
        raise ValueError("expected_total_branches must remain 96")

    ppo = raw.get("ppo", {})
    if not math.isclose(
        float(ppo.get("clip_epsilon")),
        EXPECTED_CLIP_EPSILON,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("clip_epsilon must remain 0.2")
    if int(ppo.get("updates_per_old_policy", -1)) != EXPECTED_UPDATES_PER_OLD_POLICY:
        raise ValueError("updates_per_old_policy must remain 4")
    if int(ppo.get("diagnostics_interval", -1)) != EXPECTED_DIAGNOSTICS_INTERVAL:
        raise ValueError("diagnostics_interval must remain 1000")
    forbidden = {
        "kl_penalty",
        "target_kl",
        "entropy_bonus",
        "actor_gradient_clip",
        "value_clip",
    }
    if any(ppo.get(name) not in (None, False, 0, 0.0) for name in forbidden):
        raise ValueError("PPO stability V1 forbids auxiliary PPO tricks")

    if not math.isclose(
        float(raw.get("canonical_alpha")),
        0.11,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("canonical_alpha must remain 0.11")
    if not math.isclose(
        float(raw.get("reference_distance")),
        2.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("reference_distance must remain 2.0")
    return raw, sha256_file(source)


def load_ppo_run_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    raw, digest = _BASE_LOAD_RUN_SPEC(path)
    run_spec = copy.deepcopy(raw)
    if run_spec.get("experiment_id") != "EXT-H-E7-BENCH-01":
        raise ValueError("source run_spec experiment_id changed")
    source_dataset_ids = tuple(str(item["id"]) for item in run_spec["datasets"])
    if source_dataset_ids != EXPECTED_SOURCE_DATASETS:
        raise ValueError(f"source run_spec datasets changed: {source_dataset_ids}")
    by_id = {str(item["id"]): item for item in run_spec["datasets"]}
    run_spec["datasets"] = [copy.deepcopy(by_id[name]) for name in EXPECTED_DATASETS]

    source_seeds = tuple(int(value) for value in run_spec["seeds"])
    if source_seeds != SOURCE_RUN_SPEC_SEEDS:
        raise ValueError(
            f"source run_spec seeds changed: {source_seeds}; "
            f"expected {SOURCE_RUN_SPEC_SEEDS}"
        )
    run_spec["seeds"] = list(EXPECTED_SEEDS)

    environment = run_spec.get("environment", {})
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if str(environment.get(name)) != "1":
            raise ValueError(f"run_spec {name} must remain 1")

    argv = [str(item) for item in run_spec["trainer_argv_template"]]
    if _flag_value(argv, "--alpha") != "0.11":
        raise ValueError("trainer alpha must remain 0.11")
    if _flag_value(argv, "--batch") != "256":
        raise ValueError("trainer batch must remain 256")
    if _flag_value(argv, "--lr") != "0.0003":
        raise ValueError("trainer learning rate must remain 0.0003")
    if _flag_value(argv, "--eval_interval") != "50000":
        raise ValueError("trainer eval_interval must remain 50000")
    if _flag_value(argv, "--eval_episodes") != "10":
        raise ValueError("trainer eval_episodes must remain 10")
    if _flag_value(argv, "--steps") != "1000000":
        raise ValueError("source run_spec --steps must remain 1000000")
    step_index = argv.index("--steps")
    argv[step_index + 1] = "{steps}"
    run_spec["trainer_argv_template"] = argv
    run_spec["passthrough_variants"] = []
    return run_spec, digest


def _negative_controls(grid: Mapping[str, Any]) -> list[tuple[str, NegativeControl]]:
    alpha = float(grid["canonical_alpha"])
    reference_distance = float(grid["reference_distance"])
    controls: list[tuple[str, NegativeControl]] = [
        (
            "positive_only__scale0",
            NegativeControl(
                method="positive_only",
                negative_scale=0.0,
                canonical_alpha=alpha,
                reference_distance=reference_distance,
                exponential_coefficient=EXPECTED_COEFFICIENTS[0],
            ),
        )
    ]
    for coefficient in EXPECTED_COEFFICIENTS:
        controls.append(
            (
                f"exponential__scale1__coef{_label(coefficient)}",
                NegativeControl(
                    method="exponential",
                    negative_scale=1.0,
                    canonical_alpha=alpha,
                    reference_distance=reference_distance,
                    exponential_coefficient=coefficient,
                ),
            )
        )
    return controls


def build_ppo_branches(
    contract: base.CanonicalContract,
    run_spec: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> list[base.Branch]:
    datasets = [base.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
    if tuple(item.id for item in datasets) != EXPECTED_DATASETS:
        raise ValueError("expanded dataset subset changed")
    seeds = [int(value) for value in run_spec["seeds"]]
    if tuple(seeds) != EXPECTED_SEEDS:
        raise ValueError("expanded development seeds changed")
    if not math.isclose(
        float(grid["canonical_alpha"]),
        contract.expected_canonical_alpha,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("grid canonical_alpha does not match contract")

    ppo = grid["ppo"]
    common_template = {
        "steps": str(EXPECTED_STEPS),
        "clip_epsilon": str(EXPECTED_CLIP_EPSILON),
        "updates_per_old_policy": str(EXPECTED_UPDATES_PER_OLD_POLICY),
        "diagnostics_interval": str(EXPECTED_DIAGNOSTICS_INTERVAL),
    }
    branches: list[base.Branch] = []
    for control_label, control in _negative_controls(grid):
        for actor_update_mode in EXPECTED_ACTOR_UPDATES:
            actor_label = (
                "a2c"
                if actor_update_mode == "a2c"
                else (
                    f"ppo_clip_eps{_label(float(ppo['clip_epsilon']))}__"
                    f"k{int(ppo['updates_per_old_policy'])}"
                )
            )
            for dataset in datasets:
                for seed in seeds:
                    branches.append(
                        base.Branch(
                            branch_id=(
                                f"{dataset.id}__seed{seed}__{control_label}__"
                                f"{actor_label}__steps1m"
                            ),
                            branch_kind="injected",
                            dataset=dataset,
                            seed=seed,
                            template_values={
                                **common_template,
                                "actor_update_mode": actor_update_mode,
                            },
                            negative_control=control,
                        )
                    )
    ids = [branch.branch_id for branch in branches]
    if len(ids) != len(set(ids)):
        raise ValueError("PPO stability branch IDs are not unique")
    if len(branches) != EXPECTED_TOTAL_BRANCHES:
        raise ValueError(
            f"expected {EXPECTED_TOTAL_BRANCHES} branches but built {len(branches)}"
        )
    return branches


def ppo_branch_command(
    *,
    contract_path: Path,
    contract: base.CanonicalContract,
    branch: base.Branch,
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
    trainer_args = [
        base._format_value(str(item), context)  # noqa: SLF001
        for item in trainer_argv_template
    ]
    branch_config = {
        "branch_id": branch.branch_id,
        "branch_kind": branch.branch_kind,
        "dataset_id": branch.dataset.id,
        "dataset_sha256": branch.dataset.sha256,
        "seed": branch.seed,
        "template_values": branch.template_values,
        "negative_control": dataclasses.asdict(branch.negative_control),
    }
    branch_config_path = branch_dir / "branch_config.json"
    base.atomic_write_json(branch_config_path, branch_config)
    command = [
        sys.executable,
        "-m",
        "drpo.e7_canonical_ppo_bootstrap",
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
    previous_experiment_id = base.EXPERIMENT_ID
    previous_scientific_status = base.SCIENTIFIC_STATUS
    previous_runner_version = base.RUNNER_VERSION
    previous_load_grid = base.load_grid
    previous_load_run_spec = base.load_run_spec
    previous_build_branches = base.build_branches
    previous_branch_command = base.branch_command
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.SCIENTIFIC_STATUS = SCIENTIFIC_STATUS
    base.RUNNER_VERSION = RUNNER_VERSION
    base.load_grid = load_ppo_grid
    base.load_run_spec = load_ppo_run_spec
    base.build_branches = build_ppo_branches
    base.branch_command = ppo_branch_command
    try:
        delegated_argv = list(sys.argv[1:] if argv is None else argv)
        result = base.main(delegated_argv)
        if delegated_argv and delegated_argv[0] == "run":
            if "--work-dir" not in delegated_argv:
                raise ValueError("run command is missing --work-dir")
            work_index = delegated_argv.index("--work-dir")
            if work_index + 1 >= len(delegated_argv):
                raise ValueError("run command has no --work-dir value")
            aggregate_results(delegated_argv[work_index + 1])
        return result
    finally:
        base.EXPERIMENT_ID = previous_experiment_id
        base.SCIENTIFIC_STATUS = previous_scientific_status
        base.RUNNER_VERSION = previous_runner_version
        base.load_grid = previous_load_grid
        base.load_run_spec = previous_load_run_spec
        base.build_branches = previous_build_branches
        base.branch_command = previous_branch_command


if __name__ == "__main__":
    raise SystemExit(main())
