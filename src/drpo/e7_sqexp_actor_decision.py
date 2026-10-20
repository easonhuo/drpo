"""Code-first runner for the 192-branch E7 actor/high-c decision pilot."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from drpo import e7_canonical_sweep as base
from drpo import e7_w0_highc_actor as source
from drpo.e7_canonical_injection import sha256_file
from drpo.e7_sqexp_actor_decision_aggregate import aggregate as aggregate_results


EXPERIMENT_ID = "EXT-H-E7-SQEXP-ACTOR-DECISION-01"
SCIENTIFIC_STATUS = "four_seed_actor_and_high_c_decision_screening_only"
RUNNER_VERSION = "1.0.0-e7-sqexp-actor-decision-192"

EXPECTED_DATASETS = (
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
)
SOURCE_SEEDS = (200, 201)
EXPECTED_SEEDS = (200, 201, 202, 203)
HELD_OUT_SEEDS = (204, 205, 206, 207)
EXPECTED_ACTOR_MODES = ("a2c", "ppo_clip_kl_k4")
EXPECTED_CONTROL_IDS = (
    "positive_only",
    "linear_c12",
    "squared_c4",
    "squared_c8",
    "squared_c16",
    "squared_c32",
    "squared_c64",
    "squared_c128",
)
EXPECTED_STEPS = 1_000_000
EXPECTED_TOTAL_BRANCHES = 192
REFERENCE_DISTANCE = 2.0
INTERNAL_CANONICAL_ALPHA = 0.11
DIAGNOSTICS_INTERVAL = 10_000
SAMPLED_VALUES_PER_UPDATE = 16

LINEAR_FORMULA = "w(d)=w(0)*exp(-c*(d/2))"
SQUARED_FORMULA = "w(d)=w(0)*exp(-c*(d/2)^2)"
POSITIVE_ONLY_FORMULA = "w(d)=0"


def _flag_value(argv: list[str], flag: str) -> str:
    return source._flag_value(argv, flag)  # noqa: SLF001


def _validate_control(raw: Mapping[str, Any]) -> dict[str, Any]:
    control_id = str(raw.get("id"))
    family = str(raw.get("family"))
    w0 = float(raw.get("weight_at_zero"))
    coefficient = float(raw.get("exp_coefficient"))
    reference_distance = float(raw.get("reference_distance"))
    formula = str(raw.get("formula"))
    expected: dict[str, tuple[str, float, float, str]] = {
        "positive_only": ("positive_only", 0.0, 0.0, POSITIVE_ONLY_FORMULA),
        "linear_c12": ("linear_exponential", 1.0, 12.0, LINEAR_FORMULA),
        "squared_c4": ("squared_exponential", 1.0, 4.0, SQUARED_FORMULA),
        "squared_c8": ("squared_exponential", 1.0, 8.0, SQUARED_FORMULA),
        "squared_c16": ("squared_exponential", 1.0, 16.0, SQUARED_FORMULA),
        "squared_c32": ("squared_exponential", 1.0, 32.0, SQUARED_FORMULA),
        "squared_c64": ("squared_exponential", 1.0, 64.0, SQUARED_FORMULA),
        "squared_c128": ("squared_exponential", 1.0, 128.0, SQUARED_FORMULA),
    }
    if control_id not in expected:
        raise ValueError(f"unsupported control id: {control_id}")
    expected_family, expected_w0, expected_c, expected_formula = expected[control_id]
    if family != expected_family:
        raise ValueError(f"{control_id} family changed")
    if not math.isclose(w0, expected_w0, abs_tol=1e-12):
        raise ValueError(f"{control_id} weight_at_zero changed")
    if not math.isclose(coefficient, expected_c, abs_tol=1e-12):
        raise ValueError(f"{control_id} coefficient changed")
    if not math.isclose(reference_distance, REFERENCE_DISTANCE, abs_tol=1e-12):
        raise ValueError(f"{control_id} reference distance changed")
    if formula != expected_formula:
        raise ValueError(f"{control_id} formula changed")
    return {
        "id": control_id,
        "family": family,
        "weight_at_zero": w0,
        "exp_coefficient": coefficient,
        "reference_distance": reference_distance,
        "formula": formula,
    }


def load_grid(path: str | Path) -> tuple[dict[str, Any], str]:
    source_path = Path(path)
    raw = json.loads(source_path.read_text())
    if raw.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"grid experiment_id must be {EXPERIMENT_ID}")
    if raw.get("run_kind") != "pilot" or raw.get("scientific_status") != SCIENTIFIC_STATUS:
        raise ValueError("actor-decision grid must remain the frozen pilot")
    if tuple(raw.get("datasets", ())) != EXPECTED_DATASETS:
        raise ValueError("dataset set changed")
    if tuple(int(value) for value in raw.get("development_seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("development seeds changed")
    if tuple(int(value) for value in raw.get("held_out_seeds", ())) != HELD_OUT_SEEDS:
        raise ValueError("held-out seed reservation changed")
    if int(raw.get("steps", -1)) != EXPECTED_STEPS:
        raise ValueError("steps must remain 1,000,000")
    if int(raw.get("evaluation_interval", -1)) != 50_000:
        raise ValueError("evaluation_interval must remain 50,000")
    if int(raw.get("evaluation_episodes", -1)) != 10:
        raise ValueError("evaluation_episodes must remain 10")

    controls = raw.get("controls")
    if not isinstance(controls, list):
        raise ValueError("controls must be a list")
    validated_controls = [_validate_control(item) for item in controls]
    if tuple(item["id"] for item in validated_controls) != EXPECTED_CONTROL_IDS:
        raise ValueError("control order or membership changed")

    actor_modes = raw.get("actor_modes")
    if not isinstance(actor_modes, list):
        raise ValueError("actor_modes must be a list")
    actor_ids = tuple(str(item.get("id")) for item in actor_modes)
    if actor_ids != EXPECTED_ACTOR_MODES:
        raise ValueError("actor mode set changed")
    ppo = actor_modes[1]
    if not math.isclose(float(ppo.get("clip_epsilon")), 0.2, abs_tol=1e-12):
        raise ValueError("PPO clip epsilon changed")
    if int(ppo.get("max_updates_per_old_policy", -1)) != 4:
        raise ValueError("PPO K max changed")
    if ppo.get("analytic_kl_early_refresh") is not True:
        raise ValueError("PPO KL early refresh must remain enabled")
    if not math.isclose(float(ppo.get("target_kl")), 0.01, abs_tol=1e-12):
        raise ValueError("PPO target_kl changed")
    if ppo.get("kl_penalty") is not False:
        raise ValueError("KL penalty must remain disabled")

    diagnostics = raw.get("diagnostics", {})
    if int(diagnostics.get("interval", -1)) != DIAGNOSTICS_INTERVAL:
        raise ValueError("diagnostics interval changed")
    if int(diagnostics.get("sampled_values_per_update", -1)) != SAMPLED_VALUES_PER_UPDATE:
        raise ValueError("sampled values per update changed")
    if diagnostics.get("kl_event_jsonl") is not False:
        raise ValueError("per-trigger KL JSONL must remain disabled")
    if int(diagnostics.get("late_window_start", -1)) != 800_000:
        raise ValueError("late-window start changed")

    if int(raw.get("expected_controls", -1)) != len(EXPECTED_CONTROL_IDS):
        raise ValueError("control count changed")
    if int(raw.get("expected_actor_modes", -1)) != len(EXPECTED_ACTOR_MODES):
        raise ValueError("actor count changed")
    if int(raw.get("expected_runnable_branches", -1)) != EXPECTED_TOTAL_BRANCHES:
        raise ValueError("branch count changed")
    if raw.get("formal_evidence_allowed") is not False:
        raise ValueError("development pilot cannot allow formal evidence")
    if raw.get("registration_blocks_launch") is not False:
        raise ValueError("code-first launch cannot be registration-blocked")
    if raw.get("gae_included") is not False:
        raise ValueError("GAE must remain outside this experiment")
    return raw, sha256_file(source_path)


def load_run_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    raw, digest = source._BASE_LOAD_RUN_SPEC(path)  # noqa: SLF001
    run_spec = copy.deepcopy(raw)
    if run_spec.get("experiment_id") != "EXT-H-E7-BENCH-01":
        raise ValueError("source run_spec experiment_id changed")
    source_ids = tuple(str(item["id"]) for item in run_spec["datasets"])
    if source_ids != source.EXPECTED_SOURCE_DATASETS:
        raise ValueError("source run_spec dataset order changed")
    by_id = {str(item["id"]): item for item in run_spec["datasets"]}
    run_spec["datasets"] = [copy.deepcopy(by_id[name]) for name in EXPECTED_DATASETS]
    if tuple(int(value) for value in run_spec["seeds"]) != SOURCE_SEEDS:
        raise ValueError("source run_spec seed contract changed")
    run_spec["seeds"] = list(EXPECTED_SEEDS)
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
    argv[argv.index("--steps") + 1] = "{steps}"
    run_spec["trainer_argv_template"] = argv
    run_spec["passthrough_variants"] = []
    return run_spec, digest


def controls(grid: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = [_validate_control(item) for item in grid["controls"]]
    if tuple(item["id"] for item in values) != EXPECTED_CONTROL_IDS:
        raise ValueError("control matrix changed")
    return values


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
        raise ValueError("canonical alpha changed from 0.11")
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
        "clip_epsilon": "0.2",
        "updates_per_old_policy": "4",
        "target_kl": "0.01",
    }
    branches: list[base.Branch] = []
    for actor_mode in EXPECTED_ACTOR_MODES:
        for control in controls(grid):
            for dataset in datasets:
                for seed in seeds:
                    branch_id = (
                        f"{dataset.id}__seed{seed}__{control['id']}__"
                        f"{actor_mode}__steps1m"
                    )
                    branches.append(
                        base.Branch(
                            branch_id=branch_id,
                            branch_kind="injected",
                            dataset=dataset,
                            seed=seed,
                            template_values={
                                **common,
                                "actor_update_mode": actor_mode,
                                "control_id": str(control["id"]),
                                "weight_family": str(control["family"]),
                                "weight_at_zero": f"{float(control['weight_at_zero']):.17g}",
                                "exp_coefficient": f"{float(control['exp_coefficient']):.17g}",
                                "reference_distance": f"{REFERENCE_DISTANCE:.17g}",
                                "formula": str(control["formula"]),
                            },
                            negative_control=None,
                        )
                    )
    ids = [branch.branch_id for branch in branches]
    if len(ids) != len(set(ids)):
        raise ValueError("branch IDs are not unique")
    if len(branches) != EXPECTED_TOTAL_BRANCHES:
        raise ValueError(f"expected {EXPECTED_TOTAL_BRANCHES} branches, built {len(branches)}")
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
    actor_mode = str(values["actor_update_mode"])
    if actor_mode not in EXPECTED_ACTOR_MODES:
        raise ValueError("actor mode changed")
    control = _validate_control(
        {
            "id": values["control_id"],
            "family": values["weight_family"],
            "weight_at_zero": values["weight_at_zero"],
            "exp_coefficient": values["exp_coefficient"],
            "reference_distance": values["reference_distance"],
            "formula": values["formula"],
        }
    )
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
    branch_config = {
        "experiment_id": EXPERIMENT_ID,
        "branch_id": branch.branch_id,
        "branch_kind": branch.branch_kind,
        "dataset_id": branch.dataset.id,
        "dataset_sha256": branch.dataset.sha256,
        "seed": branch.seed,
        "template_values": values,
        "weight_control": control,
        "actor_update": {
            "id": actor_mode,
            "clip_epsilon": None if actor_mode == "a2c" else 0.2,
            "max_updates_per_old_policy": None if actor_mode == "a2c" else 4,
            "analytic_kl_early_refresh": actor_mode == "ppo_clip_kl_k4",
            "target_kl": None if actor_mode == "a2c" else 0.01,
            "kl_penalty": False,
        },
    }
    branch_config_path = branch_dir / "branch_config.json"
    base.atomic_write_json(branch_config_path, branch_config)
    command = [
        sys.executable,
        "-m",
        "drpo.e7_sqexp_actor_decision_bootstrap",
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
