"""Code-first runner for the 48-branch E7 squared-EXP high-c boundary pilot."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from drpo import e7_canonical_sweep as canonical
from drpo import e7_sqexp_actor_decision as predecessor
from drpo.e7_canonical_injection import sha256_file
from drpo.e7_sqexp_highc_boundary_aggregate import aggregate as aggregate_results


EXPERIMENT_ID = "EXT-H-E7-SQEXP-HIGHC-BOUNDARY-01"
SCIENTIFIC_STATUS = "four_seed_high_c_boundary_extension_screening_only"
RUNNER_VERSION = "1.0.0-e7-sqexp-highc-boundary-48"
PREDECESSOR_EXPERIMENT_ID = predecessor.EXPERIMENT_ID
PREDECESSOR_IMPLEMENTATION_COMMIT = "d1afb5ff094f69986e0ecc3bf7f9385485add62b"

EXPECTED_DATASETS = predecessor.EXPECTED_DATASETS
EXPECTED_SEEDS = predecessor.EXPECTED_SEEDS
HELD_OUT_SEEDS = predecessor.HELD_OUT_SEEDS
EXPECTED_ACTOR_MODES = predecessor.EXPECTED_ACTOR_MODES
EXPECTED_CONTROL_IDS = ("squared_c256", "squared_c512")
EXPECTED_STEPS = predecessor.EXPECTED_STEPS
EXPECTED_TOTAL_BRANCHES = 48
REFERENCE_DISTANCE = predecessor.REFERENCE_DISTANCE
INTERNAL_CANONICAL_ALPHA = predecessor.INTERNAL_CANONICAL_ALPHA
DIAGNOSTICS_INTERVAL = predecessor.DIAGNOSTICS_INTERVAL
SAMPLED_VALUES_PER_UPDATE = predecessor.SAMPLED_VALUES_PER_UPDATE
SQUARED_FORMULA = predecessor.SQUARED_FORMULA


def _flag_value(argv: list[str], flag: str) -> str:
    return predecessor._flag_value(argv, flag)  # noqa: SLF001


def _validate_control(raw: Mapping[str, Any]) -> dict[str, Any]:
    control_id = str(raw.get("id"))
    family = str(raw.get("family"))
    weight_at_zero = float(raw.get("weight_at_zero"))
    coefficient = float(raw.get("exp_coefficient"))
    reference_distance = float(raw.get("reference_distance"))
    formula = str(raw.get("formula"))
    expected_coefficients = {
        "squared_c256": 256.0,
        "squared_c512": 512.0,
    }
    if control_id not in expected_coefficients:
        raise ValueError(f"unsupported high-c control id: {control_id}")
    if family != "squared_exponential":
        raise ValueError(f"{control_id} family must remain squared_exponential")
    if not math.isclose(weight_at_zero, 1.0, abs_tol=1e-12):
        raise ValueError(f"{control_id} weight_at_zero must remain 1")
    if not math.isclose(
        coefficient,
        expected_coefficients[control_id],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{control_id} coefficient changed")
    if not math.isclose(
        reference_distance,
        REFERENCE_DISTANCE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{control_id} reference distance changed")
    if formula != SQUARED_FORMULA:
        raise ValueError(f"{control_id} formula changed")
    return {
        "id": control_id,
        "family": family,
        "weight_at_zero": weight_at_zero,
        "exp_coefficient": coefficient,
        "reference_distance": reference_distance,
        "formula": formula,
    }


def _validate_actor_modes(raw: Any) -> None:
    if not isinstance(raw, list):
        raise ValueError("actor_modes must be a list")
    actor_ids = tuple(str(item.get("id")) for item in raw)
    if actor_ids != EXPECTED_ACTOR_MODES:
        raise ValueError("actor mode set changed")
    ppo = raw[1]
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


def load_grid(path: str | Path) -> tuple[dict[str, Any], str]:
    source_path = Path(path)
    raw = json.loads(source_path.read_text())
    if raw.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"grid experiment_id must be {EXPERIMENT_ID}")
    if raw.get("run_kind") != "pilot" or raw.get("scientific_status") != SCIENTIFIC_STATUS:
        raise ValueError("high-c boundary grid must remain the frozen pilot")
    predecessor_record = raw.get("predecessor", {})
    if predecessor_record.get("experiment_id") != PREDECESSOR_EXPERIMENT_ID:
        raise ValueError("predecessor experiment id changed")
    if predecessor_record.get("implementation_commit") != PREDECESSOR_IMPLEMENTATION_COMMIT:
        raise ValueError("predecessor implementation commit changed")
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

    raw_controls = raw.get("controls")
    if not isinstance(raw_controls, list):
        raise ValueError("controls must be a list")
    validated_controls = [_validate_control(item) for item in raw_controls]
    if tuple(item["id"] for item in validated_controls) != EXPECTED_CONTROL_IDS:
        raise ValueError("control order or membership changed")
    _validate_actor_modes(raw.get("actor_modes"))

    diagnostics = raw.get("diagnostics", {})
    if int(diagnostics.get("interval", -1)) != DIAGNOSTICS_INTERVAL:
        raise ValueError("diagnostics interval changed")
    if int(diagnostics.get("sampled_values_per_update", -1)) != SAMPLED_VALUES_PER_UPDATE:
        raise ValueError("sampled values per update changed")
    if diagnostics.get("kl_event_jsonl") is not False:
        raise ValueError("per-trigger KL JSONL must remain disabled")
    if int(diagnostics.get("late_window_start", -1)) != 800_000:
        raise ValueError("late-window start changed")

    analysis_contract = raw.get("analysis_contract", {})
    if analysis_contract.get("primary_within_run_comparison") != (
        "squared_c512_minus_squared_c256"
    ):
        raise ValueError("primary high-c comparison changed")
    if analysis_contract.get("no_automatic_common_c_selection") is not True:
        raise ValueError("common-c auto selection must remain disabled")
    if analysis_contract.get("no_automatic_ppo_selection") is not True:
        raise ValueError("PPO auto selection must remain disabled")
    if analysis_contract.get("c1024_extension_not_authorized") is not True:
        raise ValueError("c1024 must remain outside the current authorization")

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
    return predecessor.load_run_spec(path)


def controls(grid: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = [_validate_control(item) for item in grid["controls"]]
    if tuple(item["id"] for item in values) != EXPECTED_CONTROL_IDS:
        raise ValueError("control matrix changed")
    return values


def build_branches(
    contract: canonical.CanonicalContract,
    run_spec: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> list[canonical.Branch]:
    if not math.isclose(
        contract.expected_canonical_alpha,
        INTERNAL_CANONICAL_ALPHA,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("canonical alpha changed from 0.11")
    datasets = [canonical.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
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
    branches: list[canonical.Branch] = []
    for actor_mode in EXPECTED_ACTOR_MODES:
        for control in controls(grid):
            for dataset in datasets:
                for seed in seeds:
                    branch_id = (
                        f"{dataset.id}__seed{seed}__{control['id']}__"
                        f"{actor_mode}__steps1m"
                    )
                    branches.append(
                        canonical.Branch(
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
    contract: canonical.CanonicalContract,
    branch: canonical.Branch,
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
        canonical._format_value(str(item), context)  # noqa: SLF001
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
        "predecessor_implementation_commit": PREDECESSOR_IMPLEMENTATION_COMMIT,
    }
    branch_config_path = branch_dir / "branch_config.json"
    canonical.atomic_write_json(branch_config_path, branch_config)
    command = [
        sys.executable,
        "-m",
        "drpo.e7_sqexp_highc_boundary_bootstrap",
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
        canonical.EXPERIMENT_ID,
        canonical.SCIENTIFIC_STATUS,
        canonical.RUNNER_VERSION,
        canonical.load_grid,
        canonical.load_run_spec,
        canonical.build_branches,
        canonical.branch_command,
    )
    canonical.EXPERIMENT_ID = EXPERIMENT_ID
    canonical.SCIENTIFIC_STATUS = SCIENTIFIC_STATUS
    canonical.RUNNER_VERSION = RUNNER_VERSION
    canonical.load_grid = load_grid
    canonical.load_run_spec = load_run_spec
    canonical.build_branches = build_branches
    canonical.branch_command = branch_command
    try:
        delegated = list(sys.argv[1:] if argv is None else argv)
        result = canonical.main(delegated)
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
            canonical.EXPERIMENT_ID,
            canonical.SCIENTIFIC_STATUS,
            canonical.RUNNER_VERSION,
            canonical.load_grid,
            canonical.load_run_spec,
            canonical.build_branches,
            canonical.branch_command,
        ) = previous


if __name__ == "__main__":
    raise SystemExit(main())
