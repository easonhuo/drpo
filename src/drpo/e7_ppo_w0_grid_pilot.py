"""Direct-w(0) exponential grid pilot for canonical E7 PPO training.

The experiment-facing parameterization is exactly

    u = d / 2
    w(d) = w(0) * exp(-c * u)

with ``w(0)`` in ``[0, 1]``. Legacy ``negative_scale`` and
``canonical_alpha`` are not accepted in the grid or written to branch identity.
A private compatibility conversion is performed only inside the bootstrap because
the frozen canonical agent still exposes ``alpha=0.11``.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from drpo import e7_canonical_sweep as base
from drpo.e7_canonical_injection import sha256_file
from drpo.e7_ppo_w0_aggregate import aggregate as aggregate_results

EXPERIMENT_ID = "EXT-H-E7-PPO-W0-EXP-GRID-01"
SCIENTIFIC_STATUS = "ppo_w0_exp_grid_screening_pilot_only"
RUNNER_VERSION = "1.0.0-e7-ppo-w0-exp-grid"

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
EXPECTED_SEEDS = (200, 201)
HELD_OUT_SEEDS = (204, 205, 206, 207)
EXPECTED_W0 = (0.0, 0.025, 0.05, 0.11, 0.25, 0.5, 1.0)
EXPECTED_COEFFICIENTS = (0.0, 0.25, 0.5, 1.0, 1.5)
EXPECTED_PARAMETER_POINTS = 31
EXPECTED_TOTAL_BRANCHES = 186
EXPECTED_STEPS = 500_000
EXPECTED_CLIP_EPSILON = 0.2
EXPECTED_UPDATES_PER_OLD_POLICY = 4
EXPECTED_DIAGNOSTICS_INTERVAL = 1000
INTERNAL_CANONICAL_ALPHA = 0.11
REFERENCE_DISTANCE = 2.0

_BASE_LOAD_RUN_SPEC = base.load_run_spec


def _label(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def _flag_value(argv: list[str], flag: str) -> str:
    positions = [index for index, token in enumerate(argv) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"trainer_argv_template must contain exactly one {flag}")
    return argv[positions[0] + 1]


def load_w0_grid(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = json.loads(source.read_text())
    if raw.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"grid experiment_id must be {EXPERIMENT_ID}")
    if raw.get("run_kind") != "pilot":
        raise ValueError("w(0) grid must remain a pilot")
    if raw.get("scientific_status") != SCIENTIFIC_STATUS:
        raise ValueError("unexpected scientific_status")

    forbidden_legacy = {
        "negative_scale",
        "negative_scale_grid",
        "canonical_alpha",
        "effective_alpha",
    }
    present = sorted(forbidden_legacy & set(raw))
    if present:
        raise ValueError(
            "direct-w(0) grid forbids legacy scale/alpha fields: " + ", ".join(present)
        )

    if tuple(raw.get("datasets", ())) != EXPECTED_DATASETS:
        raise ValueError("w(0) pilot datasets changed")
    if tuple(int(v) for v in raw.get("development_seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("w(0) pilot development seeds changed")
    if tuple(int(v) for v in raw.get("held_out_seeds", ())) != HELD_OUT_SEEDS:
        raise ValueError("held-out seed reservation changed")

    w0_values = tuple(float(v) for v in raw.get("weight_at_zero_grid", ()))
    coefficients = tuple(float(v) for v in raw.get("exp_coefficients", ()))
    if w0_values != EXPECTED_W0:
        raise ValueError("weight_at_zero_grid changed")
    if coefficients != EXPECTED_COEFFICIENTS:
        raise ValueError("EXP coefficient grid changed")
    if len(set(w0_values)) != len(w0_values):
        raise ValueError("weight_at_zero_grid contains duplicates")
    if any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in w0_values):
        raise ValueError("every w(0) must be finite and in [0, 1]")
    if any(not math.isfinite(v) or v < 0.0 for v in coefficients):
        raise ValueError("every EXP coefficient must be finite and non-negative")

    if raw.get("actor_update_mode") != "ppo_clip":
        raise ValueError("screening pilot is PPO-only")
    if int(raw.get("steps", -1)) != EXPECTED_STEPS:
        raise ValueError("steps must remain 500,000")
    if int(raw.get("evaluation_interval", -1)) != 50_000:
        raise ValueError("evaluation_interval must remain 50,000")
    if int(raw.get("evaluation_episodes", -1)) != 10:
        raise ValueError("evaluation_episodes must remain 10")
    if int(raw.get("expected_unique_parameter_points", -1)) != EXPECTED_PARAMETER_POINTS:
        raise ValueError("expected_unique_parameter_points must remain 31")
    if int(raw.get("expected_total_branches", -1)) != EXPECTED_TOTAL_BRANCHES:
        raise ValueError("expected_total_branches must remain 186")
    if raw.get("formal_evidence_allowed") is not False:
        raise ValueError("500k screening pilot cannot allow formal evidence")

    ppo = raw.get("ppo", {})
    if not math.isclose(
        float(ppo.get("clip_epsilon")), EXPECTED_CLIP_EPSILON, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("clip_epsilon must remain 0.2")
    if int(ppo.get("updates_per_old_policy", -1)) != EXPECTED_UPDATES_PER_OLD_POLICY:
        raise ValueError("updates_per_old_policy must remain 4")
    if int(ppo.get("diagnostics_interval", -1)) != EXPECTED_DIAGNOSTICS_INTERVAL:
        raise ValueError("diagnostics_interval must remain 1000")
    forbidden_tricks = {
        "kl_penalty",
        "target_kl",
        "entropy_bonus",
        "actor_gradient_clip",
        "value_clip",
    }
    if any(ppo.get(name) not in (None, False, 0, 0.0) for name in forbidden_tricks):
        raise ValueError("pilot forbids auxiliary PPO tricks")
    return raw, sha256_file(source)


def load_w0_run_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    raw, digest = _BASE_LOAD_RUN_SPEC(path)
    run_spec = copy.deepcopy(raw)
    if run_spec.get("experiment_id") != "EXT-H-E7-BENCH-01":
        raise ValueError("source run_spec experiment_id changed")
    source_dataset_ids = tuple(str(item["id"]) for item in run_spec["datasets"])
    if source_dataset_ids != EXPECTED_SOURCE_DATASETS:
        raise ValueError(f"source run_spec datasets changed: {source_dataset_ids}")
    by_id = {str(item["id"]): item for item in run_spec["datasets"]}
    run_spec["datasets"] = [copy.deepcopy(by_id[name]) for name in EXPECTED_DATASETS]

    source_seeds = tuple(int(v) for v in run_spec["seeds"])
    if source_seeds != EXPECTED_SEEDS:
        raise ValueError(
            f"source run_spec seeds changed: {source_seeds}; expected {EXPECTED_SEEDS}"
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
    argv[argv.index("--steps") + 1] = "{steps}"
    run_spec["trainer_argv_template"] = argv
    run_spec["passthrough_variants"] = []
    return run_spec, digest


def parameter_points(grid: Mapping[str, Any]) -> list[tuple[float, float | None]]:
    w0_values = [float(v) for v in grid["weight_at_zero_grid"]]
    coefficients = [float(v) for v in grid["exp_coefficients"]]
    points: list[tuple[float, float | None]] = [(0.0, None)]
    for w0 in w0_values:
        if w0 == 0.0:
            continue
        for coefficient in coefficients:
            points.append((w0, coefficient))
    if len(points) != EXPECTED_PARAMETER_POINTS or len(points) != len(set(points)):
        raise ValueError("direct-w(0) parameter grid must contain 31 unique points")
    return points


def build_w0_branches(
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
        raise ValueError("canonical source alpha changed from the frozen 0.11 contract")
    datasets = [base.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
    if tuple(item.id for item in datasets) != EXPECTED_DATASETS:
        raise ValueError("expanded dataset subset changed")
    seeds = [int(v) for v in run_spec["seeds"]]
    if tuple(seeds) != EXPECTED_SEEDS:
        raise ValueError("expanded development seeds changed")

    common_template = {
        "steps": str(EXPECTED_STEPS),
        "actor_update_mode": "ppo_clip",
        "clip_epsilon": str(EXPECTED_CLIP_EPSILON),
        "updates_per_old_policy": str(EXPECTED_UPDATES_PER_OLD_POLICY),
        "diagnostics_interval": str(EXPECTED_DIAGNOSTICS_INTERVAL),
    }
    branches: list[base.Branch] = []
    for w0, coefficient in parameter_points(grid):
        if w0 == 0.0:
            control_label = "positive_only__w0_0"
            method = "positive_only"
            coefficient_value = 0.0
            coefficient_label = "none"
        else:
            assert coefficient is not None
            control_label = f"exp__w0_{_label(w0)}__c_{_label(coefficient)}"
            method = "exponential"
            coefficient_value = coefficient
            coefficient_label = _label(coefficient)
        for dataset in datasets:
            for seed in seeds:
                branches.append(
                    base.Branch(
                        branch_id=(
                            f"{dataset.id}__seed{seed}__{control_label}__"
                            "ppo_clip_eps0p2__k4__steps500k"
                        ),
                        branch_kind="injected",
                        dataset=dataset,
                        seed=seed,
                        template_values={
                            **common_template,
                            "weight_method": method,
                            "weight_at_zero": f"{w0:.17g}",
                            "exp_coefficient": f"{coefficient_value:.17g}",
                            "exp_coefficient_label": coefficient_label,
                            "reference_distance": f"{REFERENCE_DISTANCE:.17g}",
                        },
                        negative_control=None,
                    )
                )
    ids = [branch.branch_id for branch in branches]
    if len(ids) != len(set(ids)):
        raise ValueError("w(0) pilot branch IDs are not unique")
    if len(branches) != EXPECTED_TOTAL_BRANCHES:
        raise ValueError(
            f"expected {EXPECTED_TOTAL_BRANCHES} branches but built {len(branches)}"
        )
    return branches


def w0_branch_command(
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
    method = values["weight_method"]
    if not 0.0 <= w0 <= 1.0:
        raise ValueError("branch w(0) must be in [0, 1]")
    if method == "positive_only" and w0 != 0.0:
        raise ValueError("positive-only branch requires w(0)=0")
    if method == "exponential" and w0 <= 0.0:
        raise ValueError("exponential branch requires w(0)>0")

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
        "weight_at_zero": w0,
        "exp_coefficient": coefficient,
        "reference_distance": REFERENCE_DISTANCE,
        "formula": "w(d)=w(0)*exp(-c*(d/2))",
    }
    branch_config = {
        "experiment_id": EXPERIMENT_ID,
        "branch_id": branch.branch_id,
        "branch_kind": branch.branch_kind,
        "dataset_id": branch.dataset.id,
        "dataset_sha256": branch.dataset.sha256,
        "seed": branch.seed,
        "template_values": values,
        "weight_control": weight_control,
    }
    branch_config_path = branch_dir / "branch_config.json"
    base.atomic_write_json(branch_config_path, branch_config)
    command = [
        sys.executable,
        "-m",
        "drpo.e7_ppo_w0_bootstrap",
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
    base.load_grid = load_w0_grid
    base.load_run_spec = load_w0_run_spec
    base.build_branches = build_w0_branches
    base.branch_command = w0_branch_command
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
