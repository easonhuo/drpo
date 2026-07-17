"""Canonical joint actor--critic TD/GAE pilot for E7 squared-remoteness control."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from drpo import e7_canonical_sweep as base
from drpo import e7_squared_exp_night as night
from drpo import e7_squared_exp_night_aggregate as night_aggregate
from drpo import e7_squared_exp_night_bootstrap as canonical_bootstrap
from drpo.e7_canonical_gae_injection import (
    SnapshotEstimatorConfig,
    build_joint_snapshot_agent_class,
    load_ordered_replay,
    transition_id_channel,
)
from drpo.e7_canonical_injection import sha256_file
from drpo.e7_squared_exp_kernel import FORMULA


EXPERIMENT_ID = "EXT-H-E7-SQEXP-GAE-01"
SCIENTIFIC_STATUS = "canonical_joint_critic_trajectory_snapshot_gae_pilot_only"
RUNNER_VERSION = "4.0.0-canonical-joint-critic"
EXPECTED_DATASETS = night.EXPECTED_DATASETS
EXPECTED_SEEDS = (200, 201, 202, 203)
HELD_OUT_SEEDS = night.HELD_OUT_SEEDS
ESTIMATORS = ("td", "gae")
COEFFICIENTS = (64.0, 128.0, 256.0)
EXPECTED_BRANCHES = 96
EXPECTED_STEPS = 1_000_000
GAE_LAMBDA = 0.95
CANONICAL_BATCH_SIZE = 256
LATE_WINDOW_START = 800_000


def _trainer_flag(argv: Sequence[str], flag: str) -> str | None:
    positions = [index for index, token in enumerate(argv) if token == flag]
    if len(positions) > 1:
        raise ValueError(f"trainer args contain duplicate {flag}")
    if not positions:
        return None
    index = positions[0]
    if index + 1 >= len(argv):
        raise ValueError(f"trainer arg {flag} has no value")
    return str(argv[index + 1])


def _validate_trainer_args(argv: Sequence[str], *, runtime_probe: bool) -> None:
    values = list(argv)
    if values and values[0] == "--":
        values = values[1:]
    if _trainer_flag(values, "--variant") != "iqlv_exp_rank":
        raise ValueError("GAE pilot requires canonical iqlv_exp_rank")
    if _trainer_flag(values, "--batch") != str(CANONICAL_BATCH_SIZE):
        raise ValueError("GAE pilot requires canonical batch size 256")
    ret_weight = _trainer_flag(values, "--ret_weight_mode")
    if ret_weight not in {None, "none"}:
        raise ValueError("transition-ID channel requires uniform ret_weight_mode=none")
    steps = _trainer_flag(values, "--steps")
    if steps is None or int(steps) <= 0:
        raise ValueError("trainer args require positive --steps")
    if not runtime_probe and int(steps) != EXPECTED_STEPS:
        raise ValueError("full GAE branches require exactly 1,000,000 updates")


def _load_grid(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = json.loads(source.read_text())
    required = {
        "experiment_id": EXPERIMENT_ID,
        "run_kind": "pilot",
        "status": "not_run",
        "scientific_status": SCIENTIFIC_STATUS,
        "predecessor_experiment_id": night.EXPERIMENT_ID,
        "datasets": list(EXPECTED_DATASETS),
        "development_seeds": list(EXPECTED_SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "steps": EXPECTED_STEPS,
        "actor_update_modes": ["a2c"],
        "advantage_modes": ["one_step_td", "gae_lambda_0p95"],
        "expected_total_branches": EXPECTED_BRANCHES,
        "screening_only": True,
        "formal_evidence_allowed": False,
    }
    changed = [key for key, value in required.items() if raw.get(key) != value]
    weight = raw.get("weight_control", {})
    snapshot = raw.get("trajectory_snapshot", {})
    if (
        changed
        or weight.get("formula") != FORMULA
        or tuple(float(value) for value in weight.get("exp_coefficients", ()))
        != COEFFICIENTS
        or float(snapshot.get("gae_lambda", -1.0)) != GAE_LAMBDA
        or int(snapshot.get("canonical_batch_size", -1)) != CANONICAL_BATCH_SIZE
        or snapshot.get("critic_updated_every_step") is not True
        or snapshot.get("prepared_advantage_artifact") is not False
    ):
        raise ValueError(f"joint-critic GAE grid changed: {changed or ['nested_contract']}")
    return raw, sha256_file(source)


def _load_run_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    run_spec, digest = night.load_run_spec(path)
    run_spec["seeds"] = list(EXPECTED_SEEDS)
    return run_spec, digest


def _label(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


def _build_branches(
    contract: base.CanonicalContract,
    run_spec: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> list[base.Branch]:
    del grid
    if not math.isclose(contract.expected_canonical_alpha, 0.11, abs_tol=1e-12):
        raise ValueError("canonical source alpha changed")
    datasets = [base.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
    if tuple(item.id for item in datasets) != EXPECTED_DATASETS:
        raise ValueError("GAE dataset order changed")
    controls = [("positive_only", 0.0, 0.0)] + [
        ("squared_exponential", 1.0, coefficient) for coefficient in COEFFICIENTS
    ]
    branches: list[base.Branch] = []
    for estimator in ESTIMATORS:
        for method, weight_at_zero, coefficient in controls:
            control = (
                "positive_only"
                if method == "positive_only"
                else f"sqexp_c{_label(coefficient)}"
            )
            for dataset in datasets:
                for seed in EXPECTED_SEEDS:
                    branches.append(
                        base.Branch(
                            branch_id=(
                                f"{dataset.id}__seed{seed}__{estimator}__{control}__"
                                "a2c__steps1m"
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
                                "exp_coefficient": f"{coefficient:.17g}",
                                "reference_distance": f"{night.REFERENCE_DISTANCE:.17g}",
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


def _branch_command(
    *,
    contract_path: Path,
    contract: base.CanonicalContract,
    branch: base.Branch,
    branch_dir: Path,
    trainer_argv_template: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    values = branch.template_values
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
    trainer_args = [
        base._format_value(str(item), context)  # noqa: SLF001
        for item in trainer_argv_template
    ]
    branch_config = {
        "experiment_id": EXPERIMENT_ID,
        "branch_id": branch.branch_id,
        "branch_kind": branch.branch_kind,
        "canonical_root": context["canonical_root"],
        "dataset_id": branch.dataset.id,
        "dataset_path": context["dataset_path"],
        "dataset_sha256": branch.dataset.sha256,
        "seed": branch.seed,
        "template_values": values,
        "weight_control": {
            "method": values["weight_method"],
            "weight_at_zero": float(values["weight_at_zero"]),
            "exp_coefficient": float(values["exp_coefficient"]),
            "reference_distance": night.REFERENCE_DISTANCE,
            "formula": FORMULA,
        },
    }
    config_path = branch_dir / "branch_config.json"
    base.atomic_write_json(config_path, branch_config)
    return (
        [
            sys.executable,
            "-m",
            "drpo.e7_sqexp_gae",
            "bootstrap",
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


def _bootstrap(argv: list[str]) -> int:
    args = canonical_bootstrap.build_parser().parse_args(argv)
    branch_path = Path(args.branch_config).expanduser().resolve()
    branch = json.loads(branch_path.read_text())
    if branch.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("GAE branch experiment_id mismatch")
    values = branch.get("template_values", {})
    estimator_name = str(values.get("advantage_estimator"))
    if estimator_name not in ESTIMATORS or values.get("actor_update_mode") != "a2c":
        raise ValueError("joint GAE pilot supports only TD/GAE with canonical A2C")
    runtime_probe = os.environ.get("DRPO_RUNTIME_RESOURCE_PROBE") == "1"
    _validate_trainer_args(args.trainer_args, runtime_probe=runtime_probe)
    dataset = Path(branch["dataset_path"]).expanduser().resolve()
    if sha256_file(dataset) != branch["dataset_sha256"]:
        raise RuntimeError("ordered dataset hash mismatch")
    replay = load_ordered_replay(
        canonical_root=branch["canonical_root"],
        dataset_path=dataset,
        dataset_id=str(branch["dataset_id"]),
    )
    estimator = SnapshotEstimatorConfig(
        estimator=estimator_name,
        gae_lambda=GAE_LAMBDA,
        canonical_batch_size=CANONICAL_BATCH_SIZE,
    )
    instances: list[Any] = []
    old_experiment_id = canonical_bootstrap.EXPERIMENT_ID
    old_patch = canonical_bootstrap.patch_canonical_module
    old_returns: Any = None

    def patch_joint(module: Any, contract: Any, control: Any) -> type:
        nonlocal old_returns
        import d4rl_common.train_loop as train_loop

        old_returns = train_loop.compute_mc_returns

        def transition_ids(rewards: Any, *_: Any, **__: Any) -> Any:
            if len(rewards) != replay.size:
                raise RuntimeError("trainer replay length changed before transition-ID injection")
            return transition_id_channel(replay.size)

        train_loop.compute_mc_returns = transition_ids
        parent = getattr(module, contract.target_class)
        injected = build_joint_snapshot_agent_class(
            parent,
            replay=replay,
            negative_control=control,
            estimator=estimator,
            return_mode=contract.return_mode,
            instance_sink=instances,
        )
        setattr(module, contract.target_class, injected)
        return injected

    canonical_bootstrap.EXPERIMENT_ID = EXPERIMENT_ID
    canonical_bootstrap.patch_canonical_module = patch_joint
    manifest_path = Path(args.branch_manifest).expanduser().resolve()
    try:
        result = canonical_bootstrap.main(argv)
        if len(instances) != 1:
            raise RuntimeError(f"expected one canonical agent, found {len(instances)}")
        snapshot = instances[0]._drpo_snapshot_summary()  # noqa: SLF001
        if not runtime_probe and (
            int(snapshot["snapshot_count"]) < 2
            or snapshot["critic_evolution_observed"] is not True
        ):
            raise RuntimeError("full joint-critic branch did not prove critic evolution")
        payload = json.loads(manifest_path.read_text())
        payload.update(
            {
                "advantage_estimator": estimator_name,
                "gae_used": estimator_name == "gae",
                "critic_updated_during_actor_training": True,
                "prepared_advantage_artifact_used": False,
                "transition_id_channel": "ep_ret_exact_float32_index",
                "trajectory_snapshot": snapshot,
            }
        )
        canonical_bootstrap._atomic_json(manifest_path, payload)  # noqa: SLF001
        return result
    except BaseException as exc:
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text())
            payload.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            canonical_bootstrap._atomic_json(manifest_path, payload)  # noqa: SLF001
        raise
    finally:
        canonical_bootstrap.EXPERIMENT_ID = old_experiment_id
        canonical_bootstrap.patch_canonical_module = old_patch
        if old_returns is not None:
            import d4rl_common.train_loop as train_loop

            train_loop.compute_mc_returns = old_returns


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(work_dir: Path) -> dict[str, Any]:
    branch_dirs = sorted(path for path in (work_dir / "branches").iterdir() if path.is_dir())
    if len(branch_dirs) != EXPECTED_BRANCHES:
        raise RuntimeError(f"expected {EXPECTED_BRANCHES} branches, found {len(branch_dirs)}")
    rows: list[dict[str, Any]] = []
    for branch_dir in branch_dirs:
        if not (branch_dir / "COMPLETED.json").is_file():
            raise RuntimeError(f"incomplete branch: {branch_dir.name}")
        branch = json.loads((branch_dir / "branch_config.json").read_text())
        manifest = json.loads((branch_dir / "branch_manifest.json").read_text())
        summary_path = night_aggregate._only(  # noqa: SLF001
            (branch_dir / "trainer_output").glob("*_summary.json"),
            "trainer summary",
        )
        summary = json.loads(summary_path.read_text())
        steps, scores = night_aggregate._read_history(summary)  # noqa: SLF001
        if steps[-1] != EXPECTED_STEPS or not all(math.isfinite(score) for score in scores):
            raise RuntimeError(f"non-finite or incomplete branch: {branch_dir.name}")
        late = [
            score
            for step, score in zip(steps, scores, strict=True)
            if step >= LATE_WINDOW_START
        ]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        control = branch["weight_control"]
        coefficient = (
            None if control["method"] == "positive_only" else float(control["exp_coefficient"])
        )
        snapshot = manifest.get("trajectory_snapshot", {})
        if snapshot.get("critic_evolution_observed") is not True:
            raise RuntimeError(f"critic evolution missing: {branch_dir.name}")
        geometry = night_aggregate._aggregate_geometry(  # noqa: SLF001
            branch_dir / "geometry_diagnostics.jsonl"
        )
        rows.append(
            {
                "branch_id": branch["branch_id"],
                "dataset": branch["dataset_id"],
                "seed": int(branch["seed"]),
                "advantage_estimator": branch["template_values"]["advantage_estimator"],
                "exp_coefficient": coefficient,
                "best_score": scores[best_index],
                "best_step": steps[best_index],
                "final_score": scores[-1],
                "best_to_final_drop": scores[best_index] - scores[-1],
                "late_window_mean_800k_1m": statistics.fmean(late),
                "late_window_std_800k_1m": statistics.stdev(late),
                "late_slope_per_100k": night_aggregate._slope_per_100k(  # noqa: SLF001
                    steps,
                    scores,
                    LATE_WINDOW_START,
                ),
                "snapshot_count": int(snapshot["snapshot_count"]),
                "snapshot_refresh_interval": int(snapshot["snapshot_refresh_interval"]),
                "critic_evolution_observed": True,
                "geometry_effective_negative_mass_fraction": geometry[
                    "effective_negative_mass_fraction"
                ],
                "task_performance_collapse_event": "not_adjudicated_no_registered_threshold",
                "support_or_variance_boundary_event": "not_instrumented_in_this_pilot",
                "nan_inf_numerical_failure": False,
            }
        )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["advantage_estimator"], row["exp_coefficient"])].append(row)
    groups: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: repr(item[0])):
        seeds = tuple(sorted(int(row["seed"]) for row in values))
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(f"paired seed set changed for {key}: {seeds}")
        groups.append(
            {
                "dataset": key[0],
                "advantage_estimator": key[1],
                "exp_coefficient": key[2],
                "seeds": list(seeds),
                "final_mean": statistics.fmean(float(row["final_score"]) for row in values),
                "late_mean": statistics.fmean(
                    float(row["late_window_mean_800k_1m"]) for row in values
                ),
                "final_seed_std": statistics.stdev(
                    float(row["final_score"]) for row in values
                ),
            }
        )
    index = {
        (group["dataset"], group["advantage_estimator"], group["exp_coefficient"]): group
        for group in groups
    }
    comparisons = []
    for dataset in EXPECTED_DATASETS:
        for coefficient in (None, *COEFFICIENTS):
            td = index[(dataset, "td", coefficient)]
            gae = index[(dataset, "gae", coefficient)]
            comparisons.append(
                {
                    "dataset": dataset,
                    "exp_coefficient": coefficient,
                    "gae_minus_td_final": float(gae["final_mean"]) - float(td["final_mean"]),
                    "gae_minus_td_late": float(gae["late_mean"]) - float(td["late_mean"]),
                }
            )
    aggregate_dir = work_dir / "aggregate"
    _write_csv(aggregate_dir / "branch_results.csv", rows)
    _write_csv(aggregate_dir / "group_summary.csv", groups)
    _write_csv(aggregate_dir / "gae_td_comparisons.csv", comparisons)
    audit = {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "raw_complete": True,
        "branch_count_observed": len(rows),
        "expected_branch_count": EXPECTED_BRANCHES,
        "critic_updated_during_actor_training": True,
        "prepared_advantage_artifact_used": False,
        "held_out_seeds_touched": False,
        "task_performance_collapse_separate": True,
        "support_or_variance_boundary_separate": True,
        "nan_inf_separate": True,
        "convergence_claim_allowed": False,
        "steady_state_ranking_allowed": False,
        "universal_gae_superiority_claim_allowed": False,
        "formal_evidence_allowed": False,
    }
    base.atomic_write_json(aggregate_dir / "terminal_audit.json", audit)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "branch_count": len(rows),
        "group_count": len(groups),
        "comparison_count": len(comparisons),
        "terminal_audit": str(aggregate_dir / "terminal_audit.json"),
    }
    base.atomic_write_json(aggregate_dir / "aggregate_summary.json", summary)
    return summary


def _runner(argv: list[str] | None = None) -> int:
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
    base.load_grid = _load_grid
    base.load_run_spec = _load_run_spec
    base.build_branches = _build_branches
    base.branch_command = _branch_command
    delegated = list(sys.argv[1:] if argv is None else argv)
    try:
        result = int(base.main(delegated))
        if delegated and delegated[0] == "run":
            index = delegated.index("--work-dir")
            _aggregate(Path(delegated[index + 1]).expanduser().resolve())
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


def main(argv: list[str] | None = None) -> int:
    delegated = list(sys.argv[1:] if argv is None else argv)
    if delegated and delegated[0] == "bootstrap":
        return _bootstrap(delegated[1:])
    return _runner(delegated)


if __name__ == "__main__":
    raise SystemExit(main())
