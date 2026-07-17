"""Canonical joint actor--critic TD/GAE pilot for E7 squared-remoteness control."""
from __future__ import annotations

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
RUNNER_VERSION = "4.1.0-canonical-joint-critic-repaired"
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
    values = list(argv[1:] if argv and argv[0] == "--" else argv)
    if _trainer_flag(values, "--variant") != "iqlv_exp_rank":
        raise ValueError("GAE pilot requires canonical iqlv_exp_rank")
    if _trainer_flag(values, "--batch") != str(CANONICAL_BATCH_SIZE):
        raise ValueError("GAE pilot requires canonical batch size 256")
    if _trainer_flag(values, "--ret_weight_mode") not in {None, "none"}:
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
    weight, snapshot = raw.get("weight_control", {}), raw.get("trajectory_snapshot", {})
    nested_changed = (
        weight.get("formula") != FORMULA
        or tuple(map(float, weight.get("exp_coefficients", ()))) != COEFFICIENTS
        or float(snapshot.get("gae_lambda", -1.0)) != GAE_LAMBDA
        or int(snapshot.get("canonical_batch_size", -1)) != CANONICAL_BATCH_SIZE
        or snapshot.get("critic_updated_every_step") is not True
        or snapshot.get("prepared_advantage_artifact") is not False
    )
    if changed or nested_changed:
        raise ValueError(f"joint-critic GAE grid changed: {changed or ['nested_contract']}")
    return raw, sha256_file(source)


def _load_run_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    run_spec, digest = night.load_run_spec(path)
    run_spec["seeds"] = list(EXPECTED_SEEDS)
    return run_spec, digest


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
    controls = [("positive_only", 0.0, 0.0), *(
        ("squared_exponential", 1.0, coefficient) for coefficient in COEFFICIENTS
    )]
    branches = [
        base.Branch(
            branch_id=(
                f"{dataset.id}__seed{seed}__{estimator}__"
                f"{'positive_only' if method == 'positive_only' else f'sqexp_c{coefficient:g}'}__"
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
        for estimator in ESTIMATORS
        for method, weight_at_zero, coefficient in controls
        for dataset in datasets
        for seed in EXPECTED_SEEDS
    ]
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
    command, config = night.branch_command(
        contract_path=contract_path,
        contract=contract,
        branch=branch,
        branch_dir=branch_dir,
        trainer_argv_template=trainer_argv_template,
    )
    config.update(
        experiment_id=EXPERIMENT_ID,
        canonical_root=str(contract.source_root),
        dataset_path=str(Path(branch.dataset.path).expanduser().resolve()),
    )
    base.atomic_write_json(branch_dir / "branch_config.json", config)
    command[2] = "drpo.e7_sqexp_gae"
    command.insert(3, "bootstrap")
    return command, config


def _bootstrap(argv: list[str]) -> int:
    args = canonical_bootstrap.build_parser().parse_args(argv)
    branch = json.loads(Path(args.branch_config).expanduser().read_text())
    values = branch.get("template_values", {})
    estimator_name = str(values.get("advantage_estimator"))
    if branch.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("GAE branch experiment_id mismatch")
    if estimator_name not in ESTIMATORS or values.get("actor_update_mode") != "a2c":
        raise ValueError("joint GAE pilot supports only TD/GAE with canonical A2C")
    runtime_probe = os.getenv("DRPO_RUNTIME_RESOURCE_PROBE") == "1"
    _validate_trainer_args(args.trainer_args, runtime_probe=runtime_probe)
    dataset = Path(branch["dataset_path"]).expanduser().resolve()
    if sha256_file(dataset) != branch["dataset_sha256"]:
        raise RuntimeError("ordered dataset hash mismatch")
    replay = load_ordered_replay(
        canonical_root=branch["canonical_root"],
        dataset_path=dataset,
        dataset_id=str(branch["dataset_id"]),
    )
    estimator = SnapshotEstimatorConfig(estimator_name, GAE_LAMBDA, CANONICAL_BATCH_SIZE)
    instances: list[Any] = []
    old_experiment, old_patch, old_returns = (
        canonical_bootstrap.EXPERIMENT_ID,
        canonical_bootstrap.patch_canonical_module,
        None,
    )

    def patch_joint(module: Any, contract: Any, control: Any) -> type:
        nonlocal old_returns
        import d4rl_common.train_loop as train_loop

        old_returns = train_loop.compute_mc_returns

        def transition_ids(rewards: Any, *_: Any, **__: Any) -> Any:
            if len(rewards) != replay.size:
                raise RuntimeError("trainer replay length changed before transition-ID injection")
            return transition_id_channel(replay.size)

        train_loop.compute_mc_returns = transition_ids
        injected = build_joint_snapshot_agent_class(
            getattr(module, contract.target_class),
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
            advantage_estimator=estimator_name,
            gae_used=estimator_name == "gae",
            critic_updated_during_actor_training=True,
            prepared_advantage_artifact_used=False,
            transition_id_channel="ep_ret_exact_float32_index",
            trajectory_snapshot=snapshot,
        )
        canonical_bootstrap._atomic_json(manifest_path, payload)  # noqa: SLF001
        return result
    except BaseException as exc:
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text())
            payload.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            canonical_bootstrap._atomic_json(manifest_path, payload)  # noqa: SLF001
        raise
    finally:
        canonical_bootstrap.EXPERIMENT_ID = old_experiment
        canonical_bootstrap.patch_canonical_module = old_patch
        if old_returns is not None:
            import d4rl_common.train_loop as train_loop

            train_loop.compute_mc_returns = old_returns


def _branch_row(branch_dir: Path) -> dict[str, Any]:
    if not (branch_dir / "COMPLETED.json").is_file():
        raise RuntimeError(f"incomplete branch: {branch_dir.name}")
    branch = json.loads((branch_dir / "branch_config.json").read_text())
    manifest = json.loads((branch_dir / "branch_manifest.json").read_text())
    summary_path = night_aggregate._only(  # noqa: SLF001
        (branch_dir / "trainer_output").glob("*_summary.json"), "trainer summary"
    )
    steps, scores = night_aggregate._read_history(  # noqa: SLF001
        json.loads(summary_path.read_text())
    )
    if steps[-1] != EXPECTED_STEPS or not all(map(math.isfinite, scores)):
        raise RuntimeError(f"non-finite or incomplete branch: {branch_dir.name}")
    snapshot = manifest.get("trajectory_snapshot", {})
    if snapshot.get("critic_evolution_observed") is not True:
        raise RuntimeError(f"critic evolution missing: {branch_dir.name}")
    control = branch["weight_control"]
    coefficient = None if control["method"] == "positive_only" else float(
        control["exp_coefficient"]
    )
    late = [score for step, score in zip(steps, scores, strict=True) if step >= LATE_WINDOW_START]
    best = max(range(len(scores)), key=scores.__getitem__)
    geometry = night_aggregate._aggregate_geometry(  # noqa: SLF001
        branch_dir / "geometry_diagnostics.jsonl"
    )
    return {
        "branch_id": branch["branch_id"],
        "dataset": branch["dataset_id"],
        "seed": int(branch["seed"]),
        "advantage_estimator": branch["template_values"]["advantage_estimator"],
        "exp_coefficient": coefficient,
        "best_score": scores[best],
        "best_step": steps[best],
        "final_score": scores[-1],
        "best_to_final_drop": scores[best] - scores[-1],
        "late_window_mean_800k_1m": statistics.fmean(late),
        "late_window_std_800k_1m": statistics.stdev(late),
        "late_slope_per_100k": night_aggregate._slope_per_100k(  # noqa: SLF001
            steps, scores, LATE_WINDOW_START
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


def _aggregate(work_dir: Path) -> dict[str, Any]:
    branch_dirs = sorted(path for path in (work_dir / "branches").iterdir() if path.is_dir())
    if len(branch_dirs) != EXPECTED_BRANCHES:
        raise RuntimeError(f"expected {EXPECTED_BRANCHES} branches, found {len(branch_dirs)}")
    rows = [_branch_row(path) for path in branch_dirs]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["advantage_estimator"], row["exp_coefficient"])].append(row)
    groups = []
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
                "final_mean": statistics.fmean(row["final_score"] for row in values),
                "late_mean": statistics.fmean(
                    row["late_window_mean_800k_1m"] for row in values
                ),
                "final_seed_std": statistics.stdev(row["final_score"] for row in values),
            }
        )
    index = {
        (group["dataset"], group["advantage_estimator"], group["exp_coefficient"]): group
        for group in groups
    }
    comparisons = [
        {
            "dataset": dataset,
            "exp_coefficient": coefficient,
            "gae_minus_td_final": index[(dataset, "gae", coefficient)]["final_mean"]
            - index[(dataset, "td", coefficient)]["final_mean"],
            "gae_minus_td_late": index[(dataset, "gae", coefficient)]["late_mean"]
            - index[(dataset, "td", coefficient)]["late_mean"],
        }
        for dataset in EXPECTED_DATASETS
        for coefficient in (None, *COEFFICIENTS)
    ]
    aggregate_dir = work_dir / "aggregate"
    night_aggregate._write_csv(aggregate_dir / "branch_results.csv", rows)  # noqa: SLF001
    night_aggregate._write_csv(aggregate_dir / "group_summary.csv", groups)  # noqa: SLF001
    night_aggregate._write_csv(  # noqa: SLF001
        aggregate_dir / "gae_td_comparisons.csv", comparisons
    )
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
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "branch_count": len(rows),
        "group_count": len(groups),
        "comparison_count": len(comparisons),
        "terminal_audit": str(aggregate_dir / "terminal_audit.json"),
    }
    base.atomic_write_json(aggregate_dir / "aggregate_summary.json", result)
    return result


def _runner(argv: list[str] | None = None) -> int:
    names = (
        "EXPERIMENT_ID",
        "SCIENTIFIC_STATUS",
        "RUNNER_VERSION",
        "load_grid",
        "load_run_spec",
        "build_branches",
        "branch_command",
    )
    previous = tuple(getattr(base, name) for name in names)
    replacements = (
        EXPERIMENT_ID,
        SCIENTIFIC_STATUS,
        RUNNER_VERSION,
        _load_grid,
        _load_run_spec,
        _build_branches,
        _branch_command,
    )
    for name, value in zip(names, replacements, strict=True):
        setattr(base, name, value)
    delegated = list(sys.argv[1:] if argv is None else argv)
    try:
        result = int(base.main(delegated))
        if delegated and delegated[0] == "run":
            index = delegated.index("--work-dir")
            _aggregate(Path(delegated[index + 1]).expanduser().resolve())
        return result
    finally:
        for name, value in zip(names, previous, strict=True):
            setattr(base, name, value)


def main(argv: list[str] | None = None) -> int:
    delegated = list(sys.argv[1:] if argv is None else argv)
    return _bootstrap(delegated[1:]) if delegated and delegated[0] == "bootstrap" else _runner(delegated)


if __name__ == "__main__":
    raise SystemExit(main())
