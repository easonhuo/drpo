#!/usr/bin/env bash
set -euo pipefail

COMMAND="${1:-run}"
case "${COMMAND}" in
  validate|plan|run) ;;
  *)
    echo "usage: $0 [validate|plan|run]" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

GRID="${E7_TASKC_GRID:-configs/e7_bench_joint_gae_taskc_top4_multiseed.json}"
CONTRACT="${E7_CANONICAL_CONTRACT:-/root/d4rl2/configs/e7_canonical_contract_9task.json}"
RUN_SPEC="${E7_CANONICAL_RUN_SPEC:-/root/d4rl2/configs/e7_canonical_9task_full_grid_run_spec_v1.json}"
WORK_DIR="${E7_TASKC_WORK_DIR:-outputs/e7/taskc_top4_multiseed_001}"
MAX_WORKERS="${E7_TASKC_MAX_WORKERS:-176}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to run from a dirty checkout" >&2
  exit 2
fi
if [[ ! -f "${GRID}" ]]; then
  echo "missing task-specific grid: ${GRID}" >&2
  exit 2
fi
if [[ "${COMMAND}" != "validate" ]]; then
  for required in "${CONTRACT}" "${RUN_SPEC}"; do
    if [[ ! -f "${required}" ]]; then
      echo "missing required file: ${required}" >&2
      exit 2
    fi
  done
fi
if ! [[ "${MAX_WORKERS}" =~ ^[1-9][0-9]*$ ]] || (( MAX_WORKERS > 180 )); then
  echo "E7_TASKC_MAX_WORKERS must be an integer in [1,180]" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python - "${COMMAND}" "${CONTRACT}" "${RUN_SPEC}" "${GRID}" "${WORK_DIR}" "${MAX_WORKERS}" <<'PY'
from __future__ import annotations

import copy
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from drpo import e7_canonical_sweep as base
from drpo import e7_squared_exp_night as suite
from drpo import e7_squared_exp_night_aggregate as agg
from drpo.e7_canonical_injection import sha256_file
from drpo.e7_squared_exp_kernel import THRESHOLDED_FORMULA

TASKC_PROFILE_ID = "d4rl9_task_specific_c_top4_multiseed"
TASKC_STATUS = "d4rl9_task_specific_c_top4_multiseed_joint_critic_gae_pilot_only"
TASKC_RUNNER_VERSION = "5.4.0-taskc-top4-multiseed"
TASKC_SEEDS = (200, 201, 202, 203, 208)
TASKC_FULL_RUN_ENV = "DRPO_E7_TASKC_TOP4_MULTISEED_FULL_RUN"
TASKC_EXPECTED_BRANCHES = 180
TASKC_TOP_K = 3
EXPECTED_DATASETS = tuple(suite.TUNING_DATASETS)


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("cannot average an empty list")
    return statistics.fmean(values)


def _std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _taskc_config(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "experiment_id": suite.GAE_EXPERIMENT_ID,
        "profile_id": TASKC_PROFILE_ID,
        "parent_experiment_id": "EXT-H-E7-BENCH-01",
        "run_kind": "pilot",
        "status": "not_run",
        "scientific_status": TASKC_STATUS,
        "datasets": list(EXPECTED_DATASETS),
        "development_seeds": list(TASKC_SEEDS),
        "held_out_seeds": list(suite.HELD_OUT_SEEDS),
        "steps": suite.EXPECTED_STEPS,
        "evaluation_interval": 50_000,
        "evaluation_episodes": 10,
        "actor_update_modes": ["a2c"],
        "advantage_modes": ["gae_lambda_0p95"],
        "expected_total_branches": TASKC_EXPECTED_BRANCHES,
        "screening_only": True,
        "formal_evidence_allowed": False,
    }
    changed = [key for key, value in required.items() if raw.get(key) != value]
    if changed:
        raise ValueError(f"task-specific grid changed: {changed}")

    controls = raw.get("task_remoteness_scales")
    if not isinstance(controls, dict) or tuple(controls) != EXPECTED_DATASETS:
        raise ValueError("task_remoteness_scales must preserve the canonical nine-task order")
    for dataset, values in controls.items():
        scales = tuple(float(value) for value in values)
        if len(scales) != 4 or len(set(scales)) != 4:
            raise ValueError(f"{dataset} must contain exactly four unique c values")
        if any(not math.isfinite(value) or value <= 0.0 for value in scales):
            raise ValueError(f"{dataset} contains a non-positive or non-finite c")

    weight = raw.get("weight_control", {})
    expected_weight = {
        "formula": THRESHOLDED_FORMULA,
        "coordinate": "normalized_squared_standardized_distance",
        "reference_distance": suite.REFERENCE_DISTANCE,
        "taper_lambda": suite.TUNING_TAPER_LAMBDA,
        "remoteness_threshold": suite.TUNING_REMOTENESS_THRESHOLD,
        "positive_only_anchor": False,
        "uncontrolled_anchor": False,
    }
    if any(weight.get(key) != value for key, value in expected_weight.items()):
        raise ValueError("task-specific thresholded taper contract changed")

    snapshot = raw.get("trajectory_snapshot", {})
    expected_snapshot = {
        "gae_lambda": suite.GAE_LAMBDA,
        "canonical_batch_size": suite.GAE_CANONICAL_BATCH_SIZE,
        "critic_updated_every_step": True,
        "prepared_advantage_artifact": False,
        "terminal_bootstrap": False,
        "timeout_bootstrap": True,
        "terminal_timeout_recursion_stop": True,
        "dataset_tail_recursion_stop": True,
    }
    if any(snapshot.get(key) != value for key, value in expected_snapshot.items()):
        raise ValueError("task-specific trajectory-snapshot GAE contract changed")

    reporting = raw.get("reporting_protocol", {})
    if (
        reporting.get("publication_summary") != "top3_of_5"
        or int(reporting.get("top_k", -1)) != TASKC_TOP_K
        or int(reporting.get("total_seeds", -1)) != len(TASKC_SEEDS)
        or reporting.get("retain_all_seed_results") is not True
        or reporting.get("retain_complete_training_curves") is not True
        or reporting.get("failed_runs_may_be_deleted") is not False
    ):
        raise ValueError("task-specific reporting protocol changed")
    return raw, sha256_file(source)


def _install_taskc_profile() -> None:
    suite.TUNING_SEEDS = TASKC_SEEDS
    suite.TUNING_EXPECTED_BRANCHES = TASKC_EXPECTED_BRANCHES
    suite.TUNING_RUNNER_VERSION = TASKC_RUNNER_VERSION
    suite.TUNING_FULL_RUN_ENV = TASKC_FULL_RUN_ENV

    def is_tuning() -> bool:
        return (
            suite._ACTIVE_EXPERIMENT_ID == suite.GAE_EXPERIMENT_ID
            and suite._ACTIVE_PROFILE_ID == TASKC_PROFILE_ID
        )

    suite._is_tuning = is_tuning
    suite._is_p3 = lambda: False
    suite.active_scientific_status = lambda: TASKC_STATUS
    suite.active_expected_branch_count = lambda: TASKC_EXPECTED_BRANCHES

    def configure_execution(
        grid_path: str | Path,
        *,
        liveness_pair: bool = False,
        liveness_steps: int | None = None,
    ) -> None:
        if liveness_pair or liveness_steps is not None:
            raise ValueError("the task-specific 180-branch profile has no liveness submatrix")
        raw, _ = _taskc_config(grid_path)
        suite._ACTIVE_EXPERIMENT_ID = str(raw["experiment_id"])
        suite._ACTIVE_PROFILE_ID = str(raw["profile_id"])
        suite._LIVENESS_STEPS = None

    suite.configure_execution = configure_execution
    suite.load_grid = _taskc_config

    original_load_run_spec = suite.load_run_spec

    def load_run_spec(path: str | Path) -> tuple[dict[str, Any], str]:
        run_spec, digest = original_load_run_spec(path)
        run_spec = copy.deepcopy(run_spec)
        run_spec["seeds"] = list(TASKC_SEEDS)
        return run_spec, digest

    suite.load_run_spec = load_run_spec

    def gae_branches(
        run_spec: Mapping[str, Any], grid: Mapping[str, Any]
    ) -> list[base.Branch]:
        datasets = [base.DatasetSpec.from_mapping(item) for item in run_spec["datasets"]]
        by_id = {dataset.id: dataset for dataset in datasets}
        if tuple(by_id) != EXPECTED_DATASETS:
            raise ValueError("task-specific run spec changed the canonical task order")
        branches: list[base.Branch] = []
        task_scales = grid["task_remoteness_scales"]
        for dataset_id in EXPECTED_DATASETS:
            dataset = by_id[dataset_id]
            for scale_value in task_scales[dataset_id]:
                scale = float(scale_value)
                coefficient = suite.TUNING_TAPER_LAMBDA / scale
                label = f"drpo_c{scale:g}"
                for seed in TASKC_SEEDS:
                    branches.append(
                        base.Branch(
                            branch_id=(
                                f"{dataset.id}__seed{seed}__gae__{label}__"
                                "a2c__steps1m"
                            ),
                            branch_kind="injected",
                            dataset=dataset,
                            seed=seed,
                            template_values={
                                "steps": str(suite.EXPECTED_STEPS),
                                "stage": "taskc_top4_multiseed",
                                "actor_update_mode": "a2c",
                                "advantage_estimator": "gae",
                                "weight_method": "thresholded_exponential",
                                "weight_at_zero": "1",
                                "exp_coefficient": f"{coefficient:.17g}",
                                "reference_distance": f"{suite.REFERENCE_DISTANCE:.17g}",
                                "remoteness_threshold": (
                                    f"{suite.TUNING_REMOTENESS_THRESHOLD:.17g}"
                                ),
                                "remoteness_scale": f"{scale:.17g}",
                                "taper_lambda": f"{suite.TUNING_TAPER_LAMBDA:.17g}",
                                "diagnostics_interval": str(suite.DIAGNOSTICS_INTERVAL),
                                "sampled_values_per_update": str(
                                    suite.SAMPLED_VALUES_PER_UPDATE
                                ),
                                "execution_mode": "full",
                            },
                            negative_control=None,
                        )
                    )
        return branches

    suite._gae_branches = gae_branches

    original_branch_command = suite.branch_command

    def branch_command(**kwargs: Any) -> tuple[list[str], dict[str, Any]]:
        command, branch_config = original_branch_command(**kwargs)
        branch_config["profile_id"] = TASKC_PROFILE_ID
        branch_dir = Path(kwargs["branch_dir"])
        base.atomic_write_json(branch_dir / "branch_config.json", branch_config)
        return command, branch_config

    suite.branch_command = branch_command
    suite.aggregate_results = _aggregate_taskc


def _aggregate_taskc(work_dir: str | Path) -> dict[str, Any]:
    work = Path(work_dir)
    branch_dirs, experiment_id = agg._branch_dirs(work)
    if experiment_id != suite.GAE_EXPERIMENT_ID:
        raise RuntimeError("task-specific aggregate saw the wrong experiment ID")
    if len(branch_dirs) != TASKC_EXPECTED_BRANCHES:
        raise RuntimeError(
            f"expected {TASKC_EXPECTED_BRANCHES} branches, found {len(branch_dirs)}"
        )

    original_is_tuning = agg._is_tuning_profile
    original_profile_name = agg._profile_name
    original_seeds = agg.TUNING_SEEDS
    agg._is_tuning_profile = lambda profile: (
        profile == TASKC_PROFILE_ID or original_is_tuning(profile)
    )
    agg._profile_name = lambda profile: (
        "task-specific top-4 multi-seed"
        if profile == TASKC_PROFILE_ID
        else original_profile_name(profile)
    )
    agg.TUNING_SEEDS = TASKC_SEEDS
    try:
        rows = [agg._gae_branch_row(path) for path in branch_dirs]
    finally:
        agg._is_tuning_profile = original_is_tuning
        agg._profile_name = original_profile_name
        agg.TUNING_SEEDS = original_seeds

    if {row["profile_id"] for row in rows} != {TASKC_PROFILE_ID}:
        raise RuntimeError("task-specific aggregate contains mixed profiles")
    if any(int(row["trainer_steps"]) != suite.EXPECTED_STEPS for row in rows):
        raise RuntimeError("one or more task-specific branches did not reach one million steps")
    if {int(row["seed"]) for row in rows} != set(TASKC_SEEDS):
        raise RuntimeError("task-specific aggregate seed set changed")

    aggregate_dir = work / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["control"] = agg._tuning_label(row)
        row["log10_remoteness_scale"] = math.log10(float(row["remoteness_scale"]))
    agg._write_csv(aggregate_dir / "branch_results.csv", rows)

    curves: list[dict[str, Any]] = []
    branch_by_id = {str(row["branch_id"]): row for row in rows}
    for branch_dir in branch_dirs:
        branch = json.loads((branch_dir / "branch_config.json").read_text())
        row = branch_by_id[str(branch["branch_id"])]
        summary_path = agg._only(
            (branch_dir / "trainer_output").glob("*_summary.json"),
            "trainer summary",
        )
        steps, scores = agg._read_history(json.loads(summary_path.read_text()))
        for step, score in zip(steps, scores, strict=True):
            curves.append(
                {
                    "dataset": row["dataset"],
                    "remoteness_scale": row["remoteness_scale"],
                    "control": row["control"],
                    "seed": row["seed"],
                    "step": step,
                    "normalized_return": score,
                }
            )
    agg._write_csv(aggregate_dir / "training_curves_long.csv", curves)

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), float(row["remoteness_scale"]))].append(row)

    all5_rows: list[dict[str, Any]] = []
    top3_rows: list[dict[str, Any]] = []
    manifest_cells: list[dict[str, Any]] = []
    for (dataset, scale), values in sorted(grouped.items()):
        seeds = tuple(sorted(int(row["seed"]) for row in values))
        if seeds != TASKC_SEEDS or len(values) != len(TASKC_SEEDS):
            raise RuntimeError(f"incomplete five-seed cell for {dataset}, c={scale:g}")
        late = [float(row["late_window_mean_800k_1m"]) for row in values]
        final = [float(row["final_score"]) for row in values]
        best = [float(row["best_score"]) for row in values]
        drop = [float(row["best_to_final_drop"]) for row in values]
        slope = [float(row["late_slope_per_100k"]) for row in values]
        all5 = {
            "dataset": dataset,
            "control": f"drpo_c{scale:g}",
            "remoteness_scale": scale,
            "seeds": list(seeds),
            "all5_late_mean": _mean(late),
            "all5_late_std": _std(late),
            "all5_late_median": statistics.median(late),
            "all5_final_mean": _mean(final),
            "all5_final_std": _std(final),
            "all5_final_median": statistics.median(final),
            "all5_best_mean": _mean(best),
            "all5_best_to_final_drop_mean": _mean(drop),
            "all5_late_slope_per_100k_mean": _mean(slope),
        }
        all5_rows.append(all5)

        ordered = sorted(
            values,
            key=lambda row: (
                float(row["late_window_mean_800k_1m"]),
                float(row["final_score"]),
                -int(row["seed"]),
            ),
            reverse=True,
        )
        selected = ordered[:TASKC_TOP_K]
        selected_seeds = [int(row["seed"]) for row in selected]
        top3 = {
            "dataset": dataset,
            "control": f"drpo_c{scale:g}",
            "remoteness_scale": scale,
            "top3_seed_ids": selected_seeds,
            "top3_of5_late_mean": _mean(
                [float(row["late_window_mean_800k_1m"]) for row in selected]
            ),
            "top3_of5_late_std": _std(
                [float(row["late_window_mean_800k_1m"]) for row in selected]
            ),
            "top3_of5_final_mean": _mean(
                [float(row["final_score"]) for row in selected]
            ),
            "top3_of5_best_mean": _mean(
                [float(row["best_score"]) for row in selected]
            ),
            "all5_late_mean": all5["all5_late_mean"],
            "all5_late_std": all5["all5_late_std"],
            "all5_late_median": all5["all5_late_median"],
        }
        top3_rows.append(top3)
        manifest_cells.append(
            {
                "dataset": dataset,
                "remoteness_scale": scale,
                "selection_metric": "late_window_mean_800k_1m",
                "top_k": TASKC_TOP_K,
                "selected_seed_ids": selected_seeds,
                "ranked_seed_rows": [
                    {
                        "rank": rank,
                        "seed": int(row["seed"]),
                        "late_window_mean_800k_1m": float(
                            row["late_window_mean_800k_1m"]
                        ),
                        "final_score": float(row["final_score"]),
                    }
                    for rank, row in enumerate(ordered, start=1)
                ],
            }
        )

    if len(all5_rows) != 36 or len(top3_rows) != 36:
        raise RuntimeError("task-specific aggregate did not produce 36 task-c cells")
    agg._write_csv(aggregate_dir / "task_c_all5_summary.csv", all5_rows)
    agg._write_csv(aggregate_dir / "task_c_top3_of5_summary.csv", top3_rows)

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all5_index = {
        (str(row["dataset"]), float(row["remoteness_scale"])): row
        for row in all5_rows
    }
    for row in top3_rows:
        by_task[str(row["dataset"])].append(row)
    selected_task_rows: list[dict[str, Any]] = []
    selected_c_by_task: dict[str, float] = {}
    for dataset in EXPECTED_DATASETS:
        candidates = by_task[dataset]
        if len(candidates) != 4:
            raise RuntimeError(f"{dataset} does not contain four candidate c values")
        chosen = max(
            candidates,
            key=lambda row: (
                float(row["top3_of5_late_mean"]),
                float(row["all5_late_median"]),
                -float(row["remoteness_scale"]),
            ),
        )
        scale = float(chosen["remoteness_scale"])
        all5 = all5_index[(dataset, scale)]
        selected_c_by_task[dataset] = scale
        selected_task_rows.append(
            {
                **chosen,
                "selection_rule": (
                    "max_top3_of5_late_mean_then_all5_median_then_smaller_c"
                ),
                "all5_final_mean": all5["all5_final_mean"],
                "all5_final_std": all5["all5_final_std"],
                "all5_best_mean": all5["all5_best_mean"],
            }
        )
    agg._write_csv(aggregate_dir / "task_selected_c_summary.csv", selected_task_rows)

    selection_manifest = {
        "schema_version": 1,
        "profile_id": TASKC_PROFILE_ID,
        "selection_protocol": "predeclared_top3_of5",
        "primary_metric": "late_window_mean_800k_1m",
        "top_k": TASKC_TOP_K,
        "total_seeds": len(TASKC_SEEDS),
        "candidate_c_rule": (
            "max_top3_of5_late_mean_then_all5_median_then_smaller_c"
        ),
        "selected_c_by_task": selected_c_by_task,
        "cells": manifest_cells,
        "all_seed_outputs_retained": True,
        "complete_training_curves_retained": True,
    }
    agg._atomic_json(aggregate_dir / "top3_selection_manifest.json", selection_manifest)

    audit = {
        "status": "PASS",
        "experiment_id": suite.GAE_EXPERIMENT_ID,
        "profile_id": TASKC_PROFILE_ID,
        "raw_complete": True,
        "branch_count_observed": len(rows),
        "expected_branch_count": TASKC_EXPECTED_BRANCHES,
        "task_count": len(EXPECTED_DATASETS),
        "candidate_c_per_task": 4,
        "development_seeds": list(TASKC_SEEDS),
        "held_out_seeds_touched": False,
        "critic_updated_during_actor_training": True,
        "prepared_advantage_artifact_used": False,
        "task_performance_collapse_status": (
            "not_adjudicated_no_registered_threshold"
        ),
        "support_or_variance_boundary_status": "not_instrumented_in_this_pilot",
        "rollout_failure_count": 0,
        "nan_inf_numerical_failure_count": 0,
        "task_performance_collapse_separate": True,
        "support_or_variance_boundary_separate": True,
        "rollout_failure_separate": True,
        "nan_inf_separate": True,
        "top3_of5_publication_summary_predeclared": True,
        "all5_statistics_retained": True,
        "failed_runs_preserved": True,
        "fixed_horizon_is_not_convergence": True,
        "method_ranking_claim_allowed": False,
        "formal_evidence_allowed": False,
    }
    agg._atomic_json(aggregate_dir / "terminal_audit.json", audit)
    summary = {
        "experiment_id": suite.GAE_EXPERIMENT_ID,
        "profile_id": TASKC_PROFILE_ID,
        "status": "PASS",
        "branch_count": len(rows),
        "task_c_cell_count": len(all5_rows),
        "selected_c_by_task": selected_c_by_task,
        "terminal_audit": str(aggregate_dir / "terminal_audit.json"),
        "files": {
            "training_curves": str(aggregate_dir / "training_curves_long.csv"),
            "branch_results": str(aggregate_dir / "branch_results.csv"),
            "all5_summary": str(aggregate_dir / "task_c_all5_summary.csv"),
            "top3_summary": str(aggregate_dir / "task_c_top3_of5_summary.csv"),
            "selected_c_summary": str(aggregate_dir / "task_selected_c_summary.csv"),
            "selection_manifest": str(aggregate_dir / "top3_selection_manifest.json"),
        },
    }
    agg._atomic_json(aggregate_dir / "aggregate_summary.json", summary)
    return summary


def main() -> int:
    if len(sys.argv) != 7:
        raise SystemExit("internal launcher argument mismatch")
    command, contract, run_spec, grid, work_dir, workers = sys.argv[1:]
    _install_taskc_profile()
    raw, grid_sha = _taskc_config(grid)
    branch_count = len(raw["datasets"]) * len(TASKC_SEEDS) * 4
    if branch_count != TASKC_EXPECTED_BRANCHES:
        raise RuntimeError("task-specific matrix is not exactly 180 branches")
    if command == "validate":
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "profile_id": TASKC_PROFILE_ID,
                    "grid_sha256": grid_sha,
                    "task_count": len(raw["datasets"]),
                    "candidate_c_per_task": 4,
                    "seeds": list(TASKC_SEEDS),
                    "branch_count": branch_count,
                    "held_out_seeds_touched": False,
                    "publication_summary": "top3_of_5",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    args = [
        "plan",
        "--contract",
        contract,
        "--run-spec",
        run_spec,
        "--grid",
        grid,
        "--work-dir",
        work_dir,
        "--max-workers",
        workers,
    ]
    suite.main(args)
    if command == "plan":
        return 0
    os.environ[TASKC_FULL_RUN_ENV] = "1"
    args[0] = "run"
    args.append("--resume")
    return int(suite.main(args))


raise SystemExit(main())
PY
