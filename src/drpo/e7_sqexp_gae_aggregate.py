"""Paired terminal aggregation without failed-cell imputation."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any
from drpo.e7_sqexp_gae_contract import (
    ESTIMATORS, EXPECTED_BRANCHES, EXPERIMENT_ID, SCIENTIFIC_STATUS, FrozenProtocol,
    atomic_json, build_actor_branches, utc_now, write_csv,
)

def aggregate_results(work_dir: Path, protocol: FrozenProtocol) -> dict[str, Any]:
    branches = build_actor_branches(protocol)
    rows: list[dict[str, Any]] = []
    for branch in branches:
        path = work_dir / "branches" / branch.dataset_id / f"seed_{branch.seed}" / branch.id / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing branch summary: {path}")
        row = json.loads(path.read_text())
        if row.get("experiment_id") != EXPERIMENT_ID:
            raise RuntimeError(f"branch summary experiment identity mismatch: {path}")
        if row.get("branch_id") != branch.id:
            raise RuntimeError(f"branch summary branch identity mismatch: {path}")
        required_terminal_fields = (
            "fixed_budget_completed",
            "terminal_state",
            "task_performance_collapse",
            "support_or_variance_boundary_event",
            "nan_inf_numerical_failure",
            "critic_immutability_verified",
        )
        missing = [name for name in required_terminal_fields if name not in row]
        if missing:
            raise RuntimeError(f"branch summary lacks terminal fields {missing}: {path}")
        row.update(dataclasses.asdict(branch))
        rows.append(row)
    by_pair: dict[tuple[str, int, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["dataset_id"], int(row["seed"]), row["actor_mode"], row["control_id"])
        by_pair.setdefault(key, {})[row["estimator"]] = row
    paired: list[dict[str, Any]] = []
    failed_pairs = 0
    for key, estimators in sorted(by_pair.items()):
        if set(estimators) != set(ESTIMATORS):
            failed_pairs += 1
            continue
        td = estimators["one_step_td"]
        gae = estimators["behavior_gae"]
        if not bool(td["fixed_budget_completed"]) or not bool(gae["fixed_budget_completed"]):
            failed_pairs += 1
            continue
        td_value = float(td["late_window_normalized_return_mean"])
        gae_value = float(gae["late_window_normalized_return_mean"])
        if not (math.isfinite(td_value) and math.isfinite(gae_value)):
            failed_pairs += 1
            continue
        paired.append(
            {
                "dataset_id": key[0],
                "seed": key[1],
                "actor_mode": key[2],
                "control_id": key[3],
                "td_late_window_normalized_return": td_value,
                "gae_late_window_normalized_return": gae_value,
                "gae_minus_td": gae_value - td_value,
            }
        )
    write_csv(work_dir / "aggregate" / "branch_summary.csv", rows)
    write_csv(work_dir / "aggregate" / "paired_gae_minus_td.csv", paired)
    terminal_state_counts: dict[str, int] = {}
    for row in rows:
        state = str(row["terminal_state"])
        terminal_state_counts[state] = terminal_state_counts.get(state, 0) + 1
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": SCIENTIFIC_STATUS,
        "expected_branches": EXPECTED_BRANCHES,
        "observed_branches": len(rows),
        "fixed_budget_completed_branches": sum(bool(row["fixed_budget_completed"]) for row in rows),
        "fixed_budget_incomplete_branches": sum(not bool(row["fixed_budget_completed"]) for row in rows),
        "terminal_state_counts": terminal_state_counts,
        "task_performance_collapse_adjudicated_branches": sum(
            row["task_performance_collapse"]
            != "not_adjudicated_no_frozen_task_collapse_threshold"
            for row in rows
        ),
        "task_performance_collapse_not_adjudicated_branches": sum(
            row["task_performance_collapse"]
            == "not_adjudicated_no_frozen_task_collapse_threshold"
            for row in rows
        ),
        "support_or_variance_boundary_event_branches": sum(
            bool(row["support_or_variance_boundary_event"]) for row in rows
        ),
        "nan_inf_numerical_failure_branches": sum(
            bool(row["nan_inf_numerical_failure"]) for row in rows
        ),
        "critic_immutability_failures": sum(
            not bool(row["critic_immutability_verified"]) for row in rows
        ),
        "paired_cells_included": len(paired),
        "paired_cells_excluded_without_imputation": failed_pairs,
        "failed_cell_imputation_used": False,
        "held_out_seeds_touched": False,
        "fixed_horizon_is_convergence": False,
        "formal_evidence_allowed": False,
        "claim_limits": [
            "no_universal_estimator_superiority",
            "no_convergence_or_steady_state_claim",
            "no_causal_identification_claim",
            "no_ood_generalization_claim",
            "hopper_and_walker_external_validity_only",
        ],
        "reporting_separation": [
            "task_performance_collapse",
            "support_or_variance_boundary_event",
            "nan_inf_numerical_failure",
        ],
        "completed_utc": utc_now(),
    }
    atomic_json(work_dir / "aggregate" / "terminal_audit.json", payload)
    return payload


