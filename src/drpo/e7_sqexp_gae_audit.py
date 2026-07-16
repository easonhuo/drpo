"""Terminal engineering audit for the E7 TD/GAE development pilot."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from drpo import e7_canonical_sweep as base
from drpo.e7_sqexp_gae_protocol import EXPECTED_BRANCHES, EXPERIMENT_ID


def terminal_audit(work_dir: Path) -> dict[str, Any]:
    summary = json.loads((work_dir / "RUN_SUMMARY.json").read_text())
    failures: list[str] = []
    estimator_counts = {"td": 0, "gae": 0}
    actor_counts = {"a2c": 0, "ppo_clip_k4": 0}
    seeds: set[int] = set()
    for result in summary.get("results", []):
        branch_id = str(result["branch_id"])
        manifest_path = work_dir / "branches" / branch_id / "branch_manifest.json"
        if not manifest_path.is_file():
            failures.append(f"{branch_id}:missing_manifest")
            continue
        manifest = json.loads(manifest_path.read_text())
        branch = manifest.get("branch", {})
        values = branch.get("template_values", {})
        estimator = str(manifest.get("advantage_estimator"))
        actor_mode = str(values.get("actor_update_mode"))
        if estimator in estimator_counts:
            estimator_counts[estimator] += 1
        else:
            failures.append(f"{branch_id}:bad_estimator")
        if actor_mode in actor_counts:
            actor_counts[actor_mode] += 1
        else:
            failures.append(f"{branch_id}:bad_actor_mode")
        seeds.add(int(branch.get("seed", -1)))
        if manifest.get("gae_used") != (estimator == "gae"):
            failures.append(f"{branch_id}:gae_flag")
        if manifest.get("critic_immutability_verified") is not True:
            failures.append(f"{branch_id}:critic_changed")
        if manifest.get("critic_initial_state_sha256") != manifest.get(
            "critic_final_state_sha256"
        ):
            failures.append(f"{branch_id}:critic_hash")
        provenance = manifest.get("advantage_provenance", {})
        if provenance.get("gae_recomputed_from_td_and_boundaries") is not True:
            failures.append(f"{branch_id}:gae_not_recomputed")
        if provenance.get("gae_matches_prepared_artifact") is not True:
            failures.append(f"{branch_id}:gae_artifact_mismatch")
    expected_half = EXPECTED_BRANCHES // 2
    if (
        summary.get("branch_count") != EXPECTED_BRANCHES
        or summary.get("completed") != EXPECTED_BRANCHES
        or summary.get("failed") != 0
    ):
        failures.append("run_summary")
    if estimator_counts != {"td": expected_half, "gae": expected_half}:
        failures.append("estimator_counts")
    if actor_counts != {"a2c": expected_half, "ppo_clip_k4": expected_half}:
        failures.append("actor_counts")
    held_out_touched = bool(seeds & {204, 205, 206, 207})
    if held_out_touched:
        failures.append("held_out_seed")
    audit = {
        "status": "PASS" if not failures else "FAIL",
        "experiment_id": EXPERIMENT_ID,
        "branch_count": summary.get("branch_count"),
        "completed": summary.get("completed"),
        "failed": summary.get("failed"),
        "estimator_counts": estimator_counts,
        "actor_mode_counts": actor_counts,
        "development_seeds_observed": sorted(seeds),
        "held_out_seeds_touched": held_out_touched,
        "critic_immutability_failures": sum("critic" in item for item in failures),
        "task_performance_collapse_event": "not_adjudicated_no_registered_threshold",
        "support_or_variance_boundary_event": "not_instrumented",
        "nan_inf_numerical_collapse": "not_observed_in_completed_branches",
        "fixed_1m_endpoint_is_convergence": False,
        "method_ranking_allowed": False,
        "formal_evidence_allowed": False,
        "failures": failures,
    }
    base.atomic_write_json(work_dir / "TERMINAL_AUDIT.json", audit)
    if failures:
        raise RuntimeError(f"terminal audit failed: {audit}")
    return audit
