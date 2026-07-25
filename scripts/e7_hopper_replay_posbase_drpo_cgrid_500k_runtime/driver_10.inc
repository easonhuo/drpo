        if row["comparison_family"] == "drpo_minus_positive_only_same_baseline"
    }
    screening_rows: list[dict[str, Any]] = []
    for summary in summaries:
        if summary["weight_method"] != "thresholded_exponential":
            continue
        paired = same_baseline_summary[summary["control_id"]]
        screening_rows.append(
            {
                **summary,
                "paired_late_win_count_vs_positive_only_same_baseline": paired["late_win_count"],
                "paired_late_difference_mean_vs_positive_only_same_baseline": paired[
                    "late_difference_mean"
                ],
                "median_trend_nonnegative": summary["trend_slope_median_per_100k"] >= 0.0,
            }
        )
    screening_rows.sort(
        key=lambda row: (
            float(row["late_median"]),
            float(row["worst_three_seed_late_mean"]),
            int(row["paired_late_win_count_vs_positive_only_same_baseline"]),
            bool(row["median_trend_nonnegative"]),
            float(row["final_median"]),
            -float(row["mean_best_to_final_drop"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(screening_rows, start=1):
        row["finite_horizon_screening_rank"] = rank

    aggregate_dir = work / "aggregate"
    _write_csv(aggregate_dir / "branch_results.csv", rows)
    _write_csv(aggregate_dir / "training_curves_long.csv", curves)
    _write_csv(aggregate_dir / "advantage_snapshot_curves_long.csv", snapshot_curves)
    _write_csv(aggregate_dir / "geometry_diagnostics_long.csv", geometry_curves)
    _write_csv(aggregate_dir / "actor_training_dynamics_long.csv", dynamics_curves)
    _write_csv(aggregate_dir / "control_summary.csv", summaries)
    _write_csv(aggregate_dir / "paired_drpo_vs_positive_only_same_baseline.csv", paired_same_baseline)
    _write_csv(aggregate_dir / "paired_shifted_drpo_vs_b0_same_c.csv", paired_same_c_b0)
    _write_csv(aggregate_dir / "paired_comparison_summary.csv", paired_summary)
    _write_csv(aggregate_dir / "finite_horizon_screening_order.csv", screening_rows)

    audit = {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "profile_id": PROFILE_ID,
        "raw_complete": True,
        "branch_count_observed": len(rows),
        "expected_branch_count": EXPECTED_BRANCHES,
        "control_count_observed": len(summaries),
        "expected_control_count": EXPECTED_CONTROLS,
        "controls": list(CONTROL_ORDER),
        "baselines": list(BASELINES),
        "drpo_remoteness_scales": list(REMOTENESS_SCALES),
        "seeds": list(SEEDS),
        "held_out_seeds_touched": False,
        "all_replay_samples_retained": True,
        "advantage_shift_before_sign_split": True,
        "complete_training_curves_retained": True,
        "complete_snapshot_series_retained": True,
        "complete_geometry_series_retained": True,
        "sampled_raw_gradient_and_adam_update_series_retained": True,
        "task_performance_collapse_status": "not_adjudicated_no_registered_threshold",
        "support_or_variance_boundary_branch_count": sum(
            bool(row["support_or_variance_boundary_event"]) for row in rows
        ),
        "rollout_failure_count": sum(bool(row["rollout_failure_event"]) for row in rows),
        "nan_inf_numerical_failure_count": sum(
            bool(row["nan_inf_numerical_failure"]) for row in rows
        ),
        "fixed_500k_horizon_is_not_convergence": True,
        "steady_state_ranking_allowed": False,
        "held_out_confirmation_completed": False,
        "formal_evidence_allowed": False,
    }
    if audit["rollout_failure_count"] != 0 or audit["nan_inf_numerical_failure_count"] != 0:
        raise RuntimeError("completed aggregate contains rollout or numerical failures")
    _atomic_json(aggregate_dir / "terminal_audit.json", audit)
    summary = {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "profile_id": PROFILE_ID,
