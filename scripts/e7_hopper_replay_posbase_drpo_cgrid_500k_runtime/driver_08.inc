                    "advantage_baseline": baseline,
                    "remoteness_scale": scale,
                    **item,
                }
            )
        for item in dynamics_rows:
            dynamics_curves.append(
                {
                    "branch_id": branch["branch_id"],
                    "seed": seed,
                    "control_id": control_id,
                    "advantage_baseline": baseline,
                    "remoteness_scale": scale,
                    **item,
                }
            )

    observed_cells = {(str(row["control_id"]), int(row["seed"])) for row in rows}
    expected_cells = {(control, seed) for control in CONTROL_ORDER for seed in SEEDS}
    if observed_cells != expected_cells:
        raise RuntimeError("Positive-only/DRPO aggregate cell set is incomplete")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["control_id"])].append(row)
    summaries: list[dict[str, Any]] = []
    for control_id in CONTROL_ORDER:
        values = sorted(grouped[control_id], key=lambda row: int(row["seed"]))
        if len(values) != len(SEEDS):
            raise RuntimeError(f"control {control_id} does not contain nine seeds")
        late_values = [float(row["late_window_mean_360k_500k"]) for row in values]
        final_values = [float(row["final_score_500k"]) for row in values]
        slope_values = [float(row["trend_slope_per_100k_200k_500k"]) for row in values]
        drop_values = [float(row["best_to_final_drop"]) for row in values]
        effective_positive = [float(row["latest_effective_positive_fraction"]) for row in values]
        effective_negative = [
            float(row["effective_negative_mass_fraction"])
            for row in values
            if row["effective_negative_mass_fraction"] is not None
        ]
        worst_three = sorted(late_values)[:3]
        control = CONTROL_BY_ID[control_id]
        summaries.append(
            {
                "control_id": control_id,
                "weight_method": control["weight_method"],
                "advantage_baseline": control["advantage_baseline"],
                "remoteness_scale": control["remoteness_scale"],
                "seed_ids": list(SEEDS),
                "late_mean": _mean(late_values),
                "late_median": _median(late_values),
                "late_std": _std(late_values),
                "worst_three_seed_late_mean": _mean(worst_three),
                "final_mean": _mean(final_values),
                "final_median": _median(final_values),
                "final_std": _std(final_values),
                "trend_slope_mean_per_100k": _mean(slope_values),
                "trend_slope_median_per_100k": _median(slope_values),
                "nonnegative_trend_seed_count": sum(value >= 0.0 for value in slope_values),
                "mean_best_to_final_drop": _mean(drop_values),
                "latest_effective_positive_fraction_mean": _mean(effective_positive),
                "effective_negative_mass_fraction_mean": (
                    _mean(effective_negative) if effective_negative else 0.0
                ),
                "support_or_variance_boundary_seed_count": sum(
                    bool(row["support_or_variance_boundary_event"]) for row in values
                ),
                "nan_inf_numerical_failure_seed_count": sum(
                    bool(row["nan_inf_numerical_failure"]) for row in values
                ),
            }
        )

    index = {(str(row["control_id"]), int(row["seed"])): row for row in rows}
    paired_same_baseline: list[dict[str, Any]] = []
    paired_same_c_b0: list[dict[str, Any]] = []
    paired_summary: list[dict[str, Any]] = []

    for baseline in BASELINES:
        comparator_id = _positive_control_id(baseline)
        for scale in REMOTENESS_SCALES:
            candidate_id = _drpo_control_id(baseline, scale)
            comparisons = []
            for seed in SEEDS:
