                left = index[(candidate_id, seed)]
                right = index[(comparator_id, seed)]
                record = {
                    "candidate": candidate_id,
                    "comparator": comparator_id,
                    "advantage_baseline": baseline,
                    "remoteness_scale": scale,
                    "seed": seed,
                    "late_difference": float(left["late_window_mean_360k_500k"])
                    - float(right["late_window_mean_360k_500k"]),
                    "final_difference": float(left["final_score_500k"])
                    - float(right["final_score_500k"]),
                    "trend_slope_difference_per_100k": float(
                        left["trend_slope_per_100k_200k_500k"]
                    )
                    - float(right["trend_slope_per_100k_200k_500k"]),
                }
                paired_same_baseline.append(record)
                comparisons.append(record)
            paired_summary.append(
                {
                    "candidate": candidate_id,
                    "comparator": comparator_id,
                    "comparison_family": "drpo_minus_positive_only_same_baseline",
                    "advantage_baseline": baseline,
                    "remoteness_scale": scale,
                    "late_difference_mean": _mean([row["late_difference"] for row in comparisons]),
                    "late_difference_median": _median([row["late_difference"] for row in comparisons]),
                    "late_win_count": sum(row["late_difference"] > 0.0 for row in comparisons),
                    "final_difference_mean": _mean([row["final_difference"] for row in comparisons]),
                    "final_win_count": sum(row["final_difference"] > 0.0 for row in comparisons),
                }
            )

    for baseline in BASELINES[1:]:
        for scale in REMOTENESS_SCALES:
            candidate_id = _drpo_control_id(baseline, scale)
            comparator_id = _drpo_control_id(0.0, scale)
            comparisons = []
            for seed in SEEDS:
                left = index[(candidate_id, seed)]
                right = index[(comparator_id, seed)]
                record = {
                    "candidate": candidate_id,
                    "comparator": comparator_id,
                    "advantage_baseline": baseline,
                    "remoteness_scale": scale,
                    "seed": seed,
                    "late_difference": float(left["late_window_mean_360k_500k"])
                    - float(right["late_window_mean_360k_500k"]),
                    "final_difference": float(left["final_score_500k"])
                    - float(right["final_score_500k"]),
                    "trend_slope_difference_per_100k": float(
                        left["trend_slope_per_100k_200k_500k"]
                    )
                    - float(right["trend_slope_per_100k_200k_500k"]),
                }
                paired_same_c_b0.append(record)
                comparisons.append(record)
            paired_summary.append(
                {
                    "candidate": candidate_id,
                    "comparator": comparator_id,
                    "comparison_family": "shifted_drpo_minus_b0_same_c",
                    "advantage_baseline": baseline,
                    "remoteness_scale": scale,
                    "late_difference_mean": _mean([row["late_difference"] for row in comparisons]),
                    "late_difference_median": _median([row["late_difference"] for row in comparisons]),
                    "late_win_count": sum(row["late_difference"] > 0.0 for row in comparisons),
                    "final_difference_mean": _mean([row["final_difference"] for row in comparisons]),
                    "final_win_count": sum(row["final_difference"] > 0.0 for row in comparisons),
                }
            )

    same_baseline_summary = {
        row["candidate"]: row
        for row in paired_summary
