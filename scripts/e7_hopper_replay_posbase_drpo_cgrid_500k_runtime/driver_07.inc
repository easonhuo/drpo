        snapshot_rows = _load_jsonl(branch_dir / "ADVANTAGE_SHIFT_SNAPSHOTS.jsonl")
        geometry_rows = _load_jsonl(branch_dir / "geometry_diagnostics.jsonl")
        dynamics_rows = _load_jsonl(branch_dir / "ACTOR_TRAINING_DYNAMICS.jsonl")
        shift = manifest["trajectory_snapshot"]["advantage_shift"]
        if len(snapshot_rows) != int(shift.get("snapshot_stat_count", -1)):
            raise RuntimeError("snapshot-series count does not match branch manifest")
        latest_shift = shift["latest_snapshot"]
        geometry = _aggregate_geometry(geometry_rows)
        dynamics = _aggregate_training_dynamics(dynamics_rows)

        row = {
            "branch_id": branch["branch_id"],
            "experiment_id": EXPERIMENT_ID,
            "profile_id": PROFILE_ID,
            "dataset": DATASET_ID,
            "seed": seed,
            "control_id": control_id,
            "weight_method": method,
            "advantage_baseline": baseline,
            "remoteness_scale": scale,
            "derived_exp_coefficient": 0.0 if scale is None else TAPER_LAMBDA / scale,
            "trainer_steps": STEPS,
            "evaluation_interval": EVAL_INTERVAL,
            "evaluation_episodes": EVAL_EPISODES,
            "batch_size": BATCH_SIZE,
            "actor_lr": ACTOR_LR,
            "critic_lr": CRITIC_LR,
            "early_window_mean_20k_200k": _mean(early),
            "middle_window_mean_220k_340k": _mean(middle),
            "late_window_mean_360k_500k": _mean(late),
            "late_window_std_360k_500k": _std(late),
            "trend_slope_per_100k_200k_500k": trend,
            "best_score": scores[best_index],
            "best_step": steps[best_index],
            "final_score_500k": scores[-1],
            "best_to_final_drop": scores[best_index] - scores[-1],
            "snapshot_stat_count": len(snapshot_rows),
            "latest_raw_advantage_mean": latest_shift.get("raw_mean"),
            "latest_raw_advantage_std": latest_shift.get("raw_std"),
            "latest_raw_positive_fraction": latest_shift.get("raw_positive_fraction"),
            "latest_effective_positive_fraction": latest_shift.get("effective_positive_fraction"),
            "latest_effective_negative_fraction": latest_shift.get("effective_negative_fraction"),
            "latest_crossed_positive_to_nonpositive_fraction": latest_shift.get(
                "crossed_positive_to_nonpositive_fraction"
            ),
            "latest_baseline_over_raw_std": latest_shift.get("baseline_over_raw_std"),
            "latest_raw_negative_abs_mass": latest_shift.get("raw_negative_abs_mass"),
            "latest_shifted_negative_abs_mass": latest_shift.get("shifted_negative_abs_mass"),
            "task_performance_collapse_event": "not_adjudicated_no_registered_threshold",
            "rollout_failure_event": False,
            **geometry,
            **dynamics,
        }
        rows.append(row)
        for step, score in zip(steps, scores, strict=True):
            curves.append(
                {
                    "branch_id": branch["branch_id"],
                    "seed": seed,
                    "control_id": control_id,
                    "advantage_baseline": baseline,
                    "remoteness_scale": scale,
                    "step": step,
                    "normalized_score": score,
                }
            )
        for item in snapshot_rows:
            snapshot_curves.append(
                {
                    "branch_id": branch["branch_id"],
                    "seed": seed,
                    "control_id": control_id,
                    "advantage_baseline": baseline,
                    "remoteness_scale": scale,
                    **item,
                }
            )
        for item in geometry_rows:
            geometry_curves.append(
                {
                    "branch_id": branch["branch_id"],
                    "seed": seed,
                    "control_id": control_id,
