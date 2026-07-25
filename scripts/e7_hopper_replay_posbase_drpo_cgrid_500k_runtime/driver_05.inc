def _aggregate_geometry(records: list[dict[str, Any]]) -> dict[str, Any]:
    final = records[-1]
    if final.get("status") != "complete" or int(final.get("update", -1)) != STEPS:
        raise RuntimeError("geometry diagnostics did not complete at 500k")
    sums = {
        "negative_samples": 0,
        "raw": 0.0,
        "weighted": 0.0,
        "near_raw": 0.0,
        "far_raw": 0.0,
        "near_weighted": 0.0,
        "far_weighted": 0.0,
    }
    for record in records:
        sums["negative_samples"] += int(record.get("negative_samples", 0))
        sums["raw"] += float(record.get("negative_abs_advantage_sum", 0.0))
        sums["weighted"] += float(record.get("weighted_negative_abs_advantage_sum", 0.0))
        sums["near_raw"] += float(record.get("near_raw_negative_mass", 0.0))
        sums["far_raw"] += float(record.get("far_raw_negative_mass", 0.0))
        sums["near_weighted"] += float(record.get("near_effective_negative_mass", 0.0))
        sums["far_weighted"] += float(record.get("far_effective_negative_mass", 0.0))
    return {
        "geometry_record_count": len(records),
        "negative_samples": sums["negative_samples"],
        "raw_shifted_negative_mass": sums["raw"],
        "effective_shifted_negative_mass": sums["weighted"],
        "effective_negative_mass_fraction": (
            sums["weighted"] / sums["raw"] if sums["raw"] > 0.0 else None
        ),
        "near_raw_shifted_negative_mass": sums["near_raw"],
        "far_raw_shifted_negative_mass": sums["far_raw"],
        "near_effective_shifted_negative_mass": sums["near_weighted"],
        "far_effective_shifted_negative_mass": sums["far_weighted"],
        "near_effective_negative_mass_fraction": (
            sums["near_weighted"] / sums["near_raw"] if sums["near_raw"] > 0.0 else None
        ),
        "far_effective_negative_mass_fraction": (
            sums["far_weighted"] / sums["far_raw"] if sums["far_raw"] > 0.0 else None
        ),
        "terminal_negative_distance_p50": final.get("negative_distance_p50"),
        "terminal_negative_distance_p90": final.get("negative_distance_p90"),
        "terminal_negative_weight_p50": final.get("negative_weight_p50"),
        "terminal_negative_weight_p90": final.get("negative_weight_p90"),
    }


def _aggregate_training_dynamics(records: list[dict[str, Any]]) -> dict[str, Any]:
    final = records[-1]
    if final.get("status") != "complete" or int(final.get("update", -1)) != STEPS:
        raise RuntimeError("actor training dynamics did not complete at 500k")
    gradients = [float(record["raw_actor_gradient_norm"]) for record in records]
    updates = [float(record["adam_actor_update_norm"]) for record in records]
    lower = [float(record["lower_support_boundary_fraction"]) for record in records]
    upper = [float(record["upper_support_boundary_fraction"]) for record in records]
    if not all(math.isfinite(value) for value in gradients + updates + lower + upper):
        raise RuntimeError("non-finite actor training dynamics in completed branch")
    return {
        "actor_training_dynamics_record_count": len(records),
        "raw_actor_gradient_norm_mean": _mean(gradients),
        "raw_actor_gradient_norm_median": _median(gradients),
        "raw_actor_gradient_norm_max": max(gradients),
        "raw_actor_gradient_norm_terminal": gradients[-1],
        "adam_actor_update_norm_mean": _mean(updates),
        "adam_actor_update_norm_median": _median(updates),
        "adam_actor_update_norm_max": max(updates),
        "adam_actor_update_norm_terminal": updates[-1],
        "lower_support_boundary_fraction_max": max(lower),
        "upper_support_boundary_fraction_max": max(upper),
        "support_or_variance_boundary_event": any(
            bool(record["support_or_variance_boundary_event"]) for record in records
        ),
        "nan_inf_numerical_failure": any(
            bool(record["nan_inf_numerical_failure"]) for record in records
        ),
    }


