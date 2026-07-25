                        raise RuntimeError("actor output contract changed during audit")
                    _, log_std = output
                    log_std = log_std.detach().float()
                    finite = bool(torch.isfinite(log_std).all())
                    lower_fraction = float((log_std <= LOG_STD_LOWER_BOUND).float().mean().cpu())
                    upper_fraction = float((log_std >= LOG_STD_UPPER_BOUND).float().mean().cpu())
                    log_std_min = float(log_std.min().cpu())
                    log_std_max = float(log_std.max().cpu())
                if not finite or not math.isfinite(raw_gradient_norm) or not math.isfinite(update_norm):
                    raise FloatingPointError("non-finite actor training dynamics")
                payload = {
                    "schema_version": 1,
                    "status": "complete" if next_update == audit.total_steps else "running",
                    "update": next_update,
                    "total_steps": audit.total_steps,
                    "raw_actor_gradient_norm": raw_gradient_norm,
                    "adam_actor_update_norm": update_norm,
                    "log_std_min_on_update_batch": log_std_min,
                    "log_std_max_on_update_batch": log_std_max,
                    "lower_support_boundary_fraction": lower_fraction,
                    "upper_support_boundary_fraction": upper_fraction,
                    "support_or_variance_boundary_event": bool(
                        lower_fraction > 0.0 or upper_fraction > 0.0
                    ),
                    "nan_inf_numerical_failure": False,
                }
                _append_jsonl(audit.jsonl_path, payload)
                _atomic_json(audit.latest_path, payload)
                audit.records += 1
            return result

        target.__init__ = wrapped_init
        target.update = wrapped_update
        return {"installed": True}

    def validate_complete(self) -> dict[str, Any]:
        if self.update_count != self.total_steps:
            raise RuntimeError(
                f"actor training audit stopped at {self.update_count}, expected {self.total_steps}"
            )
        if self.optimizer_control is None or self.optimizer_control.get("applied_count") != 1:
            raise RuntimeError("optimizer audit was not applied exactly once")
        if not self.latest_path.is_file():
            raise RuntimeError("actor training dynamics latest record is missing")
        latest = json.loads(self.latest_path.read_text())
        if latest.get("status") != "complete" or int(latest.get("update", -1)) != self.total_steps:
            raise RuntimeError("actor training dynamics did not complete")
        return {
            "schema_version": 1,
            "jsonl": self.jsonl_path.name,
            "latest": self.latest_path.name,
            "records": self.records,
            "diagnostic_interval": self.interval,
            "optimizer_control": self.optimizer_control,
            "final": latest,
        }


def _near_far_mass(
    advantage: torch.Tensor,
    distance: torch.Tensor,
    factor: torch.Tensor,
) -> dict[str, float | int]:
    flat_advantage = advantage.detach().reshape(-1)
    flat_distance = distance.detach().reshape(-1)
    flat_factor = factor.detach().reshape(-1)
    negative = flat_advantage < 0
    abs_advantage = flat_advantage[negative].abs()
    negative_distance = flat_distance[negative]
    negative_factor = flat_factor[negative]
    near = negative_distance <= REFERENCE_DISTANCE
    far = ~near
    return {
        "near_negative_samples": int(near.sum().cpu()),
        "far_negative_samples": int(far.sum().cpu()),
        "near_raw_negative_mass": float(abs_advantage[near].sum().cpu()),
        "far_raw_negative_mass": float(abs_advantage[far].sum().cpu()),
        "near_effective_negative_mass": float(
            (abs_advantage[near] * negative_factor[near]).sum().cpu()
        ),
        "far_effective_negative_mass": float(
