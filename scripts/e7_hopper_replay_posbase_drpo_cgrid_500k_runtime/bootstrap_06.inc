                "snapshot_series_file": snapshots_path.name,
                "first_snapshot": self.shift_snapshot_stats[0],
                "latest_snapshot": latest,
            }
            return payload

    bootstrap.TrajectorySnapshotAdvantage = ShiftedBatchAwareProvider

    base_geometry = bootstrap.GeometryDiagnostics

    class NearFarGeometryDiagnostics(base_geometry):
        def _reset_interval(self) -> None:
            super()._reset_interval()
            self.near_negative_samples = 0
            self.far_negative_samples = 0
            self.near_raw_negative_mass = 0.0
            self.far_raw_negative_mass = 0.0
            self.near_effective_negative_mass = 0.0
            self.far_effective_negative_mass = 0.0

        def observe(
            self,
            advantage: torch.Tensor,
            distance: torch.Tensor,
            factor: torch.Tensor,
        ) -> None:
            with torch.no_grad():
                mass = _near_far_mass(advantage, distance, factor)
                self.near_negative_samples += int(mass["near_negative_samples"])
                self.far_negative_samples += int(mass["far_negative_samples"])
                self.near_raw_negative_mass += float(mass["near_raw_negative_mass"])
                self.far_raw_negative_mass += float(mass["far_raw_negative_mass"])
                self.near_effective_negative_mass += float(
                    mass["near_effective_negative_mass"]
                )
                self.far_effective_negative_mass += float(
                    mass["far_effective_negative_mass"]
                )
            super().observe(advantage, distance, factor)

        def _flush(self) -> None:
            count = self.negative_samples
            payload: dict[str, Any] = {
                "schema_version": 2,
                "status": "complete" if self.update_count == self.total_steps else "running",
                "update": self.update_count,
                "total_steps": self.total_steps,
                "interval_start_update": self.interval_start_update,
                "interval_end_update": self.update_count,
                "actor_update_mode": self.actor_update_mode,
                "weight_control": self.public_control,
                "negative_samples": count,
                "negative_distance_mean": self.distance_sum / count if count else None,
                "negative_weight_mean": self.factor_sum / count if count else None,
                "negative_abs_advantage_sum": self.abs_advantage_sum,
                "weighted_negative_abs_advantage_sum": self.weighted_abs_advantage_sum,
                "effective_negative_mass_fraction": (
                    self.weighted_abs_advantage_sum / self.abs_advantage_sum
                    if self.abs_advantage_sum > 0.0
                    else None
                ),
                "near_far_boundary": REFERENCE_DISTANCE,
                "near_negative_samples": self.near_negative_samples,
                "far_negative_samples": self.far_negative_samples,
                "near_raw_negative_mass": self.near_raw_negative_mass,
                "far_raw_negative_mass": self.far_raw_negative_mass,
                "near_effective_negative_mass": self.near_effective_negative_mass,
                "far_effective_negative_mass": self.far_effective_negative_mass,
                "near_effective_negative_mass_fraction": (
                    self.near_effective_negative_mass / self.near_raw_negative_mass
                    if self.near_raw_negative_mass > 0.0
                    else None
                ),
                "far_effective_negative_mass_fraction": (
                    self.far_effective_negative_mass / self.far_raw_negative_mass
                    if self.far_raw_negative_mass > 0.0
                    else None
                ),
                "sampled_negative_values": sum(
                    int(value.numel()) for value in self.distance_samples
                ),
