    snapshots_path.unlink(missing_ok=True)

    branch = json.loads(branch_config_path.read_text(encoding="utf-8"))
    if branch.get("experiment_id") != EXPERIMENT_ID or branch.get("profile_id") != PROFILE_ID:
        raise ValueError("Hopper replay Positive-only/DRPO bootstrap identity mismatch")
    values = branch.get("template_values", {})
    batch_size = int(values.get("batch_size", -1))
    baseline = float(values.get("advantage_baseline", float("nan")))
    method = str(values.get("weight_method"))
    scale = float(values.get("remoteness_scale", float("nan")))
    if batch_size != EXPECTED_BATCH:
        raise ValueError(f"unsupported Positive-only/DRPO batch_size={batch_size}")
    if baseline not in ALLOWED_BASELINES:
        raise ValueError(f"unsupported fixed advantage baseline={baseline}")
    if method not in ALLOWED_METHODS:
        raise ValueError(f"unsupported method={method}")
    if method == "thresholded_exponential" and scale not in ALLOWED_SCALES:
        raise ValueError(f"unsupported DRPO remoteness_scale={scale}")
    if method == "positive_only" and scale != 1.0:
        raise ValueError("Positive-only must use the inert remoteness_scale=1 placeholder")

    original_flag_value = suite._flag_value  # noqa: SLF001

    def compatible_flag_value(tokens: list[str], flag: str) -> str:
        value = original_flag_value(tokens, flag)
        return "256" if flag == "--batch" else value

    suite._flag_value = compatible_flag_value  # type: ignore[assignment]  # noqa: SLF001

    original_provider = bootstrap.TrajectorySnapshotAdvantage

    class ShiftedBatchAwareProvider(original_provider):
        def __init__(self, replay: Any, estimator: str, batch_size_arg: int = 256) -> None:
            del batch_size_arg
            super().__init__(replay, estimator, batch_size=EXPECTED_BATCH)
            self.fixed_baseline = baseline
            self.shift_snapshot_stats: list[dict[str, Any]] = []

        def _refresh(self, agent: Any) -> None:
            super()._refresh(agent)
            if self.table is None:
                raise RuntimeError("trajectory snapshot table was not materialized")
            stats = _snapshot_stats(
                self.table,
                self.fixed_baseline,
                provider_update=self.update_count,
            )
            self.shift_snapshot_stats.append(stats)
            _append_jsonl(snapshots_path, stats)

        def __call__(self, agent: Any, context: Any, default: torch.Tensor) -> torch.Tensor:
            raw = super().__call__(agent, context, default)
            shifted = raw - self.fixed_baseline
            if shifted.shape != raw.shape or shifted.numel() != raw.numel():
                raise RuntimeError("advantage shift changed the sampled batch")
            return shifted

        def summary(self) -> dict[str, Any]:
            payload = super().summary()
            if not self.shift_snapshot_stats or self.table is None:
                raise RuntimeError("advantage-shift statistics were never recorded")
            latest = _snapshot_stats(
                self.table,
                self.fixed_baseline,
                provider_update=self.update_count,
                include_quantiles=True,
            )
            payload["advantage_shift"] = {
                "formula": "shifted_advantage = raw_trajectory_snapshot_gae - fixed_baseline",
                "actor_rule": (
                    "positive shifted advantages remain unattenuated; negative shifted advantages "
                    "are either zeroed by Positive-only or controlled by DRPO"
                ),
                "fixed_baseline": self.fixed_baseline,
                "baseline_units": "raw_trajectory_snapshot_gae",
                "shift_before_sign_split": True,
                "replay_sample_deletion": False,
                "replay_sample_masking": False,
                "snapshot_stat_count": len(self.shift_snapshot_stats),
